"""Module 4: NLLB translation service (CTranslate2) - backends, batching queue, chunking.

Test layers
-----------
* **Mock backend** - ``LexiconMockBackend`` is a MOCK (phrase/word lexicon), not a neural model. It lets
  the quality checks (key tokens + sacreBLEU) and the service logic run offline. Because the mock
  reproduces its own reference table, its BLEU of ~100 only proves the metric plumbing works.
* **CTranslate2 contract** - ``CTranslate2NLLBBackend`` driven by a fake translator/tokenizer (MOCKS)
  that follow the documented interfaces, to pin down the exact call arguments and post-processing.
* **Real model** - opt-in: ``NMT_TEST_MODEL_DIR=/path/to/nllb-ct2 pytest services/translation -k real``
  (needs the converted model - see ``scripts/convert_nllb.sh`` - and the Hugging Face tokenizer).
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
import sacrebleu

from services.translation.backends import CTranslate2NLLBBackend, LexiconMockBackend, NMTItem
from services.translation.chunking import pack_chunks, split_sentences
from services.translation.service import TranslationService, create_backend
from shared.config import Settings
from shared.languages import UnsupportedLanguage


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


SENTENCES = [
    "Hello, how are you?",
    "Good morning everyone.",
    "The meeting starts at nine o'clock.",
    "Please share your screen.",
    "Thank you for joining us today.",
    "Can you hear me?",
]
REFS = {
    "es": ["Hola, ¿cómo estás?", "Buenos días a todos.", "La reunión empieza a las nueve.",
           "Por favor, comparte tu pantalla.", "Gracias por acompañarnos hoy.", "¿Me oyes?"],
    "fr": ["Bonjour, comment allez-vous ?", "Bonjour à tous.", "La réunion commence à neuf heures.",
           "Veuillez partager votre écran.", "Merci de nous avoir rejoints aujourd'hui.", "Vous m'entendez ?"],
}
KEY_TOKENS = {
    "es": ["hola", "buenos", "reunión", "pantalla", "gracias", "oyes"],
    "fr": ["bonjour", "bonjour", "réunion", "écran", "merci", "entendez"],
}


class EchoBackend:
    """MOCK: returns '[tgt] text', counts words as tokens, records every batch."""

    is_mock = True

    def __init__(self, fail_first: int = 0, delay: float = 0.0):
        self.batches: list[list[NMTItem]] = []
        self.fail_first = fail_first
        self.delay = delay

    def count_tokens(self, text, src):
        return len(text.split())

    def translate_batch(self, items):
        import time

        self.batches.append(list(items))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError("backend exploded")
        if self.delay:
            time.sleep(self.delay)
        return [f"[{it.tgt}] {it.text}" for it in items]


@pytest.fixture
async def service():
    svc = TranslationService(settings(), LexiconMockBackend())
    yield svc
    await svc.aclose()


# =================================================== translation quality (mock backend)
@pytest.mark.parametrize("lang", ["es", "fr"])
async def test_english_to_spanish_and_french_key_tokens_and_bleu(service, lang):
    out = await asyncio.gather(*(service.translate(s, "en", lang) for s in SENTENCES))
    for text, token in zip(out, KEY_TOKENS[lang]):
        assert token in text.lower(), f"{token!r} missing in {text!r}"
    bleu = sacrebleu.corpus_bleu(out, [REFS[lang]]).score
    assert bleu >= 90.0  # circular for the mock (see module docstring); real-model threshold below


async def test_unknown_sentences_fall_back_to_word_lexicon_and_copy_unknown_words(service):
    assert await service.translate("Hello Maria", "en", "es") == "hola Maria"
    back = await service.translate("Buenos días a todos.", "es", "en")
    assert back == "Good morning everyone."


async def test_language_codes_are_normalised_and_validated(service):
    assert await service.translate("Can you hear me?", "en-US", "es-ES") == "¿Me oyes?"
    assert await service.translate("Can you hear me?", "eng_Latn", "fra_Latn") == "Vous m'entendez ?"
    with pytest.raises(UnsupportedLanguage):
        await service.translate("hello", "en", "klingon")
    with pytest.raises(UnsupportedLanguage):
        await service.translate("hello", "", "es")


async def test_same_language_and_empty_text_skip_the_model():
    backend = EchoBackend()
    svc = TranslationService(settings(), backend)
    assert await svc.translate("  keep me  ", "en", "en") == "keep me"
    assert await svc.translate("   ", "en", "es") == ""
    assert backend.batches == []
    await svc.aclose()


# =========================================================== batching and queueing
async def test_concurrent_requests_share_one_batch():
    backend = EchoBackend()
    svc = TranslationService(settings(nmt_batch_window_ms=50), backend)
    texts = [f"sentence number {i}" for i in range(10)]
    out = await asyncio.gather(*(svc.translate(t, "en", "es") for t in texts))
    assert out == [f"[spa_Latn] {t}" for t in texts]  # results returned to the right callers
    assert len(backend.batches) == 1 and len(backend.batches[0]) == 10
    assert svc.stats.batch_sizes == [10]
    await svc.aclose()


async def test_batch_size_limit_splits_large_bursts():
    backend = EchoBackend()
    svc = TranslationService(settings(nmt_batch_window_ms=100, nmt_max_batch_size=4), backend)
    out = await asyncio.gather(*(svc.translate(f"text {i}", "en", "fr") for i in range(10)))
    assert out == [f"[fra_Latn] text {i}" for i in range(10)]
    sizes = [len(b) for b in backend.batches]
    assert sum(sizes) == 10 and max(sizes) <= 4 and len(sizes) >= 3
    await svc.aclose()


async def test_mixed_language_pairs_are_batched_together():
    backend = EchoBackend()
    svc = TranslationService(settings(nmt_batch_window_ms=50), backend)
    jobs = [("hi", "en", "es"), ("hi", "en", "fr"), ("hallo", "de", "en"), ("hola", "es", "de")]
    out = await asyncio.gather(*(svc.translate(*j) for j in jobs))
    assert out == ["[spa_Latn] hi", "[fra_Latn] hi", "[eng_Latn] hallo", "[deu_Latn] hola"]
    assert len(backend.batches) == 1
    assert {(i.src, i.tgt) for i in backend.batches[0]} == {
        ("eng_Latn", "spa_Latn"), ("eng_Latn", "fra_Latn"), ("deu_Latn", "eng_Latn"), ("spa_Latn", "deu_Latn")}
    await svc.aclose()


async def test_identical_requests_in_a_batch_run_once_and_repeats_hit_the_cache():
    backend = EchoBackend()
    svc = TranslationService(settings(nmt_batch_window_ms=50), backend)
    a = await asyncio.gather(*(svc.translate("same text", "en", "es") for _ in range(5)))
    assert a == ["[spa_Latn] same text"] * 5
    assert sum(len(b) for b in backend.batches) == 1  # deduplicated
    await svc.translate("same text", "en", "es")
    assert len(backend.batches) == 1 and svc.stats.cache_hits == 1  # served from cache
    await svc.translate("same text", "en", "fr")  # different target -> real call
    assert len(backend.batches) == 2
    await svc.aclose()


async def test_cache_is_bounded():
    backend = EchoBackend()
    svc = TranslationService(settings(), backend, cache_size=3)
    for i in range(6):
        await svc.translate(f"t{i}", "en", "es")
    assert len(svc._cache) == 3
    await svc.translate("t0", "en", "es")  # evicted earlier -> backend again
    assert sum(len(b) for b in backend.batches) == 7
    await svc.aclose()


async def test_translate_segment_fans_out_to_all_listener_languages_in_one_batch():
    backend = EchoBackend()
    svc = TranslationService(settings(nmt_batch_window_ms=50), backend)
    res = await svc.translate_segment("alice-1", "Good morning", "en", ["es", "fr", "es", "de", "en"], is_final=True)
    assert [r.target_lang for r in res] == ["es", "fr", "de", "en"]  # de-duplicated, order kept
    assert res[0].text == "[spa_Latn] Good morning" and res[3].text == "Good morning"  # en->en skipped
    assert all(r.segment_id == "alice-1" and r.is_final and r.source_lang == "en" and r.latency_ms >= 0 for r in res)
    assert len(backend.batches) == 1 and len(backend.batches[0]) == 3
    await svc.aclose()


async def test_backend_failure_fails_only_that_batch_and_service_recovers():
    backend = EchoBackend(fail_first=1)
    svc = TranslationService(settings(), backend)
    with pytest.raises(RuntimeError, match="exploded"):
        await svc.translate("first", "en", "es")
    assert await svc.translate("second", "en", "es") == "[spa_Latn] second"
    await svc.aclose()


async def test_backend_returning_wrong_number_of_results_is_an_error():
    class Bad(EchoBackend):
        def translate_batch(self, items):
            return ["only one"]

    svc = TranslationService(settings(nmt_batch_window_ms=30), Bad())
    res = await asyncio.gather(svc.translate("a", "en", "es"), svc.translate("b", "en", "es"), return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in res)
    await svc.aclose()


async def test_closed_service_rejects_requests_and_fails_pending():
    svc = TranslationService(settings(nmt_batch_window_ms=500), EchoBackend(delay=0.2))
    task = asyncio.create_task(svc.translate("x", "en", "es"))
    await asyncio.sleep(0.01)
    await svc.aclose()
    with pytest.raises(Exception):
        await asyncio.wait_for(task, 2)
    with pytest.raises(RuntimeError, match="closed"):
        await svc.translate("y", "en", "es")


async def test_event_loop_stays_responsive_during_slow_translation():
    svc = TranslationService(settings(), EchoBackend(delay=0.3))
    task = asyncio.create_task(svc.translate("slow one", "en", "es"))
    ticks = 0
    while not task.done():
        await asyncio.sleep(0.02)
        ticks += 1
    assert ticks >= 8  # the model runs in a worker thread, the loop kept ticking
    await svc.aclose()


# ======================================================================== chunking
def test_split_sentences_latin_and_cjk():
    assert split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]
    assert split_sentences("你好。今天开会！好吗？") == ["你好。", "今天开会！", "好吗？"]
    assert split_sentences("   ") == []


def test_pack_chunks_respects_token_budget_and_preserves_order():
    count = lambda t: len(t.split())
    text = "Aa bb cc. Dd ee ff. Gg hh ii. Jj kk ll."
    chunks = pack_chunks(text, count, 6)
    assert chunks == ["Aa bb cc. Dd ee ff.", "Gg hh ii. Jj kk ll."]
    assert " ".join(chunks) == text
    assert pack_chunks("short text", count, 6) == ["short text"]
    assert pack_chunks("", count, 6) == []


def test_pack_chunks_splits_a_single_overlong_sentence_by_words():
    count = lambda t: len(t.split())
    chunks = pack_chunks("one two three four five six seven eight nine", count, 4)
    assert all(count(c) <= 4 for c in chunks) and " ".join(chunks) == "one two three four five six seven eight nine"
    assert pack_chunks("unbreakable", count, 0) == ["unbreakable"]  # last resort: left whole


async def test_long_input_is_chunked_translated_and_rejoined_in_order():
    backend = EchoBackend()
    svc = TranslationService(settings(nmt_max_input_tokens=12, nmt_batch_window_ms=30), backend)  # budget = 8 words
    text = "Aa bb cc dd. Ee ff gg hh. Ii jj kk ll. Mm nn oo pp."
    out = await svc.translate(text, "en", "es")
    items = [i.text for b in backend.batches for i in b]
    assert len(items) == 2 and all(len(t.split()) <= 8 for t in items)
    assert out == "[spa_Latn] Aa bb cc dd. Ee ff gg hh. [spa_Latn] Ii jj kk ll. Mm nn oo pp."
    out_zh = await svc.translate(text, "en", "zh")  # CJK targets are joined without a space
    assert "[zho_Hans] Aa bb cc dd. Ee ff gg hh.[zho_Hans] Ii" in out_zh
    await svc.aclose()


# ============================================================ CTranslate2 backend contract
class FakeTokenizer:
    """MOCK of the HF NLLB tokenizer interface used by the CTranslate2 guide."""

    def __init__(self):
        self.src_lang = None
        self.seen_src: list[str] = []
        self.decoded: list[list[int]] = []

    def encode(self, text):
        self.seen_src.append(self.src_lang)
        return [10 + i for i, _ in enumerate(text.split())]

    def convert_ids_to_tokens(self, ids):
        return [f"▁t{i}" for i in ids]

    def convert_tokens_to_ids(self, tokens):
        return [int(t.lstrip("▁t") or 0) if t.lstrip("▁t").isdigit() else 99 for t in tokens]

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is True
        self.decoded.append(list(ids))
        return " " + " ".join(f"w{i}" for i in ids) + " "


class FakeTranslator:
    """MOCK of ctranslate2.Translator: echoes the source tokens after the forced target prefix."""

    def __init__(self):
        self.calls: list[dict] = []

    def translate_batch(self, source, target_prefix=None, **kw):
        self.calls.append(dict(source=source, target_prefix=target_prefix, **kw))
        return [SimpleNamespace(hypotheses=[[pre[0]] + src[1:]]) for src, pre in zip(source, target_prefix)]


def test_ct2_backend_follows_the_documented_recipe():
    tok, tr = FakeTokenizer(), FakeTranslator()
    backend = CTranslate2NLLBBackend(settings(nmt_beam_size=3, nmt_max_batch_size=8, nmt_max_input_tokens=100), tr, tok)
    items = [NMTItem("a b c", "eng_Latn", "spa_Latn"), NMTItem("d e", "deu_Latn", "fra_Latn")]
    out = backend.translate_batch(items)
    call = tr.calls[0]
    assert tok.seen_src == ["eng_Latn", "deu_Latn"]  # src_lang set per item before encoding
    assert call["target_prefix"] == [["spa_Latn"], ["fra_Latn"]]  # target language forced as prefix
    assert call["beam_size"] == 3 and call["max_batch_size"] == 8
    assert call["max_input_length"] == 100 and call["max_decoding_length"] == 256
    assert call["source"][0] == ["▁t10", "▁t11", "▁t12"]
    # first hypothesis token (the language code) is dropped, then ids are decoded and stripped
    assert tok.decoded[0] == [11, 12] and out[0] == "w11 w12"
    assert out[1] == "w11"  # "d e" -> source tokens [t10, t11]; hypothesis keeps [t11] after the prefix
    assert backend.translate_batch([]) == []
    assert backend.count_tokens("x y", "eng_Latn") == 2 and tok.src_lang == "eng_Latn"


def test_ct2_backend_keeps_tokens_when_no_language_prefix_is_returned():
    class NoPrefix(FakeTranslator):
        def translate_batch(self, source, target_prefix=None, **kw):
            return [SimpleNamespace(hypotheses=[["▁t5", "▁t6"]])]

    tok = FakeTokenizer()
    out = CTranslate2NLLBBackend(settings(), NoPrefix(), tok).translate_batch([NMTItem("x", "eng_Latn", "spa_Latn")])
    assert tok.decoded[0] == [5, 6] and out == ["w5 w6"]


async def test_service_works_through_the_ct2_backend_wrapper():
    backend = CTranslate2NLLBBackend(settings(), FakeTranslator(), FakeTokenizer())
    svc = TranslationService(settings(nmt_batch_window_ms=30), backend)
    res = await asyncio.gather(svc.translate("a b c", "en", "es"), svc.translate("d e f g", "de", "fr"))
    assert all(r.startswith("w") for r in res)
    await svc.aclose()


def test_missing_model_directory_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="convert_nllb.sh"):
        CTranslate2NLLBBackend(settings(nmt_model_dir=tmp_path / "nope"))


def test_create_backend_selects_mock_or_real(tmp_path):
    assert isinstance(create_backend(settings(backend_mode="mock")), LexiconMockBackend)
    assert create_backend(settings(backend_mode="mock")).is_mock is True
    with pytest.raises(FileNotFoundError):
        create_backend(settings(backend_mode="real", nmt_model_dir=tmp_path / "missing"))


# ============================================================ real model (opt-in)
REAL_DIR = os.environ.get("NMT_TEST_MODEL_DIR")


@pytest.mark.slow
@pytest.mark.skipif(not REAL_DIR, reason="set NMT_TEST_MODEL_DIR to a CTranslate2 NLLB model to run")
@pytest.mark.parametrize("lang", ["es", "fr"])
async def test_real_nllb_translation_quality(lang):
    svc = TranslationService(settings(nmt_model_dir=REAL_DIR), CTranslate2NLLBBackend(settings(nmt_model_dir=REAL_DIR)))
    out = await asyncio.gather(*(svc.translate(s, "en", lang) for s in SENTENCES))
    bleu = sacrebleu.corpus_bleu(out, [REFS[lang]]).score
    assert bleu >= 25.0, f"BLEU {bleu:.1f}: {out}"
    await svc.aclose()


# ============================================================ mock backend itself
def test_lexicon_mock_backend_is_labelled_and_records_batches():
    backend = LexiconMockBackend()
    assert backend.is_mock is True
    out = backend.translate_batch([NMTItem("Can you hear me?", "eng_Latn", "spa_Latn"),
                                   NMTItem("whatever", "eng_Latn", "jpn_Jpan")])
    assert out == ["¿Me oyes?", "whatever"]  # unsupported direction: pass-through
    assert len(backend.batches) == 1
