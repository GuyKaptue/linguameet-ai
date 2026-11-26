"""Module 3: streaming ASR (LocalAgreement-n over faster-whisper).

Test layers
-----------
* **Pure logic** - ``HypothesisBuffer`` with hand-written hypotheses.
* **Streaming logic with a MOCK backend** - ``ReplayASRBackend`` does NOT recognise speech; it replays
  a scripted transcript (synthetic word timings) over real recordings so that buffering, commit,
  trimming, prompting, language locking and the async service can be tested deterministically.
* **Backend contract** - ``FasterWhisperBackend`` driven by a fake ``WhisperModel`` (MOCK) to check the
  exact faster-whisper call arguments and generator handling.
* **Real model** - skipped unless ``ASR_TEST_MODEL`` points to a faster-whisper model (name or CT2
  directory). It transcribes real English / French / Chinese recordings and checks keywords and
  language codes. Weights cannot be downloaded in the offline CI sandbox, so this is opt-in:
  ``ASR_TEST_MODEL=tiny pytest services/asr -k real``.
"""

from __future__ import annotations

import asyncio
import os
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gateway.core import GatewayCore
from services.asr.backends import ASRResult, FasterWhisperBackend, ReplayASRBackend
from services.asr.hypothesis import HypothesisBuffer, Word, join_words
from services.asr.service import ASRService
from services.asr.streaming import StreamingASR
from services.vad.streaming_vad import SpeechSegmenter, StreamingVAD
from shared.audio import pcm16_to_float32
from shared.config import Settings
from shared.types import Transcript

SR = 16000
DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
JFK_TEXT = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."


def load(name: str) -> np.ndarray:
    with wave.open(str(DATA / name)) as w:
        return pcm16_to_float32(w.readframes(w.getnframes()))


def norm(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())


def W(text: str, start: float, end: float) -> Word:
    return Word(start, end, text)


def words_of(sentence: str, start: float = 0.2, step: float = 0.45) -> list[Word]:
    """Scripted words with SYNTHETIC timings (not alignments of the audio)."""
    return [Word(start + i * step, start + i * step + 0.4, " " + w) for i, w in enumerate(sentence.split())]


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def stream(asr: StreamingASR, audio: np.ndarray, chunk_s: float = 0.5) -> list[Transcript]:
    """Feed audio like a live stream, running a pass whenever the ASR asks for one."""
    out: list[Transcript] = []
    step = int(chunk_s * SR)
    for i in range(0, len(audio), step):
        asr.accept(audio[i : i + step])
        if asr.should_run():
            tr = asr.process_iter()
            if tr:
                out.append(tr)
    out.append(asr.finish())
    return out


# ======================================================== HypothesisBuffer (pure)
def hyp(buf: HypothesisBuffer, words, offset=0.0):
    buf.insert(words, offset)
    return buf.flush()


def test_first_hypothesis_commits_nothing_then_agreement_commits_prefix():
    b = HypothesisBuffer(2)
    assert hyp(b, [W(" the", 0, 0.3), W(" cat", 0.3, 0.6)]) == []
    out = hyp(b, [W(" the", 0, 0.3), W(" cat", 0.3, 0.6), W(" sat", 0.6, 0.9)])
    assert join_words(out) == "the cat"  # "sat" has only been seen once
    assert join_words(b.unconfirmed()) == "sat"
    out = hyp(b, [W(" sat", 0.6, 0.9), W(" down", 0.9, 1.2)])
    assert join_words(out) == "sat" and b.last_committed_time == pytest.approx(0.9)


def test_disagreement_holds_words_back_and_comparison_ignores_case_and_punctuation():
    b = HypothesisBuffer(2)
    hyp(b, [W(" Hello,", 0, 0.4), W(" word", 0.4, 0.8)])
    out = hyp(b, [W(" hello", 0, 0.4), W(" world", 0.4, 0.8)])
    assert join_words(out) == "hello"  # 'word' != 'world' so only the first word agrees
    assert join_words(b.unconfirmed()) == "world"


def test_already_committed_words_are_removed_from_later_hypotheses():
    b = HypothesisBuffer(2)
    hyp(b, [W(" a", 0, 0.3), W(" b", 0.3, 0.6)])
    hyp(b, [W(" a", 0, 0.3), W(" b", 0.3, 0.6), W(" c", 0.6, 0.9)])  # commits a b
    # buffer restarts at t=0.6 in a new pass; Whisper re-emits "b" at the start (overlap)
    out = hyp(b, [W(" b", 0.0, 0.3), W(" c", 0.3, 0.6), W(" d", 0.6, 0.9)], offset=0.3)
    assert join_words(out) == "c"  # 'b' dropped by timestamp / n-gram overlap, 'c' confirmed, 'd' pending


def test_offset_shifts_to_absolute_time():
    b = HypothesisBuffer(1)
    out = hyp(b, [W(" x", 0.0, 0.5)], offset=10.0)
    assert out == [Word(10.0, 10.5, " x")]


def test_n1_commits_immediately_and_n3_needs_two_agreeing_predecessors():
    b1 = HypothesisBuffer(1)
    assert join_words(hyp(b1, words_of("one two"))) == "one two"
    b3 = HypothesisBuffer(3)
    w = words_of("one two")
    assert hyp(b3, w) == []
    assert hyp(b3, w) == []  # only 2 hypotheses so far
    assert join_words(hyp(b3, w)) == "one two"
    with pytest.raises(ValueError):
        HypothesisBuffer(0)


def test_complete_accepts_the_tail_and_reset_clears_everything():
    b = HypothesisBuffer(2)
    hyp(b, words_of("alpha beta"))
    assert join_words(b.complete()) == "alpha beta"
    assert b.unconfirmed() == []
    b.reset()
    assert b.last_committed_time == 0.0


# ================================================== StreamingASR with replay MOCK
@pytest.fixture(scope="module")
def jfk() -> np.ndarray:
    return load("jfk.wav")


def jfk_backend(jfk, **kw) -> ReplayASRBackend:
    # spread the 22 words over the clip (synthetic timings): 0.45 s per word starting at 0.2 s
    return ReplayASRBackend(jfk, words_of(JFK_TEXT), language="en", **kw)


def test_streaming_produces_growing_stable_partials_and_exact_final(jfk):
    asr = StreamingASR(jfk_backend(jfk), min_chunk_s=1.0)
    outs = stream(asr, jfk)
    *partials, final = outs
    assert final.is_final and norm(final.text) == norm(JFK_TEXT)
    assert final.unconfirmed == "" and final.language == "en"
    assert len(partials) >= 5
    # confirmed text only ever grows (stability: committed words never change)
    texts = [p.text for p in partials] + [final.text]
    assert all(b.startswith(a) for a, b in zip(texts, texts[1:]))
    # low latency: words are confirmed long before the utterance ends
    first_confirmed = next(i for i, p in enumerate(partials) if p.text)
    assert first_confirmed <= 3
    mid = partials[len(partials) // 2]
    assert 0 < len(mid.text) < len(final.text)
    assert all(p.compute_ms >= 0 for p in outs)


def test_unstable_last_word_is_never_committed_prematurely(jfk):
    backend = jfk_backend(jfk, flaky_last_word=True)
    outs = stream(StreamingASR(backend), jfk)
    *partials, final = outs
    # while streaming, the flaky word is only ever in `unconfirmed`, never in the confirmed text
    assert all("xxx" not in p.text for p in partials)
    assert any("xxx" in p.unconfirmed for p in partials)
    # finish() deliberately accepts the last hypothesis as-is, so only the flaky tail may differ
    assert norm(final.text).split()[:-1] == norm(JFK_TEXT).split()[:-1]


def test_buffer_is_trimmed_at_sentence_ends_and_prompt_carries_context():
    rng = np.random.default_rng(1)
    jfk = load("jfk.wav")
    gap = (rng.normal(0, 0.003, SR).astype(np.float32))  # unique, non-silent separators
    audio = np.concatenate([jfk, gap, jfk, gap, jfk])
    period = len(jfk) / SR + 1.0
    script: list[Word] = []
    for k in range(3):
        script += [Word(w.start + k * period, w.end + k * period, w.text) for w in words_of(JFK_TEXT)]
    backend = ReplayASRBackend(audio, script, language="en")
    asr = StreamingASR(backend, buffer_max_s=30.0)
    max_buf = 0.0
    out = []
    for i in range(0, len(audio), SR // 2):
        asr.accept(audio[i : i + SR // 2])
        max_buf = max(max_buf, asr.buffer_seconds)
        if asr.should_run():
            out.append(asr.process_iter())
    final = asr.finish()
    assert asr.trims >= 2  # trimmed after each completed sentence
    assert max_buf <= 16.0  # buffer never grows with the utterance (35 s of audio in total)
    assert norm(final.text) == norm(" ".join([JFK_TEXT] * 3))
    prompts = [c.prompt for c in backend.calls]
    assert prompts[0] == "" and any("Americans" in p for p in prompts[1:])
    assert all(len(p) <= 200 for p in prompts)


def test_hard_cap_drops_oldest_audio_when_nothing_commits():
    audio = np.random.default_rng(2).normal(0, 0.01, 40 * SR).astype(np.float32)
    backend = ReplayASRBackend(audio, [], language="en")  # never returns words
    asr = StreamingASR(backend, buffer_max_s=10.0, min_chunk_s=1.0)
    for i in range(0, len(audio), SR):
        asr.accept(audio[i : i + SR])
        if asr.should_run():
            asr.process_iter()
        assert asr.buffer_seconds <= 10.0 + 1.0
    assert asr.trims > 0


# ---------------------------------------------------------- language handling
def test_language_is_detected_first_then_locked(jfk):
    backend = jfk_backend(jfk)
    asr = StreamingASR(backend)
    outs = stream(asr, jfk)
    assert backend.calls[0].language_arg is None  # detection pass
    assert all(c.language_arg == "en" for c in backend.calls[1:])  # locked afterwards
    assert asr.language_locked and outs[-1].language == "en"
    assert outs[-1].language_probability == pytest.approx(0.98)


def test_low_confidence_detection_is_retried_then_locked(jfk):
    backend = jfk_backend(jfk, low_confidence_calls=1)
    asr = StreamingASR(backend, lang_confidence=0.6)
    stream(asr, jfk)
    assert [c.language_arg for c in backend.calls[:3]] == [None, None, "en"]


def test_detection_gives_up_after_max_attempts(jfk):
    backend = jfk_backend(jfk, low_confidence_calls=50)
    asr = StreamingASR(backend, max_lang_attempts=3)
    stream(asr, jfk)
    assert [c.language_arg for c in backend.calls[:4]] == [None, None, None, "en"]
    assert asr.language == "en"  # best (even if weak) detection is used


def test_language_hint_skips_detection(jfk):
    backend = jfk_backend(jfk)
    outs = stream(StreamingASR(backend, language_hint="de"), jfk)
    assert all(c.language_arg == "de" for c in backend.calls)
    assert outs[-1].language == "de"


# --------------------------------------------------------------- validation
def test_streaming_asr_input_validation_and_edges():
    asr = StreamingASR(ReplayASRBackend(np.zeros(10, dtype=np.float32), []), min_chunk_s=1.0)
    assert asr.process_iter() is None and not asr.should_run()
    with pytest.raises(ValueError):
        asr.accept(np.zeros((2, 2), dtype=np.float32))
    asr.accept(np.zeros(SR // 2, dtype=np.float32))
    assert asr.pending_seconds == pytest.approx(0.5) and not asr.should_run()
    asr.accept(np.zeros(SR // 2, dtype=np.float32))
    assert asr.should_run()
    final = asr.finish()
    assert final.is_final and final.text == "" and final.language in ("en", "und")
    with pytest.raises(RuntimeError):
        asr.accept(np.zeros(10, dtype=np.float32))
    with pytest.raises(RuntimeError):
        asr.finish()
    empty = StreamingASR(ReplayASRBackend(np.zeros(10, dtype=np.float32), []))
    f = empty.finish()  # finishing without any audio is fine
    assert f.is_final and f.text == "" and f.language == "und"


# ======================================================== ASRService (async)
def segment_script(reference: np.ndarray, text: str):
    """Spread ``text`` over the VAD segments of ``reference`` (SYNTHETIC timings inside each
    detected speech span, so no word straddles a segment boundary)."""
    seg = SpeechSegmenter(StreamingVAD(), "p")
    segs = seg.feed(reference) + seg.flush()
    words = text.split()
    total = sum(s.duration for s in segs)
    counts = [max(1, round(len(words) * s.duration / total)) for s in segs]
    counts[-1] += len(words) - sum(counts)
    script, i = [], 0
    for s, n in zip(segs, counts):
        span = (s.end_time - s.start_time) - 0.1
        for j in range(n):
            a = s.start_time + 0.05 + span * j / n
            script.append(Word(a, a + span / n * 0.9, " " + words[i]))
            i += 1
    return script, segs


async def run_service(audio, backend, frame=160, **svc_kw):
    got: list[Transcript] = []
    asr = ASRService(settings(), backend, got.append, **svc_kw)
    core = GatewayCore(settings(), asr.on_gate_event)
    pcm = (audio * 32768).astype(np.int16)
    for i in range(0, len(pcm), frame):
        await core.push_pcm16("jfk", pcm[i : i + frame], SR)
    await core.remove_participant("jfk")
    await asr.drain()
    await asr.aclose()
    return got, asr


async def test_service_end_to_end_with_real_vad_gate(jfk):
    ref = pcm16_to_float32((jfk * 32768).astype(np.int16))  # exactly what the gateway will see
    script, segs = segment_script(ref, JFK_TEXT)
    backend = ReplayASRBackend(ref, script, language="en")
    got, asr = await run_service(ref, backend)
    finals = [t for t in got if t.is_final]
    assert [t.segment_id for t in finals] == [f"jfk-{i + 1}" for i in range(len(segs))]
    assert norm(" ".join(t.text for t in finals)) == norm(JFK_TEXT)
    assert all(t.language == "en" for t in finals)
    assert asr.stats.finals == len(segs) and len(got) >= len(finals)
    # partials precede their final, never the other way round
    for f in finals:
        idx = [i for i, t in enumerate(got) if t.segment_id == f.segment_id]
        assert got[idx[-1]] is f


async def test_service_follows_code_switching_between_utterances():
    en, fr, zh = load("speech_en.wav"), load("speech_fr.wav"), load("speech_zh.wav")
    sil = np.zeros(SR, dtype=np.float32)
    audio = np.concatenate([en, sil, fr, sil, zh, sil])
    audio = pcm16_to_float32((audio * 32768).astype(np.int16))
    # scripted placeholder words per language region; the mock does not recognise the audio
    seg = SpeechSegmenter(StreamingVAD(), "p")
    segs = seg.feed(audio) + seg.flush()
    # boundaries in the middle of the 1 s silences (segments start slightly early because of the VAD pre-roll)
    t_fr, t_zh = len(en) / SR + 0.5, (len(en) + len(fr)) / SR + 1.5
    regions = [(0, t_fr, "en"), (t_fr, t_zh, "fr"), (t_zh, 1e9, "zh")]
    script = []
    for s in segs:
        lang = next(l for a, b, l in regions if a <= s.start_time + 0.01 < b)
        n = 2
        step = (s.end_time - s.start_time - 0.1) / n
        script += [Word(s.start_time + 0.05 + k * step, s.start_time + 0.05 + (k + 1) * step * 0.9, f" {lang}{k}") for k in range(n)]
    backend = ReplayASRBackend(audio, script, language="en", regions=regions)
    got, _ = await run_service(audio, backend)
    finals = [t for t in got if t.is_final]
    order = []
    for f in finals:
        if f.language not in order:
            order.append(f.language)
    assert order == ["en", "fr", "zh"]  # each utterance detected on its own
    assert all(f.text.split()[0].startswith(f.language) for f in finals if f.text)


async def test_service_never_blocks_audio_intake_on_slow_asr(jfk):
    ref = pcm16_to_float32((jfk * 32768).astype(np.int16))
    script, _ = segment_script(ref, JFK_TEXT)
    inner = ReplayASRBackend(ref, script, language="en")

    class SlowBackend:  # MOCK: simulates a slow model
        def transcribe(self, audio, *, language, prompt=""):
            time.sleep(0.25)
            return inner.transcribe(audio, language=language, prompt=prompt)

    got: list[Transcript] = []
    svc = ASRService(settings(), SlowBackend(), got.append)
    core = GatewayCore(settings(), svc.on_gate_event)
    pcm = (ref[: 4 * SR] * 32768).astype(np.int16)
    worst = 0.0
    for i in range(0, len(pcm), 320):
        t0 = time.perf_counter()
        await core.push_pcm16("jfk", pcm[i : i + 320], SR)
        worst = max(worst, time.perf_counter() - t0)
    await core.remove_participant("jfk")
    assert worst < 0.15  # intake keeps up while transcription runs in a worker thread
    await svc.drain()
    await svc.aclose()
    assert any(t.is_final for t in got)


async def test_service_language_hint_and_backend_error_isolation(jfk):
    ref = pcm16_to_float32((jfk * 32768).astype(np.int16))
    script, _ = segment_script(ref, JFK_TEXT)
    inner = ReplayASRBackend(ref, script, language="en")
    svc_hint = ASRService(settings(), inner, lambda t: None)
    svc_hint.set_language_hint("jfk", "de-DE")
    assert svc_hint._hints["jfk"] == "de"
    svc_hint.set_language_hint("jfk", None)
    assert "jfk" not in svc_hint._hints
    await svc_hint.aclose()

    class Exploding:
        def transcribe(self, audio, *, language, prompt=""):
            raise RuntimeError("model crashed")

    got: list[Transcript] = []
    svc = ASRService(settings(), Exploding(), got.append)
    core = GatewayCore(settings(), svc.on_gate_event)
    pcm = (ref[: 3 * SR] * 32768).astype(np.int16)
    for i in range(0, len(pcm), 1600):
        await core.push_pcm16("jfk", pcm[i : i + 1600], SR)  # must not raise
    await core.remove_participant("jfk")
    await svc.drain()
    await svc.aclose()
    assert got == []  # failure is logged, intake survives


# ===================================================== FasterWhisperBackend contract
class FakeWhisperModel:
    """MOCK of faster_whisper.WhisperModel with the documented return shape."""

    def __init__(self, language="fr", prob=0.91):
        self.calls: list[dict] = []
        self._info = SimpleNamespace(language=language, language_probability=prob)

    def transcribe(self, audio, **kwargs):
        self.calls.append(dict(audio=audio, **kwargs))

        def gen():  # lazy generator, like the real one
            yield SimpleNamespace(words=[SimpleNamespace(start=0.0, end=0.4, word=" Bonjour", probability=0.9)])
            yield SimpleNamespace(words=None)  # a segment without word timestamps
            yield SimpleNamespace(words=[SimpleNamespace(start=0.5, end=0.9, word=" monde", probability=0.8)])

        return gen(), self._info


def test_faster_whisper_backend_maps_arguments_and_consumes_generator():
    model = FakeWhisperModel()
    backend = FasterWhisperBackend(settings(asr_beam_size=3), model=model)
    audio = (np.arange(1600) % 100).astype(np.float64)  # wrong dtype on purpose
    res = backend.transcribe(audio, language=None, prompt="")
    call = model.calls[0]
    assert call["audio"].dtype == np.float32 and call["audio"].flags["C_CONTIGUOUS"]
    assert call["language"] is None and call["beam_size"] == 3
    assert call["word_timestamps"] is True and call["vad_filter"] is False
    assert call["condition_on_previous_text"] is False and call["initial_prompt"] is None
    assert res.language == "fr" and res.language_probability == pytest.approx(0.91)
    assert [w.text for w in res.words] == [" Bonjour", " monde"]
    assert res.words[1] == Word(0.5, 0.9, " monde")
    backend.transcribe(audio, language="fr", prompt="Bonjour le")
    assert model.calls[1]["language"] == "fr" and model.calls[1]["initial_prompt"] == "Bonjour le"


def test_faster_whisper_backend_keeps_unknown_language_codes():
    backend = FasterWhisperBackend(settings(), model=FakeWhisperModel(language="yue", prob=0.7))
    assert backend.transcribe(np.zeros(160, dtype=np.float32), language=None).language == "yue"


def test_streaming_asr_works_through_the_faster_whisper_wrapper(jfk):
    """Wrapper + streaming together: the fake model reveals one more word per call."""

    class Growing(FakeWhisperModel):
        def transcribe(self, audio, **kw):
            self.calls.append(dict(audio=audio, **kw))
            n = min(len(JFK_TEXT.split()), 2 + 2 * len(self.calls))
            words = [SimpleNamespace(start=w.start, end=w.end, word=w.text, probability=0.9) for w in words_of(JFK_TEXT)[:n]]
            return iter([SimpleNamespace(words=words)]), SimpleNamespace(language="en", language_probability=0.99)

    asr = StreamingASR(FasterWhisperBackend(settings(), model=Growing()))
    outs = stream(asr, jfk[: 6 * SR])
    assert outs[-1].is_final and outs[-1].text.startswith("And so my fellow")


# ============================================================ real model (opt-in)
REAL = os.environ.get("ASR_TEST_MODEL")


@pytest.mark.slow
@pytest.mark.skipif(not REAL, reason="set ASR_TEST_MODEL (faster-whisper model name or CT2 dir) to run")
@pytest.mark.parametrize(
    "file,lang,keywords",
    [("jfk.wav", "en", ["country", "ask"]), ("speech_fr.wav", "fr", []), ("speech_zh.wav", "zh", []), ("speech_en.wav", "en", [])],
)
def test_real_whisper_detects_language_and_transcribes(file, lang, keywords):
    backend = FasterWhisperBackend(settings(asr_model_size=REAL, asr_beam_size=1))
    audio = load(file)
    asr = StreamingASR(backend, language_hint=None, min_chunk_s=1.0)
    outs = stream(asr, audio)
    final = outs[-1]
    assert final.language == lang, f"detected {final.language} for {file}"
    text = norm(final.text)
    assert text, "empty transcript"
    for kw in keywords:
        assert kw in text, f"{kw!r} missing in {text!r}"
