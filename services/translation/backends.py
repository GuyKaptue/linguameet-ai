"""NMT backends: NLLB-200 via CTranslate2 (real) and a small lexicon MOCK for tests/offline use."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from shared.config import Settings
from shared.languages import FLORES_TO_ISO


@dataclass(frozen=True)
class NMTItem:
    text: str
    src: str  # FLORES-200 code, e.g. "eng_Latn"
    tgt: str  # FLORES-200 code, e.g. "spa_Latn"


class NMTBackend(Protocol):
    is_mock: bool

    def translate_batch(self, items: list[NMTItem]) -> list[str]: ...

    def count_tokens(self, text: str, src: str) -> int: ...


class CTranslate2NLLBBackend:
    """NLLB-200 through CTranslate2, following the official OpenNMT/CTranslate2 recipe.

    Conversion (once)::

        ct2-transformers-converter --model facebook/nllb-200-distilled-600M \\
            --quantization int8 --output_dir models/nllb-200-distilled-600M-ct2

    Inference (CTranslate2 docs, "Transformers" guide, NLLB section): the Hugging Face tokenizer
    is used with ``src_lang`` set, ``convert_ids_to_tokens(encode(text))`` gives the source tokens,
    ``Translator.translate_batch(source, target_prefix=[[tgt_lang]])`` decodes with the target
    language token as forced prefix, the first output token (that language code) is dropped and
    the rest is decoded with the tokenizer.
    """

    is_mock = False

    def __init__(self, settings: Settings, translator: Any | None = None, tokenizer: Any | None = None):
        self.settings = settings
        if translator is None:
            import ctranslate2

            model_dir = Path(settings.nmt_model_dir)
            if not (model_dir / "model.bin").is_file():
                raise FileNotFoundError(
                    f"No CTranslate2 model in {model_dir}. Convert NLLB first: scripts/convert_nllb.sh "
                    "(ct2-transformers-converter --model facebook/nllb-200-distilled-600M --quantization int8 ...)"
                )
            device = settings.resolve_device(settings.nmt_device)
            translator = ctranslate2.Translator(
                str(model_dir),
                device=device,
                compute_type=settings.resolve_compute_type(settings.nmt_compute_type, device),
            )
        if tokenizer is None:
            from transformers import AutoTokenizer  # optional extra: pip install '.[nmt]'

            tokenizer = AutoTokenizer.from_pretrained(settings.nmt_tokenizer)
        self.translator = translator
        self.tokenizer = tokenizer
        self._lock = threading.Lock()  # the tokenizer's src_lang is mutable state

    def _source_tokens(self, text: str, src: str) -> list[str]:
        self.tokenizer.src_lang = src
        return self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))

    def count_tokens(self, text: str, src: str) -> int:
        with self._lock:
            return len(self._source_tokens(text, src))

    def translate_batch(self, items: list[NMTItem]) -> list[str]:
        if not items:
            return []
        with self._lock:
            sources = [self._source_tokens(it.text, it.src) for it in items]
            prefixes = [[it.tgt] for it in items]
            results = self.translator.translate_batch(
                sources,
                target_prefix=prefixes,
                beam_size=self.settings.nmt_beam_size,
                max_batch_size=self.settings.nmt_max_batch_size,
                max_input_length=self.settings.nmt_max_input_tokens,
                max_decoding_length=256,
            )
            out: list[str] = []
            for item, res in zip(items, results):
                tokens = list(res.hypotheses[0])
                if tokens and tokens[0] == item.tgt:  # drop the forced target-language token
                    tokens = tokens[1:]
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                out.append(self.tokenizer.decode(ids, skip_special_tokens=True).strip())
            return out


# --------------------------------------------------------------------- MOCK
_PHRASES: dict[str, dict[str, str]] = {
    # English source (normalised: lower-case, no trailing punctuation) -> {iso: translation}
    "hello, how are you": {"es": "Hola, ¿cómo estás?", "fr": "Bonjour, comment allez-vous ?", "de": "Hallo, wie geht es dir?"},
    "good morning everyone": {"es": "Buenos días a todos.", "fr": "Bonjour à tous.", "de": "Guten Morgen zusammen."},
    "the meeting starts at nine o'clock": {
        "es": "La reunión empieza a las nueve.",
        "fr": "La réunion commence à neuf heures.",
        "de": "Das Treffen beginnt um neun Uhr.",
    },
    "please share your screen": {
        "es": "Por favor, comparte tu pantalla.",
        "fr": "Veuillez partager votre écran.",
        "de": "Bitte teile deinen Bildschirm.",
    },
    "thank you for joining us today": {
        "es": "Gracias por acompañarnos hoy.",
        "fr": "Merci de nous avoir rejoints aujourd'hui.",
        "de": "Danke, dass Sie heute bei uns sind.",
    },
    "can you hear me": {"es": "¿Me oyes?", "fr": "Vous m'entendez ?", "de": "Kannst du mich hören?"},
    "ask not what your country can do for you": {
        "es": "No preguntes qué puede hacer tu país por ti.",
        "fr": "Ne demandez pas ce que votre pays peut faire pour vous.",
        "de": "Frage nicht, was dein Land für dich tun kann.",
    },
}
_WORDS: dict[str, dict[str, str]] = {
    "es": {"hello": "hola", "thanks": "gracias", "yes": "sí", "no": "no", "meeting": "reunión", "today": "hoy",
           "tomorrow": "mañana", "the": "el", "and": "y", "we": "nosotros", "you": "tú", "good": "bueno"},
    "fr": {"hello": "bonjour", "thanks": "merci", "yes": "oui", "no": "non", "meeting": "réunion", "today": "aujourd'hui",
           "tomorrow": "demain", "the": "le", "and": "et", "we": "nous", "you": "vous", "good": "bon"},
    "de": {"hello": "hallo", "thanks": "danke", "yes": "ja", "no": "nein", "meeting": "Treffen", "today": "heute",
           "tomorrow": "morgen", "the": "der", "and": "und", "we": "wir", "you": "du", "good": "gut"},
}
_TOKEN = re.compile(r"\w+(?:'\w+)?|[^\w\s]", re.UNICODE)


def _norm(text: str) -> str:
    return re.sub(r"[.!?¿¡\s]+$", "", text.strip().lower())


class LexiconMockBackend:
    """MOCK NMT backend (NOT a neural model). Known meeting phrases from English are translated
    exactly; everything else goes through a tiny word lexicon, unknown words are copied. Reverse
    directions (es/fr/de -> en) use the inverted phrase table. Used when ``BACKEND_MODE=mock`` and
    in tests, where it exercises the service (batching, caching, chunking) deterministically."""

    is_mock = True

    def __init__(self) -> None:
        self.batches: list[list[NMTItem]] = []  # recorded for tests
        self._reverse: dict[str, dict[str, str]] = {}
        for en, tr in _PHRASES.items():
            for iso, text in tr.items():
                self._reverse.setdefault(iso, {})[_norm(text)] = en.capitalize() + "."

    def count_tokens(self, text: str, src: str) -> int:
        return len(_TOKEN.findall(text)) + 2  # + language token and </s>, like NLLB

    def translate_batch(self, items: list[NMTItem]) -> list[str]:
        self.batches.append(list(items))
        return [self._one(it) for it in items]

    def _one(self, it: NMTItem) -> str:
        src, tgt = FLORES_TO_ISO.get(it.src), FLORES_TO_ISO.get(it.tgt)
        key = _norm(it.text)
        if src == "en" and tgt in _WORDS:
            if key in _PHRASES and tgt in _PHRASES[key]:
                return _PHRASES[key][tgt]
            lex = _WORDS[tgt]
            return " ".join(lex.get(t.lower(), t) for t in _TOKEN.findall(it.text))
        if tgt == "en" and src in self._reverse and key in self._reverse[src]:
            return self._reverse[src][key]
        if tgt == "en" and src in _WORDS:
            inv = {v.lower(): k for k, v in _WORDS[src].items()}
            return " ".join(inv.get(t.lower(), t) for t in _TOKEN.findall(it.text))
        return it.text  # unsupported direction in the mock: pass through
