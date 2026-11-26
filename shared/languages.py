"""Language-code handling shared by ASR, NMT, TTS and the client.

Whisper reports ISO 639-1 codes (``en``, ``de``, ``zh``), NLLB-200 expects FLORES-200
codes (``eng_Latn``) as documented on its model card, and XTTS-v2 has its own list.
"""

from __future__ import annotations

import re

# ISO 639-1 -> FLORES-200 (NLLB). Codes follow the ``xxx_Yyyy`` pattern from the model card.
ISO_TO_FLORES: dict[str, str] = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "pl": "pol_Latn",
    "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "tr": "tur_Latn", "ar": "arb_Arab",
    "he": "heb_Hebr", "hi": "hin_Deva", "bn": "ben_Beng", "ur": "urd_Arab",
    "fa": "pes_Arab", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "vi": "vie_Latn", "th": "tha_Thai", "id": "ind_Latn", "sv": "swe_Latn",
    "da": "dan_Latn", "no": "nob_Latn", "fi": "fin_Latn", "cs": "ces_Latn",
    "el": "ell_Grek", "hu": "hun_Latn", "ro": "ron_Latn", "bg": "bul_Cyrl",
}
FLORES_TO_ISO: dict[str, str] = {v: k for k, v in ISO_TO_FLORES.items()}

# XTTS-v2 languages as listed in the Coqui TTS documentation (17).
XTTS_LANGUAGES: frozenset[str] = frozenset(
    {"ar", "zh", "cs", "nl", "en", "fr", "de", "hi", "hu", "it", "ja", "ko", "pl", "pt", "ru", "es", "tr"}
)

_FLORES_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")
_ALIASES = {"iw": "he", "in": "id", "nb": "no", "nn": "no", "jw": "id"}


class UnsupportedLanguage(ValueError):
    """Raised for language codes the platform cannot route."""


def is_flores(code: str) -> bool:
    return bool(_FLORES_RE.match(code))


def normalize_lang(code: str) -> str:
    """Return a lower-case ISO 639-1 code for ``en``, ``en-US``, ``zh_CN`` or ``eng_Latn`` input."""
    if not code or not code.strip():
        raise UnsupportedLanguage("empty language code")
    raw = code.strip()
    if is_flores(raw):
        try:
            return FLORES_TO_ISO[raw]
        except KeyError:
            raise UnsupportedLanguage(f"unsupported FLORES code: {raw}") from None
    base = re.split(r"[-_]", raw.lower())[0]
    base = _ALIASES.get(base, base)
    if base not in ISO_TO_FLORES:
        raise UnsupportedLanguage(f"unsupported language: {code}")
    return base


def to_flores(code: str) -> str:
    """ISO / locale / FLORES code -> FLORES-200 code used by NLLB."""
    return ISO_TO_FLORES[normalize_lang(code)]


def supports_tts(code: str) -> bool:
    """Whether XTTS-v2 can speak this language (Piper voices are checked by the TTS service)."""
    try:
        return normalize_lang(code) in XTTS_LANGUAGES
    except UnsupportedLanguage:
        return False


def supported_languages() -> list[str]:
    return sorted(ISO_TO_FLORES)
