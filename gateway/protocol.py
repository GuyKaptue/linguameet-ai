"""Wire conventions shared by the gateway and the web client (client/app.js uses the same names)."""

from __future__ import annotations

from typing import Mapping

SUBTITLE_TOPIC = "subtitles"
ATTR_LISTEN_LANG = "listen_lang"  # language the participant wants to read / hear
ATTR_SPEAK_LANG = "speak_lang"  # language the participant speaks ("" or missing = auto-detect)
TTS_TRACK_PREFIX = "tts-"  # the gateway publishes one audio track per language: "tts-es", "tts-fr", ...


def tts_track_name(lang: str) -> str:
    return f"{TTS_TRACK_PREFIX}{lang}"


def parse_prefs(attributes: Mapping[str, str]) -> tuple[str | None, str | None]:
    """(listen_lang, speak_lang) from participant attributes; empty / absent -> None."""
    listen = (attributes.get(ATTR_LISTEN_LANG) or "").strip() or None
    speak = (attributes.get(ATTR_SPEAK_LANG) or "").strip() or None
    return listen, speak
