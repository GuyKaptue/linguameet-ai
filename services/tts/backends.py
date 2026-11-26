"""TTS backends: Piper (real, CPU-friendly, GPLv3) and a labelled tone MOCK.

A backend turns text into a *stream of float32 mono chunks* at its own sample rate. Piper's
``PiperVoice.synthesize(text)`` is a generator that yields one ``AudioChunk`` per sentence
(``audio_float_array``, ``sample_rate``), so the first audio is available after the first sentence
has been synthesized - the service then slices it into small frames for the listener.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import numpy as np

from services.translation.chunking import split_sentences
from shared.config import Settings
from shared.languages import normalize_lang


class UnsupportedVoice(LookupError):
    """No voice available for the requested language."""


class TTSBackend(Protocol):
    is_mock: bool
    sample_rate: int  # nominal; chunks may carry their own via ``stream`` (see TTSService)

    def supports(self, lang: str) -> bool: ...

    def stream(self, text: str, lang: str) -> Iterator[tuple[np.ndarray, int]]:
        """Yield ``(float32 mono audio, sample_rate)`` pieces as soon as they are ready."""
        ...


class PiperBackend:
    """Piper voices from ``TTS_VOICES_DIR``.

    Voice files follow Piper's naming: ``<lang>_<REGION>-<name>-<quality>.onnx`` with a sibling
    ``.onnx.json`` (e.g. ``en_US-lessac-low.onnx``); a plain ``<lang>.onnx`` also works. The first
    match (sorted) is used per language. API (piper-tts 1.8.0): ``PiperVoice.load(model_path,
    config_path=None, use_cuda=False)`` and ``voice.synthesize(text)`` -> iterable of ``AudioChunk``.
    """

    is_mock = False
    sample_rate = 22050  # typical Piper voice rate; each chunk reports its true rate

    def __init__(self, settings: Settings, voice_loader: Callable[[Path], Any] | None = None):
        self.settings = settings
        self._loader = voice_loader or self._default_loader
        self._voices: dict[str, Any] = {}

    @staticmethod
    def _default_loader(path: Path) -> Any:
        from piper import PiperVoice  # lazy: pulls in onnxruntime + bundled espeak-ng data

        return PiperVoice.load(path)

    def voice_file(self, lang: str) -> Path | None:
        iso = normalize_lang(lang)
        d = self.settings.tts_voices_dir
        if not d.is_dir():
            return None
        candidates = sorted(d.glob(f"{iso}.onnx")) + sorted(d.glob(f"{iso}_*.onnx"))
        return candidates[0] if candidates else None

    def supports(self, lang: str) -> bool:
        try:
            return self.voice_file(lang) is not None
        except ValueError:
            return False

    def _voice(self, lang: str) -> Any:
        iso = normalize_lang(lang)
        if iso not in self._voices:
            path = self.voice_file(iso)
            if path is None:
                raise UnsupportedVoice(f"no Piper voice for '{iso}' in {self.settings.tts_voices_dir}")
            self._voices[iso] = self._loader(path)
        return self._voices[iso]

    def stream(self, text: str, lang: str) -> Iterator[tuple[np.ndarray, int]]:
        voice = self._voice(lang)
        for chunk in voice.synthesize(text):
            yield np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1), int(chunk.sample_rate)


class ToneMockBackend:
    """MOCK TTS (NOT speech): every word becomes a short sine burst whose pitch is derived from the
    word (deterministic), with a speaking-rate-like duration (``chars_per_second``). It streams one
    piece per sentence like Piper does, supports every language and needs no model files."""

    is_mock = True
    sample_rate = 22050

    def __init__(self, chars_per_second: float = 15.0):
        self.chars_per_second = chars_per_second
        self.calls: list[tuple[str, str]] = []

    def supports(self, lang: str) -> bool:
        return True

    def stream(self, text: str, lang: str) -> Iterator[tuple[np.ndarray, int]]:
        self.calls.append((text, normalize_lang(lang)))
        for sentence in split_sentences(text) or ([text.strip()] if text.strip() else []):
            yield self._sentence(sentence), self.sample_rate

    def _sentence(self, sentence: str) -> np.ndarray:
        parts: list[np.ndarray] = []
        gap = np.zeros(int(0.04 * self.sample_rate), dtype=np.float32)
        for word in sentence.split():
            dur = max(0.08, len(word) / self.chars_per_second)
            n = int(dur * self.sample_rate)
            t = np.arange(n, dtype=np.float32) / self.sample_rate
            freq = 150.0 + (zlib.crc32(word.lower().encode("utf-8")) % 250)
            env = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.01 * self.sample_rate))
            parts.append((0.3 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32))
            parts.append(gap)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def create_backend(settings: Settings) -> TTSBackend:
    if settings.is_mock or settings.tts_engine == "mock":
        return ToneMockBackend()
    return PiperBackend(settings)
