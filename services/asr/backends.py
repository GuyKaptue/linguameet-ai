"""ASR backends: the real faster-whisper wrapper and a labelled replay MOCK for tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from services.asr.hypothesis import Word
from shared.config import Settings
from shared.languages import normalize_lang


@dataclass
class ASRResult:
    words: list[Word]  # times relative to the start of the audio passed in
    language: str  # ISO 639-1
    language_probability: float


class ASRBackend(Protocol):
    def transcribe(self, audio: np.ndarray, *, language: str | None, prompt: str = "") -> ASRResult: ...


class FasterWhisperBackend:
    """Thin wrapper over ``faster_whisper.WhisperModel`` (SYSTRAN/faster-whisper, MIT).

    API used (verified against faster-whisper 1.2.1): ``WhisperModel(size_or_path, device=,
    compute_type=, download_root=)``; ``model.transcribe(audio_ndarray, language=, beam_size=,
    word_timestamps=True, condition_on_previous_text=False, initial_prompt=, vad_filter=False)``
    returns ``(segments_generator, info)`` where ``info.language`` / ``info.language_probability``
    describe the detected (or given) language and each ``segment.words`` item has
    ``start``, ``end``, ``word``, ``probability``. The generator is lazy, so it is consumed here.
    """

    def __init__(self, settings: Settings, model: Any | None = None):
        self.settings = settings
        if model is None:
            from faster_whisper import WhisperModel  # imported lazily: heavy and optional in tests

            device = settings.resolve_device(settings.asr_device)
            compute = settings.resolve_compute_type(settings.asr_compute_type, device)
            model = WhisperModel(
                settings.asr_model_size,
                device=device,
                compute_type=compute,
                download_root=str(settings.models_dir),
            )
        self.model = model

    def transcribe(self, audio: np.ndarray, *, language: str | None, prompt: str = "") -> ASRResult:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=self.settings.asr_beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            vad_filter=False,  # audio is already gated by our streaming VAD
        )
        words = [
            Word(float(w.start), float(w.end), w.word)
            for seg in segments  # consuming the generator is what runs the transcription
            for w in (seg.words or [])
        ]
        try:
            lang = normalize_lang(info.language)
        except ValueError:  # a Whisper language outside our routing table
            lang = info.language
        return ASRResult(words, lang, float(info.language_probability))


@dataclass
class ReplayCall:
    buffer_seconds: float
    language_arg: str | None
    prompt: str


@dataclass
class ReplayASRBackend:
    """MOCK ASR backend for tests - it does NOT recognise speech.

    It is constructed with a reference recording and a scripted word list (with synthetic
    timings). Given an audio buffer that is an exact slice of the reference, it locates the slice
    and returns the scripted words that lie fully inside it, imitating how Whisper reveals more
    words as the buffer grows. ``flaky_last_word`` makes the final word unstable (changes on every
    second call) to exercise LocalAgreement. Times are relative to the buffer, as with Whisper.
    """

    reference: np.ndarray
    script: list[Word]  # absolute times within the reference
    language: str = "en"
    language_probability: float = 0.98
    flaky_last_word: bool = False
    low_confidence_calls: int = 0  # first N detection calls report probability 0.3
    regions: list[tuple[float, float, str]] | None = None  # (start_s, end_s, language) of the reference
    sample_rate: int = 16000
    calls: list[ReplayCall] = field(default_factory=list)

    def _locate(self, audio: np.ndarray) -> int:
        n = min(len(audio), 4096)
        if n == 0:
            return -1
        needle = np.ascontiguousarray(audio[:n], dtype=np.float32).tobytes()
        hay = np.ascontiguousarray(self.reference, dtype=np.float32).tobytes()
        pos = hay.find(needle)
        return -1 if pos < 0 else pos // 4

    def transcribe(self, audio: np.ndarray, *, language: str | None, prompt: str = "") -> ASRResult:
        self.calls.append(ReplayCall(len(audio) / self.sample_rate, language, prompt))
        start = self._locate(audio)
        detecting = language is None
        prob = 1.0 if not detecting else self.language_probability  # given language => probability 1, like faster-whisper
        if detecting and self.low_confidence_calls > 0:
            self.low_confidence_calls -= 1
            prob = 0.3
        if start < 0:
            return ASRResult([], language or self.language, prob)
        t0 = start / self.sample_rate
        t1 = t0 + len(audio) / self.sample_rate
        words = [Word(w.start - t0, w.end - t0, w.text) for w in self.script if w.start >= t0 - 1e-6 and w.end <= t1 + 1e-6]
        if self.flaky_last_word and words and len(self.calls) % 2 == 1:
            last = words[-1]
            words[-1] = Word(last.start, last.end, " " + "x" * len(last.text.strip()))
        detected = self.language
        if self.regions:
            # the language of the region where the buffer's first *word* lies (code-switching tests)
            first = next((w for w in self.script if w.end > t0), None)
            probe = first.start if first else t0
            detected = next((lang for a, b, lang in self.regions if a <= probe < b), self.language)
        return ASRResult(words, language or detected, prob)
