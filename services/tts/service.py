"""Streaming TTS service.

``TTSService.synthesize_stream`` is an async generator of small :class:`AudioOut` frames
(``TTS_CHUNK_MS`` long, at ``TTS_OUTPUT_SAMPLE_RATE``). Synthesis runs in a worker thread and frames
are delivered as soon as the backend produces audio - the caller never waits for a complete WAV.
Closing the generator early (barge-in, newer segment) stops the worker between backend pieces.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np

from services.tts.backends import TTSBackend, UnsupportedVoice
from shared.audio import resample
from shared.config import Settings
from shared.languages import normalize_lang
from shared.types import AudioOut

log = logging.getLogger(__name__)
_DONE = object()


@dataclass
class SynthesisRecord:
    segment_id: str
    language: str
    chars: int
    first_chunk_ms: float  # request -> first audio frame available
    total_ms: float
    audio_seconds: float

    @property
    def rtf(self) -> float:
        """Real-time factor (<1 = faster than real time)."""
        return (self.total_ms / 1000.0) / self.audio_seconds if self.audio_seconds else 0.0


@dataclass
class TTSStats:
    records: list[SynthesisRecord] = field(default_factory=list)
    cancelled: int = 0


class TTSService:
    def __init__(self, settings: Settings, backend: TTSBackend, *, workers: int = 2):
        self.settings = settings
        self.backend = backend
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tts")
        self.stats = TTSStats()

    @property
    def frame_samples(self) -> int:
        return max(1, int(self.settings.tts_output_sample_rate * self.settings.tts_chunk_ms / 1000))

    def supports(self, lang: str) -> bool:
        try:
            return self.backend.supports(normalize_lang(lang))
        except ValueError:
            return False

    async def aclose(self) -> None:
        self._executor.shutdown(wait=True)

    async def synthesize_stream(self, text: str, lang: str, segment_id: str = "seg") -> AsyncIterator[AudioOut]:
        iso = normalize_lang(lang)
        text = text.strip()
        if not self.backend.supports(iso):
            raise UnsupportedVoice(f"no voice for '{iso}'")
        if not text:
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()
        out_rate = self.settings.tts_output_sample_rate
        t0 = time.perf_counter()

        def produce() -> None:
            try:
                for audio, rate in self.backend.stream(text, iso):
                    if cancel.is_set():
                        return
                    if len(audio):
                        loop.call_soon_threadsafe(queue.put_nowait, resample(audio, rate, out_rate))
            except Exception as exc:  # delivered to the consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        job = loop.run_in_executor(self._executor, produce)
        buf = np.empty(0, dtype=np.float32)
        held: np.ndarray | None = None  # one frame is held back so the final frame can carry is_last
        first_ms: float | None = None
        samples = 0
        completed = False
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    completed = True
                    break
                if isinstance(item, Exception):
                    raise item
                buf = np.concatenate([buf, item]) if buf.size else item
                while len(buf) >= self.frame_samples:
                    frame, buf = buf[: self.frame_samples], buf[self.frame_samples :]
                    if held is not None:
                        samples += len(held)
                        yield AudioOut(segment_id, iso, held, out_rate, is_last=False)
                    held = frame
                    if first_ms is None:
                        first_ms = (time.perf_counter() - t0) * 1000
            # end of stream: held frame + partial remainder
            tail = [f for f in (held, buf if buf.size else None) if f is not None]
            for i, frame in enumerate(tail):
                samples += len(frame)
                if first_ms is None:
                    first_ms = (time.perf_counter() - t0) * 1000
                yield AudioOut(segment_id, iso, frame, out_rate, is_last=(i == len(tail) - 1))
        finally:
            if not completed:
                cancel.set()
                self.stats.cancelled += 1
            if completed and first_ms is not None:
                self.stats.records.append(
                    SynthesisRecord(segment_id, iso, len(text), first_ms, (time.perf_counter() - t0) * 1000, samples / out_rate)
                )

    async def synthesize(self, text: str, lang: str, segment_id: str = "seg") -> np.ndarray:
        """Convenience: collect the whole stream into one float32 array (tests, benchmarks)."""
        frames = [f.audio async for f in self.synthesize_stream(text, lang, segment_id)]
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
