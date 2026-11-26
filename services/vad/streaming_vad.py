"""Streaming voice-activity gate in front of the ASR.

``StreamingVAD`` accepts audio of any chunk size, scores it in fixed Silero windows
(32 ms) and emits ordered events so that downstream ASR only ever receives speech:

* ``start``  - speech began; ``audio`` holds the pre-roll (so word onsets are not clipped)
* ``audio``  - further speech audio (including the short silence tail until the end is confirmed)
* ``end``    - speech ended (``forced=True`` when the segment hit the maximum length and was split)

Hysteresis follows Silero's own ``VADIterator`` (speech at ``p >= threshold``, silence only
below ``threshold - 0.15``, end after ``min_silence_ms`` of silence); on top of that a
``min_speech_ms`` debounce suppresses clicks and a ``max_segment_s`` cap bounds buffers.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from services.vad.silero_onnx import SileroOnnxModel, VADModel
from shared.config import Settings
from shared.types import SpeechSegment

HYSTERESIS = 0.15  # silence threshold = speech threshold - 0.15 (same as silero VADIterator)


@dataclass
class GateEvent:
    kind: Literal["start", "audio", "end"]
    time: float  # stream time (s): first sample of ``audio`` for start/audio; for end: when the gate CLOSED
    audio: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    forced: bool = False
    # end events only: stream time (s) where speech actually stopped. ``time - speech_end`` is the silence
    # hang-over the gate needs before it can be sure the speaker finished (latency measured from the
    # speaker's last word must include it).
    speech_end: float | None = None


class StreamingVAD:
    def __init__(
        self,
        model: VADModel | None = None,
        *,
        threshold: float = 0.5,
        min_silence_ms: int = 300,
        speech_pad_ms: int = 100,
        min_speech_ms: int = 64,
        max_segment_s: float = 15.0,
    ):
        self.model = model or SileroOnnxModel(16000)
        self.sample_rate = self.model.sample_rate
        self._w = self.model.window_samples
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        self.threshold = threshold
        self._neg_threshold = max(0.01, threshold - HYSTERESIS)
        self._min_silence = int(self.sample_rate * min_silence_ms / 1000)
        pad_windows = math.ceil(self.sample_rate * speech_pad_ms / 1000 / self._w)
        self._min_speech_windows = max(1, math.ceil(self.sample_rate * min_speech_ms / 1000 / self._w))
        self._max_segment = int(max_segment_s * self.sample_rate)
        self._ring: deque[np.ndarray] = deque(maxlen=pad_windows + self._min_speech_windows)
        # benchmarking counters
        self.windows_processed = 0
        self.inference_seconds = 0.0
        self.reset()

    @classmethod
    def from_settings(cls, s: Settings, model: VADModel | None = None) -> "StreamingVAD":
        return cls(
            model or SileroOnnxModel(s.audio_sample_rate),
            threshold=s.vad_threshold,
            min_silence_ms=s.vad_min_silence_ms,
            speech_pad_ms=s.vad_speech_pad_ms,
            min_speech_ms=s.vad_min_speech_ms,
            max_segment_s=s.vad_max_segment_seconds,
        )

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.model.reset_states()
        self._buf = np.empty(0, dtype=np.float32)  # samples not yet forming a full window
        self._pos = 0  # samples consumed into windows so far
        self._triggered = False
        self._run = 0  # consecutive speech windows while not yet triggered
        self._temp_end: int | None = None  # sample where the current silence run began
        self._seg_samples = 0
        self._ring.clear()

    @property
    def in_speech(self) -> bool:
        return self._triggered

    @property
    def stream_time(self) -> float:
        return self._pos / self.sample_rate

    # --------------------------------------------------------------- public API
    def feed(self, audio: np.ndarray) -> list[GateEvent]:
        """Consume float32 mono audio at ``sample_rate``; return the events it produced."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be 1-D mono")
        self._buf = np.concatenate([self._buf, audio]) if self._buf.size else audio
        events: list[GateEvent] = []
        pending: list[np.ndarray] = []
        pending_start = 0.0

        def flush_pending() -> None:
            nonlocal pending
            if pending:
                events.append(GateEvent("audio", pending_start, np.concatenate(pending)))
                pending = []

        n_windows = len(self._buf) // self._w
        for i in range(n_windows):
            win = self._buf[i * self._w : (i + 1) * self._w]
            start_sample = self._pos
            end_sample = start_sample + self._w
            t0 = time.perf_counter()
            prob = self.model(win)
            self.inference_seconds += time.perf_counter() - t0
            self.windows_processed += 1
            self._pos = end_sample

            if not self._triggered:
                self._ring.append(win)
                self._run = self._run + 1 if prob >= self.threshold else 0
                if self._run >= self._min_speech_windows:
                    flush_pending()
                    pre = np.concatenate(list(self._ring))
                    seg_start = end_sample - len(pre)
                    events.append(GateEvent("start", seg_start / self.sample_rate, pre))
                    self._ring.clear()
                    self._triggered = True
                    self._temp_end = None
                    self._seg_samples = len(pre)
                continue

            # --- inside a speech segment ---
            if not pending:
                pending_start = start_sample / self.sample_rate
            pending.append(win)
            self._seg_samples += self._w
            if prob >= self.threshold:
                self._temp_end = None
            elif prob < self._neg_threshold:
                if self._temp_end is None:
                    self._temp_end = end_sample
                if end_sample - self._temp_end >= self._min_silence:
                    flush_pending()
                    events.append(GateEvent("end", end_sample / self.sample_rate, speech_end=self._temp_end / self.sample_rate))
                    self._reset_segment()
                    continue
            if self._seg_samples >= self._max_segment:
                flush_pending()
                events.append(GateEvent("end", end_sample / self.sample_rate, forced=True))
                events.append(GateEvent("start", end_sample / self.sample_rate))
                self._seg_samples = 0
                self._temp_end = None

        self._buf = self._buf[n_windows * self._w :]
        flush_pending()
        return events

    def flush(self) -> list[GateEvent]:
        """End of stream: score the zero-padded remainder and close any open segment."""
        events: list[GateEvent] = []
        if self._buf.size:
            pad = np.zeros(self._w - len(self._buf), dtype=np.float32)
            events.extend(self.feed(pad))  # completes the final window
        if self._triggered:
            events.append(GateEvent("end", self._pos / self.sample_rate, speech_end=self._pos / self.sample_rate))
        self.reset()
        return events

    def _reset_segment(self) -> None:
        self._triggered = False
        self._run = 0
        self._temp_end = None
        self._seg_samples = 0
        self._ring.clear()


class SpeechSegmenter:
    """Assembles ``StreamingVAD`` events into complete :class:`SpeechSegment` objects."""

    def __init__(self, vad: StreamingVAD, participant_id: str = "speaker"):
        self.vad = vad
        self.participant_id = participant_id
        self._count = 0
        self._chunks: list[np.ndarray] = []
        self._start = 0.0

    def _consume(self, events: list[GateEvent]) -> list[SpeechSegment]:
        out: list[SpeechSegment] = []
        for ev in events:
            if ev.kind == "start":
                self._chunks = [ev.audio] if ev.audio.size else []
                self._start = ev.time
            elif ev.kind == "audio":
                self._chunks.append(ev.audio)
            else:  # end
                audio = np.concatenate(self._chunks) if self._chunks else np.empty(0, dtype=np.float32)
                self._chunks = []
                if audio.size:
                    self._count += 1
                    out.append(
                        SpeechSegment(
                            segment_id=f"{self.participant_id}-{self._count}",
                            participant_id=self.participant_id,
                            audio=audio,
                            sample_rate=self.vad.sample_rate,
                            start_time=self._start,
                            end_time=ev.time,
                            is_final=True,
                            created_at=time.monotonic(),
                        )
                    )
        return out

    def feed(self, audio: np.ndarray) -> list[SpeechSegment]:
        return self._consume(self.vad.feed(audio))

    def flush(self) -> list[SpeechSegment]:
        return self._consume(self.vad.flush())
