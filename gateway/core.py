"""Transport-independent ingestion core: per-participant VAD gating.

LiveKit delivers audio per participant track, so each participant gets its own
``StreamingVAD`` (own LSTM state, own segmenting). Speech events are handed to a sink
(the ASR stage); silence never leaves the gateway.
"""

from __future__ import annotations

import inspect
import logging
from typing import Awaitable, Callable, Union

import numpy as np

from services.vad.streaming_vad import GateEvent, StreamingVAD
from shared.audio import pcm16_to_float32, resample
from shared.config import Settings

log = logging.getLogger(__name__)

Sink = Callable[[str, GateEvent], Union[None, Awaitable[None]]]
VADFactory = Callable[[], StreamingVAD]


class GatewayCore:
    def __init__(self, settings: Settings, sink: Sink, vad_factory: VADFactory | None = None):
        self.settings = settings
        self.sink = sink
        self._vad_factory = vad_factory or (lambda: StreamingVAD.from_settings(settings))
        self._vads: dict[str, StreamingVAD] = {}
        self.stats = {"samples_in": 0, "speech_samples_out": 0}

    @property
    def participants(self) -> list[str]:
        return sorted(self._vads)

    def add_participant(self, participant_id: str) -> StreamingVAD:
        if participant_id not in self._vads:
            self._vads[participant_id] = self._vad_factory()
        return self._vads[participant_id]

    async def remove_participant(self, participant_id: str) -> list[GateEvent]:
        """Flush the participant's VAD (closing an open segment) and forget it."""
        vad = self._vads.pop(participant_id, None)
        if vad is None:
            return []
        events = vad.flush()
        await self._deliver(participant_id, events)
        return events

    async def push_pcm16(
        self, participant_id: str, pcm: Union[bytes, bytearray, memoryview, np.ndarray], sample_rate: int | None = None
    ) -> list[GateEvent]:
        """Feed int16 PCM (mono) from a participant. Resamples when ``sample_rate`` differs from the VAD rate."""
        audio = pcm16_to_float32(np.asarray(pcm) if isinstance(pcm, memoryview) else pcm)
        return await self.push_float(participant_id, audio, sample_rate)

    async def push_float(self, participant_id: str, audio: np.ndarray, sample_rate: int | None = None) -> list[GateEvent]:
        vad = self.add_participant(participant_id)
        if sample_rate and sample_rate != vad.sample_rate:
            audio = resample(audio, sample_rate, vad.sample_rate)
        self.stats["samples_in"] += len(audio)
        events = vad.feed(audio)
        await self._deliver(participant_id, events)
        return events

    async def _deliver(self, participant_id: str, events: list[GateEvent]) -> None:
        for ev in events:
            self.stats["speech_samples_out"] += len(ev.audio)
            result = self.sink(participant_id, ev)
            if inspect.isawaitable(result):
                await result
