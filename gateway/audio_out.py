"""Publishes synthesized speech into the LiveKit room: one audio track per target language.

LiveKit Python SDK calls (checked against ``livekit`` 1.3.0 by introspection):
``rtc.AudioSource(sample_rate, num_channels, queue_size_ms=1000)`` with ``await source.capture_frame(frame)``,
``rtc.LocalAudioTrack.create_audio_track(name, source)``,
``await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=...))`` and
``rtc.AudioFrame(data, sample_rate, num_channels, samples_per_channel)`` (int16 little-endian PCM).
Listeners pick the track named ``tts-<their language>``; the web client ignores the others.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import numpy as np
from livekit import rtc

from gateway.protocol import tts_track_name
from shared.audio import float32_to_pcm16
from shared.types import AudioOut

log = logging.getLogger(__name__)
SourceFactory = Callable[[int], Any]
TrackFactory = Callable[[str, Any], Any]


class LiveKitAudioPublisher:
    def __init__(
        self,
        room: Any,
        sample_rate: int,
        *,
        source_factory: SourceFactory | None = None,
        track_factory: TrackFactory | None = None,
    ):
        self.room = room
        self.sample_rate = sample_rate
        self._source_factory = source_factory or (lambda sr: rtc.AudioSource(sr, 1))
        self._track_factory = track_factory or (lambda name, src: rtc.LocalAudioTrack.create_audio_track(name, src))
        self._sources: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def languages(self) -> list[str]:
        return sorted(self._sources)

    async def _source_for(self, lang: str) -> Any:
        lock = self._locks.setdefault(lang, asyncio.Lock())
        async with lock:
            if lang not in self._sources:
                source = self._source_factory(self.sample_rate)
                track = self._track_factory(tts_track_name(lang), source)
                options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
                await self.room.local_participant.publish_track(track, options)
                self._sources[lang] = source
                log.info("published TTS track %s", tts_track_name(lang))
            return self._sources[lang]

    async def publish_frame(self, out: AudioOut) -> None:
        if out.sample_rate != self.sample_rate:
            raise ValueError(f"audio at {out.sample_rate} Hz does not match the track rate {self.sample_rate} Hz")
        source = await self._source_for(out.language)
        pcm = float32_to_pcm16(np.asarray(out.audio, dtype=np.float32))
        frame = rtc.AudioFrame(pcm.tobytes(), self.sample_rate, 1, len(pcm))
        await source.capture_frame(frame)  # waits while the source queue is full (natural back-pressure)

    async def aclose(self) -> None:
        for source in self._sources.values():
            if hasattr(source, "aclose"):
                await source.aclose()
        self._sources.clear()
