"""Message types passed between pipeline stages and to clients."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# LiveKit reliable data packets: 16 KiB protocol limit, ~15 KiB usable (docs.livekit.io/transport/data/packets).
MAX_DATA_PACKET_BYTES = 15 * 1024
SUBTITLE_SCHEMA_VERSION = 1


@dataclass
class SpeechSegment:
    """Audio gated by the VAD. ``is_final`` is False for in-progress (growing) segments."""

    segment_id: str
    participant_id: str
    audio: np.ndarray  # float32 mono
    sample_rate: int
    start_time: float  # seconds since the track started
    end_time: float
    is_final: bool = True
    created_at: float = 0.0  # time.monotonic() when the audio ended (for latency accounting)

    @property
    def duration(self) -> float:
        return len(self.audio) / float(self.sample_rate)


@dataclass
class Transcript:
    """ASR output. ``text`` is the confirmed prefix; ``unconfirmed`` is the volatile tail."""

    segment_id: str
    participant_id: str
    text: str
    language: str  # ISO 639-1
    language_probability: float = 1.0
    start_time: float = 0.0
    end_time: float = 0.0
    unconfirmed: str = ""
    is_final: bool = False
    compute_ms: float = 0.0  # time spent in the ASR backend for this update


@dataclass
class Translation:
    segment_id: str
    source_lang: str
    target_lang: str
    text: str
    is_final: bool = False
    latency_ms: float = 0.0


@dataclass
class AudioOut:
    """A synthesized audio chunk (float32 mono) for one listener language."""

    segment_id: str
    language: str
    audio: np.ndarray
    sample_rate: int
    is_last: bool = False


@dataclass
class SubtitleMessage:
    """JSON payload published on the LiveKit data channel (topic ``subtitles``)."""

    segment_id: str
    speaker: str
    source_lang: str
    text: str
    translations: dict[str, str] = field(default_factory=dict)  # ISO code -> text
    is_final: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    version: int = SUBTITLE_SCHEMA_VERSION
    type: str = "subtitle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "version": self.version,
            "segment_id": self.segment_id,
            "speaker": self.speaker,
            "source_lang": self.source_lang,
            "text": self.text,
            "translations": dict(self.translations),
            "is_final": self.is_final,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
        }

    def to_bytes(self) -> bytes:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_DATA_PACKET_BYTES:
            raise ValueError(
                f"subtitle payload is {len(payload)} bytes, above the {MAX_DATA_PACKET_BYTES}-byte data-packet limit"
            )
        return payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "SubtitleMessage":
        obj = json.loads(data.decode("utf-8"))
        if obj.get("type") != "subtitle":
            raise ValueError(f"not a subtitle message: type={obj.get('type')!r}")
        return cls(
            segment_id=obj["segment_id"],
            speaker=obj["speaker"],
            source_lang=obj["source_lang"],
            text=obj["text"],
            translations=dict(obj.get("translations", {})),
            is_final=bool(obj.get("is_final", False)),
            start_time=float(obj.get("start_time", 0.0)),
            end_time=float(obj.get("end_time", 0.0)),
            version=int(obj.get("version", SUBTITLE_SCHEMA_VERSION)),
        )


@dataclass
class LatencyReport:
    """Per-stage timings in milliseconds for one segment."""

    segment_id: str
    stages: dict[str, float] = field(default_factory=dict)

    def add(self, stage: str, ms: float) -> None:
        self.stages[stage] = self.stages.get(stage, 0.0) + ms

    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())
