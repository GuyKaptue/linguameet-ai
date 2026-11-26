"""Small, dependency-light audio helpers (PCM conversion, resampling, framing)."""

from __future__ import annotations

from math import gcd
from typing import Iterator

import numpy as np
from scipy.signal import resample_poly


def pcm16_to_float32(pcm: np.ndarray | bytes) -> np.ndarray:
    """int16 PCM (array or little-endian bytes) -> float32 in [-1, 1)."""
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        pcm = np.frombuffer(pcm, dtype="<i2")
    return np.asarray(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> np.ndarray:
    """float32 in [-1, 1] -> int16, clipping instead of wrapping."""
    audio = np.asarray(audio, dtype=np.float32)
    return np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Average channels of a (samples, channels) array; 1-D input is returned unchanged."""
    audio = np.asarray(audio)
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got {audio.ndim}-D")
    return audio.mean(axis=1).astype(audio.dtype if audio.dtype.kind == "f" else np.float32)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase resample of mono float audio."""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    audio = np.asarray(audio, dtype=np.float32)
    if src_rate == dst_rate or audio.size == 0:
        return audio
    g = gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)


def duration_seconds(num_samples: int, sample_rate: int) -> float:
    return num_samples / float(sample_rate)


def frames(audio: np.ndarray, size: int) -> Iterator[np.ndarray]:
    """Yield consecutive full frames of ``size`` samples; a trailing partial frame is dropped."""
    if size <= 0:
        raise ValueError("frame size must be positive")
    for start in range(0, len(audio) - size + 1, size):
        yield audio[start : start + size]


def sine(freq: float, seconds: float, sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    """Deterministic test tone (used by unit tests and benchmarks, never as 'speech')."""
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
