"""Module 5: streaming TTS service.

* ``ToneMockBackend`` is a MOCK (sine bursts, not speech). It streams per sentence like Piper and makes
  chunking, latency accounting, cancellation and error handling testable without voice files.
* ``PiperBackend`` is exercised through an injected fake ``PiperVoice`` (MOCK following the piper-tts 1.8.0
  API: ``synthesize(text)`` -> iterable of chunks with ``audio_float_array`` / ``sample_rate``).
* Real Piper voice: opt-in with ``TTS_TEST_VOICE_DIR=/dir/with/<lang>_*.onnx`` (+ ``.onnx.json``) and
  ``TTS_TEST_LANG`` (default ``en``); it checks real synthesis latency against the 500 ms budget.
  Piper voices are hosted on Hugging Face, which is not reachable from the build sandbox.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from services.tts.backends import PiperBackend, ToneMockBackend, UnsupportedVoice, create_backend
from services.tts.service import TTSService
from shared.audio import float32_to_pcm16
from shared.config import Settings
from shared.languages import UnsupportedLanguage

OUT_RATE = 24000
LONG_TEXT = "Good morning everyone. The meeting starts at nine o'clock. Please share your screen. Thank you for joining us today."


def settings(**kw) -> Settings:
    kw.setdefault("tts_engine", "mock")
    return Settings(_env_file=None, **kw)


@pytest.fixture
async def svc():
    s = TTSService(settings(), ToneMockBackend())
    yield s
    await s.aclose()


# ============================================================ valid audio stream
async def test_output_is_a_valid_stream_of_fixed_size_frames(svc):
    frames = [f async for f in svc.synthesize_stream(LONG_TEXT, "en", "seg-1")]
    n = svc.frame_samples
    assert n == 2400  # 100 ms at 24 kHz
    assert len(frames) > 5
    assert all(f.audio.dtype == np.float32 and f.audio.ndim == 1 for f in frames)
    assert all(len(f.audio) == n for f in frames[:-1])  # only the last frame may be short
    assert 0 < len(frames[-1].audio) <= n
    assert [f.is_last for f in frames] == [False] * (len(frames) - 1) + [True]
    assert all(f.sample_rate == OUT_RATE and f.language == "en" and f.segment_id == "seg-1" for f in frames)
    audio = np.concatenate([f.audio for f in frames])
    assert np.isfinite(audio).all() and np.abs(audio).max() <= 1.0 and np.abs(audio).max() > 0.1  # not silent
    assert 0.1 < np.sqrt(np.mean(audio**2)) < 0.5
    pcm = float32_to_pcm16(audio)  # convertible to what LiveKit publishes
    assert pcm.dtype == np.int16 and np.abs(pcm).max() > 3000


async def test_audio_duration_scales_with_text_length(svc):
    short = await svc.synthesize("Hello there.", "en")
    long = await svc.synthesize(LONG_TEXT, "en")
    assert len(long) > 3 * len(short) > 0


async def test_synthesis_is_deterministic_for_the_same_text(svc):
    a = await svc.synthesize("Can you hear me?", "en")
    b = await svc.synthesize("Can you hear me?", "en")
    assert np.array_equal(a, b)


async def test_output_sample_rate_and_frame_size_follow_settings():
    s = TTSService(settings(tts_output_sample_rate=16000, tts_chunk_ms=50), ToneMockBackend())
    frames = [f async for f in s.synthesize_stream("One two three four five six.", "en")]
    assert all(f.sample_rate == 16000 for f in frames) and len(frames[0].audio) == 800
    await s.aclose()


async def test_pitch_differs_between_different_words(svc):
    a = await svc.synthesize("alpha", "en")
    b = await svc.synthesize("zebra", "en")
    fa, fb = np.argmax(np.abs(np.fft.rfft(a))) / (len(a) / OUT_RATE), np.argmax(np.abs(np.fft.rfft(b))) / (len(b) / OUT_RATE)
    assert abs(fa - fb) > 5


# =============================================================== streaming + latency
async def test_first_audio_arrives_long_before_synthesis_finishes():
    class Slow(ToneMockBackend):
        def stream(self, text, lang):
            for piece in super().stream(text, lang):
                time.sleep(0.15)  # simulate compute per sentence
                yield piece

    s = TTSService(settings(), Slow())
    t0 = time.perf_counter()
    first_at = None
    count = 0
    async for f in s.synthesize_stream(LONG_TEXT, "en"):
        if first_at is None:
            first_at = time.perf_counter() - t0
        count += 1
    total = time.perf_counter() - t0
    assert count > 10
    assert first_at < 0.35 < total  # first frame after ~1 sentence of compute, not after all four
    rec = s.stats.records[-1]
    assert rec.first_chunk_ms < rec.total_ms and rec.first_chunk_ms < 500
    await s.aclose()


async def test_time_to_first_chunk_is_within_the_latency_budget(svc):
    budget_ms = svc.settings.budget.tts_ms
    assert budget_ms == 500
    for text in ["Hello.", "Can you hear me?", LONG_TEXT]:
        await svc.synthesize(text, "en", "lat")
    recs = svc.stats.records
    assert len(recs) == 3
    assert all(r.first_chunk_ms < budget_ms for r in recs), [round(r.first_chunk_ms, 1) for r in recs]
    assert all(r.audio_seconds > 0 and r.rtf < 1.0 for r in recs)  # faster than real time


async def test_event_loop_stays_responsive_while_backend_computes():
    class Slow(ToneMockBackend):
        def stream(self, text, lang):
            time.sleep(0.3)
            yield from super().stream(text, lang)

    s = TTSService(settings(), Slow())
    task = asyncio.create_task(s.synthesize("Hello world.", "en"))
    ticks = 0
    while not task.done():
        await asyncio.sleep(0.02)
        ticks += 1
    assert ticks >= 8 and len(task.result()) > 0
    await s.aclose()


async def test_concurrent_streams_are_independent(svc):
    texts = ["Hello there.", "Good morning everyone.", "Can you hear me?", LONG_TEXT]
    results = await asyncio.gather(*(svc.synthesize(t, "en", f"s{i}") for i, t in enumerate(texts)))
    singles = [await svc.synthesize(t, "en") for t in texts]
    assert all(np.array_equal(a, b) for a, b in zip(results, singles))


# ======================================================== cancellation and errors
async def test_closing_the_stream_early_cancels_the_worker():
    pieces = []

    class Endless(ToneMockBackend):
        def stream(self, text, lang):
            for i in range(200):
                pieces.append(i)
                time.sleep(0.01)
                yield np.full(OUT_RATE // 10, 0.1, dtype=np.float32), OUT_RATE

    s = TTSService(settings(), Endless())
    agen = s.synthesize_stream("never ending", "en")
    first = await agen.__anext__()
    await agen.aclose()  # consumer walked away (e.g. newer segment arrived)
    await asyncio.sleep(0.3)
    assert first.audio.size > 0 and len(pieces) < 100  # worker stopped long before 200 pieces
    assert s.stats.cancelled == 1 and s.stats.records == []
    await s.aclose()


async def test_backend_error_is_raised_to_the_consumer():
    class Boom(ToneMockBackend):
        def stream(self, text, lang):
            raise RuntimeError("voice crashed")
            yield  # pragma: no cover

    s = TTSService(settings(), Boom())
    with pytest.raises(RuntimeError, match="voice crashed"):
        await s.synthesize("hello", "en")
    assert await TTSService(settings(), ToneMockBackend()).synthesize("still works", "en") is not None
    await s.aclose()


async def test_empty_text_yields_nothing_and_bad_languages_raise(svc):
    assert [f async for f in svc.synthesize_stream("   ", "en")] == []
    with pytest.raises(UnsupportedLanguage):
        await svc.synthesize("hello", "klingon")
    assert svc.supports("de") and not svc.supports("klingon")


async def test_unsupported_voice_is_reported(tmp_path):
    s = TTSService(settings(tts_engine="piper", tts_voices_dir=tmp_path), PiperBackend(settings(tts_voices_dir=tmp_path)))
    assert not s.supports("en")
    with pytest.raises(UnsupportedVoice, match="no voice"):
        await s.synthesize("hello", "en")
    await s.aclose()


# ==================================================================== Piper backend
class FakePiperVoice:
    """MOCK of piper.PiperVoice (piper-tts 1.8.0): synthesize() yields one AudioChunk per sentence."""

    def __init__(self, rate=22050, path=None):
        self.rate, self.path, self.texts = rate, path, []

    def synthesize(self, text, syn_config=None, include_alignments=False):
        self.texts.append(text)
        for sentence in [s for s in text.replace("?", ".").split(".") if s.strip()]:
            n = int(self.rate * 0.05 * len(sentence.split()))
            audio = (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / self.rate)).astype(np.float32)
            yield SimpleNamespace(audio_float_array=audio, sample_rate=self.rate, sample_width=2, sample_channels=1)


def make_voices(tmp_path: Path, names=("en_US-lessac-low", "de_DE-thorsten-medium", "es")):
    for n in names:
        (tmp_path / f"{n}.onnx").write_bytes(b"x")
        (tmp_path / f"{n}.onnx.json").write_text("{}")


def test_piper_backend_discovers_voices_by_language(tmp_path):
    make_voices(tmp_path)
    b = PiperBackend(settings(tts_voices_dir=tmp_path), voice_loader=lambda p: FakePiperVoice(path=p))
    assert b.supports("en") and b.supports("en-US") and b.supports("de") and b.supports("es")
    assert not b.supports("fr") and not b.supports("klingon")
    assert b.voice_file("en").name == "en_US-lessac-low.onnx"
    assert b.voice_file("es").name == "es.onnx"
    assert PiperBackend(settings(tts_voices_dir=tmp_path / "missing")).supports("en") is False


def test_piper_backend_loads_each_voice_once_and_streams_chunks(tmp_path):
    make_voices(tmp_path)
    loads = []
    b = PiperBackend(settings(tts_voices_dir=tmp_path), voice_loader=lambda p: (loads.append(p), FakePiperVoice(path=p))[1])
    pieces = list(b.stream("First sentence here. Second one?", "en"))
    assert len(pieces) == 2 and all(a.dtype == np.float32 and r == 22050 for a, r in pieces)
    list(b.stream("Again.", "en"))
    list(b.stream("Hallo.", "de"))
    assert [p.name for p in loads] == ["en_US-lessac-low.onnx", "de_DE-thorsten-medium.onnx"]


async def test_service_resamples_piper_output_to_the_output_rate(tmp_path):
    make_voices(tmp_path)
    backend = PiperBackend(settings(tts_voices_dir=tmp_path), voice_loader=lambda p: FakePiperVoice(rate=22050, path=p))
    s = TTSService(settings(tts_voices_dir=tmp_path), backend)
    frames = [f async for f in s.synthesize_stream("One two three. Four five six seven.", "en")]
    assert all(f.sample_rate == OUT_RATE for f in frames)
    total_s = sum(len(f.audio) for f in frames) / OUT_RATE
    expected = (3 + 4) * 0.05  # fake voice emits 50 ms per word
    assert total_s == pytest.approx(expected, abs=0.02)  # resampling preserved duration
    await s.aclose()


def test_create_backend_selection(tmp_path):
    assert isinstance(create_backend(settings(tts_engine="mock")), ToneMockBackend)
    assert isinstance(create_backend(settings(tts_engine="piper", backend_mode="mock")), ToneMockBackend)
    assert isinstance(create_backend(settings(tts_engine="piper", tts_voices_dir=tmp_path)), PiperBackend)


def test_the_real_piper_package_is_importable_with_the_documented_api():
    """Guards the API assumptions in PiperBackend against the installed piper-tts."""
    import inspect

    from piper import PiperVoice

    assert "model_path" in inspect.signature(PiperVoice.load).parameters
    assert "text" in inspect.signature(PiperVoice.synthesize).parameters


# ============================================================== real voice (opt-in)
REAL_DIR = os.environ.get("TTS_TEST_VOICE_DIR")


@pytest.mark.slow
@pytest.mark.skipif(not REAL_DIR, reason="set TTS_TEST_VOICE_DIR to a directory with a Piper voice to run")
async def test_real_piper_voice_streams_audio_within_the_latency_budget():
    lang = os.environ.get("TTS_TEST_LANG", "en")
    cfg = settings(tts_engine="piper", tts_voices_dir=Path(REAL_DIR))
    s = TTSService(cfg, PiperBackend(cfg))
    await s.synthesize("Warm up.", lang)  # model load excluded from the latency measurement
    audio = await s.synthesize("Good morning everyone. Please share your screen.", lang, "real")
    rec = s.stats.records[-1]
    assert len(audio) > OUT_RATE * 0.5 and np.isfinite(audio).all() and np.abs(audio).max() > 0.01
    assert rec.first_chunk_ms < cfg.budget.tts_ms, f"first chunk {rec.first_chunk_ms:.0f} ms"
    await s.aclose()
