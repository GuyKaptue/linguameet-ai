"""Module 2: streaming VAD.

Two kinds of tests:
* **Real model** - the actual Silero VAD ONNX model on real speech (public-domain JFK clip,
  ``tests/data/jfk.wav``) and on non-speech buffers (silence, noise, tone).
* **Mock model** - ``MockEnergyVAD`` below is a MOCK (energy threshold, not Silero). It makes the
  gate's state machine (debounce, hysteresis, pre-roll, forced split, flush) exactly predictable.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from services.vad.silero_onnx import SileroOnnxModel
from services.vad.streaming_vad import GateEvent, SpeechSegmenter, StreamingVAD
from shared.audio import pcm16_to_float32, sine
from shared.config import Settings

SR = 16000
W = 512
JFK = Path(__file__).resolve().parents[2] / "tests" / "data" / "jfk.wav"


def load_jfk() -> np.ndarray:
    with wave.open(str(JFK)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        return pcm16_to_float32(w.readframes(w.getnframes()))


def segments_of(audio: np.ndarray, chunk: int, **vad_kwargs):
    seg = SpeechSegmenter(StreamingVAD(**vad_kwargs), "p")
    out = []
    for i in range(0, len(audio), chunk):
        out += seg.feed(audio[i : i + chunk])
    out += seg.flush()
    return out


# =========================================================== MOCK model (labelled)
class MockEnergyVAD:
    """MOCK VAD model: probability 0.95 if window RMS > 0.05 else 0.02. Not Silero."""

    sample_rate = SR
    window_samples = W

    def __init__(self):
        self.resets = 0

    def __call__(self, window: np.ndarray) -> float:
        return 0.95 if float(np.sqrt(np.mean(window**2))) > 0.05 else 0.02

    def reset_states(self) -> None:
        self.resets += 1


def loud(windows: int, value: float = 0.3) -> np.ndarray:
    return np.full(windows * W, value, dtype=np.float32)


def quiet(windows: int) -> np.ndarray:
    return np.zeros(windows * W, dtype=np.float32)


def mock_vad(**kw) -> StreamingVAD:
    kw.setdefault("min_silence_ms", 320)  # 10 windows
    kw.setdefault("speech_pad_ms", 96)  # 3 windows
    kw.setdefault("min_speech_ms", 64)  # 2 windows
    return StreamingVAD(MockEnergyVAD(), **kw)


def kinds(events: list[GateEvent]) -> list[str]:
    return [e.kind for e in events]


# ============================================================ real Silero model
@pytest.fixture(scope="module")
def jfk() -> np.ndarray:
    return load_jfk()


def test_real_speech_is_detected_as_separate_phrases(jfk):
    segs = segments_of(jfk, 320)
    assert 3 <= len(segs) <= 6  # the clip has four phrases separated by pauses
    assert 0.0 <= segs[0].start_time < 0.6  # speech starts almost immediately
    assert segs[-1].end_time <= len(jfk) / SR + 0.05
    for a, b in zip(segs, segs[1:]):
        assert a.end_time <= b.start_time + 0.2  # ordered, essentially non-overlapping
    speech_s = sum(s.duration for s in segs)
    assert 0.7 * len(jfk) / SR < speech_s <= len(jfk) / SR + 0.1  # most of the clip is speech


def test_segmentation_is_independent_of_chunk_size(jfk):
    ref = [(round(s.start_time, 2), round(s.end_time, 2)) for s in segments_of(jfk, 512)]
    for chunk in (160, 333, 1000, 4096):
        got = [(round(s.start_time, 2), round(s.end_time, 2)) for s in segments_of(jfk, chunk)]
        assert got == ref, f"chunk={chunk}"


def test_silence_noise_and_tone_are_filtered_out():
    rng = np.random.default_rng(42)
    for name, buf in {
        "zeros": np.zeros(5 * SR, dtype=np.float32),
        "noise": rng.normal(0, 0.01, 5 * SR).astype(np.float32),
        "tone": sine(440, 5.0, SR),
    }.items():
        vad = StreamingVAD()
        events = vad.feed(buf) + vad.flush()
        assert events == [], f"{name} produced events"


def test_speech_surrounded_by_silence_is_located_correctly(jfk):
    pad = np.zeros(3 * SR, dtype=np.float32)
    audio = np.concatenate([pad, jfk, pad])
    segs = segments_of(audio, 480)
    assert segs and segs[0].start_time >= 3.0 - 0.01  # nothing emitted from the leading silence
    assert segs[-1].end_time <= 3.0 + len(jfk) / SR + 1.0  # at most the silence tail afterwards
    emitted = sum(len(s.audio) for s in segs) / SR
    assert emitted < len(jfk) / SR + 0.5  # the 6 s of silence did not leak through


def test_gate_events_are_well_ordered(jfk):
    vad = StreamingVAD()
    events = []
    for i in range(0, len(jfk), 480):
        events += vad.feed(jfk[i : i + 480])
    events += vad.flush()
    state = "idle"
    for ev in events:
        if ev.kind == "start":
            assert state == "idle"
            state = "open"
        elif ev.kind == "audio":
            assert state == "open"
        else:
            assert state == "open"
            state = "idle"
    assert state == "idle"
    assert events[0].kind == "start" and events[0].audio.size > 0  # pre-roll attached to start


def test_end_event_reports_where_speech_stopped_and_the_gate_hangover(jfk):
    audio = np.concatenate([jfk, np.zeros(16000, dtype=np.float32)])
    vad = StreamingVAD()
    ends = [e for e in vad.feed(audio) + vad.flush() if e.kind == "end" and not e.forced]
    assert len(ends) >= 3
    window_s = 512 / 16000
    for e in ends:
        hangover = e.time - e.speech_end
        # the gate closes after min_silence (300 ms) of silence, give or take one 32 ms window
        assert 0.3 - 1e-6 <= hangover <= 0.3 + 2 * window_s, hangover


def test_real_model_is_fast_enough_for_realtime(jfk):
    vad = StreamingVAD()
    vad.feed(jfk)
    ms_per_window = vad.inference_seconds / vad.windows_processed * 1000
    assert vad.windows_processed == len(jfk) // W
    assert ms_per_window < 10.0  # a window is 32 ms of audio; ~0.1 ms measured on CPU


def test_silero_wrapper_validates_input_and_resets_state(jfk):
    model = SileroOnnxModel(16000)
    with pytest.raises(ValueError, match="512-sample"):
        model(np.zeros(100, dtype=np.float32))
    with pytest.raises(ValueError):
        SileroOnnxModel(44100)
    with pytest.raises(FileNotFoundError):
        SileroOnnxModel(16000, model_path="/nonexistent.onnx")
    wins = [jfk[i * W : (i + 1) * W] for i in range(60, 70)]
    first = [model(w) for w in wins]
    assert all(0.0 <= p <= 1.0 for p in first)
    model.reset_states()
    assert [model(w) for w in wins] == pytest.approx(first)  # reset => reproducible
    # state really is carried between calls: same window, different state, different output
    model.reset_states()
    a = model(wins[0])
    b = model(wins[0])
    assert a != b


def test_silero_8khz_model_variant():
    model = SileroOnnxModel(8000)
    assert model.window_samples == 256
    p = model(np.zeros(256, dtype=np.float32))
    assert 0.0 <= p < 0.5


def test_streams_do_not_share_state(jfk):
    a, b = SileroOnnxModel(16000), SileroOnnxModel(16000)
    for i in range(40, 50):
        a(jfk[i * W : (i + 1) * W])
    assert b(jfk[40 * W : 41 * W]) != a(jfk[40 * W : 41 * W])


def test_from_settings_uses_configured_values():
    s = Settings(_env_file=None, vad_threshold=0.6, vad_min_silence_ms=500, vad_max_segment_seconds=10.0)
    vad = StreamingVAD.from_settings(s)
    assert vad.threshold == 0.6 and vad.sample_rate == 16000


# ============================================== state machine, with the MOCK model
def run(vad: StreamingVAD, audio: np.ndarray) -> list[GateEvent]:
    return vad.feed(audio) + vad.flush()


def test_mock_basic_start_audio_end_with_preroll():
    vad = mock_vad()
    events = vad.feed(np.concatenate([quiet(10), loud(20), quiet(15)]))
    assert kinds(events) == ["start", "audio", "end"]
    start, audio, end = events
    # trigger after 2 loud windows; pre-roll = those 2 windows + 3 pad windows (silence)
    assert len(start.audio) == 5 * W
    assert start.time == pytest.approx((10 - 3) * W / SR)
    assert start.audio[: 3 * W].max() == 0.0 and start.audio[3 * W :].min() == pytest.approx(0.3)
    # Like silero's VADIterator, silence is timed from the END of the first silent window, so the
    # segment closes on the 11th silent window (min_silence = 10 windows after that first one).
    assert end.time == pytest.approx((10 + 20 + 11) * W / SR)
    assert len(audio.audio) == (18 + 11) * W  # remaining speech + silence tail up to the end


def test_mock_click_shorter_than_min_speech_is_ignored():
    vad = mock_vad()
    assert run(vad, np.concatenate([quiet(5), loud(1), quiet(20)])) == []


def test_mock_short_pause_does_not_split_but_long_pause_does():
    short_gap = np.concatenate([loud(10), quiet(5), loud(10), quiet(15)])  # gap < 10 windows
    segs = SpeechSegmenter(mock_vad(), "p")
    out = segs.feed(short_gap) + segs.flush()
    assert len(out) == 1
    long_gap = np.concatenate([loud(10), quiet(14), loud(10), quiet(15)])  # gap > 10 windows
    segs = SpeechSegmenter(mock_vad(), "p")
    out = segs.feed(long_gap) + segs.flush()
    # pre-roll never reaches back into audio already delivered, so segments abut but never overlap
    assert len(out) == 2 and out[0].end_time <= out[1].start_time


def test_mock_forced_split_at_max_segment_length():
    vad = mock_vad(max_segment_s=1.0)  # 31.25 windows
    events = vad.feed(loud(100)) + vad.flush()
    forced = [e for e in events if e.kind == "end" and e.forced]
    starts = [e for e in events if e.kind == "start"]
    # 100 windows, split every 32 windows (first window count that reaches 1.0 s) -> forced ends at 32, 64, 96
    assert len(forced) == 3 and len(starts) == 4
    # every segment (start..end) is no longer than ~max length plus one window
    segs = SpeechSegmenter(mock_vad(max_segment_s=1.0), "p")
    out = segs.feed(loud(100)) + segs.flush()
    assert all(s.duration <= 1.0 + 2 * W / SR for s in out)
    assert sum(len(s.audio) for s in out) >= 95 * W  # nothing dropped by the split


def test_mock_flush_closes_open_segment_and_pads_partial_window():
    vad = mock_vad()
    events = vad.feed(np.concatenate([loud(10), np.full(100, 0.3, dtype=np.float32)]))
    assert kinds(events)[-1] != "end"
    assert vad.in_speech
    tail = vad.flush()
    assert kinds(tail)[-1] == "end"
    assert not vad.in_speech


def test_mock_chunk_boundaries_do_not_change_events():
    audio = np.concatenate([quiet(7), loud(12), quiet(14), loud(9), quiet(12)])
    ref = SpeechSegmenter(mock_vad(), "p")
    expected = [(s.start_time, s.end_time, len(s.audio)) for s in ref.feed(audio) + ref.flush()]
    for chunk in (1, 100, 511, 513, 4000):
        sg = SpeechSegmenter(mock_vad(), "p")
        got = []
        for i in range(0, len(audio), chunk):
            got += sg.feed(audio[i : i + chunk])
        got += sg.flush()
        assert [(s.start_time, s.end_time, len(s.audio)) for s in got] == expected, chunk


def test_mock_reset_clears_state_and_resets_model():
    vad = mock_vad()
    vad.feed(loud(10))
    assert vad.in_speech
    before = vad.model.resets
    vad.reset()
    assert not vad.in_speech and vad.stream_time == 0.0 and vad.model.resets == before + 1


def test_segment_ids_and_metadata():
    sg = SpeechSegmenter(mock_vad(), "alice")
    out = sg.feed(np.concatenate([loud(10), quiet(15), loud(10), quiet(15)])) + sg.flush()
    assert [s.segment_id for s in out] == ["alice-1", "alice-2"]
    assert all(s.participant_id == "alice" and s.sample_rate == SR and s.is_final for s in out)
    assert all(s.audio.dtype == np.float32 and s.created_at > 0 for s in out)


def test_input_validation():
    vad = mock_vad()
    with pytest.raises(ValueError, match="1-D"):
        vad.feed(np.zeros((2, 512), dtype=np.float32))
    with pytest.raises(ValueError):
        StreamingVAD(MockEnergyVAD(), threshold=1.5)
