"""Module 7: end-to-end meeting flow and latency.

The REAL code path runs: LiveKitGateway (track handling, attributes, data packets) -> GatewayCore ->
Silero VAD (real ONNX model) -> ASRService/StreamingASR (LocalAgreement) -> TranslationService (batching)
-> TTSService -> LiveKitAudioPublisher, all coordinated by MeetingPipeline.

What is NOT real (no LiveKit server and no model weights in the offline sandbox; all clearly mocks):
* ``FakeRoom`` stands in for ``rtc.Room`` (same call shapes the gateway uses, checked against livekit 1.3.0).
* ``ReplayASRBackend`` replays a scripted transcript over a real recording; ``LexiconMockBackend`` and
  ``ToneMockBackend`` replace NLLB and Piper. So with the plain mocks the measured latency is the cost of
  the orchestration + the VAD end-pointing hang-over (real, wall-clock). The ``Slow*`` wrappers add
  model-like delays so the < 2.0 s budget is also checked under load that looks like real models.

Latency is measured on the wall clock, in real time: audio frames are fed at 1x speed and the time
from the end of the speech (stream time at which the speaker's last word ended, ``GateEvent.speech_end``, so the VAD silence
hang-over is part of the measurement) to the moment the final subtitle packet / first TTS frame is handed to LiveKit is taken.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from livekit import rtc

from gateway.audio_out import LiveKitAudioPublisher
from gateway.livekit_gateway import LiveKitGateway
from pipeline.meeting import MeetingPipeline
from services.asr.backends import ReplayASRBackend
from services.asr.hypothesis import Word
from services.translation.backends import _PHRASES, LexiconMockBackend
from services.tts.backends import ToneMockBackend
from services.vad.streaming_vad import SpeechSegmenter, StreamingVAD
from shared.audio import pcm16_to_float32
from shared.config import Settings
from shared.types import SubtitleMessage

DATA = Path(__file__).resolve().parents[1] / "data"
SECRET = "secret_that_is_at_least_32_characters_long"
SR = 16000
FRAME = 320  # 20 ms
PHRASES = ["Good morning everyone", "Please share your screen", "Hello, how are you"]


def settings(**kw) -> Settings:
    kw.setdefault("livekit_api_key", "devkey")
    kw.setdefault("livekit_api_secret", SECRET)
    return Settings(_env_file=None, **kw)


# ------------------------------------------------------------------ fakes (MOCKS of LiveKit objects)
class FakeLocalParticipant:
    def __init__(self):
        self.data: list[dict] = []
        self.tracks: list = []

    async def publish_data(self, payload, *, reliable=True, destination_identities=(), topic=""):
        self.data.append(dict(t=time.monotonic(), payload=payload, reliable=reliable, topic=topic, dest=list(destination_identities)))

    async def publish_track(self, track, options):
        self.tracks.append((track, options))


class FakeRoom:
    def __init__(self):
        self.handlers: dict[str, list] = {}
        self.local_participant = FakeLocalParticipant()
        self.remote_participants: dict = {}

    def on(self, event, cb):
        self.handlers.setdefault(event, []).append(cb)

    def emit(self, event, *args):
        for cb in list(self.handlers.get(event, [])):
            cb(*args)

    async def connect(self, url, token):
        pass

    async def disconnect(self):
        pass


def person(identity: str, **attrs) -> SimpleNamespace:
    return SimpleNamespace(identity=identity, attributes=dict(attrs))


class FakeSource:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.frames: list[tuple[float, rtc.AudioFrame]] = []

    async def capture_frame(self, frame):
        self.frames.append((time.monotonic(), frame))


class PacedStream:
    """MOCK rtc.AudioStream that delivers 20 ms frames in real time (frame k is available at t0 + (k+1)*20 ms)."""

    def __init__(self, pcm: np.ndarray, pace: bool = True):
        self.pcm, self.pace, self.t0 = pcm, pace, None

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        self.t0 = time.monotonic()
        for k, i in enumerate(range(0, len(self.pcm), FRAME)):
            if self.pace:
                delay = self.t0 + (k + 1) * FRAME / SR - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)
            yield SimpleNamespace(frame=SimpleNamespace(data=memoryview(self.pcm[i : i + FRAME]), sample_rate=SR))

    async def aclose(self):
        pass


class Slow:
    """Adds a fixed delay to a mock backend method, imitating model compute time."""

    def __init__(self, inner, **delays):
        self._inner, self._delays = inner, delays

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name in self._delays and callable(attr):
            if name == "stream":  # generator: delay before the first audio piece
                def wrapped(*a, **k):
                    time.sleep(self._delays[name])
                    yield from attr(*a, **k)
                return wrapped

            def delayed(*a, **k):
                time.sleep(self._delays[name])
                return attr(*a, **k)
            return delayed
        return attr


# ------------------------------------------------------------------ scenario
def load_speech() -> tuple[np.ndarray, np.ndarray, list[Word]]:
    """(int16 pcm incl. 0.7 s trailing silence, the float32 reference the gateway will see, scripted words)."""
    with wave.open(str(DATA / "speech_en.wav")) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    pcm = np.concatenate([pcm, np.zeros(int(0.7 * SR), dtype=np.int16)])
    ref = pcm16_to_float32(pcm)
    seg = SpeechSegmenter(StreamingVAD(), "p")
    segs = seg.feed(ref) + seg.flush()
    assert len(segs) == len(PHRASES), "test clip is expected to contain 3 utterances"
    script: list[Word] = []
    for s, phrase in zip(segs, PHRASES):  # SYNTHETIC word timings inside each real speech span
        words = phrase.split()
        span = (s.end_time - s.start_time) - 0.1
        for j, word in enumerate(words):
            a = s.start_time + 0.05 + span * j / len(words)
            script.append(Word(a, a + span / len(words) * 0.9, " " + word))
    return pcm, ref, script


@dataclass
class MeetingRun:
    room: FakeRoom
    pipeline: MeetingPipeline
    audio: LiveKitAudioPublisher
    sources: dict[str, FakeSource]
    t0: float
    end_times: list[float]  # stream time of each VAD end event
    subtitles: list[tuple[float, SubtitleMessage]] = field(default_factory=list)

    def final_subtitles(self) -> list[tuple[float, SubtitleMessage]]:
        return [(t, m) for t, m in self.subtitles if m.is_final]

    def audio_latencies_ms(self) -> list[float]:
        """Speaker's last word -> first translated audio frame, one value per (utterance, language)."""
        out: list[float] = []
        for ms, end in zip(self.pipeline.milestones.values(), self.end_times):
            out += [(t - (self.t0 + end)) * 1000 for t in ms.audio_first.values()]
        return out

    def subtitle_latencies_ms(self) -> list[float]:
        return [(t - (self.t0 + end)) * 1000 for (t, _), end in zip(self.final_subtitles(), self.end_times)]


async def run_meeting(*, asr_delay=0.0, nmt_delay=0.0, tts_delay=0.0, pace=True, prefs=None) -> MeetingRun:
    s = settings()
    pcm, ref, script = load_speech()
    asr = ReplayASRBackend(ref, script, language="en")
    nmt = LexiconMockBackend()
    tts = ToneMockBackend()
    if asr_delay:
        asr = Slow(asr, transcribe=asr_delay)
    if nmt_delay:
        nmt = Slow(nmt, translate_batch=nmt_delay)
    if tts_delay:
        tts = Slow(tts, stream=tts_delay)

    room = FakeRoom()
    people = prefs or {
        "alice": dict(speak_lang="en", listen_lang="en"),
        "bob": dict(listen_lang="es"),
        "carol": dict(listen_lang="fr"),
        "dave": dict(listen_lang="en"),  # same as the source language: needs no translation
    }
    room.remote_participants = {name: person(name, **a) for name, a in people.items()}

    sources: dict[str, FakeSource] = {}

    def make_source(sr):
        src = FakeSource(sr)
        sources[f"src{len(sources)}"] = src
        return src

    audio = LiveKitAudioPublisher(
        room, s.tts_output_sample_rate, source_factory=make_source,
        track_factory=lambda name, src: SimpleNamespace(name=name, source=src),
    )
    end_times: list[float] = []
    holder: dict[str, MeetingPipeline] = {}

    def sink(pid, ev):
        if ev.kind == "end":
            end_times.append(ev.speech_end)  # when the speaker's last word ended, not when the gate closed
        return holder["p"].on_gate_event(pid, ev)

    stream = PacedStream(pcm, pace=pace)
    gateway = LiveKitGateway(
        s, sink, room=room, audio_stream_factory=lambda track: stream,
        on_prefs=lambda *a: holder["p"].set_prefs(*a), on_leave=lambda pid: holder["p"].forget(pid),
    )
    pipeline = holder["p"] = MeetingPipeline(s, asr, nmt, tts, gateway, audio)

    await gateway.connect("meeting")
    alice_track = SimpleNamespace(kind=rtc.TrackKind.KIND_AUDIO)
    room.emit("track_subscribed", alice_track, None, room.remote_participants["alice"])
    await asyncio.wait_for(gateway._tasks["alice"], timeout=60)
    await asyncio.wait_for(pipeline.drain(), timeout=60)
    await gateway.disconnect()
    await pipeline.aclose()

    run = MeetingRun(room, pipeline, audio, sources, stream.t0, end_times)
    for d in room.local_participant.data:
        assert d["topic"] == "subtitles" and d["reliable"] is True
        run.subtitles.append((d["t"], SubtitleMessage.from_bytes(d["payload"])))
    return run


@pytest.fixture(scope="module")
def realtime_run() -> MeetingRun:
    return asyncio.run(run_meeting())


# ------------------------------------------------------------------ tests
def test_every_utterance_is_translated_for_each_listener_language(realtime_run):
    finals = realtime_run.final_subtitles()
    assert [m.text for _, m in finals] == PHRASES
    for (_, m), phrase in zip(finals, PHRASES):
        assert m.speaker == "alice" and m.source_lang == "en"
        assert set(m.translations) == {"es", "fr"}  # carol=fr, bob=es; dave (en) and alice herself need none
        assert m.translations["es"] == _PHRASES[phrase.lower()]["es"]
        assert m.translations["fr"] == _PHRASES[phrase.lower()]["fr"]
    assert [m.segment_id for _, m in finals] == ["alice-1", "alice-2", "alice-3"]


def test_partials_precede_their_final_and_nothing_follows_a_final(realtime_run):
    by_seg: dict[str, list[bool]] = {}
    for _, m in realtime_run.subtitles:
        by_seg.setdefault(m.segment_id, []).append(m.is_final)
    assert len(by_seg) == 3
    for flags in by_seg.values():
        assert flags[-1] is True and flags.count(True) == 1


def test_subtitle_latency_is_under_two_seconds_in_real_time(realtime_run):
    lat = realtime_run.subtitle_latencies_ms()
    assert len(lat) == 3
    assert all(0 <= x < 2000 for x in lat), lat
    # the pipeline's own accounting (from the VAD decision) agrees and no budget was exceeded
    assert realtime_run.pipeline.budget_violations == []
    internal = [m.subtitle_ms for m in realtime_run.pipeline.milestones.values()]
    assert all(x is not None and x < 2000 for x in internal)
    # the speaker-perceived figure additionally contains the VAD hang-over (>= min silence, minus one window of slack)
    hangover_ms = settings().vad_min_silence_ms
    assert all(w >= i + hangover_ms - 40 for w, i in zip(lat, internal)), (lat, internal)


def test_translated_speech_is_streamed_per_language_within_budget(realtime_run):
    run = realtime_run
    assert sorted(t.name for t, _ in run.room.local_participant.tracks) == ["tts-es", "tts-fr"]  # one track per language
    assert all(o.source == rtc.TrackSource.SOURCE_MICROPHONE for _, o in run.room.local_participant.tracks)
    assert len(run.sources) == 2
    for src in run.sources.values():
        total = sum(f.samples_per_channel for _, f in src.frames)
        assert all(f.sample_rate == 24000 and f.num_channels == 1 for _, f in src.frames)
        assert total > 24000  # three phrases of audio
    for ms in run.pipeline.milestones.values():
        assert set(ms.audio_first) == {"es", "fr"} and set(ms.audio_done) == {"es", "fr"}
    audio_lat = run.audio_latencies_ms()
    assert len(audio_lat) == 6 and all(0 < x < 2000 for x in audio_lat), audio_lat
    # utterances never overlap within a language: segment 2 audio starts after segment 1 audio ended
    ms = list(run.pipeline.milestones.values())
    for a, b in zip(ms, ms[1:]):
        assert b.audio_first["es"] >= a.audio_done["es"]


def test_the_web_client_can_read_what_the_gateway_publishes(realtime_run):
    node = shutil.which("node")
    client = Path(__file__).resolve().parents[2] / "client" / "subtitles.js"
    if node is None:
        pytest.skip("node is not installed")
    _, msg = realtime_run.final_subtitles()[0]
    script = (
        "const S=require(process.argv[1]);let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{"
        "const m=S.parseSubtitlePayload(d);console.log(JSON.stringify(S.pickText(m,{mode:'translated',lang:'es'})))});"
    )
    out = subprocess.run([node, "-e", script, str(client)], input=msg.to_bytes().decode(), capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["translated"] == msg.translations["es"]


@pytest.mark.parametrize("asr_s,nmt_s,tts_s", [(0.35, 0.15, 0.12)])
def test_latency_stays_under_two_seconds_with_model_like_delays(asr_s, nmt_s, tts_s):
    """Each ASR pass costs 350 ms, each NMT batch 150 ms, TTS needs 120 ms for its first audio."""
    run = asyncio.run(run_meeting(asr_delay=asr_s, nmt_delay=nmt_s, tts_delay=tts_s))
    lat = run.subtitle_latencies_ms()
    assert len(lat) == 3 and all(x < 2000 for x in lat), lat
    assert min(x for x in lat) > (asr_s + nmt_s) * 1000 * 0.9  # the delays really are in the measurement
    assert run.pipeline.budget_violations == []
    first_audio = run.audio_latencies_ms()
    assert len(first_audio) == 6 and all(x < 2000 for x in first_audio), first_audio


def test_the_measurement_detects_a_pipeline_that_is_too_slow():
    """Negative control: with a 1.6 s ASR the budget must be reported as exceeded."""
    run = asyncio.run(run_meeting(asr_delay=1.6, pace=False))
    assert any("asr" in v for v in run.pipeline.budget_violations), run.pipeline.budget_violations
    assert max(ms.subtitle_ms for ms in run.pipeline.milestones.values()) > 1600


# ------------------------------------------------------------- behaviour without real-time pacing
def test_speaker_alone_in_the_room_gets_original_only_subtitles_and_no_audio():
    run = asyncio.run(run_meeting(pace=False, prefs={"alice": dict(speak_lang="en", listen_lang="fr")}))
    finals = run.final_subtitles()
    assert [m.text for _, m in finals] == PHRASES
    assert all(m.translations == {} for _, m in finals)  # her own listen_lang is irrelevant for her own speech
    assert run.room.local_participant.tracks == [] and run.sources == {}


def test_unsupported_listener_language_is_ignored_and_others_still_served():
    run = asyncio.run(run_meeting(pace=False, prefs={
        "alice": dict(speak_lang="en"), "bob": dict(listen_lang="klingon"), "carol": dict(listen_lang="de"),
    }))
    assert all(set(m.translations) == {"de"} for _, m in run.final_subtitles())
    assert [t.name for t, _ in run.room.local_participant.tracks] == ["tts-de"]


def test_speak_lang_attribute_pins_asr_language_and_skips_detection():
    run = asyncio.run(run_meeting(pace=False))
    # alice announced speak_lang=en -> the backend was always called with an explicit language
    inner = run.pipeline.asr.backend
    assert inner.calls and all(c.language_arg == "en" for c in inner.calls)


def test_oversized_subtitle_is_split_per_language():
    from pipeline.meeting import MeetingPipeline as MP  # noqa: F401  (documented import path)

    class PickyPublisher:
        def __init__(self):
            self.sent: list[SubtitleMessage] = []

        async def publish_subtitle(self, message, destinations=None):
            if len(message.translations) > 1:
                raise ValueError("payload too large")
            self.sent.append(message)

    async def go():
        pub = PickyPublisher()
        pipe = MeetingPipeline(settings(), ReplayASRBackend(np.zeros(10, dtype=np.float32), []), LexiconMockBackend(), None, pub)
        msg = SubtitleMessage("s", "alice", "en", "hi", {"es": "hola", "fr": "salut"}, is_final=True)
        await pipe._publish(msg)
        await pipe.aclose()
        return pub.sent

    sent = asyncio.run(go())
    assert [m.translations for m in sent] == [{}, {"es": "hola"}, {"fr": "salut"}]
    assert all(m.segment_id == "s" and m.is_final for m in sent)


def test_leaving_participants_are_forgotten():
    async def go():
        pipe = MeetingPipeline(settings(), ReplayASRBackend(np.zeros(10, dtype=np.float32), []), LexiconMockBackend(), None, SimpleNamespace())
        pipe.set_prefs("bob", "es", "en")
        assert pipe.listener_languages("alice", "en") == ["es"]
        pipe.forget("bob")
        assert pipe.listener_languages("alice", "en") == []
        await pipe.aclose()

    asyncio.run(go())
