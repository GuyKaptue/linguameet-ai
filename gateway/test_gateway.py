"""Module 2: gateway - VAD gating per participant, LiveKit track handling, tokens, data publishing.

No LiveKit server is available in the test environment, so ``FakeRoom`` / ``FakeAudioStream``
(MOCKS of ``rtc.Room`` / ``rtc.AudioStream`` with the same call shapes) drive the real
``LiveKitGateway`` code. Tokens use the real ``livekit.api`` signing and verification.
"""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from livekit import api, rtc

from gateway.core import GatewayCore
from gateway.livekit_gateway import SUBTITLE_TOPIC, LiveKitGateway
from gateway.tokens import GATEWAY_IDENTITY, create_token, gateway_token
from shared.audio import float32_to_pcm16, pcm16_to_float32, resample
from shared.config import ConfigError, Settings
from shared.types import SubtitleMessage

JFK = Path(__file__).resolve().parents[1] / "tests" / "data" / "jfk.wav"
SECRET = "secret_that_is_at_least_32_characters_long"


def settings(**kw) -> Settings:
    kw.setdefault("livekit_api_key", "devkey")
    kw.setdefault("livekit_api_secret", SECRET)
    return Settings(_env_file=None, **kw)


@pytest.fixture(scope="module")
def jfk_pcm() -> np.ndarray:
    with wave.open(str(JFK)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


class Collector:
    def __init__(self):
        self.events: list[tuple[str, str, int]] = []  # (participant, kind, n_samples)

    async def __call__(self, pid, ev):  # async sink
        self.events.append((pid, ev.kind, len(ev.audio)))

    def kinds(self, pid):
        return [k for p, k, _ in self.events if p == pid]


# --------------------------------------------------------------------- core
async def test_core_gates_real_speech_in_10ms_frames(jfk_pcm):
    sink = Collector()
    core = GatewayCore(settings(), sink)
    for i in range(0, len(jfk_pcm), 160):  # 10 ms LiveKit-sized frames
        await core.push_pcm16("alice", jfk_pcm[i : i + 160], 16000)
    await core.remove_participant("alice")
    ks = sink.kinds("alice")
    assert ks.count("start") == ks.count("end") >= 3
    assert core.participants == []
    assert core.stats["samples_in"] == len(jfk_pcm)
    assert core.stats["speech_samples_out"] < core.stats["samples_in"]  # pauses were dropped


async def test_core_resamples_48khz_input(jfk_pcm):
    sink = Collector()
    core = GatewayCore(settings(), sink)
    audio48 = resample(pcm16_to_float32(jfk_pcm), 16000, 48000)
    pcm48 = float32_to_pcm16(audio48)
    for i in range(0, len(pcm48), 480):
        await core.push_pcm16("bob", pcm48[i : i + 480], 48000)
    await core.remove_participant("bob")
    assert sink.kinds("bob").count("start") >= 3


async def test_core_keeps_participants_independent(jfk_pcm):
    sink = Collector()
    core = GatewayCore(settings(), sink)
    silence = np.zeros(16000 * 3, dtype=np.int16)
    for i in range(0, len(jfk_pcm), 1600):
        await core.push_pcm16("talker", jfk_pcm[i : i + 1600])
        await core.push_pcm16("listener", silence[i : i + 1600] if i < len(silence) else silence[:1600])
    await core.remove_participant("talker")
    await core.remove_participant("listener")
    assert sink.kinds("talker").count("start") >= 3
    assert sink.kinds("listener") == []  # silent participant never produces speech


async def test_core_accepts_sync_sink_bytes_and_memoryview(jfk_pcm):
    got = []
    core = GatewayCore(settings(), lambda pid, ev: got.append(ev.kind))
    await core.push_pcm16("x", jfk_pcm[:16000].tobytes())
    await core.push_pcm16("x", memoryview(jfk_pcm[16000:32000]))
    await core.remove_participant("x")
    assert "start" in got
    assert await core.remove_participant("unknown") == []


# --------------------------------------------------------------- LiveKit layer
class FakeParticipant:
    def __init__(self, identity):
        self.identity = identity


class FakeTrack:
    def __init__(self, kind=rtc.TrackKind.KIND_AUDIO):
        self.kind = kind


class FakeAudioStream:
    """MOCK of rtc.AudioStream: async-iterable of events with .frame.data/.sample_rate."""

    def __init__(self, pcm: np.ndarray, frame: int = 160, sample_rate: int = 16000, then_hang: bool = False):
        self._pcm, self._frame, self._sr, self._hang = pcm, frame, sample_rate, then_hang
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for i in range(0, len(self._pcm), self._frame):
            chunk = self._pcm[i : i + self._frame]
            yield SimpleNamespace(frame=SimpleNamespace(data=memoryview(chunk), sample_rate=self._sr))
            await asyncio.sleep(0)
        if self._hang:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class FakeLocalParticipant:
    def __init__(self):
        self.published: list[dict] = []

    async def publish_data(self, payload, *, reliable=True, destination_identities=(), topic=""):
        self.published.append(
            dict(payload=payload, reliable=reliable, destination_identities=list(destination_identities), topic=topic)
        )


class FakeRoom:
    """MOCK of rtc.Room: records handlers registered with .on(event, callback)."""

    def __init__(self):
        self.handlers: dict[str, list] = {}
        self.local_participant = FakeLocalParticipant()
        self.connected_with = None
        self.disconnected = False

    def on(self, event, callback=None):
        self.handlers.setdefault(event, []).append(callback)
        return callback

    async def connect(self, url, token):
        self.connected_with = (url, token)

    async def disconnect(self):
        self.disconnected = True

    def emit(self, event, *args):
        for cb in self.handlers.get(event, []):
            cb(*args)


async def test_gateway_subscribes_to_audio_tracks_and_gates_speech(jfk_pcm):
    sink = Collector()
    room = FakeRoom()
    streams = {}

    def factory(track):
        streams[id(track)] = FakeAudioStream(jfk_pcm)
        return streams[id(track)]

    gw = LiveKitGateway(settings(), sink, room=room, audio_stream_factory=factory)
    await gw.connect("meeting-1")
    assert room.connected_with[0] == "ws://localhost:7880"
    assert set(room.handlers) == {
        "track_subscribed", "track_unsubscribed", "participant_disconnected",
        "participant_connected", "participant_attributes_changed",
    }

    audio_track, video_track = FakeTrack(), FakeTrack(rtc.TrackKind.KIND_VIDEO)
    room.emit("track_subscribed", video_track, None, FakeParticipant("cam-only"))  # ignored
    room.emit("track_subscribed", audio_track, None, FakeParticipant("alice"))
    await asyncio.wait_for(gw._tasks["alice"], timeout=20)  # reader drains the finite stream
    assert gw.core.participants == ["alice"] and "cam-only" not in gw.core.participants
    await gw.disconnect()
    assert room.disconnected and gw.core.participants == []
    assert streams[id(audio_track)].closed
    assert sink.kinds("alice").count("start") >= 3
    assert sink.kinds("alice").count("start") == sink.kinds("alice").count("end")


async def test_gateway_closes_open_segment_when_participant_leaves(jfk_pcm):
    sink = Collector()
    room = FakeRoom()
    loud_part = jfk_pcm[: 16000 * 2]  # stream stops mid-speech and then hangs
    gw = LiveKitGateway(
        settings(), sink, room=room, audio_stream_factory=lambda t: FakeAudioStream(loud_part, then_hang=True)
    )
    gw.register_handlers()
    room.emit("track_subscribed", FakeTrack(), None, FakeParticipant("carol"))
    for _ in range(2000):
        await asyncio.sleep(0.001)
        if gw.core.stats["samples_in"] >= len(loud_part):
            break
    assert sink.kinds("carol")[-1] != "end"  # still inside an utterance
    room.emit("participant_disconnected", FakeParticipant("carol"))
    for _ in range(200):
        await asyncio.sleep(0.005)
        if sink.kinds("carol")[-1] == "end":
            break
    assert sink.kinds("carol")[-1] == "end"
    assert "carol" not in gw.core.participants and "carol" not in gw._tasks


async def test_gateway_survives_a_broken_stream(jfk_pcm):
    class Broken(FakeAudioStream):
        async def _gen(self):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    sink, room = Collector(), FakeRoom()
    gw = LiveKitGateway(settings(), sink, room=room, audio_stream_factory=lambda t: Broken(jfk_pcm))
    gw.register_handlers()
    room.emit("track_subscribed", FakeTrack(), None, FakeParticipant("dave"))
    await asyncio.wait_for(gw._tasks["dave"], timeout=5)  # task ends cleanly, exception is logged
    await gw.disconnect()


async def test_gateway_default_stream_factory_uses_configured_rate():
    gw = LiveKitGateway(settings(audio_sample_rate=8000), Collector(), room=FakeRoom())
    # constructing a real rtc.AudioStream needs a live room; check the arguments we would pass instead
    import inspect

    src = inspect.getsource(gw._default_stream_factory)
    assert "sample_rate=self.settings.audio_sample_rate" in src and "num_channels=1" in src


async def test_publish_subtitle_uses_reliable_topic_and_destinations():
    room = FakeRoom()
    gw = LiveKitGateway(settings(), Collector(), room=room)
    msg = SubtitleMessage("s1", "alice", "de", "Hallo", {"en": "Hello"}, True)
    await gw.publish_subtitle(msg)
    await gw.publish_subtitle(msg, destinations=["bob"])
    first, second = room.local_participant.published
    assert first["reliable"] is True and first["topic"] == SUBTITLE_TOPIC == "subtitles"
    assert first["destination_identities"] == [] and second["destination_identities"] == ["bob"]
    assert SubtitleMessage.from_bytes(first["payload"]).translations == {"en": "Hello"}


# ----------------------------------------------------------------------- tokens
def test_token_roundtrip_with_real_livekit_signing():
    s = settings()
    jwt = create_token(s, identity="alice", room="r1", name="Alice", can_publish_data=False)
    claims = api.TokenVerifier(s.livekit_api_key, s.livekit_api_secret).verify(jwt)
    assert claims.identity == "alice" and claims.name == "Alice"
    assert claims.video.room == "r1" and claims.video.room_join is True
    assert claims.video.can_publish_data is False and claims.video.can_subscribe is True


def test_gateway_token_identity_and_agent_grant():
    s = settings()
    claims = api.TokenVerifier(s.livekit_api_key, s.livekit_api_secret).verify(gateway_token(s, "r2"))
    assert claims.identity == GATEWAY_IDENTITY and claims.video.agent is True


def test_token_requires_credentials_and_arguments():
    with pytest.raises(ConfigError):
        create_token(Settings(_env_file=None), identity="a", room="r")
    with pytest.raises(ValueError):
        create_token(settings(), identity="", room="r")


def test_token_with_wrong_secret_is_rejected():
    jwt = create_token(settings(), identity="a", room="r")
    with pytest.raises(Exception):
        api.TokenVerifier("devkey", "another_secret_that_is_32_chars_long!").verify(jwt)


# ------------------------------------------------- participant preferences (attributes)
async def test_gateway_reports_language_preferences_from_participant_attributes():
    got = []
    room = FakeRoom()
    existing = FakeParticipant("early-bird")
    existing.attributes = {"listen_lang": "es", "speak_lang": "en"}
    room.remote_participants = {"early-bird": existing}
    left = []
    gw = LiveKitGateway(settings(), Collector(), room=room, on_prefs=lambda *a: got.append(a), on_leave=left.append)
    await gw.connect("m")
    assert got == [("early-bird", "es", "en")]  # people already in the room are picked up on join

    late = FakeParticipant("late")
    late.attributes = {"listen_lang": "fr"}  # speak_lang absent -> auto-detect
    room.emit("participant_connected", late)
    late.attributes = {"listen_lang": "de", "speak_lang": ""}
    room.emit("participant_attributes_changed", {"listen_lang": "de"}, late)
    assert got[1:] == [("late", "fr", None), ("late", "de", None)]

    ghost = FakeParticipant(GATEWAY_IDENTITY)
    ghost.attributes = {"listen_lang": "ja"}
    room.emit("participant_connected", ghost)  # another gateway instance is not a listener
    assert len(got) == 3
    room.emit("participant_disconnected", late)
    assert left == ["late"]
    await gw.disconnect()


async def test_gateway_never_transcribes_its_own_synthesized_tracks(jfk_pcm):
    sink = Collector()
    room = FakeRoom()
    gw = LiveKitGateway(settings(), sink, room=room, audio_stream_factory=lambda t: FakeAudioStream(jfk_pcm[:1600]))
    await gw.connect("m")
    room.emit("track_subscribed", FakeTrack(), None, FakeParticipant(GATEWAY_IDENTITY))
    assert gw._tasks == {} and gw.core.participants == []
    await gw.disconnect()


# ------------------------------------------------------------------- audio out
class FakeSource:
    def __init__(self, sr):
        self.sample_rate, self.frames, self.closed = sr, [], False

    async def capture_frame(self, frame):
        self.frames.append(frame)

    async def aclose(self):
        self.closed = True


class FakeAudioRoom:
    def __init__(self):
        self.published = []
        self.local_participant = SimpleNamespace(publish_track=self._publish)

    async def _publish(self, track, options):
        self.published.append((track, options))


async def test_audio_publisher_creates_one_track_per_language_and_sends_pcm16_frames():
    from gateway.audio_out import LiveKitAudioPublisher
    from shared.types import AudioOut

    room = FakeAudioRoom()
    sources = []

    def make_source(sr):
        sources.append(FakeSource(sr))
        return sources[-1]

    pub = LiveKitAudioPublisher(room, 24000, source_factory=make_source, track_factory=lambda name, src: SimpleNamespace(name=name, source=src))
    tone = (0.5 * np.sin(np.arange(2400) / 10)).astype(np.float32)
    await asyncio.gather(*(pub.publish_frame(AudioOut("s", "es", tone, 24000)) for _ in range(3)))  # concurrent first use
    await pub.publish_frame(AudioOut("s", "fr", tone, 24000, is_last=True))
    assert [t.name for t, _ in room.published] == ["tts-es", "tts-fr"]  # exactly one publication per language
    assert all(o.source == rtc.TrackSource.SOURCE_MICROPHONE for _, o in room.published)
    assert pub.languages == ["es", "fr"] and len(sources) == 2
    frame = sources[0].frames[0]
    assert isinstance(frame, rtc.AudioFrame)
    assert (frame.sample_rate, frame.num_channels, frame.samples_per_channel) == (24000, 1, 2400)
    assert np.array_equal(np.frombuffer(frame.data, dtype="<i2"), float32_to_pcm16(tone))
    with pytest.raises(ValueError, match="does not match"):
        await pub.publish_frame(AudioOut("s", "es", tone, 16000))
    await pub.aclose()
    assert all(s.closed for s in sources) and pub.languages == []


def test_participant_tokens_allow_publishing_language_attributes():
    s = settings()
    grants = api.TokenVerifier(s.livekit_api_key, s.livekit_api_secret).verify(create_token(s, identity="a", room="r")).video
    assert grants.can_update_own_metadata is True
    off = api.TokenVerifier(s.livekit_api_key, s.livekit_api_secret).verify(
        create_token(s, identity="a", room="r", can_update_own_metadata=False)
    ).video
    assert off.can_update_own_metadata is False
