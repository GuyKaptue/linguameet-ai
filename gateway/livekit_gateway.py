"""LiveKit gateway: joins a room as a server-side participant, subscribes to every
remote audio track, gates it with VAD and publishes subtitle data packets.

LiveKit Python SDK calls used (all checked against ``livekit`` 1.3.0 / docs.livekit.io):
``rtc.Room.connect(url, token)``, ``Room.on("track_subscribed" | "track_unsubscribed" |
"participant_disconnected", cb)``, ``rtc.AudioStream(track, sample_rate=, num_channels=)``
(async-iterable of ``AudioFrameEvent``, ``await stream.aclose()``) and
``room.local_participant.publish_data(payload, reliable=, destination_identities=, topic=)``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from livekit import rtc

from gateway.core import GatewayCore, Sink
from gateway.protocol import SUBTITLE_TOPIC, parse_prefs
from gateway.tokens import GATEWAY_IDENTITY, gateway_token
from shared.config import Settings
from shared.types import SubtitleMessage

log = logging.getLogger(__name__)

AudioStreamFactory = Callable[[Any], Any]
# (participant identity, listen_lang, speak_lang) - called on join and whenever attributes change
PrefsCallback = Callable[[str, "str | None", "str | None"], None]


class LiveKitGateway:
    def __init__(
        self,
        settings: Settings,
        sink: Sink,
        *,
        room: Any | None = None,
        audio_stream_factory: AudioStreamFactory | None = None,
        core: GatewayCore | None = None,
        on_prefs: PrefsCallback | None = None,
        on_leave: Callable[[str], None] | None = None,
    ):
        self.settings = settings
        self.room = room if room is not None else rtc.Room()
        self.core = core or GatewayCore(settings, sink)
        self._stream_factory = audio_stream_factory or self._default_stream_factory
        self._tasks: dict[str, asyncio.Task] = {}  # participant identity -> reader task
        self._streams: dict[str, Any] = {}
        self._registered = False
        self._on_prefs = on_prefs
        self._on_leave = on_leave

    # ------------------------------------------------------------- connection
    def _default_stream_factory(self, track: Any) -> rtc.AudioStream:
        # LiveKit converts to the requested rate / channel count for us.
        return rtc.AudioStream(track, sample_rate=self.settings.audio_sample_rate, num_channels=1)

    async def connect(self, room_name: str, url: str | None = None, token: str | None = None) -> None:
        self.register_handlers()
        token = token or gateway_token(self.settings, room_name)
        await self.room.connect(url or self.settings.livekit_url, token)
        log.info("gateway joined room %s", room_name)
        for participant in list(getattr(self.room, "remote_participants", {}).values()):
            self._announce_prefs(participant)  # attributes of people who were already in the room
        # Tracks that were already published before we joined arrive via track_subscribed as well.

    async def disconnect(self) -> None:
        for pid in list(self._tasks):
            await self._stop_reader(pid)
        await self.room.disconnect()

    def register_handlers(self) -> None:
        if self._registered:
            return
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)
        self.room.on("participant_disconnected", self._on_participant_disconnected)
        self.room.on("participant_connected", self._announce_prefs)
        self.room.on("participant_attributes_changed", self._on_attributes_changed)
        self._registered = True

    # --------------------------------------------------------------- handlers
    def _announce_prefs(self, participant: Any) -> None:
        if self._on_prefs is None or participant.identity == GATEWAY_IDENTITY:
            return
        listen, speak = parse_prefs(getattr(participant, "attributes", None) or {})
        self._on_prefs(participant.identity, listen, speak)

    def _on_attributes_changed(self, changed_attributes: dict, participant: Any) -> None:
        self._announce_prefs(participant)

    def _on_track_subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        pid = participant.identity
        if pid == GATEWAY_IDENTITY:  # never transcribe synthesized speech (another gateway instance)
            return
        if pid in self._tasks:  # re-subscription: restart the reader
            self._tasks[pid].cancel()
        self.core.add_participant(pid)
        stream = self._stream_factory(track)
        self._streams[pid] = stream
        self._tasks[pid] = asyncio.get_running_loop().create_task(self._read_audio(pid, stream), name=f"audio-{pid}")

    def _on_track_unsubscribed(self, track: Any, publication: Any, participant: Any) -> None:
        if getattr(track, "kind", None) == rtc.TrackKind.KIND_AUDIO:
            asyncio.get_running_loop().create_task(self._stop_reader(participant.identity))

    def _on_participant_disconnected(self, participant: Any) -> None:
        if self._on_leave is not None:
            self._on_leave(participant.identity)
        asyncio.get_running_loop().create_task(self._stop_reader(participant.identity))

    async def _read_audio(self, pid: str, stream: Any) -> None:
        try:
            async for event in stream:
                frame = event.frame
                await self.core.push_pcm16(pid, frame.data, frame.sample_rate)
        except asyncio.CancelledError:
            raise
        except Exception:  # keep the gateway alive when one stream breaks
            log.exception("audio reader for %s failed", pid)

    async def _stop_reader(self, pid: str) -> None:
        task = self._tasks.pop(pid, None)
        stream = self._streams.pop(pid, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if stream is not None and hasattr(stream, "aclose"):
            await stream.aclose()
        await self.core.remove_participant(pid)

    # --------------------------------------------------------------- publishing
    async def publish_subtitle(self, message: SubtitleMessage, destinations: list[str] | None = None) -> None:
        """Send a subtitle as a reliable data packet (room-wide, or to the given identities)."""
        await self.room.local_participant.publish_data(
            message.to_bytes(),
            reliable=True,
            destination_identities=list(destinations or []),
            topic=SUBTITLE_TOPIC,
        )
