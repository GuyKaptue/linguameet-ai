"""Standalone meeting room: the REAL MeetingPipeline / GatewayCore / VAD / streaming ASR / batching NMT / TTS
service / subtitle protocol, with scripted bots and MOCK models, and browsers connected over a websocket
instead of LiveKit. Browsers are ordinary participants (their ``listen`` language drives translation and TTS)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from aiohttp import web

from gateway.core import GatewayCore
from meetdemo.bots import SR, Bot, MultiReplayASR, PivotLexiconNMT, default_bots
from pipeline.meeting import MeetingPipeline
from services.tts.backends import ToneMockBackend
from shared.audio import float32_to_pcm16
from shared.config import Settings
from shared.types import AudioOut, SubtitleMessage

log = logging.getLogger(__name__)
LANGUAGES = {"en": "English", "es": "Español", "fr": "Français", "de": "Deutsch"}
FRAME = 320  # 20 ms of 16 kHz audio


@dataclass(eq=False)
class Client:
    id: str
    ws: web.WebSocketResponse
    listen: str = "en"


class DemoRoom:
    def __init__(self, settings: Settings | None = None, *, pace: bool = True):
        self.settings = settings or Settings(_env_file=None, backend_mode="mock")
        self.pace = pace
        self.bots: dict[str, Bot] = default_bots()
        self.clients: dict[str, Client] = {}
        self._n = 0
        self.playing: dict[str, asyncio.Task] = {}
        self.core = GatewayCore(self.settings, self._sink)
        self.pipeline = MeetingPipeline(
            self.settings, MultiReplayASR(self.bots), PivotLexiconNMT(), ToneMockBackend(), self, self
        )
        for bot in self.bots.values():
            self.pipeline.set_prefs(bot.id, bot.lang, bot.lang)

    # ------------------------------------------------------------ pipeline publishers
    async def publish_subtitle(self, message: SubtitleMessage, destinations: list[str] | None = None) -> None:
        payload: dict[str, Any] = {"kind": "subtitle", "payload": message.to_dict()}
        ms = self.pipeline.milestones.get(message.segment_id)
        if message.is_final and ms is not None and ms.vad_end is not None:
            payload["after_gate_ms"] = round((self.pipeline.clock() - ms.vad_end) * 1000, 1)
            payload["gate_hangover_ms"] = self.settings.vad_min_silence_ms
        await self._broadcast(payload)

    async def publish_frame(self, out: AudioOut) -> None:
        pcm = float32_to_pcm16(np.asarray(out.audio, dtype=np.float32)).tobytes()
        data = out.sample_rate.to_bytes(4, "little") + pcm
        for c in list(self.clients.values()):
            if c.listen == out.language:
                try:
                    await c.ws.send_bytes(data)
                except ConnectionError:
                    pass

    async def _broadcast(self, obj: dict) -> None:
        text = json.dumps(obj, ensure_ascii=False)
        for c in list(self.clients.values()):
            try:
                await c.ws.send_str(text)
            except ConnectionError:
                pass

    async def _sink(self, pid: str, ev) -> None:
        if ev.kind in ("start", "end"):
            await self._broadcast({"kind": "vad", "speaker": pid, "event": ev.kind})
        await self.pipeline.on_gate_event(pid, ev)

    # ------------------------------------------------------------------ clients
    def hello(self, client: Client) -> dict:
        return {
            "kind": "hello", "you": client.id, "languages": LANGUAGES, "mock": True,
            "participants": [{"id": b.id, "name": b.name, "lang": b.lang, "seconds": round(b.seconds, 1), "phrases": b.phrases} for b in self.bots.values()],
        }

    async def join(self, ws: web.WebSocketResponse) -> Client:
        self._n += 1
        client = Client(f"guest-{self._n}", ws)
        self.clients[client.id] = client
        self.pipeline.set_prefs(client.id, client.listen, None)
        await ws.send_str(json.dumps(self.hello(client), ensure_ascii=False))
        return client

    def leave(self, client: Client) -> None:
        self.clients.pop(client.id, None)
        self.pipeline.forget(client.id)

    async def handle(self, client: Client, msg: dict) -> None:
        op = msg.get("op")
        if op == "prefs" and msg.get("listen") in LANGUAGES:
            client.listen = msg["listen"]
            self.pipeline.set_prefs(client.id, client.listen, None)
        elif op == "play" and msg.get("who") in self.bots:
            self.play(msg["who"])
        elif op == "stop":
            for task in list(self.playing.values()):
                task.cancel()

    # ------------------------------------------------------------------- speech
    def play(self, bot_id: str) -> bool:
        if bot_id in self.playing and not self.playing[bot_id].done():
            return False
        self.playing[bot_id] = asyncio.get_running_loop().create_task(self._feed(self.bots[bot_id]))
        return True

    async def _feed(self, bot: Bot) -> None:
        """Deliver the bot's recording to the gateway core in 20 ms frames, in real time."""
        assert bot.pcm is not None
        # Fresh VAD state per recording. The scripted MOCK ASR finds its words by locating the audio inside the
        # recording, which only works when the VAD pre-roll is the recording itself, not the previous take's silence.
        await self.core.remove_participant(bot.id)
        await self._broadcast({"kind": "playing", "speaker": bot.id, "event": "start"})
        t0 = time.monotonic()
        try:
            for k, i in enumerate(range(0, len(bot.pcm), FRAME)):
                if self.pace:
                    delay = t0 + (k + 1) * FRAME / SR - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)
                await self.core.push_pcm16(bot.id, bot.pcm[i : i + FRAME], SR)
        finally:
            await self._broadcast({"kind": "playing", "speaker": bot.id, "event": "end"})

    async def aclose(self) -> None:
        for task in list(self.playing.values()):
            task.cancel()
        await asyncio.gather(*self.playing.values(), return_exceptions=True)
        await self.pipeline.aclose()
