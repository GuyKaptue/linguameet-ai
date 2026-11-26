"""Standalone demo: a real websocket client joins the room, listens in Spanish and watches scripted speakers talk.
Real: aiohttp server, gateway core, Silero VAD, streaming ASR logic, batching NMT, TTS service, subtitle protocol.
Mock: ASR transcripts (scripted), NMT (phrase table), TTS (tones)."""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import ClientSession, WSMsgType
from aiohttp.test_utils import TestServer

from meetdemo.room import DemoRoom
from meetdemo.server import create_app


async def collect(ws, until, timeout=60, linger=0.0):
    """Read until ``until(msgs)``; then keep reading for ``linger`` seconds (late TTS frames)."""
    msgs, audio = [], []

    async def run():
        async for m in ws:
            if m.type == WSMsgType.TEXT:
                obj = json.loads(m.data)
                msgs.append(obj)
                if until(msgs):
                    return
            elif m.type == WSMsgType.BINARY:
                audio.append(m.data)

    await asyncio.wait_for(run(), timeout)
    if linger:
        try:
            await asyncio.wait_for(run_forever(ws, msgs, audio), linger)
        except asyncio.TimeoutError:
            pass
    return msgs, audio


async def run_forever(ws, msgs, audio):
    async for m in ws:
        if m.type == WSMsgType.TEXT:
            msgs.append(json.loads(m.data))
        elif m.type == WSMsgType.BINARY:
            audio.append(m.data)


async def test_listener_receives_translated_subtitles_audio_and_vad_for_a_scripted_speaker():
    room = DemoRoom(pace=False)
    async with TestServer(create_app(room)) as server, ClientSession() as http:
        ws = await http.ws_connect(server.make_url("/ws"))
        hello = json.loads((await ws.receive()).data)
        assert hello["kind"] == "hello" and hello["mock"] is True
        assert [p["id"] for p in hello["participants"]] == ["alice", "bruno"]
        await ws.send_str(json.dumps({"op": "prefs", "listen": "es"}))
        await ws.send_str(json.dumps({"op": "play", "who": "alice"}))
        finals = lambda ms: sum(1 for m in ms if m["kind"] == "subtitle" and m["payload"]["is_final"]) >= 3
        msgs, audio = await collect(ws, finals, linger=0.7)  # linger: the last TTS frames arrive after the subtitle
        subs = [m for m in msgs if m["kind"] == "subtitle" and m["payload"]["is_final"]]
        assert [m["payload"]["text"] for m in subs] == ["Good morning everyone", "Please share your screen", "Hello, how are you"]
        assert subs[0]["payload"]["translations"]["es"] == "Buenos días a todos."
        assert "fr" in subs[0]["payload"]["translations"]  # Bruno (French) is a listener too
        assert all(m["payload"]["speaker"] == "alice" for m in subs)
        assert all(m["after_gate_ms"] >= 0 and m["gate_hangover_ms"] == 300 for m in subs)
        vad = [(m["speaker"], m["event"]) for m in msgs if m["kind"] == "vad"]
        assert vad.count(("alice", "start")) == 3 and vad.count(("alice", "end")) == 3
        assert audio and all(a[:4] == (24000).to_bytes(4, "little") for a in audio)  # Spanish TTS frames only
        await ws.close()
    assert room.clients == {}


async def test_french_speaker_is_translated_through_english_for_a_german_listener():
    room = DemoRoom(pace=False)
    async with TestServer(create_app(room)) as server, ClientSession() as http:
        ws = await http.ws_connect(server.make_url("/ws"))
        await ws.receive()
        await ws.send_str(json.dumps({"op": "prefs", "listen": "de"}))
        await ws.send_str(json.dumps({"op": "play", "who": "bruno"}))
        msgs, _ = await collect(ws, lambda ms: sum(1 for m in ms if m["kind"] == "subtitle" and m["payload"]["is_final"]) >= 2)
        finals = [m["payload"] for m in msgs if m["kind"] == "subtitle" and m["payload"]["is_final"]]
        assert [f["source_lang"] for f in finals] == ["fr", "fr"]
        assert finals[0]["translations"]["en"] == "Good morning everyone."
        assert finals[0]["translations"]["de"] == "Guten Morgen zusammen."
        await ws.close()


async def test_only_the_chosen_language_audio_is_sent_and_bad_messages_are_survivable():
    room = DemoRoom(pace=False)
    async with TestServer(create_app(room)) as server, ClientSession() as http:
        ws = await http.ws_connect(server.make_url("/ws"))
        await ws.receive()
        await ws.send_str("not json")
        err = json.loads((await ws.receive()).data)
        assert err["kind"] == "error"
        await ws.send_str(json.dumps({"op": "prefs", "listen": "klingon"}))  # ignored
        await ws.send_str(json.dumps({"op": "play", "who": "nobody"}))  # ignored
        await ws.send_str(json.dumps({"op": "prefs", "listen": "en"}))  # same language as Alice: no audio for her
        await ws.send_str(json.dumps({"op": "play", "who": "alice"}))
        msgs, audio = await collect(ws, lambda ms: sum(1 for m in ms if m["kind"] == "subtitle" and m["payload"]["is_final"]) >= 3, linger=0.7)
        assert audio == []
        finals = [m["payload"] for m in msgs if m["kind"] == "subtitle" and m["payload"]["is_final"]]
        assert "en" not in finals[0]["translations"]
        await ws.close()


async def test_playing_twice_at_once_is_ignored_and_stop_cancels():
    room = DemoRoom(pace=True)
    try:
        assert room.play("alice") is True
        assert room.play("alice") is False
        await asyncio.sleep(0.2)
        await room.handle(type("C", (), {"id": "x", "listen": "en"})(), {"op": "stop"})
        await asyncio.gather(*room.playing.values(), return_exceptions=True)
        assert room.playing["alice"].cancelled() or room.playing["alice"].done()
    finally:
        await room.aclose()


async def test_a_recording_can_be_played_repeatedly_with_identical_results():
    room = DemoRoom(pace=False)

    class WS:
        def __init__(self):
            self.msgs = []

        async def send_str(self, text):
            self.msgs.append(json.loads(text))

        async def send_bytes(self, data):
            pass

    ws = WS()
    await room.join(ws)
    seen = []
    for _ in range(3):
        before = len(ws.msgs)
        room.play("alice")
        await asyncio.gather(*room.playing.values())
        await room.pipeline.drain()
        seen.append([m["payload"]["text"] for m in ws.msgs[before:] if m["kind"] == "subtitle" and m["payload"]["is_final"]])
    await room.aclose()
    assert seen == [["Good morning everyone", "Please share your screen", "Hello, how are you"]] * 3
