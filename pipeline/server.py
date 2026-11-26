"""Run the translation gateway against a LiveKit server, or mint a participant token.

    python -m pipeline.server run   --room demo
    python -m pipeline.server token --room demo --identity alice

``run`` needs real model weights (faster-whisper, NLLB CT2 directory, Piper voices; see README) and
``LIVEKIT_*`` settings. With ``BACKEND_MODE=mock`` NMT/TTS use the labelled mocks, but ASR has no mock
that can recognise speech, so a real faster-whisper model is always required here.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from livekit import rtc

from gateway.audio_out import LiveKitAudioPublisher
from gateway.livekit_gateway import LiveKitGateway
from gateway.tokens import create_token
from pipeline.meeting import MeetingPipeline
from services.asr.backends import FasterWhisperBackend
from services.translation.service import create_backend as create_nmt_backend
from services.tts.backends import create_backend as create_tts_backend
from shared.config import Settings, get_settings

log = logging.getLogger("linguameet")


def build(settings: Settings) -> tuple[LiveKitGateway, MeetingPipeline]:
    room = rtc.Room()
    audio = LiveKitAudioPublisher(room, settings.tts_output_sample_rate)
    holder: dict[str, MeetingPipeline] = {}
    gateway = LiveKitGateway(
        settings,
        lambda pid, ev: holder["p"].on_gate_event(pid, ev),
        room=room,
        on_prefs=lambda pid, listen, speak: holder["p"].set_prefs(pid, listen, speak),
        on_leave=lambda pid: holder["p"].forget(pid),
    )
    holder["p"] = MeetingPipeline(
        settings,
        FasterWhisperBackend(settings),
        create_nmt_backend(settings),
        create_tts_backend(settings),
        gateway,
        audio,
    )
    return gateway, holder["p"]


async def run(settings: Settings, room_name: str) -> None:
    gateway, pipeline = build(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await gateway.connect(room_name)
    log.info("gateway running in room %r - Ctrl+C to stop", room_name)
    try:
        await stop.wait()
    finally:
        await gateway.disconnect()
        await pipeline.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline.server")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="join a room and translate")
    r.add_argument("--room", required=True)
    t = sub.add_parser("token", help="print a participant access token")
    t.add_argument("--room", required=True)
    t.add_argument("--identity", required=True)
    args = parser.parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "token":
        print(create_token(settings, identity=args.identity, room=args.room, name=args.identity))
        return 0
    asyncio.run(run(settings, args.room))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
