"""Run the standalone demo:  python -m meetdemo.server [--port 8765]   then open http://localhost:8765"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from aiohttp import WSMsgType, web

from meetdemo.room import DemoRoom

ROOM = web.AppKey("room", DemoRoom)
WEB = Path(__file__).parent / "web"
CLIENT = Path(__file__).resolve().parents[1] / "client"


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    room: DemoRoom = request.app[ROOM]
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    client = await room.join(ws)
    try:
        async for raw in ws:
            if raw.type == WSMsgType.TEXT:
                try:
                    await room.handle(client, json.loads(raw.data))
                except (ValueError, TypeError):
                    await ws.send_str(json.dumps({"kind": "error", "message": "bad message"}))
    finally:
        room.leave(client)
    return ws


def create_app(room: DemoRoom | None = None) -> web.Application:
    app = web.Application()
    app[ROOM] = room or DemoRoom()

    async def close(app: web.Application) -> None:
        await app[ROOM].aclose()

    app.on_cleanup.append(close)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/", lambda r: web.FileResponse(WEB / "index.html"))
    app.router.add_get("/lib/subtitles.js", lambda r: web.FileResponse(CLIENT / "subtitles.js"))
    app.router.add_get("/lib/style.css", lambda r: web.FileResponse(CLIENT / "style.css"))
    app.router.add_static("/static", WEB)
    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"LinguaMeet standalone demo (mock ASR/NMT/TTS models): http://{args.host}:{args.port}")
    web.run_app(create_app(), host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
