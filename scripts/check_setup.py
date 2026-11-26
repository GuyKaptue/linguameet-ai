"""Preflight for the real stack:  python scripts/check_setup.py [--room demo]

Checks settings, model files, the LiveKit server port and that tokens can be minted. Exit code 0 only if all pass.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.config import ConfigError, Settings  # noqa: E402

Result = tuple[str, bool, str]


def check_settings(s: Settings) -> list[Result]:
    out: list[Result] = []
    try:
        s.require_livekit()
        out.append(("LiveKit key/secret", True, "set"))
    except ConfigError as e:
        out.append(("LiveKit key/secret", False, str(e)))
    out.append(("BACKEND_MODE", s.backend_mode == "real", f"{s.backend_mode} (mock mode cannot recognise speech)" if s.backend_mode != "real" else "real"))
    return out


def check_models(s: Settings) -> list[Result]:
    out: list[Result] = []
    nmt = Path(s.nmt_model_dir)
    needed = ["model.bin"]
    missing = [f for f in needed if not (nmt / f).exists()]
    out.append(("NLLB CTranslate2 model", nmt.is_dir() and not missing, f"{nmt}" + (f" missing {missing}" if missing or not nmt.is_dir() else "")))
    voices = sorted(Path(s.tts_voices_dir).glob("*.onnx")) if Path(s.tts_voices_dir).is_dir() else []
    out.append(("Piper voices", bool(voices), ", ".join(v.stem for v in voices) or f"none in {s.tts_voices_dir}"))
    configs = [v for v in voices if not v.with_suffix(".onnx.json").exists()]
    out.append(("Piper voice configs (.onnx.json)", not configs, "ok" if not configs else "missing for " + ", ".join(v.stem for v in configs)))
    return out


def check_livekit(url: str, timeout: float = 2.0) -> list[Result]:
    u = urlparse(url.replace("ws://", "http://").replace("wss://", "https://"))
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        socket.create_connection((u.hostname or "localhost", port), timeout=timeout).close()
        return [("LiveKit server reachable", True, f"{u.hostname}:{port}")]
    except OSError as e:
        return [("LiveKit server reachable", False, f"{u.hostname}:{port} - {e}. Start it: docker compose up livekit")]


def check_token(s: Settings, room: str) -> list[Result]:
    from gateway.tokens import create_token

    try:
        create_token(s, identity="preflight", room=room)
        return [("Token minting", True, "ok")]
    except Exception as e:  # noqa: BLE001 - report any failure
        return [("Token minting", False, str(e))]


def run(s: Settings, room: str = "demo", livekit: bool = True) -> list[Result]:
    results = check_settings(s) + check_models(s) + check_token(s, room)
    if livekit:
        results += check_livekit(s.livekit_url)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="demo")
    ap.add_argument("--skip-livekit", action="store_true")
    a = ap.parse_args()
    results = run(Settings(), a.room, not a.skip_livekit)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
