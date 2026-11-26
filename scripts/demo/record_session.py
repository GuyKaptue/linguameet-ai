"""Record the subtitle packets the real gateway/pipeline publishes during the e2e scenario
(real VAD + streaming ASR logic + batching NMT + publishers, MOCK models) into ``session.js``.
``capture.py`` replays them into the real web client for screenshots."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.e2e.test_meeting_flow import run_meeting  # noqa: E402


def main() -> None:
    run = asyncio.run(run_meeting())
    events = [[round((d["t"] - run.t0) * 1000), d["payload"].decode("utf-8")] for d in run.room.local_participant.data]
    out = Path(__file__).with_name("session.js")
    out.write_text("window.__SESSION__ = " + json.dumps(events, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"recorded {len(events)} packets over {events[-1][0]} ms -> {out}")


if __name__ == "__main__":
    main()
