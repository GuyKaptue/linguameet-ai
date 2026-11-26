"""Start the standalone demo server, drive it with headless Chromium and save screenshots + an animated GIF.

    python scripts/demo/capture_live.py            # writes docs/screenshots/live-*.png and docs/screenshots/meeting-demo.gif
"""

from __future__ import annotations

import io
import socket
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "screenshots"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def finals(page) -> int:
    return page.evaluate("document.querySelectorAll('.subtitle[data-final=true]').length")


def main() -> None:
    for old in OUT.glob("live-*.png"):
        old.unlink()
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "meetdemo.server", "--port", str(port)], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        url = f"http://127.0.0.1:{port}/"
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])

            def new_page(scheme="dark", w=1100, h=900, dsf=2, mobile=False):
                ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=dsf, color_scheme=scheme,
                                          is_mobile=mobile, has_touch=mobile)
                pg = ctx.new_page()
                errors: list[str] = []
                pg.on("pageerror", lambda e: errors.append(str(e)))
                pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                pg.goto(url)
                pg.wait_for_selector(".tile[data-id=alice]")
                return pg, errors

            def shot(pg, name, **kw):
                pg.screenshot(path=str(OUT / name), **kw)
                print("saved", name)

            # ---- stills ----
            pg, errs = new_page()
            pg.select_option("#listen-lang", "es")
            shot(pg, "live-01-lobby.png")
            pg.click(".tile[data-id=alice] button")
            pg.wait_for_selector(".tile[data-id=alice].speaking", timeout=10000)
            shot(pg, "live-02-alice-speaking.png", animations="disabled")  # taken while the VAD says "speech"
            pg.wait_for_function("document.querySelectorAll('.subtitle[data-final=true]').length >= 3", timeout=20000)
            pg.wait_for_timeout(400)
            shot(pg, "live-03-alice-spanish-subtitles.png")
            pg.select_option("#listen-lang", "de")
            shot(pg, "live-04-switch-to-german.png")
            pg.click(".tile[data-id=bruno] button")
            pg.wait_for_function("document.querySelectorAll('.subtitle[data-final=true]').length >= 5", timeout=20000)
            pg.wait_for_timeout(400)
            shot(pg, "live-05-bruno-french-to-german.png")
            pg.select_option("#mode", "translated")
            shot(pg, "live-06-translated-only.png")
            pg.select_option("#listen-lang", "en")
            pg.select_option("#mode", "both")
            shot(pg, "live-07-english-listener.png")
            assert not errs, errs
            pg.context.close()

            pg, errs = new_page(scheme="light")
            pg.select_option("#listen-lang", "fr")
            pg.click(".tile[data-id=alice] button")
            pg.wait_for_function("document.querySelectorAll('.subtitle[data-final=true]').length >= 3", timeout=20000)
            pg.wait_for_timeout(400)
            shot(pg, "live-08-light-theme-french.png")
            assert not errs, errs
            pg.context.close()

            pg, errs = new_page(w=390, h=844, mobile=True)
            pg.select_option("#listen-lang", "es")
            pg.click(".tile[data-id=bruno] button")
            pg.wait_for_function("document.querySelectorAll('.subtitle[data-final=true]').length >= 2", timeout=20000)
            pg.wait_for_timeout(400)
            shot(pg, "live-09-mobile.png", full_page=True)
            assert not errs, errs
            pg.context.close()

            # ---- animation: real-time meeting, ~7 s -----------------------------------------
            pg, errs = new_page(dsf=1, h=820)
            pg.select_option("#listen-lang", "es")
            frames: list[tuple[Image.Image, int]] = []

            def grab(ms: int):
                frames.append((Image.open(io.BytesIO(pg.screenshot())).convert("RGB"), ms))

            for _ in range(3):
                grab(300)
            pg.click(".tile[data-id=alice] button")
            t0 = time.time()
            while finals(pg) < 3 and time.time() - t0 < 20:
                grab(140)
            for _ in range(5):
                grab(200)
            pg.select_option("#listen-lang", "de")
            for _ in range(5):
                grab(200)
            pg.click(".tile[data-id=bruno] button")
            t0 = time.time()
            while finals(pg) < 5 and time.time() - t0 < 20:
                grab(140)
            for _ in range(4):
                grab(200)
            pg.select_option("#mode", "translated")
            for _ in range(8):
                grab(250)
            frames[-1] = (frames[-1][0], 2500)
            assert not errs, errs
            browser.close()
        small = [f.resize((f.width * 2 // 3, f.height * 2 // 3), Image.LANCZOS) for f, _ in frames]
        pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in small]
        pal[0].save(OUT / "meeting-demo.gif", save_all=True, append_images=pal[1:], duration=[d for _, d in frames], loop=0, optimize=True)
        print(f"saved meeting-demo.gif ({len(pal)} frames, {(OUT / 'meeting-demo.gif').stat().st_size // 1024} KiB)")
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
