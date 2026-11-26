"""Run the real web client in headless Chromium against a replayed session and save screenshots + an animated GIF.

    python scripts/demo/record_session.py && python scripts/demo/capture.py
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "docs" / "screenshots" / "client-only"
BANNER = ('<div style="background:#ffb454;color:#222;font:13px system-ui;padding:6px 12px;text-align:center">'
          "DEMO: simulated LiveKit room replaying subtitle packets recorded from the real pipeline (mock ASR/NMT/TTS models)</div>")


def build_page() -> Path:
    html = (ROOT / "client" / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="style.css"', 'href="../../client/style.css"')
    html = re.sub(r'<script src="https://cdn\.jsdelivr[^"]*"></script>', '<script src="session.js"></script>\n  <script src="demo_shim.js"></script>', html)
    html = html.replace('src="subtitles.js"', 'src="../../client/subtitles.js"').replace('src="app.js"', 'src="../../client/app.js"')
    html = html.replace("<body>", "<body>\n" + BANNER)
    page = HERE / "demo.html"
    page.write_text(html, encoding="utf-8")
    return page


def fill(page, listen="es", speak="en", mode="both"):
    page.fill("#token", "demo-token")
    page.select_option("#speak-lang", speak)
    page.select_option("#listen-lang", listen)
    page.select_option("#mode", mode)


def shot(page, name, **kw):
    page.screenshot(path=str(OUT / name), **kw)
    print("saved", name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    url = build_page().as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()

        def new_page(scheme="dark", w=900, h=1100, dsf=2, speed=1, mobile=False):
            ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=dsf, color_scheme=scheme,
                                      is_mobile=mobile, has_touch=mobile)
            pg = ctx.new_page()
            errors: list[str] = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            pg.goto(f"{url}?speed={speed}")
            return pg, errors

        # ---- stills (replay at 1x) ------------------------------------------------
        pg, errs = new_page()
        shot(pg, "01-join.png")
        fill(pg)
        shot(pg, "02-ready-to-join.png")
        pg.click("#connect")
        pg.wait_for_selector(".subtitle .original:not([hidden])", timeout=10000)
        pg.wait_for_timeout(250)
        shot(pg, "03-first-words-arriving.png")
        pg.wait_for_function("document.querySelectorAll('.subtitle[data-final=true]').length >= 3", timeout=15000)
        pg.wait_for_timeout(300)
        shot(pg, "04-spanish-original-and-translated.png")
        pg.select_option("#mode", "translated")
        shot(pg, "05-translated-only.png")
        pg.select_option("#listen-lang", "fr")
        shot(pg, "06-switched-to-french.png")
        pg.select_option("#mode", "original")
        shot(pg, "07-original-only.png")
        pg.select_option("#mode", "both")
        pg.select_option("#listen-lang", "de")
        shot(pg, "08-language-without-translation-pending.png")
        attrs = pg.evaluate("window.__attrs")
        print("announced attributes:", attrs)
        pg.click("#disconnect")
        pg.wait_for_timeout(200)
        shot(pg, "09-left-meeting.png")
        assert not errs, errs
        pg.context.close()

        pg, errs = new_page(scheme="light")
        fill(pg)
        pg.click("#connect")
        pg.wait_for_function("document.querySelectorAll('.subtitle[data-final=true]').length >= 3", timeout=15000)
        pg.wait_for_timeout(300)
        shot(pg, "10-light-theme.png")
        assert not errs, errs
        pg.context.close()

        pg, errs = new_page(w=390, h=844, mobile=True)
        fill(pg, listen="fr")
        pg.click("#connect")
        pg.wait_for_function("document.querySelectorAll('.subtitle[data-final=true]').length >= 3", timeout=15000)
        pg.wait_for_timeout(300)
        shot(pg, "11-mobile.png", full_page=True)
        assert not errs, errs
        pg.context.close()

        # ---- animation: slow replay, a frame every ~200 ms --------------------------------
        pg, errs = new_page(dsf=1, speed=0.5, h=900)
        fill(pg)
        frames: list[Image.Image] = []

        def grab():
            frames.append(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))

        for _ in range(3):
            grab()
        pg.click("#connect")
        for _ in range(60):
            grab()
            pg.wait_for_timeout(120)
            if pg.evaluate("document.querySelectorAll('.subtitle[data-final=true]').length") >= 3:
                break
        for _ in range(4):
            grab()
            pg.wait_for_timeout(120)
        pg.select_option("#listen-lang", "fr")
        for _ in range(6):
            grab()
        pg.select_option("#mode", "translated")
        for _ in range(8):
            grab()
        assert not errs, errs
        browser.close()

    small = [f.resize((f.width * 3 // 4, f.height * 3 // 4), Image.LANCZOS) for f in frames]
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in small]
    durations = [350] * 3 + [180] * (len(pal) - 3 - 14) + [180] * 4 + [250] * 6 + [900] * 1 + [250] * 7
    durations = (durations + [250] * len(pal))[: len(pal)]
    durations[-1] = 2000
    pal[0].save(OUT / "demo.gif", save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"saved demo.gif ({len(pal)} frames, {(OUT / 'demo.gif').stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
