"""Download / convert every model the real stack needs into ./models.

    python scripts/download_models.py --langs en,es,fr,de            # everything
    python scripts/download_models.py --only asr                     # one part

Needs network access to Hugging Face. Parts:
  asr  faster-whisper model into MODELS_DIR (same ``download_root`` the ASR backend loads from)
  nmt  NLLB-200 distilled 600M converted to int8 CTranslate2 (``ct2-transformers-converter``; needs torch) + tokenizer cache
  tts  Piper voices via ``python -m piper.download_voices`` (voice names from the Piper voice list; the download fails loudly if a name is wrong)
Licences: NLLB weights CC-BY-NC-4.0 (non-commercial); Piper is GPLv3; check each voice's own licence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOICES = {"en": "en_US-lessac-medium", "es": "es_ES-davefx-medium", "fr": "fr_FR-siwis-medium", "de": "de_DE-thorsten-medium"}


def plan(models_dir: Path, asr_size: str, langs: list[str], only: set[str]) -> list[tuple[str, list[str]]]:
    """The commands that will run (pure function, unit-tested)."""
    steps: list[tuple[str, list[str]]] = []
    py = sys.executable
    if not only or "asr" in only:
        code = f"from faster_whisper.utils import download_model; print(download_model({asr_size!r}, cache_dir={str(models_dir)!r}))"
        steps.append(("asr", [py, "-c", code]))
    if not only or "nmt" in only:
        out = models_dir / "nllb-200-distilled-600M-ct2"
        steps.append(("nmt", ["bash", str(ROOT / "scripts" / "convert_nllb.sh"), "facebook/nllb-200-distilled-600M", str(out), "int8"]))
        steps.append(("nmt-tokenizer", [py, "-c", "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); print('tokenizer cached')"]))
    if not only or "tts" in only:
        unknown = [l for l in langs if l not in DEFAULT_VOICES]
        if unknown:
            raise SystemExit(f"no default Piper voice configured for: {', '.join(unknown)} (edit DEFAULT_VOICES)")
        steps.append(("tts", [py, "-m", "piper.download_voices", "--download-dir", str(models_dir / "tts"), *[DEFAULT_VOICES[l] for l in langs]]))
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--asr-size", default="base")
    ap.add_argument("--langs", default="en,es,fr,de")
    ap.add_argument("--only", default="", help="comma list of asr,nmt,tts")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    steps = plan(a.models_dir, a.asr_size, [l for l in a.langs.split(",") if l], {x for x in a.only.split(",") if x})
    a.models_dir.mkdir(parents=True, exist_ok=True)
    for name, cmd in steps:
        print(f"== {name}: {' '.join(cmd)}")
        if not a.dry_run:
            subprocess.run(cmd, check=True)
    print("done. Next: python scripts/check_setup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
