# Run LinguaMeet AI for real (LiveKit + Whisper + NLLB + Piper)

> Status: every step below uses code and files that are unit-tested offline (`pytest scripts`), but **the full stack has never been started**
> in the build sandbox (no access to Hugging Face, Docker or a LiveKit server). Expect to fix small things; `check_setup.py` is there to find them.
> Hardware: CPU works for `ASR_MODEL_SIZE=base`; larger Whisper models need a GPU to stay inside the 2 s budget.

## Quick path with make

```bash
make setup      # install + torch + .env + models + LiveKit container + preflight (stops at the first failing step)
make gateway    # terminal 1: translation gateway          (ROOM=demo by default)
make web        # terminal 2: web client on http://localhost:8080
make tokens     # prints join tokens for alice and bruno
```
`make help` lists everything (`make demo` = standalone mock demo, `make test`, `make bench REAL=asr,nmt,tts`, `make stack` = all in Docker).
The numbered steps below are the same thing done by hand.

## 0. Prerequisites
Linux with Docker (host networking), Python 3.10+, ~6 GB free disk, network access to Hugging Face. On macOS/Windows run `livekit-server --dev`
natively instead of the `livekit` container (see comments in `docker-compose.yml`).

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[nmt,tts,dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu   # only needed once, for the NLLB conversion
```

## 2. Download and convert the models (once)
```bash
python scripts/download_models.py --langs en,es,fr,de --asr-size base
```
Does three things: faster-whisper `base` into `models/`, NLLB-200 distilled 600M converted to int8 CTranslate2 in `models/nllb-200-distilled-600M-ct2`,
and Piper voices into `models/tts` (en_US-lessac-medium, es_ES-davefx-medium, fr_FR-siwis-medium, de_DE-thorsten-medium; if a voice name is
wrong for your Piper version the download stops with an error, edit `DEFAULT_VOICES`). Use `--dry-run` to see the commands, `--only asr|nmt|tts` for one part.
Licences: NLLB weights are CC-BY-NC-4.0 (non-commercial), Piper is GPLv3.

## 3. Configure
```bash
cp .env.example .env
# set LIVEKIT_URL=ws://127.0.0.1:7880, LIVEKIT_API_KEY=devkey,
# LIVEKIT_API_SECRET=devsecret_that_is_at_least_32_characters_long   (must match deploy/livekit.yaml), BACKEND_MODE=real
```

## 4. Start LiveKit
```bash
docker compose up -d livekit          # or natively: livekit-server --config deploy/livekit.yaml
```

## 5. Preflight
```bash
python scripts/check_setup.py
```
Every line must say PASS (settings, models, voices, token minting, LiveKit port). Fix the FAIL lines first.

## 6. Start the gateway and the web page
```bash
python -m pipeline.server run --room demo      # or: docker compose up -d gateway
python -m http.server 8080 -d client           # or: docker compose up -d web
```

## 7. Join with two browsers
```bash
python -m pipeline.server token --room demo --identity alice   # paste into browser 1
python -m pipeline.server token --room demo --identity bruno   # paste into browser 2
```
Open `http://localhost:8080` in two windows, URL `ws://localhost:7880`, paste each token, pick languages (window 1 speaks English and reads English,
window 2 reads Spanish), press **Join meeting**, then **Microphone** in the speaking window (allow the browser permission). Window 2 should show
Spanish subtitles and hear the Spanish `tts-es` track within about 2 s of you finishing a sentence (headphones avoid echo).

## 8. Measure
```bash
python scripts/benchmark.py --real asr,nmt,tts --runs 5      # fills BENCHMARKS.md section 5
```

## Troubleshooting
| Symptom | Check |
|---|---|
| `check_setup` FAIL LiveKit port | `docker compose logs livekit`; firewall for 7880/7881 TCP and the UDP range in `deploy/livekit.yaml` |
| No subtitles | gateway log shows `gateway joined room`? Browser token room name equals `--room`? Participant attribute `listen_lang` set (token needs `can_update_own_metadata`, already granted)? |
| Silent translated audio | click "Enable audio"; the client only plays `tts-<your listen language>` |
| Slow subtitles | `BENCHMARKS.md`; try `ASR_MODEL_SIZE=tiny`, raise `ASR_MIN_CHUNK_SECONDS`, GPU via `ASR_DEVICE=cuda` |
