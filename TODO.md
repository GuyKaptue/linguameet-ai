# TODO: Real-Time Multilingual Meeting Platform (LinguaMeet AI)

Source: implementation prompt + `docs/AI_MULTILINGGUAL_MEETING_PLATFORM_RESEARCH_SPRINT.md`.
Existing code in the zip: `backend/` (FastAPI, faster-whisper ASR, NLLB translation, VAD service, mock models, tests). Build on it where it fits.

**Strict constraints (apply to every item)**
- No placeholders (`pass`/`TODO`) in core logic; unavailable libs get a functional mock, clearly labelled as a mock.
- No hallucinated APIs: verify faster-whisper / NLLB / LiveKit usage against official docs.
- Every module has a runnable test (pytest / jest).
- Final output: full file tree + contents of each file; start with `REFERENCES.md`.

## Phase 0: Initialization & Research
- [x] Read the research sprint doc and the existing project docs/code
- [x] Research official docs, repos and papers: Faster-Whisper streaming, NLLB-200 optimization, LiveKit Agents, Silero VAD, WebRTC data channels
- [x] Write `REFERENCES.md` (title, URL, how it will be used) - do this first

## Phase 1: Scaffolding & Core Infrastructure
- [x] **Module 1**: monorepo `services/` (asr, translation, tts), `gateway/`, `shared/`, `tests/`; `pyproject.toml`; config loading
  - [x] Test: `tests/test_config.py`
- [x] **Module 2**: LiveKit/WebRTC gateway for audio ingestion + streaming Silero VAD gating ASR input
  - [x] Test: `services/vad/test_vad.py`

## Phase 2: AI Pipeline (ASR → NMT → TTS)
- [x] **Module 3**: streaming faster-whisper ASR (LocalAgreement/SimulStreaming buffering, partial transcripts, per-chunk language detection)
  - [x] Test: `services/asr/test_asr.py`
- [x] **Module 4**: quantized NLLB-200 (600M/1.3B distilled) via CTranslate2/transformers; translate endpoint; batching/request queue
  - [x] Test: `services/translation/test_nmt.py`
- [x] **Module 5**: streaming TTS (XTTS, or Piper/MeloTTS if too slow); chunked audio output
  - [x] Test: `services/tts/test_tts.py`

## Phase 3: Client & Integration
- [x] **Module 6**: web client joining LiveKit room; data-channel JSON subtitles; original/translated display preference
  - [x] Test: `client/test_ui.js` (Jest)
- [x] **Module 7**: end-to-end flow (inject audio → pipeline → audio track + subtitles), latency < 2.0s
  - [x] Test: `tests/e2e/test_meeting_flow.py`

## Phase 4: Finalization & Report
- [x] **Module 8**: `BENCHMARKS.md` (measured ASR / NMT / TTS / total latency) and cited `REPORT.md`
- [x] Final check: run pytest + jest, check constraints above, deliver file tree
