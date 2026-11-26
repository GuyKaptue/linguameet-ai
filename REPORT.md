# REPORT: LinguaMeet AI, real-time multilingual meetings

Numbered citations [n] refer to the rows of `REFERENCES.md`. All timings quoted from our own runs come from `BENCHMARKS.md`
(`scripts/benchmark.py`, 2-core CPU, no GPU).

## 1. What was built

A cascaded pipeline, one module per stage, each with its own runnable test:

| Stage | Implementation | Test |
|---|---|---|
| Transport | LiveKit room; the server-side gateway subscribes to every remote audio track (`rtc.AudioStream`), publishes subtitles as reliable data packets on topic `subtitles` and synthesized speech as one audio track per language (`tts-<lang>`) [19][20][22] | `gateway/test_gateway.py` |
| Speech gate | Silero VAD ONNX run directly with onnxruntime, one stateful instance per participant [8][9] | `services/vad/test_vad.py` |
| ASR | faster-whisper with a re-implemented LocalAgreement-2 buffer: commit the longest common prefix of two consecutive hypotheses, trim at sentence ends, feed the trimmed text back as prompt, lock the language per utterance [2][3][4] | `services/asr/test_asr.py` |
| Translation | NLLB-200 distilled 600M through CTranslate2 `translate_batch` with `target_prefix`; an async queue batches requests, de-duplicates identical items and caches repeats [10][11][12][13] | `services/translation/test_nmt.py` |
| TTS | Piper (CPU) behind a streaming service that re-slices audio into 100 ms frames and marks the last frame [16] | `services/tts/test_tts.py` |
| Client | Browser page using `livekit-client`; renders subtitles from data packets (original / translated / both), plays only the TTS track of the chosen language, announces `listen_lang` / `speak_lang` as participant attributes | `client/test_ui.js` (Jest, 38 tests) |
| Orchestration | `MeetingPipeline` ties the stages together per participant preferences | `tests/e2e/test_meeting_flow.py` |

## 2. Design decisions and why

**Gate before ASR.** Whisper hallucinates on silence, and silence is most of a meeting. The VAD discards it before ASR sees it. On this CPU a
32 ms window costs about 0.13 ms (p50), consistent with the "<1 ms per 30+ ms chunk" claim of the Silero README [8], so the gate is not a
latency factor. Its real cost is the hang-over: the gate needs `VAD_MIN_SILENCE_MS` (300 ms) of silence before it closes an utterance, which we measured
as 320 ms between the last spoken sample and the `end` event. Every end-to-end figure includes it. A shorter value cuts latency but splits utterances at breaths.

**LocalAgreement instead of fixed chunks.** Whisper is an offline model; the whisper_streaming paper reports 3.3 s average latency for English with large-v2 on an
A40 GPU and 1 s chunks [3]. That is above our 2 s budget before any translation happens, so we use a smaller model (`base` by default), a 1 s minimum chunk and
word-level agreement so that only stable words are translated. SimulStreaming [5][6] is faster but needs PyTorch Whisper and large VRAM; it is documented as future work.

**Translate only confirmed text, show everything else immediately.** A partial subtitle shows the speaker's confirmed + unconfirmed words at once; the translation
follows for the confirmed prefix only, so translations do not flicker. The final transcript is translated into all listener languages in one batch.

**Batching.** Every utterance fans out to several languages and several people speak at once. A 20 ms collection window plus a max batch of 16 turns that into one
model call. With a fixed cost model (80 ms per batch + 10 ms per item, a mock) 16 concurrent requests finished in 242 ms instead of 1440 ms (5.95x). This
demonstrates the queue; the real gain depends on the real model.

**Per-language audio tracks.** One shared track would force every listener to hear every language. One track per language lets the client subscribe to the one it
needs; the browser attaches only `tts-<listen_lang>`, and switching language swaps tracks without reconnecting.

**Subtitles over the data channel.** Reliable packets are limited to 15 KiB [22]. `SubtitleMessage.to_bytes` enforces this; if a packet is too big the pipeline sends
one packet per language with the same `segment_id`, and the client merges them.

**Safety of displayed text.** The client writes all remote text with `textContent`; a Jest test proves that `<img onerror>` and `<script>` payloads stay text.

## 3. Results

| Measurement | Value | Source |
|---|---|---|
| VAD inference per 32 ms window | 0.13 ms p50 | real model, real CPU |
| Last spoken sample -> gate closes | 320 ms | real VAD |
| Last spoken sample -> final subtitle, **mock models** | about 356 ms p50 | real orchestration, mock models |
| Same, with **assumed** model costs (ASR 350 ms, NMT 150 ms) | about 853 ms p50 | simulated |
| Last spoken sample -> first translated audio, assumed costs | about 976 ms p50 | simulated |
| End-to-end budget | 2000 ms | `shared/config.py` |

Reading: the system we wrote adds roughly 36 ms of its own work on top of the unavoidable 320 ms VAD hang-over, which leaves about 1.6 s of the
2 s budget for ASR, translation and publishing. Whether real models fit into that is **not established by this project**, see section 4.

## 4. Limitations and what was not verified

1. **No real-model numbers.** Hugging Face and the Meta download host were unreachable from the build sandbox, so Whisper, NLLB and Piper never ran here.
   The code paths are exercised through fakes that match the library call shapes (checked against the installed `faster-whisper` 1.2.1, `ctranslate2` 4.9.0,
   `livekit` 1.3.0 packages), and opt-in tests (`ASR_TEST_MODEL`, `NMT_TEST_MODEL_DIR`, `TTS_TEST_VOICE_DIR`) plus `python scripts/benchmark.py --real asr,nmt,tts`
   will measure them on a machine with the weights (that `--real` path of the benchmark script has itself never been executed against real weights). Until then the < 2.0 s claim holds for the orchestration and for the simulated delays only.
2. **No LiveKit server.** The gateway is tested against fakes of `rtc.Room`; network round-trip time and jitter are not included. The browser client was tested in jsdom,
   not in a real browser, and `index.html` loads the pinned `livekit-client@2.24.1` UMD bundle from jsdelivr.
3. **ASR mock.** `ReplayASRBackend` replays a scripted transcript over real recordings. It tests buffering and commit logic, not recognition accuracy; no WER was measured.
4. **TTS and ASR quality** (naturalness, accuracy) and translation quality (BLEU) are untested. `sacrebleu` is a dev dependency [14] for when real models are available.
5. **Scope.** No echo cancellation between the TTS track and microphones, no speaker diarization beyond LiveKit identities, no barge-in handling, no authentication service
   (the token helper is `python -m pipeline.server token`), and the TTS voice does not match the speaker.
6. **Licences.** NLLB weights are CC-BY-NC-4.0 and labelled a research model [11]; Piper is GPLv3 [16]; the XTTS model is under the non-commercial Coqui licence [15] and was
   therefore left out of `TTS_ENGINE`. The sprint document's "XTTS: MIT" is wrong. Commercial use would need a different MT model and TTS engine.
7. **Latency measurement bug found and fixed during Module 8.** The first benchmark measured from the moment the VAD gate closed, which hid its 320 ms hang-over.
   `GateEvent.speech_end` now records where speech actually stopped and all reported end-to-end numbers use it.

## 5. Next steps

1. Run `scripts/benchmark.py --real asr,nmt,tts` on the target machine and replace the simulated rows; tune `ASR_MODEL_SIZE`, `ASR_MIN_CHUNK_SECONDS`, `NMT_BEAM_SIZE`, `VAD_MIN_SILENCE_MS` against the budget.
2. Run against a local `livekit-server --dev` and two real browsers; measure network and playout latency.
3. Evaluate WER on FLEURS-style data and BLEU with sacreBLEU; try SimulStreaming's AlignAtt policy if a large GPU is available [5].
4. Add echo suppression, barge-in (cancel TTS when a newer utterance arrives) and a voice per speaker.
