# REFERENCES

Sources gathered in Phase 0 for the LinguaMeet AI implementation. Each entry says what was checked and how it will be used.
**Status key:** ✅ page fetched and read; 🔎 found via search only (not opened, treat details as unverified).
Research date: 2026-11-19.

---

## 1. ASR: Whisper / Faster-Whisper / Streaming

| # | Title | URL | Status | Use in implementation |
|---|-------|-----|--------|-----------------------|
| 1 | Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper) | https://arxiv.org/abs/2212.04356 | 🔎 | Cited in REPORT.md for Whisper's multilingual training and built-in language detection. |
| 2 | SYSTRAN/faster-whisper (GitHub README) | https://github.com/SYSTRAN/faster-whisper | ✅ | **API reference for Module 3.** `WhisperModel(size, device, compute_type)`; `transcribe()` returns a lazy **generator** of segments plus `info` (`info.language`, `info.language_probability`); options `beam_size`, `language`, `condition_on_previous_text`, `word_timestamps`, `vad_filter`, `vad_parameters`. MIT license. README claims "up to 4 times faster than openai/whisper for the same accuracy" (large-v2, RTX 3070 Ti: 2m23s vs 1m03s fp16, 59s int8). Python ≥3.9; GPU needs CUDA 12 + cuDNN 9. |
| 3 | Macháček, Dabre, Bojar, *Turning Whisper into Real-Time Transcription System* (whisper_streaming paper) | https://arxiv.org/pdf/2307.14743v2.pdf | ✅ | Basis for the **LocalAgreement-2** buffering policy: emit the longest common prefix of the outputs on 2 consecutive chunks; buffer capped at ~30 s and trimmed at sentence ends. Reports 3.3 s average latency (English, large-v2, A40 GPU, 1 s chunk). Shows that our 1.2–2.0 s target needs smaller models or shorter segments. |
| 4 | ufal/whisper_streaming (GitHub) | https://github.com/ufal/whisper_streaming | ✅ | Reference implementation pattern: `FasterWhisperASR` → `OnlineASRProcessor` with `insert_audio_chunk()`, `process_iter()`, `finish()`, `init()`. MIT. **Note:** the README says it is outdated in 2025 and replaced by SimulStreaming. We re-implement LocalAgreement-2 ourselves on top of faster-whisper rather than depend on this repo. |
| 5 | ufal/SimulStreaming (GitHub) | https://github.com/ufal/SimulStreaming | ✅ | Successor using the AlignAtt policy. Claimed ~5× faster than WhisperStreaming. Uses PyTorch Whisper, **not faster-whisper**, and recommends ≥10 GB VRAM for large-v3. MIT. Documented as an alternative and future work, not a dependency. |
| 6 | Macháček & Polák, *Simultaneous Translation with Offline Speech and LLM Models in CUNI Submission to IWSLT 2025* | https://arxiv.org/abs/2506.17077 | 🔎 | Paper behind SimulStreaming/AlignAtt; cited in the report. |
| 7 | QuentinFuxa/WhisperLiveKit | https://github.com/QuentinFuxa/WhisperLiveKit | 🔎 | Prior art for WebSocket/FastAPI streaming on top of these policies; comparison in the report only. |

## 2. Voice Activity Detection

| # | Title | URL | Status | Use |
|---|-------|-----|--------|-----|
| 8 | snakers4/silero-vad (GitHub) | https://github.com/snakers4/silero-vad | ✅ | `pip install silero-vad`; `from silero_vad import load_silero_vad`; `get_speech_timestamps(wav, model, return_seconds=True)`. Claims a 30+ ms chunk is processed in <1 ms on one CPU thread. MIT. Requires `torch>=1.12`. |
| 9 | Silero VAD wiki: Examples and Dependencies | https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies | ✅ | **Streaming API for Module 2:** `VADIterator(model, sampling_rate=16000)`; chunk size **512 samples at 16 kHz / 256 at 8 kHz (32 ms)**; call `vad_iterator(chunk, return_seconds=True)`; `reset_states()` between streams. ONNX (opset 15/16) optional via `onnxruntime`. The return dict's keys are not documented on the page (verify in source when implementing). |

## 3. Translation: NLLB-200 / CTranslate2

| # | Title | URL | Status | Use |
|---|-------|-----|--------|-----|
| 10 | NLLB Team, *No Language Left Behind: Scaling Human-Centered Machine Translation* | https://arxiv.org/abs/2207.04672 | ✅ | Cited for 200-language coverage and the reported +44% BLEU over previous state of the art (Flores-200). |
| 11 | facebook/nllb-200-distilled-600M (model card) | https://huggingface.co/facebook/nllb-200-distilled-600M | ✅ | Language codes use FLORES-200 form (`eng_Latn`, `spa_Latn`, `fra_Latn`). **License CC-BY-NC-4.0** (non-commercial). Sentence-level; trained with inputs ≤512 tokens; "research model, not released for production deployment". The model card warns the `translation` pipeline is gone in transformers v5, so we load with `AutoModelForSeq2SeqLM` or use CTranslate2. |
| 12 | CTranslate2: Transformers guide (NLLB section) | https://opennmt.net/CTranslate2/guides/transformers.html | ✅ | **Official conversion and inference recipe for Module 4.** `ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir ...` (the page's NLLB example shows no `--quantization` flag; quantization option to be checked against the converter's `--help` when implementing). Inference: tokenizer with `src_lang`, `convert_ids_to_tokens`, `translate_batch([...], target_prefix=[[tgt_lang]])`, drop the first target token, decode. Needs `transformers>=4.21.0`. |
| 13 | OpenNMT forum: *NLLB-200 with CTranslate2* | https://forum.opennmt.net/t/nllb-200-with-ctranslate2/5090 | ✅ | Community recipe using SentencePiece + `batch_type="tokens"`, `max_batch_size`, `beam_size`. Pre-converted int8 models exist, though not used here without checking their licenses. |
| 14 | Post, *A Call for Clarity in Reporting BLEU Scores* (sacreBLEU) | https://arxiv.org/abs/1804.08771 | 🔎 | Justifies sacreBLEU for the translation test threshold. |

## 4. Text-to-Speech

| # | Title | URL | Status | Use |
|---|-------|-----|--------|-----|
| 15 | Coqui TTS docs: XTTS | https://coqui-tts.readthedocs.io/en/latest/_sources/models/xtts.md.txt | ✅ | XTTS-v2: **17 languages** (ar, zh-cn, cs, nl, en, fr, de, hi, hu, it, ja, ko, pl, pt, ru, es, tr). Streaming via `model.inference_stream(text, language, gpt_cond_latent, speaker_embedding)` (24 kHz chunks). Documentation claims "<200 ms" streaming latency (needs GPU; to be measured, not assumed). **Model license: Coqui Public Model License (non-commercial).** The sprint doc's "XTTS: MIT" is therefore incorrect. |
| 16 | OHF-Voice/piper1-gpl releases (Piper) | https://github.com/OHF-Voice/piper1-gpl/releases | ✅ (partial) | Maintained Piper fork; v1.8.0 is latest shown. Python API: `PiperVoice.synthesize(...)` yields `AudioChunk` objects (since v1.3.0). **GPLv3**. CPU-friendly candidate if XTTS is too slow; chunk-streaming semantics to be verified in code before relying on it. |
| 17 | idiap/coqui-ai-TTS (maintained Coqui fork) | https://github.com/idiap/coqui-ai-TTS | 🔎 (fetch blocked) | Likely the installable `coqui-tts` package. Unverified; confirm the package name via `pip index` or PyPI before adding to `pyproject.toml`. |
| 18 | MeloTTS | not yet researched | ❌ | Listed in the sprint doc and the prompt. Will be researched if Module 5 needs a second fallback. |

## 5. Real-Time Transport: LiveKit and WebRTC

| # | Title | URL | Status | Use |
|---|-------|-----|--------|-----|
| 19 | livekit/agents (GitHub) | https://github.com/livekit/agents | ✅ | Apache-2.0. Framework for server-side programmable participants: `AgentServer`, `@server.rtc_session()`, `JobContext` (`ctx.room`), `AgentSession.start(agent=..., room=...)`. The page does not document raw audio-track handling, so the gateway uses the low-level `livekit.rtc` API below. |
| 20 | LiveKit docs: Processing raw tracks | https://docs.livekit.io/transport/media/raw-tracks/ | ✅ | Gateway audio: receive with `rtc.AudioStream(track, sample_rate=..., num_channels=...)` and `async for frame_event in stream: frame = frame_event.frame`; publish with `rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)`, `rtc.LocalAudioTrack.create_audio_track(name, source)`, `room.local_participant.publish_track(track, options)`. Frame-push method and the `track_subscribed` handler are **not shown**; confirm in the SDK reference. |
| 21 | LiveKit Python SDK reference: `livekit.rtc` | https://docs.livekit.io/reference/python/livekit/rtc/index.html | 🔎 (fetch blocked) | To confirm `capture_frame`, `AudioFrame`, `track_subscribed`. Alternatively inspect the installed `livekit` package source. |
| 22 | LiveKit docs: Data packets | https://docs.livekit.io/transport/data/packets/ | ✅ | **Subtitle transport (Module 6).** JS: `room.localParticipant.publishData(Uint8Array, {reliable, destinationIdentities})`; receive `RoomEvent.DataReceived (payload, participant, kind)`. Python: `room.local_participant.publish_data(...)` with `reliable`, `destination_identities`, `topic`; receive `@room.on("data_received")`. **Limits: reliable ≤15 KiB, lossy ≤1300 bytes.** Per-listener subtitle payloads must stay under 15 KiB. The page recommends text streams for reliable delivery (not yet researched). |
| 23 | MDN: RTCDataChannel | https://developer.mozilla.org/docs/Web/API/RTCDataChannel | ✅ | Background for the report: ordered by default; `maxRetransmits` / `maxPacketLifeTime` for partial reliability; `message` event. |

---

## Sprint-document claims to correct or flag (found during research)

1. **XTTS license:** the sprint doc says MIT; the XTTS docs say Coqui Public Model License (non-commercial).
2. **NLLB license and status:** CC-BY-NC-4.0, labelled a research model. Fine for this academic project, but commercial deployment needs a different model or licence review.
3. **Faster-Whisper "requires PyTorch ecosystem"** (sprint §10.1.2): the README lists no PyTorch requirement (CTranslate2 + PyAV). Only Silero VAD needs torch (or onnxruntime).
4. **Latency budget:** the sprint's 1.2–2.0 s is an estimate (its Appendix C figures are not backed by cited measurements). The whisper_streaming paper measured 3.3 s on a large GPU with large-v2, so BENCHMARKS.md must report our own measured numbers.
5. **Streaming ASR:** whisper_streaming is marked outdated by its authors; SimulStreaming uses PyTorch Whisper, not faster-whisper. We implement LocalAgreement-2 on faster-whisper.

## Still to verify during implementation (all items below were closed, see the next section)
- Silero `VADIterator` return format (read source after install).
- `livekit` Python: `AudioSource.capture_frame`, `AudioFrame` constructor, `track_subscribed` event.
- `ct2-transformers-converter --quantization` flag (check `--help`).
- Piper streaming semantics and the `coqui-tts` PyPI package name.
- Whether the sandbox can download model weights (Hugging Face fetches were blocked in this session); if not, Modules 3–5 will use clearly labelled mocks for model-dependent tests.

## Resolved during implementation (checked against installed packages)
- **Silero ONNX I/O** (silero-vad 6.2.3 wheel): inputs `input` [1, 64 context + 512 window], `state` [2,1,128], `sr`; outputs `output`, `state`. Source of `VADIterator` / `OnnxWrapper` read to match hysteresis (`threshold - 0.15`) and context handling. Model file reused from the wheel (MIT).
- **LiveKit Python SDK 1.3.0**: `rtc.Room.connect(url, token)`, `Room.on(event, cb)` with events `track_subscribed`, `track_unsubscribed`, `participant_disconnected`, `data_received`; `rtc.AudioStream(track, sample_rate=, num_channels=)` async-iterable, `await aclose()`; `AudioSource.capture_frame` (coroutine); `LocalParticipant.publish_data` (coroutine, `reliable`, `destination_identities`, `topic`); `livekit.api.AccessToken` / `VideoGrants` / `TokenVerifier`.
- **faster-whisper 1.2.1**: `WhisperModel.transcribe(...)` returns `(generator, TranscriptionInfo)`; `Word` has `start`, `end`, `word`, `probability`.
- **CTranslate2 4.9.0**: `ct2-transformers-converter --quantization {int8,int8_float32,int8_float16,int8_bfloat16,int16,float16,bfloat16,float32}`; `Translator(model_path, device, compute_type=...)`; `translate_batch(source, target_prefix, beam_size, max_batch_size, max_input_length=1024 default, max_decoding_length=256 default)` -> results with `.hypotheses`.
- **Test audio** (real speech, 16 kHz mono in `tests/data/`): `jfk.wav` from ggerganov/whisper.cpp `samples/` (public-domain speech), `speech_en.wav`, `speech_fr.wav`, `speech_zh.wav` converted from Uberi/speech_recognition `examples/` (BSD-3).
- **Not reachable from the build sandbox:** Hugging Face and dl.fbaipublicfiles.com (no Whisper / NLLB / TTS weights, no NLLB tokenizer). Model-dependent tests therefore run against clearly labelled mocks, and real-model tests are opt-in via `ASR_TEST_MODEL` / `NMT_TEST_MODEL_DIR`.
- **LiveKit JS 2.24.1** (read from the installed `.d.ts`): `Room.connect(url, token)`, `RoomEvent.DataReceived = "dataReceived"` with `(payload, participant, kind, topic)`, `RoomEvent.TrackSubscribed(track, publication, participant)`, `RoomEvent.AudioPlaybackStatusChanged = "audioPlaybackChanged"`, `Room.canPlaybackAudio`, `Room.startAudio()`, `Track.Kind.Audio`, `publication.trackName`, `localParticipant.setAttributes(Record<string,string>)` (**requires the `canUpdateOwnMetadata` grant**, now set by `gateway.tokens.create_token`).
- **LiveKit Python 1.3.0, audio out and attributes:** `rtc.AudioSource(sample_rate, num_channels, queue_size_ms=1000)` + `capture_frame`, `rtc.LocalAudioTrack.create_audio_track(name, source)`, `LocalParticipant.publish_track(track, TrackPublishOptions(source=...))`, `rtc.AudioFrame(data, sample_rate, num_channels, samples_per_channel)`, `rtc.TrackSource.SOURCE_MICROPHONE`, events `participant_connected(participant)` and `participant_attributes_changed(changed_attributes, participant)`, `Room.remote_participants`, `VideoGrants.can_update_own_metadata`.
- **Silero VAD timing semantics:** a gate `end` event is emitted `VAD_MIN_SILENCE_MS` after speech stopped; `GateEvent.speech_end` carries the true end of speech.
