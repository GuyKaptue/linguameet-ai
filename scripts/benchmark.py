"""Latency benchmarks -> ``benchmarks/results.json`` and ``BENCHMARKS.md``.

    python scripts/benchmark.py                 # everything that can be measured without model weights
    python scripts/benchmark.py --real asr,nmt,tts --runs 5   # also time the real models (needs weights)

Sections and what they really are
---------------------------------
* ``vad``            REAL: Silero ONNX on this CPU, plus the end-of-speech detection delay.
* ``pipeline``       REAL orchestration code (gateway, VAD, streaming ASR, batching NMT, TTS framing, publishers)
                     driven in real time, but with MOCK models. Scenario ``mock models`` therefore measures the
                     overhead we add; ``simulated model delays`` adds fixed sleeps and shows how the budget behaves
                     if the models cost that much (an assumption, NOT a measurement of Whisper/NLLB/Piper).
* ``nmt_batching``   REAL batching queue with a MOCK model whose cost model is fixed (setup + per item).
* ``asr/nmt/tts_real`` only with ``--real``; absent otherwise (this sandbox cannot download weights).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.asr.streaming import StreamingASR  # noqa: E402
from services.translation.backends import LexiconMockBackend, NMTItem  # noqa: E402
from services.translation.service import TranslationService  # noqa: E402
from services.tts.backends import ToneMockBackend  # noqa: E402
from services.tts.service import TTSService  # noqa: E402
from services.vad.streaming_vad import StreamingVAD  # noqa: E402
from shared.audio import pcm16_to_float32  # noqa: E402
from shared.config import Settings  # noqa: E402

SR = 16000
DATA = ROOT / "tests" / "data"


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 2) if values else None,
        "p50": round(pct(values, 50), 2),
        "p95": round(pct(values, 95), 2),
        "max": round(max(values), 2) if values else None,
    }


def load_wav(name: str) -> np.ndarray:
    with wave.open(str(DATA / name)) as w:
        return pcm16_to_float32(np.frombuffer(w.readframes(w.getnframes()), dtype="<i2"))


def environment() -> dict:
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    import os

    return {"cpu": cpu, "cores": os.cpu_count(), "python": platform.python_version(), "machine": platform.machine(), "os": platform.platform()}


def settings() -> Settings:
    return Settings(_env_file=None, livekit_api_key="devkey", livekit_api_secret="secret_that_is_at_least_32_characters_long")


# ---------------------------------------------------------------------------- VAD
def bench_vad(s: Settings) -> dict:
    audio = np.concatenate([load_wav("jfk.wav"), np.zeros(SR, dtype=np.float32)])
    vad = StreamingVAD.from_settings(s)
    window = 512
    per_window: list[float] = []
    for i in range(0, 2 * SR, window):  # warm-up
        vad.feed(audio[i : i + window])
    vad = StreamingVAD.from_settings(s)
    delays: list[float] = []
    pos = 0
    for i in range(0, len(audio) - window + 1, window):
        t0 = time.perf_counter()
        events = vad.feed(audio[i : i + window])
        per_window.append((time.perf_counter() - t0) * 1000)
        pos = i + window
        for ev in events:
            if ev.kind == "end" and not ev.forced:
                delays.append((pos / SR - ev.speech_end) * 1000)  # audio that had to arrive after the last spoken sample
    return {
        "window_ms": window / SR * 1000,
        "inference_ms_per_window": summary(per_window),
        "realtime_factor": round(statistics.fmean(per_window) / (window / SR * 1000), 5),
        "end_of_speech_delay_ms": summary(delays),
        "config": {"min_silence_ms": s.vad_min_silence_ms, "speech_pad_ms": s.vad_speech_pad_ms},
    }


# ----------------------------------------------------------------------- NMT batching (mock model)
class CostModelNMT(LexiconMockBackend):
    """MOCK: each batch costs ``setup_ms + per_item_ms * items`` of real sleeping, like a GPU/CPU kernel launch + work."""

    def __init__(self, setup_ms: float, per_item_ms: float):
        super().__init__()
        self.setup_ms, self.per_item_ms = setup_ms, per_item_ms

    def translate_batch(self, items: list[NMTItem]) -> list[str]:
        time.sleep((self.setup_ms + self.per_item_ms * len(items)) / 1000)
        return super().translate_batch(items)


async def bench_nmt_batching(s: Settings, requests: int = 16, setup_ms: float = 80.0, per_item_ms: float = 10.0) -> dict:
    texts = [f"good morning everyone number {i}" for i in range(requests)]
    svc = TranslationService(s, CostModelNMT(setup_ms, per_item_ms))
    t0 = time.perf_counter()
    await asyncio.gather(*(svc.translate(t, "en", "es") for t in texts))
    batched = (time.perf_counter() - t0) * 1000
    batches = list(svc.stats.batch_sizes)
    await svc.aclose()
    one_by_one = requests * (setup_ms + per_item_ms)
    return {
        "requests": requests, "cost_model_ms": {"setup": setup_ms, "per_item": per_item_ms},
        "batched_total_ms": round(batched, 1), "batch_sizes": batches,
        "unbatched_estimate_ms": one_by_one, "speedup": round(one_by_one / batched, 2),
    }


# ----------------------------------------------------------------------- generic backend benchmarks
def bench_asr(backend, audio: np.ndarray, s: Settings, language: str | None = "en") -> dict:
    """Streams ``audio`` in 1 s steps through StreamingASR and times every backend pass and the final pass."""
    asr = StreamingASR(backend, sample_rate=SR, min_chunk_s=s.asr_min_chunk_seconds, language_hint=language)
    passes: list[float] = []
    step = SR
    for i in range(0, len(audio), step):
        asr.accept(audio[i : i + step])
        while asr.should_run():
            tr = asr.process_iter()
            if tr is not None:
                passes.append(tr.compute_ms)
    final = asr.finish()
    return {
        "audio_seconds": round(len(audio) / SR, 2), "pass_ms": summary(passes),
        "final_pass_ms": round(final.compute_ms, 2), "final_text": final.text.strip(),
    }


async def bench_nmt(backend, s: Settings, sentences: list[str], src: str, tgts: list[str]) -> dict:
    svc = TranslationService(s, backend)
    single, multi = [], []
    for text in sentences:  # one sentence -> one target
        t0 = time.perf_counter()
        await svc.translate(text, src, tgts[0])
        single.append((time.perf_counter() - t0) * 1000)
    for text in sentences:  # one sentence -> all listener languages in one batch
        t0 = time.perf_counter()
        await svc.translate_segment("b", text + " ", src, tgts)
        multi.append((time.perf_counter() - t0) * 1000)
    await svc.aclose()
    return {"sentences": len(sentences), "one_target_ms": summary(single), f"{len(tgts)}_targets_ms": summary(multi)}


async def bench_tts(backend, s: Settings, text: str, lang: str, runs: int = 5) -> dict:
    svc = TTSService(s, backend)
    for _ in range(runs):
        await svc.synthesize(text, lang)
    recs = svc.stats.records
    await svc.aclose()
    return {
        "chars": len(text), "first_chunk_ms": summary([r.first_chunk_ms for r in recs]),
        "rtf": summary([r.rtf for r in recs]), "audio_seconds": round(statistics.fmean(r.audio_seconds for r in recs), 2),
    }


# ----------------------------------------------------------------------- pipeline end to end
def bench_pipeline(runs: int, **delays) -> dict:
    from tests.e2e.test_meeting_flow import run_meeting  # the same scenario the e2e test asserts on

    wall, internal, asr_ms, nmt_ms, pub_ms, audio = [], [], [], [], [], []
    for _ in range(runs):
        run = asyncio.run(run_meeting(**delays))
        wall += run.subtitle_latencies_ms()
        audio += run.audio_latencies_ms()
        for sid, ms in run.pipeline.milestones.items():
            internal.append(ms.subtitle_ms)
            rep = run.pipeline.reports[sid]
            asr_ms.append(rep.stages["asr"]); nmt_ms.append(rep.stages["nmt"]); pub_ms.append(rep.stages["publish"])
    return {
        "utterances": len(wall), "delays_s": delays,
        "speech_end_to_subtitle_wall_ms": summary(wall),
        "from_vad_decision_ms": summary(internal),
        "stage_asr_ms": summary(asr_ms), "stage_nmt_ms": summary(nmt_ms), "stage_publish_ms": summary(pub_ms),
        "speech_end_to_first_audio_ms": summary(audio),
    }


# ----------------------------------------------------------------------------- report
def table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def fmt(d: dict, keys=("p50", "p95", "max")) -> tuple[str, ...]:
    return tuple(f"{d[k]:.0f}" if d[k] is not None and d[k] >= 10 else f"{d[k]:.2f}" for k in keys)


def render(res: dict, runs: int) -> str:
    env, vad, nb = res["environment"], res["vad"], res["nmt_batching"]
    mock, sim = res["pipeline"]["mock models"], res["pipeline"]["simulated model delays"]
    b = settings().budget
    real = res.get("real", {})
    L: list[str] = []
    L.append("# BENCHMARKS\n")
    L.append(f"Generated by `scripts/benchmark.py` ({runs} runs per pipeline scenario). Raw numbers: `benchmarks/results.json`.\n")
    L.append(f"**Machine:** {env['cpu']}, {env['cores']} cores, Python {env['python']}, no GPU.\n")
    L.append("## What is measured and what is not\n")
    L.append(table([
        ("Silero VAD inference + end-of-speech delay", "**Measured** (real model, this CPU)"),
        ("Gateway, streaming-ASR logic, batching queue, TTS framing, publishers", "**Measured** (real code, wall clock, real time)"),
        ("Whisper / NLLB / Piper compute time", "**NOT measured here**" if not real else "**Measured** (see section 5)"),
        ("GPU numbers, LiveKit network RTT", "**Not measured**"),
    ], ("Component", "Status")))
    L.append("\nThe sandbox could not download model weights (Hugging Face is unreachable), so the real models are "
             "benchmarked only when you run `python scripts/benchmark.py --real asr,nmt,tts` on a machine that has them. "
             "Nothing below pretends otherwise: rows labelled *simulated* use fixed sleeps that I chose, not measurements.\n")

    L.append("## 1. Voice activity detection (real)\n")
    L.append(table([
        ("Silero ONNX per 32 ms window", *fmt(vad["inference_ms_per_window"])),
        ("Last spoken sample -> gate closes (`end` event)", *fmt(vad["end_of_speech_delay_ms"])),
    ], ("Metric (ms)", "p50", "p95", "max")))
    L.append(f"\nReal-time factor {vad['realtime_factor']}. The end-of-speech delay is dominated by `VAD_MIN_SILENCE_MS="
             f"{vad['config']['min_silence_ms']}`: the gate has to hear that much silence before it can close the utterance. "
             "It is added to every end-to-end figure that is measured from the last spoken sample.\n")

    L.append("## 2. Pipeline overhead with mock models (real orchestration, real time)\n")
    L.append("Three utterances of real speech (`tests/data/speech_en.wav`) are fed at 1x speed through "
             "`LiveKitGateway` (fake room) -> VAD -> ASR -> NMT -> TTS -> publishers, with listeners for Spanish and French.\n")
    L.append(table([
        ("Last spoken sample -> final subtitle published", *fmt(mock["speech_end_to_subtitle_wall_ms"])),
        ("VAD gate closed -> final subtitle published", *fmt(mock["from_vad_decision_ms"])),
        ("  of which ASR stage (final transcript ready)", *fmt(mock["stage_asr_ms"])),
        ("  of which NMT stage", *fmt(mock["stage_nmt_ms"])),
        ("  of which publish", *fmt(mock["stage_publish_ms"])),
        ("Last spoken sample -> first TTS frame published", *fmt(mock["speech_end_to_first_audio_ms"])),
    ], ("Metric (ms)", "p50", "p95", "max")))
    L.append("\nWith instant models the 'ASR stage' is the time the streaming driver needs to run its final pass; the remaining "
             "wall-clock time is the VAD hang-over. This is the floor the real models add to.\n")

    L.append("## 3. Budget under assumed model costs (SIMULATED)\n")
    d = sim["delays_s"]
    L.append(f"Same run, but every ASR pass sleeps {d['asr_delay']*1000:.0f} ms, every NMT batch {d['nmt_delay']*1000:.0f} ms and "
             f"TTS {d['tts_delay']*1000:.0f} ms before its first audio. These delays are **assumptions** chosen to look like a small model on a "
             "modern CPU/GPU; replace them with section 5 numbers when you have them.\n")
    L.append(table([
        ("Last spoken sample -> final subtitle published", *fmt(sim["speech_end_to_subtitle_wall_ms"])),
        ("  ASR stage", *fmt(sim["stage_asr_ms"])),
        ("  NMT stage", *fmt(sim["stage_nmt_ms"])),
        ("Last spoken sample -> first TTS frame published", *fmt(sim["speech_end_to_first_audio_ms"])),
    ], ("Metric (ms)", "p50", "p95", "max")))
    L.append(f"\nBudget (`shared/config.py`): end-to-end {b.e2e_ms} ms, ASR {b.asr_ms} ms, NMT {b.nmt_ms} ms, TTS {b.tts_ms} ms. "
             "`tests/e2e/test_meeting_flow.py` asserts < 2000 ms for both scenarios and checks that a 1.6 s ASR is reported as a violation.\n")

    L.append("## 4. NMT batching (real queue, mock model with a fixed cost model)\n")
    L.append(f"{nb['requests']} concurrent requests, model cost = {nb['cost_model_ms']['setup']:.0f} ms per batch + "
             f"{nb['cost_model_ms']['per_item']:.0f} ms per item: batched total **{nb['batched_total_ms']:.0f} ms** "
             f"(batches {nb['batch_sizes']}) versus {nb['unbatched_estimate_ms']:.0f} ms one at a time, a {nb['speedup']}x speed-up. "
             "This shows the queue mechanics only; the cost model is an assumption.\n")

    L.append("## 5. Real models\n")
    if not real:
        L.append("Not run in this environment. Command to fill this section: `python scripts/benchmark.py --real asr,nmt,tts`.\n")
    else:
        if "asr" in real:
            a = real["asr"]
            L.append(f"**ASR** (`{a['model']}`, {a['audio_seconds']} s of speech, 1 s steps): per pass p50/p95/max "
                     f"{'/'.join(fmt(a['pass_ms']))} ms, final pass {a['final_pass_ms']} ms. Text: _{a['final_text']}_\n")
        if "nmt" in real:
            n = real["nmt"]
            keys = [k for k in n if k.endswith("targets_ms")]
            L.append(f"**NMT**: one target p50/p95/max {'/'.join(fmt(n['one_target_ms']))} ms; "
                     + "; ".join(f"{k.replace('_ms','').replace('_',' ')} {'/'.join(fmt(n[k]))} ms" for k in keys) + ".\n")
        if "tts" in real:
            t = real["tts"]
            L.append(f"**TTS**: first chunk p50/p95/max {'/'.join(fmt(t['first_chunk_ms']))} ms, real-time factor p50 {t['rtf']['p50']}.\n")
    L.append("## How to reproduce\n")
    L.append("```bash\npip install -e '.[dev]'\npython scripts/benchmark.py --runs 5\npytest tests/e2e -q\n```\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--real", default="", help="comma list of asr,nmt,tts to time with real weights")
    args = ap.parse_args()
    s = settings()
    res: dict = {"environment": environment(), "vad": bench_vad(s)}
    res["nmt_batching"] = asyncio.run(bench_nmt_batching(s))
    res["pipeline"] = {
        "mock models": bench_pipeline(args.runs),
        "simulated model delays": bench_pipeline(args.runs, asr_delay=0.35, nmt_delay=0.15, tts_delay=0.12),
    }
    wanted = {x for x in args.real.split(",") if x}
    if wanted:
        from services.asr.backends import FasterWhisperBackend
        from services.translation.service import create_backend as nmt_backend
        from services.tts.backends import create_backend as tts_backend

        real_s = Settings(_env_file=None) if not (ROOT / ".env").exists() else Settings(_env_file=ROOT / ".env")
        res["real"] = {}
        if "asr" in wanted:
            res["real"]["asr"] = {"model": real_s.asr_model_size, **bench_asr(FasterWhisperBackend(real_s), load_wav("jfk.wav")[: 8 * SR], real_s)}
        if "nmt" in wanted:
            res["real"]["nmt"] = asyncio.run(bench_nmt(
                nmt_backend(real_s), real_s,
                ["Good morning everyone.", "The meeting starts at nine o'clock.", "Please share your screen."], "en", ["es", "fr", "de"]))
        if "tts" in wanted:
            res["real"]["tts"] = asyncio.run(bench_tts(tts_backend(real_s), real_s, "Good morning everyone, the meeting starts now.", "en"))
    out = ROOT / "benchmarks"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    (ROOT / "BENCHMARKS.md").write_text(render(res, args.runs))
    print((ROOT / "BENCHMARKS.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
