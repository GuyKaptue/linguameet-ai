"""Scripted meeting participants for the standalone demo.

Each bot "speaks" a REAL recording (``tests/data``). Because the sandbox has no Whisper weights, the transcript
of each recording is SCRIPTED (``phrases`` are spread over the real speech segments the VAD finds), and
``MultiReplayASR`` replays it. So: real audio, real VAD, real streaming logic - scripted words.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from services.asr.backends import ASRResult, ReplayASRBackend
from services.asr.hypothesis import Word
from services.translation.backends import NMTItem, LexiconMockBackend
from services.vad.streaming_vad import SpeechSegmenter, StreamingVAD
from shared.audio import pcm16_to_float32

SR = 16000
DATA = Path(__file__).resolve().parents[1] / "tests" / "data"
TAIL_S = 0.8  # trailing silence so the VAD can close the last utterance


@dataclass
class Bot:
    id: str
    name: str
    lang: str  # ISO code of the language the bot speaks (and listens to)
    wav: str
    phrases: list[str]
    pcm: np.ndarray | None = None  # int16 incl. trailing silence
    asr: ReplayASRBackend | None = None

    @property
    def seconds(self) -> float:
        return 0.0 if self.pcm is None else len(self.pcm) / SR


BOT_SPECS = [
    ("alice", "Alice", "en", "speech_en.wav", ["Good morning everyone", "Please share your screen", "Hello, how are you"]),
    ("bruno", "Bruno", "fr", "speech_fr.wav", ["Bonjour à tous", "Veuillez partager votre écran"]),
]


def build_bot(bot_id: str, name: str, lang: str, wav: str, phrases: list[str]) -> Bot:
    with wave.open(str(DATA / wav)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    pcm = np.concatenate([pcm, np.zeros(int(TAIL_S * SR), dtype=np.int16)])
    ref = pcm16_to_float32(pcm)
    seg = SpeechSegmenter(StreamingVAD(), bot_id)
    segs = seg.feed(ref) + seg.flush()
    if len(segs) != len(phrases):
        raise RuntimeError(f"{wav}: VAD found {len(segs)} utterances but {len(phrases)} phrases were scripted")
    script: list[Word] = []
    for s, phrase in zip(segs, phrases):  # SYNTHETIC word timings inside each real speech span
        words = phrase.split()
        span = (s.end_time - s.start_time) - 0.1
        for j, word in enumerate(words):
            a = s.start_time + 0.05 + span * j / len(words)
            script.append(Word(a, a + span / len(words) * 0.9, " " + word))
    return Bot(bot_id, name, lang, wav, phrases, pcm, ReplayASRBackend(ref, script, language=lang))


def default_bots() -> dict[str, Bot]:
    return {spec[0]: build_bot(*spec) for spec in BOT_SPECS}


class MultiReplayASR:
    """MOCK ASR: finds which bot's recording the audio buffer came from and replays that bot's script."""

    is_mock = True

    def __init__(self, bots: dict[str, Bot]):
        self.backends = [b.asr for b in bots.values() if b.asr is not None]

    def transcribe(self, audio: np.ndarray, *, language: str | None, prompt: str = "") -> ASRResult:
        for backend in self.backends:
            if backend._locate(audio) >= 0:
                return backend.transcribe(audio, language=language, prompt=prompt)
        return ASRResult([], language or "en", 1.0)


class PivotLexiconNMT(LexiconMockBackend):
    """MOCK NMT: the phrase table only links English <-> other languages, so other pairs pivot through English."""

    EN = "eng_Latn"

    def translate_batch(self, items: list[NMTItem]) -> list[str]:
        direct = [it for it in items if it.src == self.EN or it.tgt == self.EN or it.src == it.tgt]
        out: dict[int, str] = {}
        if direct:
            for it, res in zip(direct, super().translate_batch(direct)):
                out[id(it)] = res
        pivots = [it for it in items if id(it) not in out]
        if pivots:
            to_en = super().translate_batch([NMTItem(it.text, it.src, self.EN) for it in pivots])
            from_en = super().translate_batch([NMTItem(t, self.EN, it.tgt) for t, it in zip(to_en, pivots)])
            for it, res in zip(pivots, from_en):
                out[id(it)] = res
        return [out[id(it)] for it in items]
