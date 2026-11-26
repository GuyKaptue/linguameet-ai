"""Streaming ASR for one utterance: rolling audio buffer + LocalAgreement-n + language lock.

How it works (policy from Macháček et al., 2023; see REFERENCES.md):

1. audio is appended with :meth:`accept`; once ``min_chunk_s`` of *new* audio has arrived
   :meth:`process_iter` re-transcribes the whole buffer with word timestamps;
2. the :class:`HypothesisBuffer` commits the prefix on which the last ``n`` hypotheses agree;
   the committed text is stable, the remaining ``unconfirmed`` tail may still change;
3. the buffer is trimmed at sentence ends (or at the last committed word when it grows past
   ``trim_after_s``) and the trimmed text becomes the ``initial_prompt`` of later passes;
4. :meth:`finish` accepts everything left, producing the final transcript.

Language: the first pass runs with ``language=None`` (Whisper detects it); once the detection
probability reaches ``lang_confidence`` - or after ``max_lang_attempts`` passes - the language is
locked for the rest of the utterance, which avoids hypotheses flip-flopping between languages.
Detection happens again for every utterance, so speakers who switch language between sentences
(code-switching) are followed. Mid-utterance switches are not handled.
"""

from __future__ import annotations

import time

import numpy as np

from services.asr.backends import ASRBackend, ASRResult
from services.asr.hypothesis import HypothesisBuffer, Word, join_words
from shared.types import Transcript

SENTENCE_END = (".", "?", "!", "。", "？", "！", "…")


class StreamingASR:
    def __init__(
        self,
        backend: ASRBackend,
        *,
        segment_id: str = "seg-1",
        participant_id: str = "speaker",
        sample_rate: int = 16000,
        start_time: float = 0.0,
        min_chunk_s: float = 1.0,
        buffer_max_s: float = 30.0,
        trim_after_s: float = 15.0,
        sentence_trim_s: float = 6.0,
        agreement_n: int = 2,
        language_hint: str | None = None,
        lang_confidence: float = 0.6,
        max_lang_attempts: int = 3,
        prompt_chars: int = 200,
    ):
        self.backend = backend
        self.segment_id = segment_id
        self.participant_id = participant_id
        self.sample_rate = sample_rate
        self.min_chunk_s = min_chunk_s
        self.buffer_max_s = buffer_max_s
        self.trim_after_s = trim_after_s
        self.sentence_trim_s = sentence_trim_s
        self.lang_confidence = lang_confidence
        self.max_lang_attempts = max_lang_attempts
        self.prompt_chars = prompt_chars

        self.utterance_start = start_time
        self._buffer_start = start_time
        self._audio = np.empty(0, dtype=np.float32)
        self._run_len = 0  # buffer length (samples) at the last backend call
        self._hyp = HypothesisBuffer(agreement_n)
        self._committed: list[Word] = []
        self._finished = False

        self._hint = language_hint
        self._language: str | None = language_hint
        self._lang_prob = 1.0 if language_hint else 0.0
        self._lang_locked = language_hint is not None
        self._lang_attempts = 0
        self.iterations = 0
        self.trims = 0

    # -------------------------------------------------------------- properties
    @property
    def buffer_seconds(self) -> float:
        return len(self._audio) / self.sample_rate

    @property
    def pending_seconds(self) -> float:
        return (len(self._audio) - self._run_len) / self.sample_rate

    @property
    def language(self) -> str | None:
        return self._language

    @property
    def language_locked(self) -> bool:
        return self._lang_locked

    @property
    def finished(self) -> bool:
        return self._finished

    def should_run(self) -> bool:
        return not self._finished and self.pending_seconds >= self.min_chunk_s

    # ---------------------------------------------------------------- audio in
    def accept(self, audio: np.ndarray) -> None:
        if self._finished:
            raise RuntimeError("utterance already finished")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be 1-D mono")
        if audio.size:
            self._audio = np.concatenate([self._audio, audio]) if self._audio.size else audio.copy()

    # --------------------------------------------------------------- processing
    def process_iter(self) -> Transcript | None:
        """Run one streaming pass. Returns the updated transcript, or None if there is no audio."""
        if self._finished:
            raise RuntimeError("utterance already finished")
        if not self._audio.size:
            return None
        t0 = time.perf_counter()
        self._run_backend()
        committed = self._hyp.flush()
        self._committed.extend(committed)
        self._maybe_trim()
        return self._snapshot(final=False, compute_ms=(time.perf_counter() - t0) * 1000)

    def finish(self) -> Transcript:
        """End of utterance: one last pass over unprocessed audio, then commit everything."""
        if self._finished:
            raise RuntimeError("utterance already finished")
        t0 = time.perf_counter()
        if self.pending_seconds > 0:
            self._run_backend()
            self._committed.extend(self._hyp.flush())
        self._committed.extend(self._hyp.complete())
        self._finished = True
        return self._snapshot(final=True, compute_ms=(time.perf_counter() - t0) * 1000)

    # ----------------------------------------------------------------- internals
    def _run_backend(self) -> None:
        result = self.backend.transcribe(self._audio, language=self._language if self._lang_locked else None, prompt=self._prompt())
        self.iterations += 1
        self._run_len = len(self._audio)
        self._update_language(result)
        self._hyp.insert(result.words, self._buffer_start)

    def _update_language(self, result: ASRResult) -> None:
        if self._lang_locked:
            return
        self._lang_attempts += 1
        if result.language_probability >= self._lang_prob:
            self._language, self._lang_prob = result.language, result.language_probability
        if self._lang_prob >= self.lang_confidence or self._lang_attempts >= self.max_lang_attempts:
            self._lang_locked = True

    def _prompt(self) -> str:
        """Trimmed-away committed text, used as Whisper's ``initial_prompt`` for context."""
        old = [w for w in self._committed if w.end <= self._buffer_start + 1e-3]
        return join_words(old)[-self.prompt_chars :].lstrip() if old else ""

    def _maybe_trim(self) -> None:
        dur = self.buffer_seconds
        in_buffer = [w for w in self._committed if w.end > self._buffer_start + 1e-3]
        trim_at: float | None = None
        if dur > self.trim_after_s and in_buffer:
            trim_at = in_buffer[-1].end
        elif dur > self.sentence_trim_s:
            ends = [w.end for w in in_buffer if w.text.rstrip().endswith(SENTENCE_END)]
            if ends:
                trim_at = ends[-1]
        if dur > self.buffer_max_s and (trim_at is None or self.buffer_seconds - (trim_at - self._buffer_start) > self.buffer_max_s):
            # hard cap: nothing committed recently enough, drop the oldest audio and restart agreement
            trim_at = self._buffer_start + (dur - self.buffer_max_s / 2)
            self._hyp.reset()
            self._hyp.last_committed_time = trim_at
        if trim_at is not None and trim_at > self._buffer_start:
            self._trim_to(trim_at)

    def _trim_to(self, abs_time: float) -> None:
        k = min(len(self._audio), int(round((abs_time - self._buffer_start) * self.sample_rate)))
        if k <= 0:
            return
        self._audio = self._audio[k:]
        self._buffer_start += k / self.sample_rate
        self._run_len = max(0, self._run_len - k)
        self.trims += 1

    def _snapshot(self, *, final: bool, compute_ms: float) -> Transcript:
        confirmed = join_words(self._committed)
        tail = "" if final else join_words(self._hyp.unconfirmed())
        end = self._committed[-1].end if self._committed else self._buffer_start
        return Transcript(
            segment_id=self.segment_id,
            participant_id=self.participant_id,
            text=confirmed,
            language=self._language or "und",
            language_probability=self._lang_prob,
            start_time=self.utterance_start,
            end_time=max(end, self.utterance_start),
            unconfirmed=tail,
            is_final=final,
            compute_ms=compute_ms,
        )
