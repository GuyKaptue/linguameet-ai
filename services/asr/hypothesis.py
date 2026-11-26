"""LocalAgreement-n hypothesis buffer (Macháček et al., 2023, arXiv:2307.14743).

Streaming policy: re-transcribe a growing audio buffer; a word is *committed* only when it
appears (same position, same normalised text) in ``n`` consecutive hypotheses - i.e. the
output is the longest common prefix of the last ``n`` hypotheses. ``n=2`` is the paper's
choice. Words already committed are removed from later hypotheses by timestamp and by an
n-gram overlap check, because Whisper re-emits the start of the buffer each time.

This is a re-implementation of the published policy, not a copy of the whisper_streaming code.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True)
class Word:
    start: float  # seconds (absolute stream time once inserted into the buffer)
    end: float
    text: str  # raw token incl. any leading space, exactly as Whisper produced it

    @property
    def norm(self) -> str:
        return _NON_WORD.sub("", self.text.lower())


def join_words(words: list[Word]) -> str:
    """Whisper tokens carry their own leading spaces, so plain concatenation preserves spacing
    for both spaced languages and CJK."""
    return "".join(w.text for w in words).strip()


class HypothesisBuffer:
    def __init__(self, n: int = 2, overlap_ngram: int = 5, time_tolerance: float = 0.1):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self._overlap_ngram = overlap_ngram
        self._tolerance = time_tolerance
        self._history: deque[list[Word]] = deque(maxlen=max(n - 1, 0))  # previous unconfirmed hypotheses
        self._new: list[Word] = []
        self._tail: list[Word] = []  # last committed words (for overlap removal)
        self.last_committed_time = 0.0

    def reset(self) -> None:
        self._history.clear()
        self._new = []
        self._tail = []
        self.last_committed_time = 0.0

    def insert(self, words: list[Word], offset: float) -> None:
        """Add a new hypothesis. ``words`` have times relative to the buffer; ``offset`` is the
        absolute stream time of the buffer start."""
        shifted = [Word(w.start + offset, w.end + offset, w.text) for w in words]
        new = [w for w in shifted if w.start > self.last_committed_time - self._tolerance]
        if new and self._tail and abs(new[0].start - self.last_committed_time) < 1.0:
            for k in range(min(len(self._tail), len(new), self._overlap_ngram), 0, -1):
                if [w.norm for w in self._tail[-k:]] == [w.norm for w in new[:k]]:
                    new = new[k:]
                    break
        self._new = new

    def flush(self) -> list[Word]:
        """Commit and return the prefix on which this hypothesis agrees with the previous n-1."""
        new = self._new
        k = 0
        # agreement needs n-1 earlier hypotheses; before that nothing can be committed (n=1 commits at once)
        if len(self._history) >= self.n - 1:
            while k < len(new) and all(len(h) > k and h[k].norm == new[k].norm for h in self._history):
                k += 1
        committed = new[:k]
        if committed:
            self.last_committed_time = committed[-1].end
            self._tail = (self._tail + committed)[-self._overlap_ngram :]
        remainder = new[k:]
        # histories keep only what is still unconfirmed
        trimmed = deque((h[k:] for h in self._history), maxlen=self._history.maxlen)
        self._history = trimmed
        if self._history.maxlen:
            self._history.append(remainder)
        self._new = remainder
        return committed

    def unconfirmed(self) -> list[Word]:
        """Words of the latest hypothesis that are not yet committed."""
        return list(self._new)

    def complete(self) -> list[Word]:
        """End of utterance: accept the latest hypothesis' remaining words as final."""
        rest = self.unconfirmed()
        if rest:
            self.last_committed_time = rest[-1].end
            self._tail = (self._tail + rest)[-self._overlap_ngram :]
        self._new = []
        self._history.clear()
        return rest
