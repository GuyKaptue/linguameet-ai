"""ASR service: turns VAD gate events into streaming transcripts without blocking audio intake.

``on_gate_event`` is the sink of :class:`gateway.core.GatewayCore`. It must return quickly (the
gateway reads audio frames in the same task), so transcription runs in a thread pool and each
utterance has a small driver task that performs passes while enough new audio is waiting.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Union

from services.asr.backends import ASRBackend
from services.asr.streaming import StreamingASR
from services.vad.streaming_vad import GateEvent
from shared.config import Settings
from shared.languages import normalize_lang
from shared.types import Transcript

log = logging.getLogger(__name__)
TranscriptSink = Callable[[Transcript], Union[None, Awaitable[None]]]


@dataclass(eq=False)  # identity hashing: sessions are kept in a set
class _Session:
    asr: StreamingASR
    finishing: bool = False
    task: asyncio.Task | None = None


@dataclass
class ASRStats:
    iterations: int = 0
    finals: int = 0
    compute_ms: list[float] = field(default_factory=list)


class ASRService:
    def __init__(
        self,
        settings: Settings,
        backend: ASRBackend,
        on_transcript: TranscriptSink,
        *,
        workers: int = 1,
    ):
        self.settings = settings
        self.backend = backend
        self.on_transcript = on_transcript
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asr")
        self._current: dict[str, _Session] = {}
        self._hints: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        self._all: set[_Session] = set()
        self.stats = ASRStats()

    # ----------------------------------------------------------------- control
    def set_language_hint(self, participant_id: str, language: str | None) -> None:
        """Pin a participant's spoken language (skips detection); ``None`` restores auto-detection."""
        if language is None:
            self._hints.pop(participant_id, None)
        else:
            self._hints[participant_id] = normalize_lang(language)

    async def aclose(self) -> None:
        """Wait for running drivers (no new work is accepted afterwards) and free the thread pool."""
        tasks = [s.task for s in self._all if s.task and not s.task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._executor.shutdown(wait=True)

    async def drain(self) -> None:
        """Wait until every running driver is idle (used by tests and shutdown)."""
        while True:
            tasks = [s.task for s in self._all if s.task and not s.task.done()]
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------- sink
    async def on_gate_event(self, participant_id: str, ev: GateEvent) -> None:
        if ev.kind == "start":
            await self._finish_open(participant_id)  # defensive: a start without an end
            n = self._counters.get(participant_id, 0) + 1
            self._counters[participant_id] = n
            s = self.settings
            session = _Session(
                StreamingASR(
                    self.backend,
                    segment_id=f"{participant_id}-{n}",
                    participant_id=participant_id,
                    sample_rate=s.audio_sample_rate,
                    start_time=ev.time,
                    min_chunk_s=s.asr_min_chunk_seconds,
                    buffer_max_s=s.asr_buffer_max_seconds,
                    agreement_n=s.asr_local_agreement_n,
                    language_hint=self._hints.get(participant_id),
                )
            )
            self._current[participant_id] = session
            self._all.add(session)
            session.asr.accept(ev.audio)
            self._kick(session)
        elif ev.kind == "audio":
            session = self._current.get(participant_id)
            if session is not None:
                session.asr.accept(ev.audio)
                self._kick(session)
        else:  # end
            await self._finish_open(participant_id)

    async def _finish_open(self, participant_id: str) -> None:
        session = self._current.pop(participant_id, None)
        if session is not None:
            session.finishing = True
            self._kick(session)

    # ------------------------------------------------------------------ driver
    def _kick(self, session: _Session) -> None:
        if session.task is None or session.task.done():
            session.task = asyncio.get_running_loop().create_task(self._drive(session))

    async def _drive(self, session: _Session) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                if session.finishing:
                    tr = await loop.run_in_executor(self._executor, session.asr.finish)
                    self.stats.finals += 1
                    await self._emit(tr)
                    self._all.discard(session)
                    return
                if session.asr.should_run():
                    tr = await loop.run_in_executor(self._executor, session.asr.process_iter)
                    if tr is not None:
                        await self._emit(tr)
                    continue
                return
        except Exception:
            log.exception("ASR driver failed for %s", session.asr.segment_id)
            self._all.discard(session)

    async def _emit(self, tr: Transcript) -> None:
        self.stats.iterations += 1
        self.stats.compute_ms.append(tr.compute_ms)
        result = self.on_transcript(tr)
        if inspect.isawaitable(result):
            await result
