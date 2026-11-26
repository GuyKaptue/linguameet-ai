"""Meeting pipeline: wires VAD gate events -> streaming ASR -> NMT -> subtitles (+ TTS audio).

Data flow for one utterance::

    GateEvent(start/audio/end) --ASRService--> Transcript (partials, then one final)
        partial: publish the original text at once, then translate the *confirmed* prefix and re-publish
        final  : translate into every listener language in one batch -> publish final subtitle,
                 then synthesize each language and stream it to that language's audio track

Listener languages come from participant preferences (``listen_lang`` attribute). A participant never
receives a translation into the language they speak, and the speaker's own preference is ignored for
their own utterances. Every stage timestamp is recorded in :class:`Milestones` so end-to-end latency
(end of speech -> final subtitle / first audio frame) is measured on the real code path.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

from services.asr.backends import ASRBackend
from services.asr.service import ASRService
from services.translation.backends import NMTBackend
from services.translation.service import TranslationService
from services.tts.backends import TTSBackend
from services.tts.service import TTSService
from services.vad.streaming_vad import GateEvent
from shared.config import Settings
from shared.languages import UnsupportedLanguage, normalize_lang
from shared.types import AudioOut, LatencyReport, SubtitleMessage, Transcript

log = logging.getLogger(__name__)


class SubtitlePublisher(Protocol):
    async def publish_subtitle(self, message: SubtitleMessage, destinations: list[str] | None = None) -> None: ...


class AudioPublisher(Protocol):
    async def publish_frame(self, out: AudioOut) -> None: ...


@dataclass
class Prefs:
    listen: str | None = None  # ISO code of the language the participant wants to read / hear
    speak: str | None = None  # ISO code of the language they speak (None = auto-detect)


@dataclass
class Milestones:
    """Clock readings (``clock()`` seconds) along the path of one utterance."""

    segment_id: str
    vad_end: float | None = None  # the VAD gate closed (>= VAD_MIN_SILENCE_MS after the last spoken sample)
    asr_final: float | None = None  # final transcript handed to the pipeline
    nmt_done: float | None = None  # translations for all listener languages ready
    subtitle_final: float | None = None  # final subtitle accepted by the publisher
    audio_first: dict[str, float] = field(default_factory=dict)  # lang -> first audio frame published
    audio_done: dict[str, float] = field(default_factory=dict)  # lang -> last frame published

    def _ms(self, a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else (b - a) * 1000.0

    @property
    def subtitle_ms(self) -> float | None:
        """VAD decision -> final subtitle published. Add the VAD hang-over to get the speaker-perceived latency."""
        return self._ms(self.vad_end, self.subtitle_final)

    @property
    def audio_ms(self) -> dict[str, float]:
        return {lang: (t - self.vad_end) * 1000.0 for lang, t in self.audio_first.items() if self.vad_end is not None}


@dataclass(eq=False)
class _Segment:
    segment_id: str
    task: asyncio.Task | None = None
    latest: Transcript | None = None
    sent_original: str = ""
    translated_prefix: str = ""
    translations: dict[str, str] = field(default_factory=dict)
    done: bool = False


class MeetingPipeline:
    def __init__(
        self,
        settings: Settings,
        asr_backend: ASRBackend,
        nmt_backend: NMTBackend,
        tts_backend: TTSBackend | None,
        subtitles: SubtitlePublisher,
        audio: AudioPublisher | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.settings = settings
        self.clock = clock
        self.subtitles = subtitles
        self.audio = audio
        self.nmt = TranslationService(settings, nmt_backend)
        self.tts = TTSService(settings, tts_backend) if tts_backend is not None and audio is not None else None
        self.asr = ASRService(settings, asr_backend, self._on_transcript)
        self.prefs: dict[str, Prefs] = {}
        self.milestones: dict[str, Milestones] = {}
        self.reports: dict[str, LatencyReport] = {}
        self.budget_violations: list[str] = []
        self._ends: dict[str, deque[float]] = defaultdict(deque)
        self._segments: dict[str, _Segment] = {}
        self._tts_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._background: set[asyncio.Task] = set()

    # ------------------------------------------------------------ participants
    def set_prefs(self, participant_id: str, listen: str | None, speak: str | None) -> None:
        """Record a participant's languages. Unknown codes are ignored with a warning, never fatal."""
        prefs = self.prefs.setdefault(participant_id, Prefs())
        prefs.listen = self._safe_lang(participant_id, listen)
        prefs.speak = self._safe_lang(participant_id, speak)
        self.asr.set_language_hint(participant_id, prefs.speak)

    def forget(self, participant_id: str) -> None:
        self.prefs.pop(participant_id, None)
        self._ends.pop(participant_id, None)
        self.asr.set_language_hint(participant_id, None)

    @staticmethod
    def _safe_lang(pid: str, code: str | None) -> str | None:
        if not code:
            return None
        try:
            return normalize_lang(code)
        except UnsupportedLanguage:
            log.warning("participant %s announced unsupported language %r - ignored", pid, code)
            return None

    def listener_languages(self, speaker: str, source_lang: str) -> list[str]:
        """Distinct target languages wanted by the *other* participants, excluding the source language."""
        langs = [p.listen for pid, p in self.prefs.items() if pid != speaker and p.listen and p.listen != source_lang]
        return list(dict.fromkeys(langs))

    # ------------------------------------------------------------------- input
    async def on_gate_event(self, participant_id: str, ev: GateEvent) -> None:
        """Sink for :class:`gateway.core.GatewayCore`."""
        if ev.kind == "end":
            self._ends[participant_id].append(self.clock())
        await self.asr.on_gate_event(participant_id, ev)

    # --------------------------------------------------------------- transcripts
    async def _on_transcript(self, tr: Transcript) -> None:
        seg = self._segments.get(tr.segment_id)
        if seg is None:
            seg = self._segments[tr.segment_id] = _Segment(tr.segment_id)
            self.milestones[tr.segment_id] = Milestones(tr.segment_id)
            self._trim()
        seg.latest = tr
        running = seg.task is not None and not seg.task.done()
        if tr.is_final:
            ms = self.milestones[tr.segment_id]
            ends = self._ends.get(tr.participant_id)
            ms.vad_end = ends.popleft() if ends else None
            ms.asr_final = self.clock()
            if running:
                seg.task.cancel()  # in-flight partial work is obsolete
            seg.task = asyncio.get_running_loop().create_task(self._run_final(seg, tr), name=f"final-{seg.segment_id}")
        elif not running:
            seg.task = asyncio.get_running_loop().create_task(self._run_partials(seg), name=f"partial-{seg.segment_id}")

    async def _run_partials(self, seg: _Segment) -> None:
        try:
            while seg.latest is not None and not seg.latest.is_final:
                tr, seg.latest = seg.latest, None
                await self._publish_partial(seg, tr)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("partial update for %s failed", seg.segment_id)

    async def _publish_partial(self, seg: _Segment, tr: Transcript) -> None:
        shown = (tr.text + tr.unconfirmed).strip()
        if not shown:
            return
        if shown != seg.sent_original:  # show the speaker's words immediately
            seg.sent_original = shown
            await self._publish(self._message(tr, shown, seg.translations, is_final=False))
        confirmed = tr.text.strip()
        targets = self.listener_languages(tr.participant_id, tr.language)
        if confirmed and targets and confirmed != seg.translated_prefix:
            results = await self.nmt.translate_segment(tr.segment_id, confirmed, tr.language, targets)
            seg.translated_prefix = confirmed
            seg.translations = {t.target_lang: t.text for t in results}
            await self._publish(self._message(tr, shown, seg.translations, is_final=False))

    async def _run_final(self, seg: _Segment, tr: Transcript) -> None:
        ms = self.milestones[seg.segment_id]
        try:
            text = tr.text.strip()
            targets = self.listener_languages(tr.participant_id, tr.language) if text else []
            translations: dict[str, str] = {}
            if targets:
                results = await self.nmt.translate_segment(seg.segment_id, text, tr.language, targets, is_final=True)
                translations = {t.target_lang: t.text for t in results}
            ms.nmt_done = self.clock()
            if text:
                await self._publish(self._message(tr, text, translations, is_final=True))
            ms.subtitle_final = self.clock()
            self._record(seg.segment_id, tr, ms)
            if text and self.tts is not None:
                for lang, translated in translations.items():
                    if translated.strip() and self.tts.supports(lang):
                        self._spawn(self._speak(seg.segment_id, lang, translated, ms))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("final update for %s failed", seg.segment_id)
        finally:
            seg.done = True

    # -------------------------------------------------------------------- audio
    async def _speak(self, segment_id: str, lang: str, text: str, ms: Milestones) -> None:
        assert self.tts is not None and self.audio is not None
        try:
            async with self._tts_locks[lang]:  # FIFO per language: utterances never overlap
                async for frame in self.tts.synthesize_stream(text, lang, segment_id):
                    ms.audio_first.setdefault(lang, self.clock())
                    await self.audio.publish_frame(frame)
                    if frame.is_last:
                        ms.audio_done[lang] = self.clock()
        except Exception:
            log.exception("TTS for %s/%s failed", segment_id, lang)

    # --------------------------------------------------------------- publishing
    @staticmethod
    def _message(tr: Transcript, text: str, translations: dict[str, str], *, is_final: bool) -> SubtitleMessage:
        return SubtitleMessage(
            segment_id=tr.segment_id,
            speaker=tr.participant_id,
            source_lang=tr.language,
            text=text,
            translations=dict(translations),
            is_final=is_final,
            start_time=tr.start_time,
            end_time=tr.end_time,
        )

    async def _publish(self, msg: SubtitleMessage) -> None:
        try:
            await self.subtitles.publish_subtitle(msg)
        except ValueError:
            # Over the 15 KiB data-packet limit: send the original, then one packet per language.
            # The client merges translations of the same segment_id.
            log.warning("subtitle %s too large, splitting per language", msg.segment_id)
            await self.subtitles.publish_subtitle(replace(msg, translations={}))
            for lang, text in msg.translations.items():
                await self.subtitles.publish_subtitle(replace(msg, translations={lang: text}))
        except Exception:
            log.exception("could not publish subtitle %s", msg.segment_id)

    # ---------------------------------------------------------------- accounting
    def _record(self, segment_id: str, tr: Transcript, ms: Milestones) -> None:
        if ms.vad_end is None or ms.asr_final is None or ms.nmt_done is None or ms.subtitle_final is None:
            return
        report = LatencyReport(segment_id)
        report.add("asr", (ms.asr_final - ms.vad_end) * 1000.0)
        report.add("nmt", (ms.nmt_done - ms.asr_final) * 1000.0)
        report.add("publish", (ms.subtitle_final - ms.nmt_done) * 1000.0)
        self.reports[segment_id] = report
        b = self.settings.budget
        for stage, limit in (("asr", b.asr_ms), ("nmt", b.nmt_ms)):
            if report.stages[stage] > limit:
                self._violation(f"{segment_id}: {stage} {report.stages[stage]:.0f} ms > {limit} ms")
        if report.total_ms > b.e2e_ms:
            self._violation(f"{segment_id}: end-to-end {report.total_ms:.0f} ms > {b.e2e_ms} ms")

    def _violation(self, text: str) -> None:
        self.budget_violations.append(text)
        log.warning("latency budget exceeded: %s", text)

    def _trim(self, keep: int = 512) -> None:
        """Bound memory in long meetings: forget the oldest finished segments and their records."""
        for sid in [k for k, s in self._segments.items() if s.done][: max(0, len(self._segments) - keep)]:
            del self._segments[sid]
        for store in (self.milestones, self.reports):
            while len(store) > keep * 2:
                del store[next(iter(store))]

    # ----------------------------------------------------------------- lifecycle
    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    async def drain(self) -> None:
        """Wait until ASR, translation and speech synthesis of everything received so far is finished."""
        while True:
            await self.asr.drain()
            pending = [s.task for s in self._segments.values() if s.task and not s.task.done()]
            pending += list(self._background)
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def aclose(self) -> None:
        await self.drain()
        await self.asr.aclose()
        await self.nmt.aclose()
        if self.tts is not None:
            await self.tts.aclose()
