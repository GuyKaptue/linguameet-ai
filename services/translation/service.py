"""Translation service: async API on top of a batching request queue.

Many participants speak at once and every utterance fans out to several listener languages, so
requests are collected for a short window (``NMT_BATCH_WINDOW_MS``) or until ``NMT_MAX_BATCH_SIZE``
items are waiting, then sent to the model as a single ``translate_batch`` call in a worker thread.
Identical requests inside a batch are executed once, and a small LRU cache answers repeats (the
streaming ASR re-sends the same confirmed text many times).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from services.translation.backends import CTranslate2NLLBBackend, LexiconMockBackend, NMTBackend, NMTItem
from services.translation.chunking import pack_chunks
from shared.config import Settings
from shared.languages import normalize_lang, to_flores
from shared.types import Translation

log = logging.getLogger(__name__)
_NO_SPACE_LANGS = {"zh", "ja", "th"}


def create_backend(settings: Settings) -> NMTBackend:
    """Real NLLB/CTranslate2 backend, or the labelled lexicon mock when ``BACKEND_MODE=mock``."""
    if settings.is_mock:
        log.warning("NMT running with the MOCK lexicon backend (BACKEND_MODE=mock)")
        return LexiconMockBackend()
    return CTranslate2NLLBBackend(settings)


@dataclass
class _Request:
    item: NMTItem
    future: asyncio.Future


@dataclass
class NMTStats:
    requests: int = 0
    cache_hits: int = 0
    batches: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    backend_ms: list[float] = field(default_factory=list)


class TranslationService:
    def __init__(self, settings: Settings, backend: NMTBackend, *, cache_size: int = 2048):
        self.settings = settings
        self.backend = backend
        self._queue: asyncio.Queue[_Request] | None = None
        self._worker: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nmt")
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._cache_size = cache_size
        self._closed = False
        self.stats = NMTStats()

    # --------------------------------------------------------------- lifecycle
    def _ensure_worker(self) -> asyncio.Queue:
        if self._closed:
            raise RuntimeError("TranslationService is closed")
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._run(), name="nmt-batcher")
        return self._queue

    async def aclose(self) -> None:
        self._closed = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        if self._queue is not None:
            while not self._queue.empty():
                req = self._queue.get_nowait()
                if not req.future.done():
                    req.future.set_exception(RuntimeError("TranslationService closed"))
        self._executor.shutdown(wait=True)

    # ---------------------------------------------------------------- public API
    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate ``text`` (any ISO/locale/FLORES code accepted for the languages)."""
        src_iso, tgt_iso = normalize_lang(source_lang), normalize_lang(target_lang)
        text = text.strip()
        if not text or src_iso == tgt_iso:
            return text
        src, tgt = to_flores(src_iso), to_flores(tgt_iso)
        chunks = pack_chunks(text, lambda t: self.backend.count_tokens(t, src), self.settings.nmt_max_input_tokens - 4)
        parts = await asyncio.gather(*(self._translate_chunk(c, src, tgt) for c in chunks))
        return ("" if tgt_iso in _NO_SPACE_LANGS else " ").join(p for p in parts if p)

    async def translate_segment(
        self, segment_id: str, text: str, source_lang: str, targets: list[str], *, is_final: bool = False
    ) -> list[Translation]:
        """Translate one utterance into every listener language at once (one shared batch)."""
        t0 = time.perf_counter()
        langs = list(dict.fromkeys(normalize_lang(t) for t in targets))
        texts = await asyncio.gather(*(self.translate(text, source_lang, t) for t in langs))
        ms = (time.perf_counter() - t0) * 1000
        src = normalize_lang(source_lang)
        return [Translation(segment_id, src, t, tx, is_final, ms) for t, tx in zip(langs, texts)]

    # ------------------------------------------------------------------ internals
    async def _translate_chunk(self, text: str, src: str, tgt: str) -> str:
        self.stats.requests += 1
        key = (text, src, tgt)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.stats.cache_hits += 1
            return self._cache[key]
        queue = self._ensure_worker()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        queue.put_nowait(_Request(NMTItem(text, src, tgt), fut))
        result = await fut
        self._cache[key] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result

    async def _run(self) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        window = self.settings.nmt_batch_window_ms / 1000.0
        max_batch = self.settings.nmt_max_batch_size
        while True:
            batch = [await self._queue.get()]
            deadline = loop.time() + window
            while len(batch) < max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    # drain whatever is already queued without waiting
                    try:
                        batch.append(self._queue.get_nowait())
                        continue
                    except asyncio.QueueEmpty:
                        break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            await self._dispatch(batch, loop)

    async def _dispatch(self, batch: list[_Request], loop: asyncio.AbstractEventLoop) -> None:
        # identical items in one batch are translated once
        unique: dict[NMTItem, list[_Request]] = {}
        for req in batch:
            unique.setdefault(req.item, []).append(req)
        items = list(unique)
        t0 = time.perf_counter()
        try:
            outputs = await loop.run_in_executor(self._executor, self.backend.translate_batch, items)
            if len(outputs) != len(items):
                raise RuntimeError(f"backend returned {len(outputs)} results for {len(items)} items")
        except Exception as exc:  # fail only this batch, keep the worker alive
            log.exception("NMT batch of %d failed", len(items))
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
            return
        self.stats.batches += 1
        self.stats.batch_sizes.append(len(items))
        self.stats.backend_ms.append((time.perf_counter() - t0) * 1000)
        for item, out in zip(items, outputs):
            for req in unique[item]:
                if not req.future.done():
                    req.future.set_result(out)
