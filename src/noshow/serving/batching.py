"""Micro-batching: score requests that arrive at the same time in ONE pipeline call.

Why: profiling showed the sklearn/pandas preprocessing has a fixed cost per call (~same time for
1 row or 32 rows). Under load, many /predict requests wait in the queue anyway, so we take all
of them at once instead of one by one. With no concurrent traffic a request is scored alone
immediately, so single-request latency does not get worse (MICROBATCH_MAX_WAIT_MS=0).

Environment variables
  MICROBATCH_MAX_SIZE     rows to collect per pipeline call (default 32; 1 = batching off)
  MICROBATCH_MAX_WAIT_MS  extra time to wait for more requests before scoring (default 0)
"""

from __future__ import annotations

import asyncio
import os
import time

from noshow.serving import profiling
from noshow.serving.model import ModelBundle

MAX_SIZE = int(os.getenv("MICROBATCH_MAX_SIZE", "32"))
MAX_WAIT_MS = float(os.getenv("MICROBATCH_MAX_WAIT_MS", "0"))


class MicroBatcher:
    """One per worker process. Only one pipeline call runs at a time per worker; requests that
    arrive meanwhile queue up and form the next batch."""

    def __init__(self, max_size: int = MAX_SIZE, max_wait_ms: float = MAX_WAIT_MS) -> None:
        self.max_size = max(1, max_size)
        self.max_wait = max(0.0, max_wait_ms) / 1000
        self.batch_sizes: list[int] = []  # recent batch sizes, for tests and debugging
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None  # keep a reference so it is not garbage-collected

    def _ensure_worker(self) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:  # new event loop (e.g. test client): start fresh
            self._loop = loop
            self._queue = asyncio.Queue()
            self._task = loop.create_task(self._run(self._queue))
        return self._queue

    async def submit(self, bundle: ModelBundle, records: list[dict]) -> list[dict]:
        """Queue one request (1 row for /predict, many for /predict_batch) and await its rows."""
        queue = self._ensure_worker()
        future = asyncio.get_running_loop().create_future()
        queue.put_nowait((bundle, records, future, time.perf_counter()))
        return await future

    async def _collect(self, queue: asyncio.Queue) -> list[tuple]:
        """Take queued requests until max_size rows; one request is never split."""
        batch = [await queue.get()]
        rows = len(batch[0][1])
        deadline = asyncio.get_running_loop().time() + self.max_wait
        while rows < self.max_size:
            if queue.empty():
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout)
                except TimeoutError:
                    break
            else:
                item = queue.get_nowait()
            batch.append(item)
            rows += len(item[1])
        return batch

    async def _run(self, queue: asyncio.Queue) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect(queue)
            self.batch_sizes = (self.batch_sizes + [sum(len(item[1]) for item in batch)])[-1000:]
            if profiling.ENABLED:
                started = time.perf_counter()
                for item in batch:
                    profiling.observe("queue_wait", started - item[3])
            # Usually one model; split if a reload happened while requests were queued.
            groups: dict[int, list[tuple]] = {}
            for item in batch:
                groups.setdefault(id(item[0]), []).append(item)
            for items in groups.values():
                bundle = items[0][0]
                rows = [row for item in items for row in item[1]]
                try:
                    results = await loop.run_in_executor(None, bundle.predict, rows)
                except Exception as exc:
                    for item in items:
                        if not item[2].done():
                            item[2].set_exception(exc)
                    continue
                start = 0
                for _, records, future, _ in items:
                    if not future.done():
                        future.set_result(results[start : start + len(records)])
                    start += len(records)
