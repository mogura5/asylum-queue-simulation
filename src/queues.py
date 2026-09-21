"""
Pluggable queue-discipline strategies for Stage workers to pull from.

Each strategy implements a common interface: push(case, now), pop_next(now),
remove(case), apply_aging(now) (no-op unless overridden), and __len__.

IMPORTANT: only Stage's own worker processes ever call pop_next() (see
stage.py) -- arriving cases never request their own service slot the way
the original simulation_runner.py / cel_algorithm.py / fifo_lifo_experiment.py
/ scoring_experiment.py each did (push, then separately request a
simpy.Resource, then pop). That pattern is what caused the confirmed bug
where a reneged case was never removed from the queue object, permanently
inflating backlog metrics. Under the worker-pool design, a case's fate
(served vs. reneged) is decided synchronously by the worker immediately
after it pops the case -- so there's no separate abandonment process that
needs to reach back into the queue asynchronously. `remove()` is kept here
as a general utility (e.g. useful if a future feature needs to withdraw a
case mid-wait) but is not on the critical path for reneging today.
"""

from collections import deque
import heapq
from typing import Optional
from case import Case


class QueueStrategy:
    def push(self, case: Case, now: float) -> None:
        raise NotImplementedError

    def pop_next(self, now: float) -> Optional[Case]:
        raise NotImplementedError

    def remove(self, case: Case) -> bool:
        raise NotImplementedError

    def apply_aging(self, now: float) -> None:
        pass  # default: no aging

    def __len__(self) -> int:
        raise NotImplementedError


class FIFOQueue(QueueStrategy):
    """Standard first-in, first-out queue."""

    def __init__(self):
        self._q = deque()

    def push(self, case, now):
        self._q.append(case)

    def pop_next(self, now):
        return self._q.popleft() if self._q else None

    def remove(self, case):
        try:
            self._q.remove(case)
            return True
        except ValueError:
            return False

    def __len__(self):
        return len(self._q)


class LIFOQueue(QueueStrategy):
    """Last-in, first-out stack with a starvation cap: a case that has
    waited past max_wait is forced to the front instead of being served
    in strict last-in-first-out order (this is current U.S. policy's
    real-world discipline, with the cap representing an escape valve)."""

    def __init__(self, max_wait: float = 36.0):
        self._stack = []
        self._entry_times = {}
        self._max_wait = max_wait

    def push(self, case, now):
        self._stack.append(case)
        self._entry_times[case.case_id] = now

    def pop_next(self, now):
        if not self._stack:
            return None
        oldest_idx, oldest_wait = None, -1
        for i, c in enumerate(self._stack):
            wait = now - self._entry_times.get(c.case_id, now)
            if wait > oldest_wait:
                oldest_wait, oldest_idx = wait, i
        if oldest_wait >= self._max_wait:
            case = self._stack.pop(oldest_idx)
        else:
            case = self._stack.pop()
        self._entry_times.pop(case.case_id, None)
        return case

    def remove(self, case):
        try:
            self._stack.remove(case)
            self._entry_times.pop(case.case_id, None)
            return True
        except ValueError:
            return False

    def __len__(self):
        return len(self._stack)


class PriorityQueue(QueueStrategy):
    """Non-preemptive max-priority queue using a min-heap (negated priority).
    Aging: cases waiting beyond aging_threshold get a periodic priority
    boost, preventing indefinite starvation of low-priority cases.

    PERFORMANCE NOTE (see redesign-decisions.md, "Aging-loop optimization"):
    the original apply_aging() rescanned and rebuilt the ENTIRE heap every
    call (every 1 simulated month, by default -- see stage.py's
    aging_interval), regardless of how many cases actually crossed the
    aging threshold that tick. Profiling a real-scale PRIORITY replication
    showed this consuming >50% of total replication runtime at a ~240K-case
    backlog. Fixed here with a second min-heap ("_due"), ordered by each
    case's NEXT eligible boost time (entry_time + aging_threshold), so
    apply_aging only touches the handful of cases actually due this tick
    instead of the whole queue -- O(k log n) for k newly-due cases instead
    of O(n log n) for all n queued cases. Every entry (in both heaps) is
    version-tagged so a stale copy left behind by a pop/remove/reboost is
    cheaply detected and skipped (the same lazy-deletion idea the original
    used for remove(), generalized to also cover priority updates, which a
    single removed-id set can't distinguish from a fresh re-push of the
    same case_id). This changes nothing about WHEN or BY HOW MUCH a case's
    priority is boosted -- same threshold, same boost amount, same 1-tick
    granularity -- only how cheaply it's computed, so results for a given
    seed are unchanged, just produced much faster.
    """

    def __init__(self, aging_threshold: float = 6.0, aging_boost: float = 2.0):
        self._heap = []          # (neg_priority, arrival_time, case_id, version)
        self._due = []           # (next_eligible_boost_time, case_id, version)
        self._version = {}       # case_id -> version of its currently-live entry
        self._case_by_id = {}    # case_id -> Case, for the currently-live entry
        self._entry_times = {}   # case_id -> time its current aging clock started
        self._aging_threshold = aging_threshold
        self._aging_boost = aging_boost
        self._len = 0

    def push(self, case, now):
        cid = case.case_id
        v = self._version.get(cid, 0) + 1
        self._version[cid] = v
        self._case_by_id[cid] = case
        self._entry_times[cid] = now
        heapq.heappush(self._heap, (-case.base_priority, case.arrival_time, cid, v))
        heapq.heappush(self._due, (now + self._aging_threshold, cid, v))
        self._len += 1

    def pop_next(self, now):
        while self._heap:
            _, _, cid, v = heapq.heappop(self._heap)
            if self._version.get(cid) != v:
                continue  # stale copy of a case that's since been reboosted/removed
            case = self._case_by_id.pop(cid, None)
            self._version.pop(cid, None)
            self._entry_times.pop(cid, None)
            self._len -= 1
            return case
        return None

    def remove(self, case):
        cid = case.case_id
        if cid in self._entry_times:
            self._version[cid] = self._version.get(cid, 0) + 1  # invalidates live entry
            self._case_by_id.pop(cid, None)
            self._entry_times.pop(cid, None)
            self._len -= 1
            return True
        return False

    def apply_aging(self, now):
        """Drain only the cases actually due for a boost as of `now` --
        see the class docstring for why this replaces the old full-heap
        rescan. self._due is a min-heap on due time, so its front is
        always the next case to become eligible; once the front isn't due
        yet, nothing later in the heap can be either."""
        while self._due and self._due[0][0] <= now:
            _, cid, v = heapq.heappop(self._due)
            if self._version.get(cid) != v:
                continue  # already popped/removed/reboosted since this was scheduled
            case = self._case_by_id[cid]
            case.base_priority += self._aging_boost
            new_v = v + 1
            self._version[cid] = new_v
            self._entry_times[cid] = now
            heapq.heappush(self._heap, (-case.base_priority, case.arrival_time, cid, new_v))
            heapq.heappush(self._due, (now + self._aging_threshold, cid, new_v))

    def __len__(self):
        return self._len


def make_queue(algorithm: str, **kwargs) -> QueueStrategy:
    if algorithm == "FIFO":
        return FIFOQueue()
    elif algorithm == "LIFO":
        return LIFOQueue(**kwargs)
    elif algorithm == "PRIORITY":
        return PriorityQueue(**kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")