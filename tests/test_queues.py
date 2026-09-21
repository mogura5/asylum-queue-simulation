"""
Protocol-integrity tests for queues.py, especially the PriorityQueue
aging-loop optimization (redesign-decisions.md, "Case study: profiling and
fixing the Priority queue's aging-loop bottleneck"). The whole point of that
fix was "same results, cheaper" -- these tests hold it to that promise by
comparing it against a deliberately naive reference implementation of the
ORIGINAL full-heap-rebuild apply_aging(), across randomized operation
sequences.
"""

import heapq
import random

import pytest

from src.case import Case
from src.queues import PriorityQueue, FIFOQueue, LIFOQueue


def make_case_obj(cid, arrival_time, priority):
    return Case(
        case_id=cid, arrival_time=arrival_time, has_rep=True, is_detained=False,
        has_criminal=False, nbr_charges=0, complexity_score=0.0,
        base_priority=priority, service_time_s1=1.0, service_time_s2=1.0,
    )


class NaivePriorityQueue:
    """Reference implementation matching the ORIGINAL (pre-optimization)
    apply_aging(): rebuilds the whole heap every call, no due-heap, no
    versioning. Deliberately simple/obviously-correct so it can serve as a
    ground truth for the optimized version's regression tests."""

    def __init__(self, aging_threshold: float = 6.0, aging_boost: float = 2.0):
        self._heap = []  # (neg_priority, arrival_time, case_id, case)
        self._entry_times = {}
        self._aging_threshold = aging_threshold
        self._aging_boost = aging_boost

    def push(self, case, now):
        self._entry_times[case.case_id] = now
        heapq.heappush(self._heap, (-case.base_priority, case.arrival_time, case.case_id, case))

    def pop_next(self, now):
        if not self._heap:
            return None
        _, _, cid, case = heapq.heappop(self._heap)
        self._entry_times.pop(cid, None)
        return case

    def remove(self, case):
        for i, entry in enumerate(self._heap):
            if entry[2] == case.case_id:
                self._heap.pop(i)
                heapq.heapify(self._heap)
                self._entry_times.pop(case.case_id, None)
                return True
        return False

    def apply_aging(self, now):
        new_heap = []
        for neg_pri, arr_time, cid, case in self._heap:
            if now - self._entry_times.get(cid, now) >= self._aging_threshold:
                case.base_priority += self._aging_boost
                self._entry_times[cid] = now
            heapq.heappush(new_heap, (-case.base_priority, arr_time, cid, case))
        self._heap = new_heap

    def __len__(self):
        return len(self._heap)


def _pop_all_ids(queue, now):
    ids = []
    while len(queue) > 0:
        c = queue.pop_next(now)
        if c is None:
            break
        ids.append(c.case_id)
    return ids


@pytest.mark.parametrize("seed", range(10))
def test_priority_queue_matches_naive_reference_under_random_operations(seed):
    """Randomized equivalence test: push/apply_aging/pop sequences must
    produce IDENTICAL pop order under both implementations. This is the
    correctness contract the aging-loop optimization case study promised
    ('same results, cheaper') -- verified here structurally, on top of the
    single-scenario byte-for-byte check already done at real scale."""
    rng = random.Random(seed)
    fast = PriorityQueue(aging_threshold=3.0, aging_boost=2.0)
    naive = NaivePriorityQueue(aging_threshold=3.0, aging_boost=2.0)

    now = 0.0
    next_id = 0
    for _ in range(80):
        now += rng.uniform(0.1, 1.0)
        action = rng.choice(["push", "push", "push", "age", "pop"])
        if action == "push":
            priority = rng.uniform(0, 10)
            c1 = make_case_obj(next_id, now, priority)
            c2 = make_case_obj(next_id, now, priority)  # separate objects, same values
            fast.push(c1, now)
            naive.push(c2, now)
            next_id += 1
        elif action == "age":
            fast.apply_aging(now)
            naive.apply_aging(now)
        elif action == "pop" and len(fast) > 0:
            fast_case = fast.pop_next(now)
            naive_case = naive.pop_next(now)
            assert fast_case is not None and naive_case is not None
            assert fast_case.case_id == naive_case.case_id, (
                f"seed={seed}: fast queue popped case_id={fast_case.case_id} but naive "
                f"reference popped case_id={naive_case.case_id} -- the optimization "
                f"changed pop ORDER, which would silently change which cases get served "
                f"first at real scale."
            )

    # Drain both and compare the full remaining order too.
    assert _pop_all_ids(fast, now) == _pop_all_ids(naive, now)


def test_priority_queue_remove_prevents_stale_pop():
    """A removed case must never be returned by pop_next -- this is the
    lazy-deletion/versioning contract the aging-loop fix depends on."""
    q = PriorityQueue()
    c1 = make_case_obj(1, 0.0, 5.0)
    c2 = make_case_obj(2, 0.0, 3.0)
    q.push(c1, 0.0)
    q.push(c2, 0.0)
    assert q.remove(c1) is True
    popped = q.pop_next(1.0)
    assert popped.case_id == 2
    assert q.pop_next(1.0) is None
    assert len(q) == 0


def test_priority_queue_reboost_invalidates_stale_due_entry():
    """After a case is boosted once, its OLD due-heap entry (from the first
    push) must not cause a second, incorrect boost -- this is exactly what
    the version-tagging scheme in the aging-loop fix exists to prevent."""
    q = PriorityQueue(aging_threshold=1.0, aging_boost=2.0)
    c = make_case_obj(1, 0.0, 5.0)
    q.push(c, 0.0)
    q.apply_aging(1.0)   # first boost: 5.0 -> 7.0, reschedules next boost at t=2.0
    assert c.base_priority == pytest.approx(7.0)
    q.apply_aging(1.5)   # not due yet (next boost is t=2.0) -- must be a no-op
    assert c.base_priority == pytest.approx(7.0)
    q.apply_aging(2.0)   # now due: 7.0 -> 9.0
    assert c.base_priority == pytest.approx(9.0)


def test_fifo_queue_order_and_remove():
    q = FIFOQueue()
    cases = [make_case_obj(i, 0.0, 0.0) for i in range(3)]
    for c in cases:
        q.push(c, 0.0)
    assert q.remove(cases[1]) is True
    assert [c.case_id for c in [q.pop_next(0.0), q.pop_next(0.0)]] == [0, 2]
    assert q.pop_next(0.0) is None


def test_lifo_queue_starvation_cap():
    """A case waiting past max_wait must be forced to the front instead of
    strict LIFO order -- the documented 'escape valve' behavior."""
    q = LIFOQueue(max_wait=5.0)
    old_case = make_case_obj(1, 0.0, 0.0)
    q.push(old_case, now=0.0)
    for i in range(2, 5):
        q.push(make_case_obj(i, 0.0, 0.0), now=0.0)
    # At t=10, old_case has waited 10 > max_wait=5 -- should be forced out first.
    popped = q.pop_next(now=10.0)
    assert popped.case_id == 1