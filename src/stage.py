"""
Stage: a worker-pool abstraction shared by both the Officer stage and the
Judge stage, and by every scheduling algorithm (FIFO/LIFO/Priority all plug
in the same way via a QueueStrategy -- see queues.py).

WHY THIS REPLACES THE ORIGINAL PATTERN:
The original code (simulation_runner.py, cel_algorithm.py,
fifo_lifo_experiment.py, scoring_experiment.py -- reimplemented four times)
had each arriving case push itself onto a custom queue object AND separately
request a plain simpy.Resource as a capacity counter, then pop from the
queue once granted. That mostly worked for cases that completed normally
(SimPy's own FCFS resource-grant order happened to track the custom queue's
push order closely enough), but it meant a case's own coroutine often ended
up servicing a *different* case's data, which made reneging impossible to
handle safely: a reneging case's SimPy resource request could be cleanly
released, but nothing ever removed it from the separate custom queue
object, leaving a permanent "ghost" entry that inflated backlog metrics
forever (see redesign-decisions.md).

Here, a fixed number of worker processes (matching that stage's server
capacity) pull directly from the shared queue themselves. Arriving cases
just get pushed and wait to be picked -- they never request anything. This
makes reneging trivial and safe: the same worker that popped a case decides
its fate (served vs. reneged) in one atomic step, so there's never a
window where a case is both "in the queue" and "already resolved."
"""

import simpy
from typing import Callable, Optional
from case import Case
from queues import QueueStrategy


class Stage:
    def __init__(self, env: simpy.Environment, capacity: int,
                 queue: QueueStrategy,
                 service_time_fn: Callable[[Case], float],
                 on_complete: Callable[[Case], None],
                 start_attr: str, end_attr: str, queue_enter_attr: str,
                 renege_prob: float = 0.0,
                 renege_exempt: Callable[[Case], bool] = lambda c: False,
                 on_renege: Optional[Callable[[Case], None]] = None,
                 aging_interval: Optional[float] = 1.0):
        """
        capacity          -- number of workers (officers or judges) for this stage
        queue             -- a QueueStrategy instance (FIFO/LIFO/Priority/etc.)
        service_time_fn   -- given a Case, returns how long its service takes
        on_complete       -- called with the Case once service finishes
        start_attr/end_attr/queue_enter_attr
                          -- Case attribute names this stage should stamp with
                             simulation time (e.g. "s1_start", "s1_end",
                             "s1_queue_enter" for the officer stage)
        renege_prob       -- flat probability, checked once per case the moment
                             a worker pulls it from the queue (not a function of
                             how long it waited -- see redesign-decisions.md,
                             reneging section, for why this replaced the
                             original patience-clock model)
        renege_exempt     -- predicate; cases for which this returns True never
                             renege (e.g. detained cases)
        on_renege         -- called with the Case if it reneges instead of on_complete
        aging_interval    -- how often (sim months) to call queue.apply_aging();
                             pass None to disable (irrelevant for FIFO/LIFO)
        """
        self.env = env
        self.queue = queue
        self.service_time_fn = service_time_fn
        self.on_complete = on_complete
        self.start_attr = start_attr
        self.end_attr = end_attr
        self.queue_enter_attr = queue_enter_attr
        self.renege_prob = renege_prob
        self.renege_exempt = renege_exempt
        self.on_renege = on_renege
        self.rng = None  # set via set_rng() before run() -- shared with the simulator

        # Used purely as a wake-up signal: one put() per push(), so exactly
        # one idle worker wakes per case submitted. Capacity is unbounded --
        # this is never used to hold case data, only to synchronize workers
        # with the queue's actual contents.
        self._wake = simpy.Store(env)

        # Cumulative worker-idle time (summed across all `capacity` workers):
        # how much total simulated time this stage's workers spent blocked
        # on _wake.get() waiting for a case, rather than serving one. Added
        # to investigate the CEL Stage-2-throughput-suppression hypothesis
        # in redesign-decisions.md ("A real, unresolved puzzle") -- if the
        # monthly lottery batch withholds qualifying cases from Stage 2 for
        # up to a month at a time, Stage 2 should show measurably more idle
        # time under a CEL run than under FIFO at the same scale, even
        # though both are nominally overloaded (rho >> 1). Zero cost when
        # nobody reads it, so this is safe to leave enabled unconditionally.
        self.idle_time_total = 0.0

        for _ in range(capacity):
            env.process(self._worker())
        if aging_interval is not None:
            env.process(self._aging_loop(aging_interval))

    def set_rng(self, rng):
        self.rng = rng

    def submit(self, case: Case):
        """Add a case to this stage. Called by an arrival process or by the
        upstream stage handing a case off (e.g. Stage 1 -> Stage 2)."""
        setattr(case, self.queue_enter_attr, self.env.now)
        self.queue.push(case, self.env.now)
        self._wake.put(None)

    def __len__(self):
        return len(self.queue)

    def _worker(self):
        while True:
            idle_start = self.env.now
            yield self._wake.get()
            self.idle_time_total += self.env.now - idle_start
            case = self.queue.pop_next(self.env.now)
            if case is None:
                continue  # defensive; should not happen given push/wake are 1:1

            if self.renege_prob > 0 and not self.renege_exempt(case):
                if self.rng.random() < self.renege_prob:
                    case.outcome = "reneged"
                    case.total_time = self.env.now - case.arrival_time
                    if self.on_renege:
                        self.on_renege(case)
                    continue  # no service time consumed; worker immediately free

            setattr(case, self.start_attr, self.env.now)
            yield self.env.timeout(self.service_time_fn(case))
            setattr(case, self.end_attr, self.env.now)

            self.on_complete(case)

    def _aging_loop(self, interval: float):
        while True:
            yield self.env.timeout(interval)
            self.queue.apply_aging(self.env.now)