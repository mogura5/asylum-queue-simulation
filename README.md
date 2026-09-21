# Asylum Queue Simulator

Two-stage tandem-queue simulation of the U.S. asylum adjudication process (Officer screening -> Judge hearing), comparing FIFO, LIFO, Priority, and two Conditional Economic Lottery (CEL) scheduling variants.

## File structure

```
.
├── README.md
├── calibration/
│   ├── calibration_diagnostic.py
│   ├── compute_parameter.py
│   └── data_processor
├── data/
│   ├── sim_parameters_FINAL.json
│   └── sim_results/
├── figures/
│   └── generate_figures.py
├── src/
│   ├── case.py
│   ├── cel.py
│   ├── orchestrator.py
│   ├── queues.py
│   ├── simulator.py
│   ├── smoke_test.py
│   └── stage.py
└── tests/
    ├── conftest.py
    ├── test_case.py
    ├── test_cel.py
    ├── test_integration.py
    ├── test_orchestrator.py
    ├── test_queues.py
    └── test_stage.py
```

### 1. Calibration pipeline (run in order, once, to produce `data/sim_parameters_FINAL.json`)

```bash
cd calibration
python3 data_processor.py --raw-dir <path to master.csv/reliefApplications.csv> --out ../data/mock_data_FINAL.csv
python3 compute_parameter.py --data ../data/mock_data_FINAL.csv --out ../data/sim_parameters_FINAL.json
python3 calibration_diagnostic.py --data-dir ../data   # optional, sanity-checks the above
```

### 2. Simulator / orchestrator

```bash
cd src
python3 orchestrator.py \
    --params ../data/sim_parameters_FINAL.json \
    --output ../data/sim_results/ \
    --scale 0.30 --months 100 --warmup 20 --replications 15 \
    --algorithms FIFO LIFO PRIORITY --workers <cpu count>

# CEL variants (separate call; merges into the same output dir, doesn't redo the above):
python3 orchestrator.py \
    --params ../data/sim_parameters_FINAL.json \
    --output ../data/sim_results/ \
    --scale 0.30 --months 100 --warmup 20 --replications 15 \
    --algorithms CEL_balanced CEL_broad --workers <cpu count>
```

Useful flags: `--fresh` (overwrite instead of merge), `--log-level DEBUG|INFO|WARNING|ERROR` (default INFO), `--no-ttest`.

Quick sanity run before committing to a long grid:

```bash
python3 orchestrator.py --params ../data/sim_parameters_FINAL.json --output /tmp/smoke_out \
    --scale 0.02 --months 20 --warmup 5 --replications 1 --algorithms FIFO CEL_balanced
```

Or the older manual script:

```bash
python3 smoke_test.py
```

### 3. Tests

```bash
cd src   # tests/conftest.py adds src/ to sys.path itself, so this works from repo root too
python3 -m pytest ../tests/ -v
```

Runs in ~1.5s (42 tests, synthetic smoke-scale params — not the real calibrated file).

### 4. Diagnostics (CEL Stage-2 throughput question)

```bash
cd src
python3 ../diagnostics/verify_cel_stage2_idle.py \
    --params ../data/sim_parameters_FINAL.json \
    --era post_covid_2024 --archetype high_volume \
    --scale 0.10 --months 100 --seed 42 --cel-config balanced
```

Omit `--params`/`--era`/`--archetype` to run it standalone against a small built-in synthetic overloaded scenario. Add `--skip-profile` to run only the idle-time comparison (faster).

### 5. Figures

```bash
cd figures
python3 generate_figures.py --input ../data/sim_results --output .
```

Produces `fig_headline_comparison.png`, `fig_lifo_paradox.png`, `fig_cel_welfare_targeting.png`.

### Lint / static analysis (not currently wired up -- candidates)

```bash
pip install ruff mypy
ruff check src/ calibration/ figures/ diagnostics/ tests/
mypy src/ figures/ diagnostics/
```

## Class/module structure reference

**`src/case.py`**
- `Case` (dataclass): `case_id`, `arrival_time`, `has_rep`, `is_detained`, `has_criminal`, `nbr_charges`, `complexity_score`, `base_priority`, `service_time_s1`, `service_time_s2`, `is_appeal`, `econ_need_score`, plus timestamps (`s1_queue_enter`/`s1_start`/`s1_end`/`s2_queue_enter`/`s2_start`/`s2_end`), `outcome`, `total_time`. Method: `effective_s2_time()`.
- `compute_complexity()`, `draw_service_time()`, `make_case()` -- module-level functions, no class.

**`src/queues.py`**
- `QueueStrategy` (base): `push`, `pop_next`, `remove`, `apply_aging`, `__len__`.
- `FIFOQueue(QueueStrategy)`: `deque`-backed.
- `LIFOQueue(QueueStrategy)`: list-backed stack with a `max_wait` starvation cap.
- `PriorityQueue(QueueStrategy)`: min-heap (negated priority) + a second `_due` min-heap for aging; version-tagged lazy deletion.
- `make_queue(algorithm, **kwargs)` -- factory function.

**`src/stage.py`**
- `Stage`: worker-pool abstraction (`capacity` worker processes pulling from one shared `QueueStrategy`). Public: `submit()`, `__len__`, `set_rng()`, `idle_time_total` (cumulative worker idle time). Private: `_worker()`, `_aging_loop()`.

**`src/simulator.py`**
- `AsylumSimulator`: owns `stage1`/`stage2` (`Stage` instances), the SimPy `env`, and the CEL `lottery_pool`. Methods: `run()`, `_arrival_process()`, `_advance_to_stage2()`, `_lottery_draw_process()`, `_resolve_case()`, `_collect_results()`.

**`src/cel.py`**
- `LOTTERY_CONFIGS` (dict: `"balanced"`, `"broad"`).
- `cel_config_for_algorithm(algorithm)`, `draw_econ_need_score(rng, params)` -- module-level functions, no class.

**`src/orchestrator.py`**
- `OrchestratorError(Exception)`.
- Module-level functions: `_scaled_params()`, `_summarize()`, `run_one_replication()`, `_run_task()`, `run_all_experiments()`, `run_pairwise_ttests()`.

**`figures/generate_figures.py`**
- Module-level functions only: `load_results()`, `plot_headline_comparison()`, `plot_lifo_wait_paradox()`, `plot_cel_welfare_targeting()`, `main()`.

**`diagnostics/verify_cel_stage2_idle.py`**
- Module-level functions only: `_load_scaled_params()`, `_run_and_report()`, `compare_idle_time()`, `profile_cel_replication()`, `main()`.

**`tests/`**
- `conftest.py`: `TEST_PARAMS` fixture.
- One file per `src/` module under test (`test_case.py`, `test_queues.py`, `test_stage.py`, `test_cel.py`, `test_orchestrator.py`), plus `test_integration.py` for end-to-end runs.
