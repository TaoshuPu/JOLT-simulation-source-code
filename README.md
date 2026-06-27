# JOLT Deployment Experiments

This repository contains simulation code for studying joint deployment of LLM services and tool services on mixed G-Type/C-Type server environments.

The experiments compare direct monolithic optimization baselines with JOLT, a two-stage deployment strategy:

1. Assign LLM services to G-Type servers.
2. Assign tool services to G-Type/C-Type tool hosts.

The code is source-only by design. Generated CSV/JSON/log/figure outputs and local dependency folders should not be committed.

## Repository Layout

| File | Purpose |
| --- | --- |
| `jolt_cli.py` | Command-line entry point with `smoke`, `run`, `sweep`, and `list-methods` subcommands. |
| `jolt_single_run_checkpoint_sweep.py` | Checkpoint monitoring and per-method execution logic. |
| `jolt_small_scale_experiment.py` | Instance generation, objective evaluation, feasibility checks, and solver implementations. |
| `requirements.txt` | Python dependency list. |

Most users should start with `jolt_cli.py`.

## Requirements

Use Python 3.11+.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Solver notes:

- Gurobi-based methods require `gurobipy` and a valid Gurobi license.
- SCIP-based methods require `pyscipopt`.
- CP-SAT baselines require `ortools`.
- Plotting utilities require `Pillow` and `matplotlib`.

For reproducible GitHub usage, avoid committing local dependency folders such as `.gurobi_deps`, `.scip_deps`, `.ortools_deps`, `.venv`, and generated result directories.

## Quick Start

Run a fast smoke test:

```bash
python jolt_cli.py smoke --jolt-solver scip --restart
```

PowerShell equivalent:

```powershell
python .\jolt_cli.py smoke --jolt-solver scip --restart
```

Run the default sweep:

```bash
python jolt_cli.py sweep \
  --llm-list 20,40,60,80 \
  --tool-ratio 3 \
  --time-limit-s 600 \
  --checkpoints-s 60,180,300,420,600 \
  --capacity-mode fixed_per_server \
  --methods gurobi,scip,cpsat,jolt \
  --out-dir results/mixed_gc_20_80 \
  --restart
```

## Output Files

Each run writes results into `--out-dir`.

| File | Meaning |
| --- | --- |
| `summary_long.csv` | Long-form records for each LLM scale, method, and checkpoint. |
| `avg_call_distance_wide.csv` | Pivot table of average call distance by method. |
| `status_wide.csv` | Pivot table of solver status by method. |
| `trace_L*_*.json` | Per-method checkpoint trace and terminal metadata. |
| `instance_L*.json` | Generated instance metadata. |

The main quality metric is `avg_call_distance`; lower is better.

## Command-Line Interface

The recommended entry point is:

```bash
python jolt_cli.py <command> [options]
```

Available subcommands:

| Command | Purpose |
| --- | --- |
| `smoke` | Run a tiny correctness test. |
| `run` | Run one configurable experiment. |
| `sweep` | Run multiple LLM scales and checkpoints. |
| `list-methods` | Print available methods and required solvers. |

### Smoke Test

```bash
python jolt_cli.py smoke --jolt-solver scip --restart
```

Defaults:

- `--llms 5`
- `--tool-ratio 3`
- `--time-limit-s 4`
- `--checkpoints-s 2,4`
- `--methods jolt`
- `--out-dir results/smoke`

### Single Experiment

```bash
python jolt_cli.py run \
  --llms 20 \
  --tool-ratio 3 \
  --time-limit-s 180 \
  --checkpoints-s 180 \
  --methods gurobi,scip,cpsat,jolt \
  --jolt-solver scip \
  --out-dir results/l20_3min_scip \
  --restart
```

This should generate one instance with `20` LLM services and `60` tool services, then compare the selected methods.

### Scale Sweep

```bash
python jolt_cli.py sweep \
  --llm-list 20,40,60,80 \
  --tool-ratio 3 \
  --time-limit-s 600 \
  --checkpoints-s 60,180,300,420,600 \
  --methods gurobi,scip,cpsat,jolt \
  --jolt-solver gurobi \
  --out-dir results/mixed_gc_20_80 \
  --restart
```

### JOLT Solver

JOLT can use either Gurobi or SCIP.

Options:

| Option | Values | Meaning |
| --- | --- | --- |
| `--jolt-solver` | `gurobi`, `scip` | Solver used by JOLT. |

Behavior:

- If only `--time-limit-s` is provided, use it for each selected method.

Example commands:

Run JOLT with SCIP:

```bash
python jolt_cli.py run \
  --llms 20 \
  --time-limit-s 180 \
  --methods jolt \
  --jolt-solver scip \
  --out-dir results/l20_jolt_scip
```

Run JOLT with Gurobi:

```bash
python jolt_cli.py run \
  --llms 20 \
  --time-limit-s 180 \
  --methods jolt \
  --jolt-solver gurobi \
  --out-dir results/l20_jolt_gurobi
```

Compare direct baselines with SCIP-based two-stage solving:

```bash
python jolt_cli.py run \
  --llms 20 \
  --time-limit-s 180 \
  --methods gurobi,scip,cpsat,jolt \
  --jolt-solver scip \
  --out-dir results/l20_3min_compare
```
