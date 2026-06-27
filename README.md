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

## Examples

JOLT can be solved with either Gurobi or SCIP via `--jolt-solver`.

### Smoke Example

This smoke example uses `LLM=20`, `Tools=60`, and a 3-minute limit per selected method. It compares Gurobi, SCIP, CP-SAT, and JOLT, so the expected wall-clock time is about 12-15 minutes on a typical workstation.

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

### Full Example

This full example runs the packaged scale sweep for `LLM=20,40,60,80` with 10-minute limits and five checkpoints.

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
