# JOLT Deployment Experiments

This repository contains simulation code for studying joint deployment of LLM services and tool services on mixed G-Type/C-Type server environments.

The experiments compare direct monolithic optimization baselines with JOLT, a two-stage deployment strategy:

1. Assign LLM services to G-Type servers.
2. Assign tool services to G-Type/C-Type tool hosts.

The code is source-only by design. Generated CSV/JSON/log/figure outputs and local dependency folders should not be committed.

## Repository Layout

| File | Purpose |
| --- | --- |
| `jolt_packaged_mixed_gc_experiment.py` | Main packaged experiment runner for mixed G/C environments. |
| `jolt_cli.py` | GitHub-facing command-line entry point with `smoke`, `run`, `sweep`, and `list-methods` subcommands. |
| `jolt_single_run_checkpoint_sweep.py` | Checkpoint monitoring and per-method execution logic. |
| `jolt_small_scale_experiment.py` | Instance generation, objective evaluation, feasibility checks, and solver implementations. |
| `jolt_hisc_ma_tool_mip_checkpoints.py` | HISC-MA + Tool-MIP checkpoint experiment. |
| `jolt_algorithm1_convergence_L20.py` | HISC-MA / Algorithm 1 convergence experiment. |
| `requirements.txt` | Python dependency list. |

Other `jolt_*.py` files are historical or extended experiments kept for reproducibility.

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
python jolt_cli.py smoke --two-stage-solver scip --restart
```

PowerShell equivalent:

```powershell
python .\jolt_cli.py smoke --two-stage-solver scip --restart
```

Run the default packaged experiment:

```bash
python jolt_packaged_mixed_gc_experiment.py --restart
```

The default setting is:

- LLM scales: `20,40,60,80`
- Tool count: `3 x LLM count`
- Checkpoints: `60,180,300,420,600` seconds
- Maximum runtime per algorithm: `600` seconds
- Capacity mode: `fixed_per_server`
- Methods: `gurobi,scip,cpsat,jolt`

Equivalent explicit command:

```bash
python jolt_packaged_mixed_gc_experiment.py \
  --llm-list 20,40,60,80 \
  --tool-ratio 3 \
  --checkpoints-s 60,180,300,420,600 \
  --max-time-s 600 \
  --capacity-mode fixed_per_server \
  --methods gurobi,scip,cpsat,jolt \
  --out-dir jolt_packaged_mixed_gc_20_80 \
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
| --- | --- | --- |
| `smoke` | Run a tiny correctness test. |
| `run` | Run one configurable experiment. |
| `sweep` | Run multiple LLM scales and checkpoints. |
| `list-methods` | Print available methods and required solvers. |

### Smoke Test

```bash
python jolt_cli.py smoke --two-stage-solver scip --restart
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
  --gqap-solver scip \
  --tool-mip-solver scip \
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
  --gqap-solver gurobi \
  --tool-mip-solver gurobi \
  --out-dir results/mixed_gc_20_80 \
  --restart
```

### Solver Selection

JOLT can use Gurobi or SCIP independently in each phase.

Options:

| Option | Values | Meaning |
| --- | --- | --- |
| `--gqap-solver` | `gurobi`, `scip` | Solver for Phase 1 LLM GQAP. |
| `--tool-mip-solver` | `gurobi`, `scip` | Solver for Phase 2. |
| `--two-stage-solver` | `gurobi`, `scip` | Shortcut for setting both phase solvers. |

Behavior:

- If only `--time-limit-s` is provided, use it for each selected method.
- If `--two-stage-solver scip` is provided, set both `--gqap-solver scip` and `--tool-mip-solver scip`.

Example commands:

Run both phases with SCIP:

```bash
python jolt_cli.py run \
  --llms 20 \
  --time-limit-s 180 \
  --methods jolt \
  --gqap-solver scip \
  --tool-mip-solver scip \
  --out-dir results/l20_jolt_scip
```

Run Phase 1 with Gurobi and Phase 2 with SCIP:

```bash
python jolt_cli.py run \
  --llms 20 \
  --time-limit-s 180 \
  --methods jolt \
  --gqap-solver gurobi \
  --tool-mip-solver scip \
  --out-dir results/l20_jolt_gurobi_scip
```

Compare direct baselines with SCIP-based two-stage solving:

```bash
python jolt_cli.py run \
  --llms 20 \
  --time-limit-s 180 \
  --methods gurobi,scip,cpsat,jolt \
  --two-stage-solver scip \
  --out-dir results/l20_3min_compare
```

## Python API

```python
run_gqap_tool_mip(
    inst,
    gqap_solver="gurobi",
    tool_mip_solver="gurobi",
    phase1_timeout_s=180.0,
    phase2_timeout_s=180.0,
)
```

The legacy packaged runner remains available:

```bash
python jolt_packaged_mixed_gc_experiment.py \
  --llm-list 5 \
  --checkpoints-s 2,4 \
  --max-time-s 4 \
  --methods jolt \
  --two-stage-solver scip \
  --out-dir results/packaged_smoke_scip \
  --restart
```

## Example Result

On a small LLM=20, Tools=60, 3-minute comparison, SCIP-based JOLT completed with:

```text
PHASE1_TIMELIMIT;PHASE2_OPTIMAL
avg_call_distance = 0.204807555841
deployment_feasible = True
```

This confirms that SCIP can solve the JOLT workflow on the small comparison instance.

## Git Hygiene

Before pushing to GitHub:

1. Commit source files, README, and requirements.
2. Do not commit generated output directories unless a specific result set is needed.
3. Do not commit local solver dependency folders or virtual environments.
4. Keep large logs, figures, and temporary experiment outputs out of the repository.

Recommended `.gitignore` patterns:

```gitignore
__pycache__/
*.pyc
.venv/
.gurobi_deps/
.scip_deps/
.ortools_deps/
.scipy_deps/
jolt_*_outputs/
jolt_*_smoke/
jolt_*_compare/
results/
*.log
*.err
```
