# Legacy Experiment Scripts

This directory contains historical and extended JOLT experiment scripts kept for reproducibility.

The main supported entry points live in the repository root:

- `jolt_cli.py`
- `jolt_packaged_mixed_gc_experiment.py`
- `jolt_single_run_checkpoint_sweep.py`
- `jolt_small_scale_experiment.py`

Scripts in this directory may depend on specific traces, output folders, or historical experiment settings. They are useful for auditing old runs, but new experiments should normally use `jolt_cli.py`.
