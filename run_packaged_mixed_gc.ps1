$ErrorActionPreference = "Stop"

python .\jolt_packaged_mixed_gc_experiment.py `
  --llm-list 20,40,60,80 `
  --tool-ratio 3 `
  --checkpoints-s 60,180,300,420,600 `
  --max-time-s 600 `
  --capacity-mode fixed_per_server `
  --methods gurobi,scip,cpsat,jolt `
  --out-dir jolt_packaged_mixed_gc_20_80 `
  --restart
