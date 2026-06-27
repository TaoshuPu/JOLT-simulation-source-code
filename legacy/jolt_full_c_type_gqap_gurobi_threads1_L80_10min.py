from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for deps_name in (".gurobi_deps", ".ortools_deps"):
    deps = ROOT / deps_name
    if deps.exists():
        sys.path.insert(0, str(deps))

import gurobipy as gp
from gurobipy import GRB

from jolt_full_c_type_gqap_hisc_vs_gurobi_L80_10min import (
    CHECKPOINTS,
    LLMS,
    OUT_DIR,
    SEED,
    TOOLS,
    build_gurobi_gqap_model,
    write_csv,
)
from jolt_small_scale_experiment import llm_similarity, llm_surrogate_cost, make_instance


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable fixed-thread Gurobi GQAP run for the full C-Type environment.")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    threads = max(1, int(args.threads))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inst = make_instance(SEED, llm_count=LLMS, tool_count=TOOLS, g_count=None, c_count=None, capacity_mode="fixed_per_server")
    sim = llm_similarity(inst)
    print(
        f"Gurobi GQAP Threads={threads}: LLM={inst.llm_count}, Tools={inst.tool_count}, "
        f"G={inst.g_count}, C={inst.c_count}",
        flush=True,
    )
    build_start = time.perf_counter()
    model, aux = build_gurobi_gqap_model(inst, sim, seed=SEED + 100)
    model.Params.Threads = threads
    model.Params.OutputFlag = 0
    build_s = time.perf_counter() - build_start
    print(f"model built in {build_s:.2f}s, q_terms={aux['quadratic_terms']}", flush=True)

    records: dict[float, dict] = {}
    last_place: list[int] = []
    last_cost = math.inf
    last_bound = math.nan

    def record(runtime: float, status: str, source: str) -> None:
        gap = math.nan
        if math.isfinite(last_cost) and math.isfinite(last_bound) and abs(last_cost) > 1e-12:
            gap = max(0.0, 100.0 * (last_cost - last_bound) / abs(last_cost))
        for checkpoint_s in CHECKPOINTS:
            if checkpoint_s not in records and runtime >= checkpoint_s:
                records[checkpoint_s] = {
                    "method": "Gurobi GQAP",
                    "checkpoint_s": checkpoint_s,
                    "minute": int(checkpoint_s // 60),
                    "recorded_elapsed_s": runtime,
                    "gqap_cost": last_cost,
                    "best_bound": last_bound,
                    "gap_percent": gap,
                    "status": status,
                    "source": source,
                    "assignment": json.dumps(last_place),
                }
                print(f"{int(checkpoint_s // 60)}min cost={last_cost:.6f} bound={last_bound:.6f}", flush=True)

    def callback(model_cb: gp.Model, where: int) -> None:
        nonlocal last_place, last_cost, last_bound
        if where == GRB.Callback.MIPSOL:
            x = aux["x"]
            place = [
                max(range(inst.g_count), key=lambda n: model_cb.cbGetSolution(x[i, n]))
                for i in range(inst.llm_count)
            ]
            last_place = place
            last_cost = llm_surrogate_cost(inst, place, sim)
            try:
                last_bound = float(model_cb.cbGet(GRB.Callback.MIPSOL_OBJBND))
            except gp.GurobiError:
                last_bound = math.nan
            runtime = float(model_cb.cbGet(GRB.Callback.RUNTIME))
            record(runtime, "RUNNING", "mipsol")
        elif where == GRB.Callback.MIP:
            runtime = float(model_cb.cbGet(GRB.Callback.RUNTIME))
            try:
                last_bound = float(model_cb.cbGet(GRB.Callback.MIP_OBJBND))
            except gp.GurobiError:
                pass
            record(runtime, "RUNNING", "mip")

    model.optimize(callback)

    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }.get(model.Status, str(model.Status))
    if model.SolCount > 0:
        x = aux["x"]
        last_place = [max(range(inst.g_count), key=lambda n: x[i, n].X) for i in range(inst.llm_count)]
        last_cost = llm_surrogate_cost(inst, last_place, sim)
        last_bound = float(model.ObjBound)
    runtime = float(model.Runtime)
    for checkpoint_s in CHECKPOINTS:
        if checkpoint_s not in records:
            records[checkpoint_s] = {
                "method": "Gurobi GQAP",
                "checkpoint_s": checkpoint_s,
                "minute": int(checkpoint_s // 60),
                "recorded_elapsed_s": runtime,
                "gqap_cost": last_cost,
                "best_bound": last_bound,
                "gap_percent": math.nan,
                "status": status_name,
                "source": "final_fill",
                "assignment": json.dumps(last_place),
            }
    rows = [records[c] for c in CHECKPOINTS]
    write_csv(OUT_DIR / f"gurobi_gqap_threads{threads}_checkpoints.csv", rows)
    write_csv(
        OUT_DIR / f"gurobi_gqap_threads{threads}_final.csv",
        [
            {
                "method": "Gurobi GQAP",
                "final_gqap_cost": last_cost,
                "best_bound": last_bound,
                "solver_runtime_s": runtime,
                "build_time_s": build_s,
                "status": status_name,
                "sol_count": int(model.SolCount),
                "node_count": float(model.NodeCount),
                "threads": threads,
            }
        ],
    )
    print(f"done status={status_name} cost={last_cost:.6f} runtime={runtime:.2f}s", flush=True)


if __name__ == "__main__":
    main()
