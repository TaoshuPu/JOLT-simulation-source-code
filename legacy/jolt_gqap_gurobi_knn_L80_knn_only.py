from __future__ import annotations

import csv
import json
import math
import random
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

from jolt_algorithm1_convergence_L20 import (
    instance_to_jsonable,
    is_feasible,
    knn_diverse_initialization,
    make_llm_instance,
    normalized_cost,
    preference_similarity,
)
from jolt_gqap_gurobi_knn_init_experiment import (
    build_gqap_model,
    gap_percent,
    norm_value,
    status_name,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_knn_start(inst, sim, seed: int, samples: int) -> tuple[list[int], list[dict]]:
    rows = []
    best_place = None
    best_cost = math.inf
    for sample_idx in range(samples):
        rng = random.Random(seed + sample_idx)
        place = knn_diverse_initialization(inst, sim, rng)
        cost = normalized_cost(inst, place, sim)
        row = {
            "sample": sample_idx,
            "seed": seed + sample_idx,
            "cost": cost,
            "feasible": is_feasible(inst, place),
            "assignment": json.dumps(place),
        }
        rows.append(row)
        if cost < best_cost:
            best_cost = cost
            best_place = place
    if best_place is None:
        raise RuntimeError("No KNN start was generated.")
    return best_place, rows


def main() -> None:
    out_dir = ROOT / "jolt_gqap_gurobi_knn_L80"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = 20260529
    checkpoints = [120, 240, 360, 480, 600]
    time_limit_s = 600
    method = "Gurobi + KNN init"

    print("KNN-only rerun: LLM=80, tools=240, G=20, 10 min", flush=True)
    inst = make_llm_instance(seed=seed, llm_count=80, tool_count=240, g_count=20, gpu_cap=4, mem_cap=48)
    sim = preference_similarity(inst)
    (out_dir / "instance_meta.json").write_text(json.dumps(instance_to_jsonable(inst), indent=2), encoding="utf-8")

    knn_start, knn_rows = best_knn_start(inst, sim, seed + 50_000, 20)
    knn_cost = normalized_cost(inst, knn_start, sim)
    write_csv(out_dir / "knn_start_candidates.csv", knn_rows)
    (out_dir / "best_knn_start.json").write_text(
        json.dumps({"cost": knn_cost, "assignment": knn_start, "feasible": is_feasible(inst, knn_start)}, indent=2),
        encoding="utf-8",
    )
    print(f"Best KNN start cost={knn_cost:.6f}", flush=True)

    denom = float(sim.sum())
    build_start = time.perf_counter()
    model, aux = build_gqap_model(inst, sim, knn_start, seed=seed + 100, output_flag=0)
    build_time = time.perf_counter() - build_start
    model.Params.TimeLimit = float(time_limit_s)
    print(f"Model built in {build_time:.2f}s, optimizing {time_limit_s}s...", flush=True)

    records: dict[int, dict] = {}

    def persist() -> None:
        rows = [records[c] for c in checkpoints if c in records]
        write_csv(out_dir / "checkpoint_results_knn_only.csv", rows)

    def remember(runtime: float, best_raw: float, bound_raw: float) -> None:
        for checkpoint in checkpoints:
            if checkpoint not in records and runtime >= checkpoint:
                best_norm = norm_value(best_raw, denom)
                bound_norm = norm_value(bound_raw, denom)
                records[checkpoint] = {
                    "method": method,
                    "checkpoint_s": checkpoint,
                    "recorded_runtime_s": runtime,
                    "incumbent_cost": best_norm,
                    "best_bound_cost": bound_norm,
                    "gap_percent": gap_percent(best_norm, bound_norm),
                }
                persist()
                print(
                    f"[{method}] checkpoint {checkpoint}s: incumbent={best_norm:.6f}, "
                    f"bound={bound_norm:.6f}, gap={records[checkpoint]['gap_percent']:.2f}%",
                    flush=True,
                )

    def callback(model_cb: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIP:
            runtime = float(model_cb.cbGet(GRB.Callback.RUNTIME))
            best_raw = float(model_cb.cbGet(GRB.Callback.MIP_OBJBST))
            bound_raw = float(model_cb.cbGet(GRB.Callback.MIP_OBJBND))
            remember(runtime, best_raw, bound_raw)
        elif where == GRB.Callback.MIPSOL:
            runtime = float(model_cb.cbGet(GRB.Callback.RUNTIME))
            best_raw = float(model_cb.cbGet(GRB.Callback.MIPSOL_OBJBST))
            try:
                bound_raw = float(model_cb.cbGet(GRB.Callback.MIPSOL_OBJBND))
            except gp.GurobiError:
                bound_raw = math.nan
            remember(runtime, best_raw, bound_raw)

    optimize_start = time.perf_counter()
    model.optimize(callback)
    optimize_wall = time.perf_counter() - optimize_start

    final_best_raw = float(model.ObjVal) if model.SolCount > 0 else math.nan
    final_bound_raw = float(model.ObjBound) if model.SolCount > 0 or math.isfinite(model.ObjBound) else math.nan
    final_best = norm_value(final_best_raw, denom)
    final_bound = norm_value(final_bound_raw, denom)
    final_gap = gap_percent(final_best, final_bound)

    for checkpoint in checkpoints:
        if checkpoint not in records:
            records[checkpoint] = {
                "method": method,
                "checkpoint_s": checkpoint,
                "recorded_runtime_s": float(model.Runtime),
                "incumbent_cost": final_best,
                "best_bound_cost": final_bound,
                "gap_percent": final_gap,
            }
    persist()

    assignment = []
    feasible = False
    if model.SolCount > 0:
        x = aux["x"]
        assignment = [max(range(inst.g_count), key=lambda n: x[i, n].X) for i in range(inst.llm_count)]
        feasible = is_feasible(inst, assignment)

    final = {
        "method": method,
        "status": status_name(model.Status),
        "solver_runtime_s": float(model.Runtime),
        "optimize_wall_s": optimize_wall,
        "build_time_s": build_time,
        "sol_count": int(model.SolCount),
        "node_count": float(model.NodeCount),
        "final_incumbent_cost": final_best,
        "final_bound_cost": final_bound,
        "final_gap_percent": final_gap,
        "mip_start_cost": knn_cost,
        "final_feasible": feasible,
        "final_assignment": json.dumps(assignment),
        "quadratic_terms": aux["quadratic_terms"],
    }
    write_csv(out_dir / "final_results_knn_only.csv", [final])
    print(
        f"Done: status={final['status']}, incumbent={final_best:.6f}, "
        f"bound={final_bound:.6f}, gap={final_gap:.2f}%, runtime={model.Runtime:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
