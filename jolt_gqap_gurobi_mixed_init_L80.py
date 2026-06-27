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
    random_feasible_assignment,
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


def generate_mixed_starts(inst, sim, seed: int, knn_count: int, random_count: int) -> tuple[list[list[int]], list[dict]]:
    total_starts = knn_count + random_count
    starts: list[list[int]] = []
    rows: list[dict] = []

    for idx in range(total_starts):
        kind = "KNN-diverse" if idx < knn_count else "Random feasible"
        rng = random.Random(seed + idx)
        place = knn_diverse_initialization(inst, sim, rng) if idx < knn_count else random_feasible_assignment(inst, rng)
        cost = normalized_cost(inst, place, sim)
        starts.append(place)
        rows.append(
            {
                "start_idx": idx,
                "kind": kind,
                "cost": cost,
                "feasible": is_feasible(inst, place),
                "assignment": json.dumps(place),
            }
        )

    ranked = sorted(zip(starts, rows), key=lambda item: float(item[1]["cost"]))
    starts = [item[0] for item in ranked]
    rows = []
    for rank, (_, row) in enumerate(ranked):
        row = dict(row)
        row["rank"] = rank
        rows.append(row)
    return starts, rows


def set_multiple_mip_starts(model: gp.Model, x, starts: list[list[int]], g_count: int) -> None:
    model.NumStart = len(starts)
    model.update()
    for start_idx, place in enumerate(starts):
        model.Params.StartNumber = start_idx
        for i, server in enumerate(place):
            for n in range(g_count):
                x[i, n].Start = 1.0 if n == server else 0.0


def main() -> None:
    out_dir = ROOT / "jolt_gqap_gurobi_mixed_init_L80"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = 20260529
    time_limit_s = 600
    checkpoints = [120, 240, 360, 480, 600]
    knn_count = 20
    random_count = 20
    total_starts = knn_count + random_count
    method = "Gurobi + mixed init"

    print(
        "Mixed-init rerun: LLM=80, tools=240, G=20, "
        f"starts={total_starts}, KNN={knn_count}, random={random_count}, 10 min",
        flush=True,
    )
    inst = make_llm_instance(seed=seed, llm_count=80, tool_count=240, g_count=20, gpu_cap=4, mem_cap=48)
    sim = preference_similarity(inst)
    (out_dir / "instance_meta.json").write_text(json.dumps(instance_to_jsonable(inst), indent=2), encoding="utf-8")

    starts, start_rows = generate_mixed_starts(inst, sim, seed + 50_000, knn_count, random_count)
    write_csv(out_dir / "mixed_start_candidates.csv", start_rows)
    best_start_cost = float(start_rows[0]["cost"])
    print(
        f"Generated {len(starts)} starts: "
        f"{sum(1 for r in start_rows if r['kind'] == 'KNN-diverse')} KNN-diverse, "
        f"{sum(1 for r in start_rows if r['kind'] == 'Random feasible')} random; "
        f"best start={best_start_cost:.6f}",
        flush=True,
    )

    denom = float(sim.sum())
    print(f"[{method}] building GQAP model...", flush=True)
    build_start = time.perf_counter()
    model, aux = build_gqap_model(inst, sim, mip_start=None, seed=seed + 100, output_flag=0)
    set_multiple_mip_starts(model, aux["x"], starts, inst.g_count)
    build_time = time.perf_counter() - build_start
    model.Params.TimeLimit = float(time_limit_s)
    print(
        f"[{method}] model built in {build_time:.2f}s, x={inst.llm_count * inst.g_count}, "
        f"quadratic_terms={aux['quadratic_terms']}, optimizing {time_limit_s}s...",
        flush=True,
    )

    records: dict[int, dict] = {}

    def persist_checkpoints() -> None:
        rows = [records[c] for c in checkpoints if c in records]
        write_csv(out_dir / "checkpoint_results.csv", rows)

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
                persist_checkpoints()
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
    persist_checkpoints()

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
        "best_mixed_start_cost": best_start_cost,
        "mixed_start_count": total_starts,
        "knn_start_count": sum(1 for row in start_rows if row["kind"] == "KNN-diverse"),
        "random_start_count": sum(1 for row in start_rows if row["kind"] == "Random feasible"),
        "final_feasible": feasible,
        "final_assignment": json.dumps(assignment),
        "quadratic_terms": aux["quadratic_terms"],
    }
    write_csv(out_dir / "final_results.csv", [final])
    print(
        f"Done: status={final['status']}, incumbent={final_best:.6f}, "
        f"bound={final_bound:.6f}, gap={final_gap:.2f}%, runtime={model.Runtime:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
