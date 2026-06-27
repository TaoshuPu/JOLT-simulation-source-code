from __future__ import annotations

import argparse
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


DEFAULT_CHECKPOINTS = [60, 180, 300, 600]


def status_name(status: int) -> str:
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }.get(status, str(status))


def norm_value(raw: float, denom: float) -> float:
    if raw is None or not math.isfinite(raw) or abs(raw) >= GRB.INFINITY * 0.5:
        return math.nan
    return raw / denom


def gap_percent(best: float, bound: float) -> float:
    if not math.isfinite(best) or not math.isfinite(bound) or abs(best) < 1e-12:
        return math.nan
    return max(0.0, 100.0 * (best - bound) / abs(best))


def generate_best_knn_start(inst, sim, seed: int, samples: int) -> tuple[list[int], list[dict]]:
    rows = []
    best_place = None
    best_cost = math.inf
    for sample_idx in range(samples):
        rng = random.Random(seed + sample_idx)
        place = knn_diverse_initialization(inst, sim, rng)
        cost = normalized_cost(inst, place, sim)
        rows.append(
            {
                "sample": sample_idx,
                "seed": seed + sample_idx,
                "cost": cost,
                "feasible": is_feasible(inst, place),
                "assignment": json.dumps(place),
            }
        )
        if cost < best_cost:
            best_cost = cost
            best_place = place
    if best_place is None:
        raise RuntimeError("No KNN start was generated.")
    return best_place, rows


def build_gqap_model(inst, sim, mip_start: list[int] | None, seed: int, output_flag: int) -> tuple[gp.Model, dict]:
    model = gp.Model("algorithm1_llm_gqap")
    model.Params.OutputFlag = output_flag
    model.Params.NonConvex = 2
    model.Params.Seed = seed
    model.Params.Threads = 0

    x = model.addVars(inst.llm_count, inst.g_count, vtype=GRB.BINARY, name="x")
    model.addConstrs((gp.quicksum(x[i, n] for n in range(inst.g_count)) == 1 for i in range(inst.llm_count)), name="assign")
    model.addConstrs(
        (
            gp.quicksum(int(inst.llm_gpu[i]) * x[i, n] for i in range(inst.llm_count))
            <= int(inst.g_gpu_cap[n])
            for n in range(inst.g_count)
        ),
        name="gpu_cap",
    )
    model.addConstrs(
        (
            gp.quicksum(int(inst.llm_mem[i]) * x[i, n] for i in range(inst.llm_count))
            <= int(inst.g_mem_cap[n])
            for n in range(inst.g_count)
        ),
        name="mem_cap",
    )

    objective = gp.QuadExpr()
    coeff_batch = []
    left_batch = []
    right_batch = []
    q_terms = 0

    def flush() -> None:
        nonlocal coeff_batch, left_batch, right_batch
        if coeff_batch:
            objective.addTerms(coeff_batch, left_batch, right_batch)
            coeff_batch = []
            left_batch = []
            right_batch = []

    # Original pairwise GQAP objective:
    # sum_{i,k} sim_ik * D_{p_i,p_k}; using i<k with coefficient 2*sim_ik.
    for i in range(inst.llm_count):
        for k in range(i + 1, inst.llm_count):
            sim_coeff = float(2.0 * sim[i, k])
            if sim_coeff <= 1e-14:
                continue
            for n in range(inst.g_count):
                for q in range(inst.g_count):
                    dist = float(inst.d_gg[n, q])
                    if dist <= 1e-14:
                        continue
                    coeff_batch.append(sim_coeff * dist)
                    left_batch.append(x[i, n])
                    right_batch.append(x[k, q])
                    q_terms += 1
                    if len(coeff_batch) >= 100_000:
                        flush()
    flush()
    model.setObjective(objective, GRB.MINIMIZE)

    if mip_start is not None:
        for i, server in enumerate(mip_start):
            for n in range(inst.g_count):
                x[i, n].Start = 1.0 if n == server else 0.0

    model.update()
    return model, {"x": x, "quadratic_terms": q_terms}


def solve_with_checkpoints(
    inst,
    sim,
    method: str,
    mip_start: list[int] | None,
    mip_start_cost: float | None,
    seed: int,
    time_limit_s: int,
    checkpoints: list[int],
    output_flag: int,
) -> tuple[list[dict], dict]:
    denom = float(sim.sum())
    print(f"[{method}] building GQAP model...", flush=True)
    build_start = time.perf_counter()
    model, aux = build_gqap_model(inst, sim, mip_start, seed=seed, output_flag=output_flag)
    build_time = time.perf_counter() - build_start
    model.Params.TimeLimit = float(time_limit_s)
    print(
        f"[{method}] model built in {build_time:.2f}s, x={inst.llm_count * inst.g_count}, "
        f"quadratic_terms={aux['quadratic_terms']}, optimizing {time_limit_s}s...",
        flush=True,
    )

    checkpoints = sorted({checkpoint for checkpoint in checkpoints if 0 < checkpoint <= time_limit_s})
    records: dict[int, dict] = {}

    def remember(runtime: float, best_raw: float, bound_raw: float) -> None:
        for checkpoint in checkpoints:
            if checkpoint <= time_limit_s and checkpoint not in records and runtime >= checkpoint:
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

    # Fill checkpoints not reached exactly by callback with the final incumbent/bound.
    runtime_for_fill = float(model.Runtime)
    for checkpoint in checkpoints:
        if checkpoint <= time_limit_s and checkpoint not in records:
            records[checkpoint] = {
                "method": method,
                "checkpoint_s": checkpoint,
                "recorded_runtime_s": runtime_for_fill,
                "incumbent_cost": final_best,
                "best_bound_cost": final_bound,
                "gap_percent": final_gap,
            }

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
        "mip_start_cost": mip_start_cost if mip_start_cost is not None else math.nan,
        "final_feasible": feasible,
        "final_assignment": json.dumps(assignment),
        "quadratic_terms": aux["quadratic_terms"],
    }
    print(
        f"[{method}] done: status={final['status']}, incumbent={final_best:.6f}, "
        f"bound={final_bound:.6f}, gap={final_gap:.2f}%, runtime={model.Runtime:.2f}s",
        flush=True,
    )
    return [records[c] for c in checkpoints if c <= time_limit_s], final


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gurobi GQAP with/without KNN initialization.")
    parser.add_argument("--llms", type=int, default=100)
    parser.add_argument("--tools", type=int, default=300)
    parser.add_argument("--g-servers", type=int, default=25)
    parser.add_argument("--gpu-cap", type=int, default=4)
    parser.add_argument("--mem-cap", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--knn-samples", type=int, default=20)
    parser.add_argument("--time-limit-s", type=int, default=600)
    parser.add_argument("--checkpoints-s", default=",".join(str(item) for item in DEFAULT_CHECKPOINTS))
    parser.add_argument("--out-dir", type=Path, default=Path("jolt_gqap_gurobi_knn_L100"))
    parser.add_argument("--output-flag", type=int, default=0)
    args = parser.parse_args()
    checkpoints = [int(item.strip()) for item in args.checkpoints_s.split(",") if item.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Instance: LLM={args.llms}, tools={args.tools}, G={args.g_servers}, "
        f"GPU cap={args.gpu_cap}, MEM cap={args.mem_cap}, seed={args.seed}",
        flush=True,
    )
    inst = make_llm_instance(
        seed=args.seed,
        llm_count=args.llms,
        tool_count=args.tools,
        g_count=args.g_servers,
        gpu_cap=args.gpu_cap,
        mem_cap=args.mem_cap,
    )
    sim = preference_similarity(inst)
    (args.out_dir / "instance_meta.json").write_text(
        json.dumps(instance_to_jsonable(inst), indent=2),
        encoding="utf-8",
    )

    print(f"Generating {args.knn_samples} KNN starts...", flush=True)
    knn_start, knn_rows = generate_best_knn_start(inst, sim, args.seed + 50_000, args.knn_samples)
    knn_cost = normalized_cost(inst, knn_start, sim)
    write_csv(args.out_dir / "knn_start_candidates.csv", knn_rows)
    (args.out_dir / "best_knn_start.json").write_text(
        json.dumps({"cost": knn_cost, "assignment": knn_start, "feasible": is_feasible(inst, knn_start)}, indent=2),
        encoding="utf-8",
    )
    print(f"Best KNN start cost={knn_cost:.6f}, feasible={is_feasible(inst, knn_start)}", flush=True)

    checkpoint_rows: list[dict] = []
    final_rows: list[dict] = []

    rows, final = solve_with_checkpoints(
        inst,
        sim,
        method="Gurobi direct",
        mip_start=None,
        mip_start_cost=None,
        seed=args.seed + 100,
        time_limit_s=args.time_limit_s,
        checkpoints=checkpoints,
        output_flag=args.output_flag,
    )
    checkpoint_rows.extend(rows)
    final_rows.append(final)
    write_csv(args.out_dir / "checkpoint_results.csv", checkpoint_rows)
    write_csv(args.out_dir / "final_results.csv", final_rows)

    rows, final = solve_with_checkpoints(
        inst,
        sim,
        method="Gurobi + KNN init",
        mip_start=knn_start,
        mip_start_cost=knn_cost,
        seed=args.seed + 100,
        time_limit_s=args.time_limit_s,
        checkpoints=checkpoints,
        output_flag=args.output_flag,
    )
    checkpoint_rows.extend(rows)
    final_rows.append(final)
    write_csv(args.out_dir / "checkpoint_results.csv", checkpoint_rows)
    write_csv(args.out_dir / "final_results.csv", final_rows)

    print("\nCheckpoint results")
    for row in checkpoint_rows:
        print(
            f"{row['method']:<20} t={row['checkpoint_s']:>4}s "
            f"cost={float(row['incumbent_cost']):.6f} "
            f"bound={float(row['best_bound_cost']):.6f} gap={float(row['gap_percent']):.2f}%",
            flush=True,
        )
    print(f"\nWrote outputs to {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
