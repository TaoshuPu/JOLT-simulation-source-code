from __future__ import annotations

import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for deps_name in (".scipy_deps", ".gurobi_deps", ".ortools_deps"):
    deps = ROOT / deps_name
    if deps.exists():
        sys.path.insert(0, str(deps))

import numpy as np

from jolt_flp_iwc_gs_best_L80_T240 import SOURCE_TRACE, load_best_llm_assignment
from jolt_small_scale_experiment import (
    Instance,
    article_objective_weights,
    instance_to_jsonable,
    is_llm_feasible,
    is_tool_feasible,
    make_instance,
    tool_distance_matrix,
    tool_host_capacity,
    tool_host_coords,
    tool_host_count,
    weighted_calls,
)


LLMS = 80
TOOLS = 10_000
SEED = 20340528
TIME_LIMIT_S = 120.0
CHECKPOINTS = [0, 1, 5, 10, 30, 60, 120]
OUT_DIR = ROOT / "jolt_flp_iwc_gs_init_L80_T10000_2min_fast"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def emit(stage: str, **payload: object) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row = {"stage": stage, **payload}
    (OUT_DIR / "stage_status.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def status_name(status: int) -> str:
    import gurobipy as gp
    from gurobipy import GRB

    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.MEM_LIMIT: "MEM_LIMIT",
    }.get(status, str(status))


def safe_avg(raw_objective: float, total_calls: float) -> float:
    if not math.isfinite(raw_objective):
        return math.nan
    return raw_objective / total_calls


def gap_percent(best: float, bound: float) -> float:
    if not math.isfinite(best) or not math.isfinite(bound) or abs(best) < 1e-12:
        return math.nan
    return max(0.0, 100.0 * (best - bound) / abs(best))


def fast_tool_cost_matrix(inst: Instance, llm_place: list[int]) -> np.ndarray:
    # Equivalent to tool_cost_matrix, but uses BLAS instead of a Python triple loop.
    llm_host_dist = tool_distance_matrix(inst)[np.asarray(llm_place, dtype=int), :]
    return article_objective_weights(inst).T @ llm_host_dist


def fast_avg_from_cost(tool_cost: np.ndarray, tool_place: list[int], total_calls: float) -> float:
    indices = np.arange(len(tool_place), dtype=int)
    raw = float(tool_cost[indices, np.asarray(tool_place, dtype=int)].sum())
    return raw / total_calls


def fast_best_fit_tool_assignment(inst: Instance, llm_place: list[int], tool_cost: np.ndarray) -> list[int] | None:
    freq = weighted_calls(inst).sum(axis=0)
    orders = [
        sorted(
            range(inst.tool_count),
            key=lambda j: (int(inst.tool_cpu[j]) + int(inst.tool_mem[j]), int(inst.tool_cpu[j]), float(freq[j])),
            reverse=True,
        ),
        sorted(
            range(inst.tool_count),
            key=lambda j: (float(np.max(tool_cost[j]) - np.min(tool_cost[j])), float(freq[j])),
            reverse=True,
        ),
        sorted(range(inst.tool_count), key=lambda j: float(freq[j]), reverse=True),
        sorted(range(inst.tool_count), key=lambda j: (int(inst.tool_cpu[j]), int(inst.tool_mem[j])), reverse=True),
        sorted(range(inst.tool_count), key=lambda j: (int(inst.tool_mem[j]), int(inst.tool_cpu[j])), reverse=True),
    ]
    rem_cpu_base, rem_mem_base = tool_host_capacity(inst, llm_place)
    modes = ["cost", "tight_sum", "tight_cpu", "tight_mem", "most_room"]
    for order in orders:
        for mode in modes:
            rem_cpu = rem_cpu_base.astype(int, copy=True)
            rem_mem = rem_mem_base.astype(int, copy=True)
            place = np.full(inst.tool_count, -1, dtype=np.int32)
            failed = False
            for j in order:
                cpu_need = int(inst.tool_cpu[j])
                mem_need = int(inst.tool_mem[j])
                feasible = np.flatnonzero((rem_cpu >= cpu_need) & (rem_mem >= mem_need))
                if feasible.size == 0:
                    failed = True
                    break
                cpu_left = rem_cpu[feasible] - cpu_need
                mem_left = rem_mem[feasible] - mem_need
                if mode == "cost":
                    score = tool_cost[j, feasible]
                    chosen = int(feasible[int(np.argmin(score))])
                elif mode == "tight_sum":
                    score = cpu_left + mem_left
                    chosen = int(feasible[int(np.argmin(score))])
                elif mode == "tight_cpu":
                    score = cpu_left * 1000 + mem_left
                    chosen = int(feasible[int(np.argmin(score))])
                elif mode == "tight_mem":
                    score = mem_left * 1000 + cpu_left
                    chosen = int(feasible[int(np.argmin(score))])
                else:
                    score = rem_cpu[feasible] + rem_mem[feasible]
                    chosen = int(feasible[int(np.argmax(score))])
                place[j] = chosen
                rem_cpu[chosen] -= cpu_need
                rem_mem[chosen] -= mem_need
            if not failed:
                return place.astype(int).tolist()
    return None


def fast_constructive_repair_assignment(inst: Instance, llm_place: list[int], tool_cost: np.ndarray) -> list[int] | None:
    """A deterministic multi-pass repair for tight two-resource tool packing."""
    rem_cpu_base, rem_mem_base = tool_host_capacity(inst, llm_place)
    h_count = tool_host_count(inst)
    orders = [
        sorted(range(inst.tool_count), key=lambda j: (int(inst.tool_cpu[j]), int(inst.tool_mem[j])), reverse=True),
        sorted(range(inst.tool_count), key=lambda j: (int(inst.tool_mem[j]), int(inst.tool_cpu[j])), reverse=True),
        sorted(
            range(inst.tool_count),
            key=lambda j: (int(inst.tool_cpu[j]) + int(inst.tool_mem[j]), int(inst.tool_cpu[j])),
            reverse=True,
        ),
    ]
    for order in orders:
        rem_cpu = rem_cpu_base.astype(int, copy=True)
        rem_mem = rem_mem_base.astype(int, copy=True)
        place = np.full(inst.tool_count, -1, dtype=np.int32)
        failed = False
        for j in order:
            cpu_need = int(inst.tool_cpu[j])
            mem_need = int(inst.tool_mem[j])
            feasible = np.flatnonzero((rem_cpu >= cpu_need) & (rem_mem >= mem_need))
            if feasible.size == 0:
                failed = True
                break
            cpu_left = rem_cpu[feasible] - cpu_need
            mem_left = rem_mem[feasible] - mem_need
            # Tight resource fit first; among close fits, prefer lower communication cost.
            tight_score = cpu_left + mem_left
            best_tight = np.min(tight_score)
            close = feasible[tight_score <= best_tight + 2]
            chosen = int(close[int(np.argmin(tool_cost[j, close]))])
            place[j] = chosen
            rem_cpu[chosen] -= cpu_need
            rem_mem[chosen] -= mem_need
        if not failed:
            return place.astype(int).tolist()
    return None


def fast_iwc_gs_tool_deployment(inst: Instance, llm_place: list[int]) -> tuple[list[int], dict]:
    start = time.perf_counter()
    w = weighted_calls(inst)
    freq = w.sum(axis=0)
    h_count = tool_host_count(inst)
    host_coords = tool_host_coords(inst)
    llm_coords = inst.g_coords[np.asarray(llm_place, dtype=int)]
    centroid = w.T @ llm_coords
    nonzero = freq > 1e-12
    centroid[nonzero] /= freq[nonzero, None]
    if not np.all(nonzero):
        centroid[~nonzero] = llm_coords.mean(axis=0)

    dist = np.linalg.norm(centroid[:, None, :] - host_coords[None, :, :], axis=2)
    server_orders = np.argsort(dist, axis=1).astype(np.int32, copy=False)
    rem_cpu_base, rem_mem_base = tool_host_capacity(inst, llm_place)

    def try_order(order: list[int]) -> list[int] | None:
        rem_cpu = rem_cpu_base.astype(int, copy=True)
        rem_mem = rem_mem_base.astype(int, copy=True)
        place = np.full(inst.tool_count, -1, dtype=np.int32)
        for j in order:
            cpu_need = int(inst.tool_cpu[j])
            mem_need = int(inst.tool_mem[j])
            for m in server_orders[j]:
                if rem_cpu[m] >= cpu_need and rem_mem[m] >= mem_need:
                    place[j] = int(m)
                    rem_cpu[m] -= cpu_need
                    rem_mem[m] -= mem_need
                    break
            if place[j] < 0:
                return None
        return place.astype(int).tolist()

    freq_order = sorted(range(inst.tool_count), key=lambda j: float(freq[j]), reverse=True)
    resource_order = sorted(
        range(inst.tool_count),
        key=lambda j: (int(inst.tool_cpu[j]) + int(inst.tool_mem[j]), int(inst.tool_cpu[j]), float(freq[j])),
        reverse=True,
    )
    combined_order = sorted(
        range(inst.tool_count),
        key=lambda j: (float(freq[j]), int(inst.tool_cpu[j]) + int(inst.tool_mem[j])),
        reverse=True,
    )
    tool_cost = fast_tool_cost_matrix(inst, llm_place)
    total_calls = float(weighted_calls(inst).sum())
    candidates = [place for place in (try_order(freq_order), try_order(resource_order), try_order(combined_order)) if place]
    fallback_used = False
    if not candidates:
        fallback = fast_best_fit_tool_assignment(inst, llm_place, tool_cost)
        if fallback is None:
            fallback = fast_constructive_repair_assignment(inst, llm_place, tool_cost)
        if fallback is None:
            raise RuntimeError("No feasible IWC-GS or best-fit fallback candidate found.")
        candidates = [fallback]
        fallback_used = True
    best = min(candidates, key=lambda place: fast_avg_from_cost(tool_cost, place, total_calls))
    elapsed = time.perf_counter() - start
    return best, {
        "iwc_gs_runtime_s": elapsed,
        "candidate_count": len(candidates),
        "fallback_used": fallback_used,
        "iwc_gs_initial_avg_call_distance": fast_avg_from_cost(tool_cost, best, total_calls),
    }


def solve_flp_mvar(
    *,
    inst: Instance,
    llm_place: list[int],
    method: str,
    warm_start: list[int] | None,
    seed: int,
    time_limit_s: float,
    checkpoints: list[int],
) -> tuple[list[dict], dict]:
    import gurobipy as gp
    from gurobipy import GRB

    h_count = tool_host_count(inst)
    total_calls = float(weighted_calls(inst).sum())
    overall_start = time.perf_counter()
    cost_start = time.perf_counter()
    tool_cost = fast_tool_cost_matrix(inst, llm_place)
    cost_time = time.perf_counter() - cost_start

    model = gp.Model(f"jolt_flp_L80_T10000_{method.replace(' ', '_')}")
    model.Params.OutputFlag = 0
    model.Params.Seed = seed
    model.Params.Threads = 0
    model.Params.TimeLimit = float(time_limit_s)
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0

    build_start = time.perf_counter()
    y = model.addMVar((inst.tool_count, h_count), vtype=GRB.BINARY, name="y")
    model.addConstr(y.sum(axis=1) == 1, name="assign_tool")
    rem_cpu, rem_mem = tool_host_capacity(inst, llm_place)
    model.addConstr(np.asarray(inst.tool_cpu, dtype=float) @ y <= rem_cpu.astype(float), name="cpu_cap")
    model.addConstr(np.asarray(inst.tool_mem, dtype=float) @ y <= rem_mem.astype(float), name="mem_cap")
    if warm_start is not None:
        start_values = np.zeros((inst.tool_count, h_count), dtype=float)
        start_values[np.arange(inst.tool_count), np.asarray(warm_start, dtype=int)] = 1.0
        y.Start = start_values
    model.setObjective((y * tool_cost).sum(), GRB.MINIMIZE)
    model.update()
    build_time = time.perf_counter() - build_start

    records: dict[int, dict] = {}
    if 0 in checkpoints:
        if warm_start is not None:
            start_raw = float(tool_cost[np.arange(inst.tool_count), np.asarray(warm_start, dtype=int)].sum())
            records[0] = {
                "method": method,
                "checkpoint_s": 0,
                "recorded_runtime_s": 0.0,
                "avg_call_distance": safe_avg(start_raw, total_calls),
                "raw_objective": start_raw,
                "best_bound_avg": math.nan,
                "gap_percent": math.nan,
                "status": "IWC_GS_INITIAL_INCUMBENT",
                "source": "before_optimize",
            }
        else:
            records[0] = {
                "method": method,
                "checkpoint_s": 0,
                "recorded_runtime_s": 0.0,
                "avg_call_distance": math.nan,
                "raw_objective": math.nan,
                "best_bound_avg": math.nan,
                "gap_percent": math.nan,
                "status": "NO_INITIAL_INCUMBENT",
                "source": "before_optimize",
            }

    def remember(runtime: float, best_raw: float, bound_raw: float, source: str) -> None:
        best_avg = safe_avg(best_raw, total_calls)
        bound_avg = safe_avg(bound_raw, total_calls)
        for checkpoint in checkpoints:
            if checkpoint == 0:
                continue
            if checkpoint not in records and runtime >= checkpoint:
                records[checkpoint] = {
                    "method": method,
                    "checkpoint_s": checkpoint,
                    "recorded_runtime_s": runtime,
                    "avg_call_distance": best_avg,
                    "raw_objective": best_raw if math.isfinite(best_raw) else math.nan,
                    "best_bound_avg": bound_avg,
                    "gap_percent": gap_percent(best_avg, bound_avg),
                    "status": "RUNNING_INCUMBENT" if math.isfinite(best_avg) else "NO_INCUMBENT",
                    "source": source,
                }

    def callback(model_cb: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIP:
            runtime = float(model_cb.cbGet(GRB.Callback.RUNTIME))
            best_raw = float(model_cb.cbGet(GRB.Callback.MIP_OBJBST))
            bound_raw = float(model_cb.cbGet(GRB.Callback.MIP_OBJBND))
            remember(runtime, best_raw, bound_raw, "mip_callback")
        elif where == GRB.Callback.MIPSOL:
            runtime = float(model_cb.cbGet(GRB.Callback.RUNTIME))
            sol_raw = float(model_cb.cbGet(GRB.Callback.MIPSOL_OBJ))
            try:
                bound_raw = float(model_cb.cbGet(GRB.Callback.MIPSOL_OBJBND))
            except gp.GurobiError:
                bound_raw = math.nan
            remember(runtime, sol_raw, bound_raw, "mipsol_callback")

    emit("gurobi_optimize_start", method=method, cost_matrix_s=cost_time, build_time_s=build_time)
    opt_start = time.perf_counter()
    model.optimize(callback)
    opt_wall = time.perf_counter() - opt_start
    status = status_name(model.Status)

    if model.SolCount > 0:
        final_raw = float(model.ObjVal)
        final_bound_raw = float(model.ObjBound)
    else:
        final_raw = math.nan
        final_bound_raw = math.nan
    final_avg = safe_avg(final_raw, total_calls)
    final_bound_avg = safe_avg(final_bound_raw, total_calls)
    final_gap = gap_percent(final_avg, final_bound_avg)
    for checkpoint in checkpoints:
        if checkpoint not in records:
            records[checkpoint] = {
                "method": method,
                "checkpoint_s": checkpoint,
                "recorded_runtime_s": float(model.Runtime),
                "avg_call_distance": final_avg,
                "raw_objective": final_raw,
                "best_bound_avg": final_bound_avg,
                "gap_percent": final_gap,
                "status": status,
                "source": "final_fill_after_solver_end",
            }

    final = {
        "method": method,
        "llms": inst.llm_count,
        "tools": inst.tool_count,
        "g_servers": inst.g_count,
        "c_servers": inst.c_count,
        "tool_hosts": h_count,
        "status": status,
        "proven_optimal": model.Status == GRB.OPTIMAL,
        "solver_runtime_s": float(model.Runtime),
        "optimize_wall_s": opt_wall,
        "total_wall_s": time.perf_counter() - overall_start,
        "cost_matrix_s": cost_time,
        "build_time_s": build_time,
        "sol_count": int(model.SolCount),
        "node_count": float(model.NodeCount),
        "raw_objective": final_raw,
        "best_bound_raw": final_bound_raw,
        "final_avg_call_distance": final_avg,
        "best_bound_avg_call_distance": final_bound_avg,
        "final_gap_percent": final_gap,
        "warm_start_used": warm_start is not None,
    }
    rows = [records[c] for c in sorted(records)]
    model.dispose()
    return rows, final


def percent_gap(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(candidate) or abs(reference) < 1e-12:
        return math.nan
    return 100.0 * (candidate - reference) / abs(reference)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    llm_place, source_snapshot = load_best_llm_assignment(SOURCE_TRACE)
    emit("make_instance_start", llms=LLMS, tools=TOOLS)
    t0 = time.perf_counter()
    inst = make_instance(SEED, llm_count=LLMS, tool_count=TOOLS, g_count=None, c_count=None, capacity_mode="fixed_per_server")
    meta = {
        "llms": LLMS,
        "tools": TOOLS,
        "g_servers": inst.g_count,
        "c_servers": inst.c_count,
        "tool_hosts": tool_host_count(inst),
        "y_variables": TOOLS * tool_host_count(inst),
        "tool_cpu_sum": int(inst.tool_cpu.sum()),
        "tool_mem_sum": int(inst.tool_mem.sum()),
        "make_instance_s": time.perf_counter() - t0,
        "llm_feasible": is_llm_feasible(inst, llm_place),
        "fixed_llm_source": str(SOURCE_TRACE),
        "source_snapshot_elapsed_s": source_snapshot.get("elapsed_s"),
    }
    emit("make_instance_done", **meta)
    (OUT_DIR / "instance_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    emit("iwc_gs_start", **meta)
    iwc_tool, iwc_extra = fast_iwc_gs_tool_deployment(inst, llm_place)
    iwc_row = {
        **meta,
        **iwc_extra,
        "iwc_gs_feasible": is_tool_feasible(inst, iwc_tool, llm_place),
        "iwc_gs_tool_assignment": json.dumps(iwc_tool),
    }
    write_csv(OUT_DIR / "iwc_gs_initial_result.csv", [iwc_row])
    emit("iwc_gs_done", **{k: v for k, v in iwc_row.items() if k != "iwc_gs_tool_assignment"})

    checkpoint_rows: list[dict] = []
    final_rows: list[dict] = []
    for method, warm_start in [("Gurobi FLP direct", None), ("Gurobi FLP + IWC-GS init", iwc_tool)]:
        try:
            emit("gurobi_build_start", method=method, **meta)
            rows, final = solve_flp_mvar(
                inst=inst,
                llm_place=llm_place,
                method=method,
                warm_start=warm_start,
                seed=SEED + 700,
                time_limit_s=TIME_LIMIT_S,
                checkpoints=CHECKPOINTS,
            )
            for row in rows:
                row.update({"llms": LLMS, "tools": TOOLS, "g_servers": inst.g_count, "c_servers": inst.c_count, "tool_hosts": tool_host_count(inst)})
            checkpoint_rows.extend(rows)
            final.update(
                {
                    "iwc_gs_runtime_s": iwc_extra["iwc_gs_runtime_s"],
                    "iwc_gs_initial_avg_call_distance": iwc_extra["iwc_gs_initial_avg_call_distance"],
                }
            )
            final_rows.append(final)
            write_csv(OUT_DIR / "flp_checkpoint_results.csv", checkpoint_rows)
            write_csv(OUT_DIR / "flp_final_results.csv", final_rows)
            emit("gurobi_done", method=method, status=final["status"], avg=final["final_avg_call_distance"], gap=final["final_gap_percent"])
        except Exception as exc:
            error = {"method": method, "error": repr(exc), "traceback": traceback.format_exc(), **meta}
            (OUT_DIR / f"{method.replace(' ', '_').replace('+', 'plus')}_error.json").write_text(
                json.dumps(error, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            emit("gurobi_failed", **error)

    if len(final_rows) == 2:
        direct = next(row for row in final_rows if row["method"] == "Gurobi FLP direct")
        warm = next(row for row in final_rows if row["method"] == "Gurobi FLP + IWC-GS init")
        summary = {
            **meta,
            "time_limit_s": TIME_LIMIT_S,
            "iwc_gs_runtime_s": iwc_extra["iwc_gs_runtime_s"],
            "iwc_gs_initial_avg_call_distance": iwc_extra["iwc_gs_initial_avg_call_distance"],
            "direct_status": direct["status"],
            "direct_final_avg_call_distance": direct["final_avg_call_distance"],
            "direct_gap_percent": direct["final_gap_percent"],
            "direct_runtime_s": direct["solver_runtime_s"],
            "direct_build_time_s": direct["build_time_s"],
            "warm_status": warm["status"],
            "warm_final_avg_call_distance": warm["final_avg_call_distance"],
            "warm_gap_percent": warm["final_gap_percent"],
            "warm_runtime_s": warm["solver_runtime_s"],
            "warm_build_time_s": warm["build_time_s"],
            "warm_minus_direct_avg": warm["final_avg_call_distance"] - direct["final_avg_call_distance"],
            "warm_vs_direct_percent": percent_gap(direct["final_avg_call_distance"], warm["final_avg_call_distance"]),
        }
        write_csv(OUT_DIR / "summary.csv", [summary])
        emit("done", **summary)
    else:
        emit("done_partial", final_rows=len(final_rows), **meta)


if __name__ == "__main__":
    main()
