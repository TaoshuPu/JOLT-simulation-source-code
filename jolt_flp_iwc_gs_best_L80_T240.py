from __future__ import annotations

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

from jolt_small_scale_experiment import (
    average_call_distance,
    instance_to_jsonable,
    is_deployment_feasible,
    is_llm_feasible,
    is_tool_feasible,
    iwc_gs_tool_deployment,
    make_instance,
    tool_cost_matrix,
    tool_host_capacity,
    tool_host_count,
)


LLMS = 80
TOOLS = 240
SEED = 20340528
TIME_LIMIT_S = 600.0
CHECKPOINTS = [0, 1, 5, 10, 30, 60, 120, 180, 300, 420, 600]
SOURCE_TRACE = ROOT / "jolt_single_run_checkpoint_20_80_10min_rerun" / "trace_L80_gqap_tool_mip.json"
OUT_DIR = ROOT / "jolt_flp_iwc_gs_best_L80_T240"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def safe_avg(raw_objective: float, denominator: float) -> float:
    if raw_objective is None or not math.isfinite(raw_objective) or abs(raw_objective) >= GRB.INFINITY * 0.5:
        return math.nan
    return raw_objective / denominator


def gap_percent(best: float, bound: float) -> float:
    if not math.isfinite(best) or not math.isfinite(bound) or abs(best) < 1e-12:
        return math.nan
    return max(0.0, 100.0 * (best - bound) / abs(best))


def load_best_llm_assignment(trace_path: Path) -> tuple[list[int], dict]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    snapshots = trace.get("snapshots", [])
    candidates = [
        snap
        for snap in snapshots
        if len(snap.get("llm_place", [])) == LLMS
        and math.isfinite(float(snap.get("extra", {}).get("llm_surrogate_cost", math.inf)))
    ]
    if not candidates:
        candidates = [snap for snap in snapshots if len(snap.get("llm_place", [])) == LLMS]
    if not candidates:
        raise RuntimeError(f"No LLM={LLMS} snapshot found in {trace_path}")

    def rank_key(snap: dict) -> tuple[float, float]:
        extra = snap.get("extra", {})
        surrogate = float(extra.get("llm_surrogate_cost", math.inf))
        distance = float(snap.get("avg_call_distance", math.inf))
        return surrogate, distance

    best = min(candidates, key=rank_key)
    return [int(x) for x in best["llm_place"]], best


def raw_tool_objective(tool_cost, tool_place: list[int]) -> float:
    return sum(float(tool_cost[j, h]) for j, h in enumerate(tool_place))


def solve_flp_with_checkpoints(
    *,
    inst,
    llm_place: list[int],
    method: str,
    warm_start: list[int] | None,
    seed: int,
    time_limit_s: float,
    checkpoints: list[int],
) -> tuple[list[dict], dict]:
    start_wall = time.perf_counter()
    tool_cost = tool_cost_matrix(inst, llm_place)
    total_calls = float((inst.arrival[:, None] * inst.pref).sum())
    h_count = tool_host_count(inst)
    rem_cpu, rem_mem = tool_host_capacity(inst, llm_place)

    model = gp.Model("jolt_flp_fixed_llm_tool_assignment")
    model.Params.OutputFlag = 0
    model.Params.Seed = seed
    model.Params.Threads = 0
    model.Params.TimeLimit = float(time_limit_s)
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0

    build_start = time.perf_counter()
    y = model.addVars(inst.tool_count, h_count, vtype=GRB.BINARY, name="y")
    model.addConstrs((gp.quicksum(y[j, h] for h in range(h_count)) == 1 for j in range(inst.tool_count)), name="assign_tool")
    model.addConstrs(
        (
            gp.quicksum(int(inst.tool_cpu[j]) * y[j, h] for j in range(inst.tool_count))
            <= int(rem_cpu[h])
            for h in range(h_count)
        ),
        name="cpu_cap",
    )
    model.addConstrs(
        (
            gp.quicksum(int(inst.tool_mem[j]) * y[j, h] for j in range(inst.tool_count))
            <= int(rem_mem[h])
            for h in range(h_count)
        ),
        name="mem_cap",
    )
    if warm_start is not None:
        for j, host in enumerate(warm_start):
            for h in range(h_count):
                y[j, h].Start = 1.0 if h == host else 0.0

    model.setObjective(
        gp.quicksum(float(tool_cost[j, h]) * y[j, h] for j in range(inst.tool_count) for h in range(h_count)),
        GRB.MINIMIZE,
    )
    model.update()
    build_time = time.perf_counter() - build_start

    records: dict[int, dict] = {}
    first_incumbent_runtime = math.nan
    first_incumbent_avg = math.nan

    if 0 in checkpoints:
        if warm_start is not None and is_tool_feasible(inst, warm_start, llm_place):
            start_raw = raw_tool_objective(tool_cost, warm_start)
            start_avg = safe_avg(start_raw, total_calls)
            records[0] = {
                "method": method,
                "checkpoint_s": 0,
                "recorded_runtime_s": 0.0,
                "avg_call_distance": start_avg,
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
        nonlocal first_incumbent_runtime, first_incumbent_avg
        if where == GRB.Callback.MIP:
            runtime = float(model_cb.cbGet(GRB.Callback.RUNTIME))
            best_raw = float(model_cb.cbGet(GRB.Callback.MIP_OBJBST))
            bound_raw = float(model_cb.cbGet(GRB.Callback.MIP_OBJBND))
            remember(runtime, best_raw, bound_raw, "mip_callback")
        elif where == GRB.Callback.MIPSOL:
            runtime = float(model_cb.cbGet(GRB.Callback.RUNTIME))
            sol_raw = float(model_cb.cbGet(GRB.Callback.MIPSOL_OBJ))
            if not math.isfinite(first_incumbent_runtime):
                first_incumbent_runtime = runtime
                first_incumbent_avg = safe_avg(sol_raw, total_calls)
            try:
                bound_raw = float(model_cb.cbGet(GRB.Callback.MIPSOL_OBJBND))
            except gp.GurobiError:
                bound_raw = math.nan
            remember(runtime, sol_raw, bound_raw, "mipsol_callback")

    print(f"[{method}] building done in {build_time:.3f}s; optimizing up to {time_limit_s:g}s", flush=True)
    optimize_start = time.perf_counter()
    model.optimize(callback)
    optimize_wall = time.perf_counter() - optimize_start
    total_wall = time.perf_counter() - start_wall

    if model.SolCount > 0:
        tool_place = [max(range(h_count), key=lambda h: y[j, h].X) for j in range(inst.tool_count)]
        final_raw = float(model.ObjVal)
        final_bound_raw = float(model.ObjBound)
    else:
        tool_place = []
        final_raw = math.nan
        final_bound_raw = math.nan

    final_avg = safe_avg(final_raw, total_calls)
    final_bound_avg = safe_avg(final_bound_raw, total_calls)
    final_gap = gap_percent(final_avg, final_bound_avg)
    final_status = status_name(model.Status)

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
                "status": final_status,
                "source": "final_fill_after_solver_end",
            }

    warm_start_avg = math.nan
    if warm_start is not None:
        warm_start_avg = average_call_distance(inst, llm_place, warm_start)

    final = {
        "method": method,
        "llms": inst.llm_count,
        "tools": inst.tool_count,
        "g_servers": inst.g_count,
        "c_servers": inst.c_count,
        "tool_hosts": h_count,
        "status": final_status,
        "proven_optimal": model.Status == GRB.OPTIMAL,
        "solver_runtime_s": float(model.Runtime),
        "optimize_wall_s": optimize_wall,
        "total_wall_s": total_wall,
        "build_time_s": build_time,
        "sol_count": int(model.SolCount),
        "node_count": float(model.NodeCount),
        "raw_objective": final_raw,
        "best_bound_raw": final_bound_raw,
        "final_avg_call_distance": final_avg,
        "best_bound_avg_call_distance": final_bound_avg,
        "final_gap_percent": final_gap,
        "warm_start_used": warm_start is not None,
        "warm_start_avg_call_distance": warm_start_avg,
        "first_incumbent_runtime_s": first_incumbent_runtime,
        "first_incumbent_avg_call_distance": first_incumbent_avg,
        "llm_feasible": is_llm_feasible(inst, llm_place),
        "tool_feasible": bool(tool_place) and is_tool_feasible(inst, tool_place, llm_place),
        "deployment_feasible": bool(tool_place) and is_deployment_feasible(inst, llm_place, tool_place),
        "final_tool_assignment": json.dumps(tool_place),
    }
    rows = [records[c] for c in sorted(records)]
    print(
        f"[{method}] {final_status}: avg={final_avg:.6f}, bound={final_bound_avg:.6f}, "
        f"gap={final_gap:.4f}%, runtime={model.Runtime:.3f}s, nodes={model.NodeCount:.0f}",
        flush=True,
    )
    return rows, final


def plot_results(out_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: {exc}", flush=True)
        return

    methods = list(dict.fromkeys(row["method"] for row in rows))
    colors = {
        "Gurobi FLP direct": "#F2B880",
        "Gurobi FLP + IWC-GS init": "#6AAED6",
    }
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        xs = [float(row["checkpoint_s"]) for row in method_rows]
        ys = [float(row["avg_call_distance"]) if str(row["avg_call_distance"]) != "nan" else math.nan for row in method_rows]
        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=4.5, label=method, color=colors.get(method))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average Call Distance")
    ax.set_title("FLP Solving with Best Fixed LLM Deployment")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "flp_iwc_gs_init_comparison_300dpi.png", dpi=300)
    fig.savefig(out_dir / "flp_iwc_gs_init_comparison_300dpi.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inst = make_instance(
        SEED,
        llm_count=LLMS,
        tool_count=TOOLS,
        g_count=None,
        c_count=None,
        capacity_mode="fixed_per_server",
    )
    llm_place, source_snapshot = load_best_llm_assignment(SOURCE_TRACE)
    if not is_llm_feasible(inst, llm_place):
        raise RuntimeError("Selected best LLM deployment is infeasible.")

    iwc_tool_place = iwc_gs_tool_deployment(inst, llm_place)
    if not is_tool_feasible(inst, iwc_tool_place, llm_place):
        raise RuntimeError("IWC-GS warm start is infeasible.")

    source_meta = {
        "fixed_llm_source": str(SOURCE_TRACE),
        "selection_rule": "minimum phase-1 LLM surrogate cost among JOLT snapshots",
        "selected_snapshot_elapsed_s": source_snapshot.get("elapsed_s"),
        "selected_snapshot_iwc_gs_avg_call_distance": source_snapshot.get("avg_call_distance"),
        "selected_snapshot_llm_surrogate_cost": source_snapshot.get("extra", {}).get("llm_surrogate_cost"),
        "llm_assignment": llm_place,
        "iwc_gs_tool_assignment": iwc_tool_place,
        "iwc_gs_avg_call_distance": average_call_distance(inst, llm_place, iwc_tool_place),
    }
    (OUT_DIR / "instance_L80_T240.json").write_text(
        json.dumps(instance_to_jsonable(inst), ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT_DIR / "fixed_llm_and_iwc_gs_start.json").write_text(
        json.dumps(source_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "FLP experiment: LLM=80, tools=240, fixed LLM deployment from best existing GQAP snapshot, "
        f"IWC-GS start distance={source_meta['iwc_gs_avg_call_distance']:.6f}",
        flush=True,
    )

    all_checkpoint_rows: list[dict] = []
    final_rows: list[dict] = []
    direct_rows, direct_final = solve_flp_with_checkpoints(
        inst=inst,
        llm_place=llm_place,
        method="Gurobi FLP direct",
        warm_start=None,
        seed=SEED + 700,
        time_limit_s=TIME_LIMIT_S,
        checkpoints=CHECKPOINTS,
    )
    all_checkpoint_rows.extend(direct_rows)
    final_rows.append(direct_final)
    write_csv(OUT_DIR / "flp_checkpoint_results.csv", all_checkpoint_rows)
    write_csv(OUT_DIR / "flp_final_results.csv", final_rows)

    warm_rows, warm_final = solve_flp_with_checkpoints(
        inst=inst,
        llm_place=llm_place,
        method="Gurobi FLP + IWC-GS init",
        warm_start=iwc_tool_place,
        seed=SEED + 700,
        time_limit_s=TIME_LIMIT_S,
        checkpoints=CHECKPOINTS,
    )
    all_checkpoint_rows.extend(warm_rows)
    final_rows.append(warm_final)
    write_csv(OUT_DIR / "flp_checkpoint_results.csv", all_checkpoint_rows)
    write_csv(OUT_DIR / "flp_final_results.csv", final_rows)
    plot_results(OUT_DIR, all_checkpoint_rows)

    print(f"Wrote outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
