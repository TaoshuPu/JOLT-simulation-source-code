from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

from jolt_small_scale_experiment import (
    AlgorithmResult,
    article_objective_weights,
    average_call_distance,
    greedy_original_warm_start,
    instance_to_jsonable,
    is_deployment_feasible,
    is_llm_feasible,
    iwc_gs_tool_deployment,
    make_instance,
    random_llm_assignment,
    resource_greedy_llm_assignment,
    tool_distance,
    tool_host_capacity,
    tool_host_count,
    gurobi_tool_deployment,
)


def gurobi_llm_deployment_fixed_tools(
    inst,
    tool_place: list[int],
    timeout_s: float | None = None,
    warm_start: list[int] | None = None,
) -> tuple[list[int], dict]:
    start = time.perf_counter()
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        fallback = warm_start if warm_start and is_llm_feasible(inst, warm_start) else resource_greedy_llm_assignment(inst)
        return fallback, {
            "phase": "llm",
            "solver": "gurobi_llm_mip_fixed_tools",
            "status": "IMPORT_ERROR",
            "error": str(exc),
            "solve_time_s": time.perf_counter() - start,
        }

    fixed_tool_cpu_on_g = [0] * inst.g_count
    fixed_tool_mem_on_g = [0] * inst.g_count
    for j, host in enumerate(tool_place):
        if host < inst.g_count:
            fixed_tool_cpu_on_g[host] += int(inst.tool_cpu[j])
            fixed_tool_mem_on_g[host] += int(inst.tool_mem[j])

    try:
        model = gp.Model("jolt_llm_assignment_fixed_tools")
        model.Params.OutputFlag = 0
        model.Params.MIPFocus = 1
        if timeout_s is not None and timeout_s > 0:
            model.Params.TimeLimit = float(timeout_s)

        x = model.addVars(inst.llm_count, inst.g_count, vtype=GRB.BINARY, name="x")
        model.addConstrs((gp.quicksum(x[i, n] for n in range(inst.g_count)) == 1 for i in range(inst.llm_count)))
        model.addConstrs(
            (
                gp.quicksum(int(inst.llm_gpu[i]) * x[i, n] for i in range(inst.llm_count))
                <= int(inst.g_gpu_cap[n])
                for n in range(inst.g_count)
            )
        )
        model.addConstrs(
            (
                gp.quicksum(int(inst.llm_cpu[i]) * x[i, n] for i in range(inst.llm_count))
                + fixed_tool_cpu_on_g[n]
                <= int(inst.g_cpu_cap[n])
                for n in range(inst.g_count)
            )
        )
        model.addConstrs(
            (
                gp.quicksum(int(inst.llm_mem[i]) * x[i, n] for i in range(inst.llm_count))
                + fixed_tool_mem_on_g[n]
                <= int(inst.g_mem_cap[n])
                for n in range(inst.g_count)
            )
        )

        if warm_start:
            for i in range(inst.llm_count):
                for n in range(inst.g_count):
                    x[i, n].Start = 1.0 if warm_start[i] == n else 0.0

        w = article_objective_weights(inst)
        model.setObjective(
            gp.quicksum(
                sum(float(w[i, j]) * tool_distance(inst, n, int(tool_place[j])) for j in range(inst.tool_count))
                * x[i, n]
                for i in range(inst.llm_count)
                for n in range(inst.g_count)
            ),
            GRB.MINIMIZE,
        )
        model.optimize()
        elapsed = time.perf_counter() - start

        if model.SolCount > 0:
            place = [max(range(inst.g_count), key=lambda n: x[i, n].X) for i in range(inst.llm_count)]
        else:
            place = warm_start.copy() if warm_start and is_llm_feasible(inst, warm_start) else resource_greedy_llm_assignment(inst)

        status_name = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(model.Status, str(model.Status))
        return place, {
            "phase": "llm",
            "solver": "gurobi_llm_mip_fixed_tools",
            "status": status_name,
            "proven_optimal": model.Status == GRB.OPTIMAL,
            "objective_value": float(model.ObjVal) if model.SolCount > 0 else math.nan,
            "best_objective_bound": float(model.ObjBound) if model.SolCount > 0 else math.nan,
            "relative_gap": float(model.MIPGap) if model.SolCount > 0 and math.isfinite(model.MIPGap) else math.nan,
            "runtime_s": float(model.Runtime),
            "solve_time_s": elapsed,
            "node_count": float(model.NodeCount),
            "x_variables": inst.llm_count * inst.g_count,
            "objective_terms": inst.llm_count * inst.g_count,
            "timeout_s": timeout_s,
        }
    except Exception as exc:
        fallback = warm_start.copy() if warm_start and is_llm_feasible(inst, warm_start) else resource_greedy_llm_assignment(inst)
        return fallback, {
            "phase": "llm",
            "solver": "gurobi_llm_mip_fixed_tools",
            "status": "ERROR",
            "error": str(exc),
            "solve_time_s": time.perf_counter() - start,
        }


def initial_candidates(inst, seed: int, random_starts: int) -> list[tuple[list[int], list[int], str]]:
    rng = random.Random(seed + 9173)
    candidates: list[tuple[list[int], list[int], str]] = []

    def add_llm(llm_place: list[int], label: str) -> None:
        if not is_llm_feasible(inst, llm_place):
            return
        tool_place = iwc_gs_tool_deployment(inst, llm_place)
        if is_deployment_feasible(inst, llm_place, tool_place):
            candidates.append((llm_place, tool_place, label))

    warm_llm, warm_tool, warm_extra = greedy_original_warm_start(inst, samples=max(8, random_starts))
    candidates.append((warm_llm, warm_tool, f"greedy_warm:{warm_extra.get('warm_start')}"))
    add_llm(resource_greedy_llm_assignment(inst), "resource_greedy")
    for idx in range(random_starts):
        try:
            add_llm(random_llm_assignment(inst, rng), f"random_{idx}")
        except RuntimeError:
            continue

    seen = set()
    unique: list[tuple[list[int], list[int], str]] = []
    for llm_place, tool_place, label in candidates:
        key = (tuple(llm_place), tuple(tool_place))
        if key in seen:
            continue
        seen.add(key)
        unique.append((llm_place, tool_place, label))
    unique.sort(key=lambda item: average_call_distance(inst, item[0], item[1]))
    return unique


def gurobi_alternating_solver(
    inst,
    seed: int,
    total_timeout_s: float = 180.0,
    subproblem_timeout_s: float = 30.0,
    max_rounds: int = 8,
    random_starts: int = 8,
) -> AlgorithmResult:
    start = time.perf_counter()
    best_llm: list[int] | None = None
    best_tool: list[int] | None = None
    best_distance = math.inf
    trace: list[dict] = []
    starts = initial_candidates(inst, seed, random_starts)

    def remaining_time() -> float:
        return max(0.0, total_timeout_s - (time.perf_counter() - start))

    for start_idx, (llm_place, tool_place, label) in enumerate(starts):
        if remaining_time() <= 1e-3:
            break
        current_distance = average_call_distance(inst, llm_place, tool_place)
        if current_distance < best_distance and is_deployment_feasible(inst, llm_place, tool_place):
            best_distance = current_distance
            best_llm = llm_place.copy()
            best_tool = tool_place.copy()
        trace.append(
            {
                "start": start_idx,
                "round": -1,
                "phase": "init",
                "label": label,
                "avg_call_distance": current_distance,
                "elapsed_s": time.perf_counter() - start,
            }
        )

        no_improve_steps = 0
        for round_idx in range(max_rounds):
            if remaining_time() <= 1e-3:
                break
            phase_timeout = min(subproblem_timeout_s, remaining_time())
            tool_place, tool_extra = gurobi_tool_deployment(
                inst,
                llm_place,
                timeout_s=phase_timeout,
                warm_start=tool_place,
            )
            distance_after_tool = average_call_distance(inst, llm_place, tool_place)
            improved = distance_after_tool < best_distance - 1e-10 and is_deployment_feasible(inst, llm_place, tool_place)
            if improved:
                best_distance = distance_after_tool
                best_llm = llm_place.copy()
                best_tool = tool_place.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            trace.append(
                {
                    "start": start_idx,
                    "round": round_idx,
                    "phase": "tool",
                    "status": tool_extra.get("phase2_solver_status"),
                    "avg_call_distance": distance_after_tool,
                    "elapsed_s": time.perf_counter() - start,
                    "runtime_s": tool_extra.get("phase2_runtime_s"),
                    "gap": tool_extra.get("phase2_relative_gap"),
                }
            )

            if remaining_time() <= 1e-3:
                break
            phase_timeout = min(subproblem_timeout_s, remaining_time())
            llm_place, llm_extra = gurobi_llm_deployment_fixed_tools(
                inst,
                tool_place,
                timeout_s=phase_timeout,
                warm_start=llm_place,
            )
            distance_after_llm = average_call_distance(inst, llm_place, tool_place)
            improved = distance_after_llm < best_distance - 1e-10 and is_deployment_feasible(inst, llm_place, tool_place)
            if improved:
                best_distance = distance_after_llm
                best_llm = llm_place.copy()
                best_tool = tool_place.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            trace.append(
                {
                    "start": start_idx,
                    "round": round_idx,
                    "phase": "llm",
                    "status": llm_extra.get("status"),
                    "avg_call_distance": distance_after_llm,
                    "elapsed_s": time.perf_counter() - start,
                    "runtime_s": llm_extra.get("runtime_s"),
                    "gap": llm_extra.get("relative_gap"),
                }
            )
            if no_improve_steps >= 2:
                break

    elapsed = time.perf_counter() - start
    if best_llm is None or best_tool is None:
        best_llm, best_tool, _ = greedy_original_warm_start(inst, samples=max(8, random_starts))
        best_distance = average_call_distance(inst, best_llm, best_tool)

    status = "TIME_LIMIT" if elapsed >= total_timeout_s - 1e-3 else "LOCAL_OPTIMUM"
    return AlgorithmResult(
        name="Gurobi alternating fix-and-optimize",
        decision_count=inst.llm_count + inst.tool_count,
        avg_call_distance=best_distance,
        solve_time_s=elapsed,
        feasible=is_deployment_feasible(inst, best_llm, best_tool),
        assignment=best_llm + best_tool,
        extra={
            "solver_status": status,
            "total_timeout_s": total_timeout_s,
            "subproblem_timeout_s": subproblem_timeout_s,
            "starts_generated": len(starts),
            "trace": trace,
            "rounds_completed": max((item["round"] for item in trace), default=-1) + 1,
            "x_variables_per_llm_phase": inst.llm_count * inst.g_count,
            "y_variables_per_tool_phase": inst.tool_count * tool_host_count(inst),
            "llm_objective_terms_per_phase": inst.llm_count * inst.g_count,
            "tool_objective_terms_per_phase": inst.tool_count * tool_host_count(inst),
            "is_direct_original_heuristic": True,
            "note": "Alternates exact Gurobi MIP subproblems on the original objective: fixed LLM -> Tool-MIP, fixed Tools -> LLM-MIP.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llms", type=int, default=60)
    parser.add_argument("--tool-ratio", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=20260528)
    parser.add_argument("--total-timeout-s", type=float, default=180.0)
    parser.add_argument("--subproblem-timeout-s", type=float, default=30.0)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--random-starts", type=int, default=8)
    parser.add_argument("--capacity-mode", choices=["fixed_per_server", "g_only_fixed_per_server"], default="fixed_per_server")
    parser.add_argument("--out-dir", type=Path, default=Path("jolt_gurobi_alternating_L60_3min"))
    args = parser.parse_args()

    tools = args.llms * args.tool_ratio
    seed = args.seed_base + args.llms * 1000
    inst = make_instance(
        seed,
        llm_count=args.llms,
        tool_count=tools,
        g_count=None,
        c_count=None,
        capacity_mode=args.capacity_mode,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / f"instance_L{args.llms}.json").open("w", encoding="utf-8") as f:
        json.dump(instance_to_jsonable(inst), f)

    print(
        f"Scale L={args.llms}, T={tools}, G={inst.g_count}, C={inst.c_count}, H={tool_host_count(inst)}",
        flush=True,
    )
    result = gurobi_alternating_solver(
        inst,
        seed,
        total_timeout_s=args.total_timeout_s,
        subproblem_timeout_s=args.subproblem_timeout_s,
        max_rounds=args.max_rounds,
        random_starts=args.random_starts,
    )
    row = {
        "llms": args.llms,
        "tools": tools,
        "g_servers": inst.g_count,
        "c_servers": inst.c_count,
        "tool_hosts": tool_host_count(inst),
        "method": result.name,
        "decision_parameters": result.decision_count,
        "actual_parameters": (
            f"per_tool_phase_y={result.extra['y_variables_per_tool_phase']};"
            f"per_llm_phase_x={result.extra['x_variables_per_llm_phase']};"
            f"starts={result.extra['starts_generated']}"
        ),
        "avg_call_distance": result.avg_call_distance if result.feasible else "NaN",
        "solve_time_s": result.solve_time_s,
        "status": result.extra["solver_status"],
        "feasible": result.feasible,
        "extra": json.dumps(result.extra, ensure_ascii=False),
    }
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    with (args.out_dir / "trace.json").open("w", encoding="utf-8") as f:
        json.dump(result.extra["trace"], f, ensure_ascii=False, indent=2)
    print(
        f"{result.name}: status={row['status']} feasible={result.feasible} "
        f"distance={result.avg_call_distance:.12g} time={result.solve_time_s:.3f}s",
        flush=True,
    )
    print(f"Wrote {(args.out_dir / 'summary.csv').resolve()}", flush=True)


if __name__ == "__main__":
    main()
