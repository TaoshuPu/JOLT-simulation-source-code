from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

from jolt_small_scale_experiment import (
    AlgorithmResult,
    cpsat_alternating_heuristic_solver,
    cpsat_original_solver,
    gurobi_original_solver,
    instance_to_jsonable,
    make_instance,
    proposed_two_stage,
    proposed_two_stage_solver,
    scip_original_solver,
)


RESULT_FIELDS = [
    "llms",
    "tools",
    "trial",
    "seed",
    "method",
    "decision_count",
    "g_servers",
    "c_servers",
    "avg_call_distance",
    "solve_time_s",
    "gap_to_gurobi_percent",
    "feasible",
    "solver_status",
    "proven_optimal",
    "relative_gap",
    "assignment",
    "extra",
]

SUMMARY_FIELDS = [
    "llms",
    "tools",
    "method",
    "decision_count",
    "g_servers",
    "c_servers",
    "trials",
    "mean_avg_call_distance",
    "std_avg_call_distance",
    "mean_solve_time_s",
    "std_solve_time_s",
    "mean_gap_to_gurobi_percent",
    "proven_optimal_trials",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    keys = sorted({(int(row["llms"]), int(row["tools"]), row["method"]) for row in rows})
    for llms, tools, method in keys:
        group = [row for row in rows if int(row["llms"]) == llms and int(row["tools"]) == tools and row["method"] == method]
        distances = [float(row["avg_call_distance"]) for row in group]
        times = [float(row["solve_time_s"]) for row in group]
        gaps = [float(row["gap_to_gurobi_percent"]) for row in group if row["gap_to_gurobi_percent"] != ""]
        proven = sum(1 for row in group if str(row["proven_optimal"]) == "True")
        summary.append(
            {
                "llms": llms,
                "tools": tools,
                "method": method,
                "decision_count": group[0]["decision_count"],
                "g_servers": group[0].get("g_servers", ""),
                "c_servers": group[0].get("c_servers", ""),
                "trials": len(group),
                "mean_avg_call_distance": statistics.mean(distances),
                "std_avg_call_distance": statistics.pstdev(distances) if len(distances) > 1 else 0.0,
                "mean_solve_time_s": statistics.mean(times),
                "std_solve_time_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
                "mean_gap_to_gurobi_percent": statistics.mean(gaps) if gaps else math.nan,
                "proven_optimal_trials": proven,
            }
        )
    return summary


def result_to_row(
    result: AlgorithmResult,
    llms: int,
    tools: int,
    trial: int,
    seed: int,
    g_servers: int,
    c_servers: int,
    gurobi_result: AlgorithmResult | None,
) -> dict:
    gap = ""
    if (
        gurobi_result is not None
        and bool(gurobi_result.extra.get("proven_optimal", False))
        and math.isfinite(gurobi_result.avg_call_distance)
    ):
        gap = 100.0 * (result.avg_call_distance - gurobi_result.avg_call_distance) / gurobi_result.avg_call_distance
    return {
        "llms": llms,
        "tools": tools,
        "trial": trial,
        "seed": seed,
        "method": result.name,
        "decision_count": result.decision_count,
        "g_servers": g_servers,
        "c_servers": c_servers,
        "avg_call_distance": result.avg_call_distance,
        "solve_time_s": result.solve_time_s,
        "gap_to_gurobi_percent": gap,
        "feasible": result.feasible,
        "solver_status": result.extra.get("solver_status", ""),
        "proven_optimal": bool(result.extra.get("proven_optimal", False)),
        "relative_gap": result.extra.get("relative_gap", ""),
        "assignment": json.dumps(result.assignment),
        "extra": json.dumps(result.extra),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental large-scale JOLT comparison runner.")
    parser.add_argument("--llms", type=int, default=60)
    parser.add_argument("--tool-ratio", type=int, default=3)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=20260528)
    parser.add_argument("--capacity-mode", choices=["scaled_capacity", "fixed_per_server"], default="fixed_per_server")
    parser.add_argument("--gurobi-timeout-s", type=float, default=120.0)
    parser.add_argument("--scip-timeout-s", type=float, default=120.0)
    parser.add_argument("--cpsat-timeout-s", type=float, default=120.0)
    parser.add_argument("--phase1-timeout-s", type=float, default=120.0)
    parser.add_argument("--phase2-timeout-s", type=float, default=120.0)
    parser.add_argument("--methods", default="gurobi,cpsat,proposed")
    parser.add_argument("--out-dir", type=Path, default=Path("jolt_article_objective_L60_120s"))
    args = parser.parse_args()

    tools = args.llms * args.tool_ratio
    seed = args.seed_base + args.llms * 1000 + args.trial
    methods = [item.strip().lower() for item in args.methods.split(",") if item.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inst = make_instance(
        seed,
        llm_count=args.llms,
        tool_count=tools,
        g_count=None if args.capacity_mode == "fixed_per_server" else 4,
        c_count=None if args.capacity_mode == "fixed_per_server" else 6,
        capacity_mode=args.capacity_mode,
    )
    with (args.out_dir / "instance_meta.json").open("w", encoding="utf-8") as f:
        json.dump(instance_to_jsonable(inst), f)

    print(
        f"Scale L={args.llms}, S={tools}, G={inst.g_count}, C={inst.c_count}, "
        f"seed={seed}, methods={','.join(methods)}",
        flush=True,
    )

    result_path = args.out_dir / "trial_results.csv"
    summary_path = args.out_dir / "summary.csv"
    rows: list[dict] = read_existing_rows(result_path)
    gurobi_result: AlgorithmResult | None = None
    for method in methods:
        method_start = time.perf_counter()
        try:
            if method == "gurobi":
                result = gurobi_original_solver(inst, timeout_s=args.gurobi_timeout_s)
                gurobi_result = result
            elif method == "scip":
                result = scip_original_solver(inst, timeout_s=args.scip_timeout_s)
            elif method == "cpsat":
                result = cpsat_original_solver(inst, timeout_s=args.cpsat_timeout_s, warm_start=True)
            elif method in {"cpsat_alt", "cpsat-alternating", "cpsat_heuristic", "cpsat-heuristic"}:
                result = cpsat_alternating_heuristic_solver(inst, timeout_s=args.cpsat_timeout_s)
            elif method in {"proposed", "hisc", "split"}:
                result = proposed_two_stage(inst, seed + 10_000, phase1_timeout_s=args.phase1_timeout_s)
            elif method in {"proposed_solver", "gqap_tool_mip", "gqap+tool-mip", "jolt"}:
                result = proposed_two_stage_solver(
                    inst,
                    seed + 10_000,
                    phase1_timeout_s=args.phase1_timeout_s,
                    phase2_timeout_s=args.phase2_timeout_s,
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as exc:
            result = AlgorithmResult(
                name=method,
                decision_count=args.llms + tools,
                avg_call_distance=math.inf,
                solve_time_s=time.perf_counter() - method_start,
                feasible=False,
                assignment=[],
                extra={"solver_status": "RUNNER_ERROR", "error": repr(exc), "proven_optimal": False},
            )
        row = result_to_row(result, args.llms, tools, args.trial, seed, inst.g_count, inst.c_count, gurobi_result)
        rows = [
            existing
            for existing in rows
            if not (
                int(existing["llms"]) == args.llms
                and int(existing["tools"]) == tools
                and int(existing["trial"]) == args.trial
                and existing["method"] == row["method"]
            )
        ]
        rows.append(row)
        write_csv(result_path, rows, RESULT_FIELDS)
        write_csv(summary_path, summarize(rows), SUMMARY_FIELDS)
        print(
            f"  {result.name:<30} distance={result.avg_call_distance:.6f} "
            f"time={result.solve_time_s:.3f}s status={row['solver_status']}",
            flush=True,
        )

    print(f"Wrote incremental results to {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
