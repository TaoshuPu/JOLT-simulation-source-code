from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from jolt_small_scale_experiment import (
    cpsat_original_solver,
    gurobi_original_solver,
    make_instance,
    proposed_two_stage,
)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def print_summary(summary: list[dict]) -> None:
    print("\nGurobi / CP-SAT / Proposed summary")
    print(
        f"{'LLMs':>4} {'Tools':>5} {'G':>3} {'C':>3} {'Method':<30} {'Decisions':>9} "
        f"{'Distance':>11} {'Time(s)':>10} {'Gap %':>9} {'Optimal':>8}"
    )
    for row in summary:
        gap = row["mean_gap_to_gurobi_percent"]
        gap_text = "NA" if isinstance(gap, float) and math.isnan(gap) else f"{gap:.3f}"
        print(
            f"{row['llms']:>4} {row['tools']:>5} {row['g_servers']:>3} {row['c_servers']:>3} "
            f"{row['method']:<30} {row['decision_count']:>9} "
            f"{row['mean_avg_call_distance']:>11.6f} {row['mean_solve_time_s']:>10.3f} "
            f"{gap_text:>9} {row['proven_optimal_trials']:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gurobi, OR-Tools CP-SAT, and proposed two-stage JOLT split.")
    parser.add_argument("--llm-list", default="5,10,15,20")
    parser.add_argument("--tool-ratio", type=int, default=3)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--g-servers", type=int, default=4)
    parser.add_argument("--c-servers", type=int, default=6)
    parser.add_argument("--capacity-mode", choices=["scaled_capacity", "fixed_per_server"], default="scaled_capacity")
    parser.add_argument("--gurobi-timeout-s", type=float, default=30.0)
    parser.add_argument("--cpsat-timeout-s", type=float, default=30.0)
    parser.add_argument("--phase1-timeout-s", type=float, default=30.0)
    parser.add_argument("--split-pop-size", type=int, default=24)
    parser.add_argument("--split-generations", type=int, default=45)
    parser.add_argument("--out-dir", type=Path, default=Path("jolt_gurobi_cpsat_outputs"))
    args = parser.parse_args()

    llm_values = [int(item.strip()) for item in args.llm_list.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for llms in llm_values:
        tools = llms * args.tool_ratio
        for trial in range(args.trials):
            seed = args.seed + llms * 1000 + trial
            print(f"\nScale L={llms}, S={tools}, trial {trial + 1}/{args.trials}, seed={seed}", flush=True)
            inst = make_instance(
                seed,
                llm_count=llms,
                tool_count=tools,
                g_count=None if args.capacity_mode == "fixed_per_server" else args.g_servers,
                c_count=None if args.capacity_mode == "fixed_per_server" else args.c_servers,
                capacity_mode=args.capacity_mode,
            )

            results = [
                gurobi_original_solver(inst, timeout_s=args.gurobi_timeout_s),
                cpsat_original_solver(inst, timeout_s=args.cpsat_timeout_s, warm_start=True),
                proposed_two_stage(
                    inst,
                    seed + 10_000,
                    split_pop_size=args.split_pop_size,
                    split_generations=args.split_generations,
                    phase1_timeout_s=args.phase1_timeout_s,
                ),
            ]
            gurobi = results[0]
            gurobi_proven = bool(gurobi.extra.get("proven_optimal", False))
            gurobi_distance = gurobi.avg_call_distance

            for result in results:
                gap = ""
                if gurobi_proven and math.isfinite(gurobi_distance):
                    gap = 100.0 * (result.avg_call_distance - gurobi_distance) / gurobi_distance
                status = result.extra.get("solver_status", "")
                proven = bool(result.extra.get("proven_optimal", False))
                relative_gap = result.extra.get("relative_gap", "")
                row = {
                    "llms": llms,
                    "tools": tools,
                    "trial": trial,
                    "seed": seed,
                    "method": result.name,
                    "decision_count": result.decision_count,
                    "g_servers": inst.g_count,
                    "c_servers": inst.c_count,
                    "avg_call_distance": result.avg_call_distance,
                    "solve_time_s": result.solve_time_s,
                    "gap_to_gurobi_percent": gap,
                    "feasible": result.feasible,
                    "solver_status": status,
                    "proven_optimal": proven,
                    "relative_gap": relative_gap,
                    "assignment": json.dumps(result.assignment),
                    "extra": json.dumps(result.extra),
                }
                rows.append(row)
                gap_text = "" if gap == "" else f" gap={gap:.3f}%"
                status_text = f" status={status}" if status else ""
                opt_text = " optimal" if proven else ""
                print(
                    f"  {result.name:<30} distance={result.avg_call_distance:.6f} "
                    f"time={result.solve_time_s:.3f}s{gap_text}{status_text}{opt_text}",
                    flush=True,
                )

    result_fields = [
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
    summary = summarize(rows)
    summary_fields = [
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
    write_csv(args.out_dir / "trial_results.csv", rows, result_fields)
    write_csv(args.out_dir / "summary.csv", summary, summary_fields)
    print_summary(summary)
    print(f"\nWrote results to {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
