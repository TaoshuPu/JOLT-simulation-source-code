from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from jolt_small_scale_experiment import run_trial


def summarize(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    keys = sorted({(int(row["llms"]), int(row["tools"]), row["method"]) for row in rows})
    for llms, tools, method in keys:
        group = [row for row in rows if int(row["llms"]) == llms and int(row["tools"]) == tools and row["method"] == method]
        distances = [float(row["avg_call_distance"]) for row in group]
        times = [float(row["solve_time_s"]) for row in group]
        proven_count = sum(1 for row in group if str(row["exact_proven_optimal"]) == "True")
        gaps = [float(row["gap_to_exact_percent"]) for row in group if row["gap_to_exact_percent"] != ""]
        summary.append(
            {
                "llms": llms,
                "tools": tools,
                "method": method,
                "decision_count": group[0]["decision_count"],
                "trials": len(group),
                "mean_avg_call_distance": statistics.mean(distances),
                "std_avg_call_distance": statistics.pstdev(distances) if len(distances) > 1 else 0.0,
                "mean_solve_time_s": statistics.mean(times),
                "std_solve_time_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
                "mean_gap_to_exact_percent": statistics.mean(gaps) if gaps else math.nan,
                "exact_proven_trials": proven_count if str(group[0]["is_exact_baseline"]) == "True" else "",
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict], emit=print) -> None:
    emit("\nScale sweep summary")
    emit(
        f"{'LLMs':>4} {'Tools':>5} {'Method':<28} {'Decisions':>9} "
        f"{'Distance':>11} {'Time(s)':>10} {'Gap %':>9} {'Exact proven':>12}"
    )
    for row in summary:
        gap = row["mean_gap_to_exact_percent"]
        gap_text = "NA" if isinstance(gap, float) and math.isnan(gap) else f"{gap:.3f}"
        emit(
            f"{row['llms']:>4} {row['tools']:>5} {row['method']:<28} {row['decision_count']:>9} "
            f"{row['mean_avg_call_distance']:>11.6f} {row['mean_solve_time_s']:>10.3f} "
            f"{gap_text:>9} {str(row['exact_proven_trials']):>12}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JOLT scale sweep experiments.")
    parser.add_argument("--llm-list", default="5,10,15,20")
    parser.add_argument("--tool-ratio", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--g-servers", type=int, default=4)
    parser.add_argument("--c-servers", type=int, default=6)
    parser.add_argument("--capacity-mode", choices=["scaled_capacity", "fixed_per_server"], default="scaled_capacity")
    parser.add_argument("--exact-timeout-s", type=float, default=30.0)
    parser.add_argument("--exact-solver", choices=["cpsat", "gurobi", "bnb"], default="cpsat")
    parser.add_argument("--ga-pop-size", type=int, default=90)
    parser.add_argument("--ga-generations", type=int, default=220)
    parser.add_argument("--split-pop-size", type=int, default=24)
    parser.add_argument("--split-generations", type=int, default=45)
    parser.add_argument("--out-dir", type=Path, default=Path("jolt_scale_sweep_outputs"))
    parser.add_argument("--progress-log", type=Path)
    args = parser.parse_args()

    llm_values = [int(item.strip()) for item in args.llm_list.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = None
    if args.progress_log is not None:
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        progress_file = args.progress_log.open("w", encoding="utf-8")

    def emit(message: str = "") -> None:
        print(message, flush=True)
        if progress_file is not None:
            progress_file.write(message + "\n")
            progress_file.flush()

    rows: list[dict] = []

    try:
        for llms in llm_values:
            tools = llms * args.tool_ratio
            for trial in range(args.trials):
                seed = args.seed + llms * 1000 + trial
                emit(f"\nScale L={llms}, S={tools}, trial {trial + 1}/{args.trials}, seed={seed}")
                _, results = run_trial(
                    seed,
                    llm_count=llms,
                    tool_count=tools,
                    g_count=args.g_servers,
                    c_count=args.c_servers,
                    exact_timeout_s=args.exact_timeout_s,
                    exact_solver=args.exact_solver,
                    ga_pop_size=args.ga_pop_size,
                    ga_generations=args.ga_generations,
                    split_pop_size=args.split_pop_size,
                    split_generations=args.split_generations,
                    capacity_mode=args.capacity_mode,
                )
                exact_result = next(result for result in results if result.extra.get("is_exact_baseline"))
                exact_distance = exact_result.avg_call_distance
                exact_proven = bool(exact_result.extra.get("proven_optimal", False))

                for result in results:
                    gap = ""
                    if exact_proven and math.isfinite(exact_distance):
                        gap = 100.0 * (result.avg_call_distance - exact_distance) / exact_distance
                    row = {
                        "llms": llms,
                        "tools": tools,
                        "trial": trial,
                        "seed": seed,
                        "method": result.name,
                        "decision_count": result.decision_count,
                        "avg_call_distance": result.avg_call_distance,
                        "solve_time_s": result.solve_time_s,
                        "gap_to_exact_percent": gap,
                        "feasible": result.feasible,
                        "exact_proven_optimal": exact_proven,
                        "is_exact_baseline": bool(result.extra.get("is_exact_baseline", False)),
                        "assignment": json.dumps(result.assignment),
                        "extra": json.dumps(result.extra),
                    }
                    rows.append(row)
                    note = " best-so-far" if result.extra.get("is_exact_baseline") and not exact_proven else ""
                    gap_text = "" if gap == "" else f" gap={gap:.3f}%"
                    emit(
                        f"  {result.name:<28} distance={result.avg_call_distance:.6f} "
                        f"time={result.solve_time_s:.3f}s{gap_text}{note}"
                    )
    finally:
        if progress_file is not None:
            progress_file.flush()

    result_fields = [
        "llms",
        "tools",
        "trial",
        "seed",
        "method",
        "decision_count",
        "avg_call_distance",
        "solve_time_s",
        "gap_to_exact_percent",
        "feasible",
        "exact_proven_optimal",
        "is_exact_baseline",
        "assignment",
        "extra",
    ]
    summary = summarize(rows)
    summary_fields = [
        "llms",
        "tools",
        "method",
        "decision_count",
        "trials",
        "mean_avg_call_distance",
        "std_avg_call_distance",
        "mean_solve_time_s",
        "std_solve_time_s",
        "mean_gap_to_exact_percent",
        "exact_proven_trials",
    ]
    write_csv(args.out_dir / "scale_trial_results.csv", rows, result_fields)
    write_csv(args.out_dir / "scale_summary.csv", summary, summary_fields)
    print_summary(summary, emit=emit)
    emit(f"\nWrote results to {args.out_dir.resolve()}")
    if progress_file is not None:
        progress_file.close()


if __name__ == "__main__":
    main()
