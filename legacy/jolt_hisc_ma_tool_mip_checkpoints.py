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

from jolt_small_scale_experiment import (
    average_call_distance,
    gurobi_tool_deployment,
    hisc_ma_llm_deployment,
    instance_to_jsonable,
    is_deployment_feasible,
    is_llm_feasible,
    is_tool_feasible,
    iwc_gs_tool_deployment,
    llm_similarity,
    llm_surrogate_cost,
    make_instance,
    random_llm_assignment,
    tool_host_count,
)


FIELDS = [
    "llms",
    "tools",
    "checkpoint_s",
    "checkpoint_label",
    "g_servers",
    "c_servers",
    "tool_hosts",
    "method",
    "decision_parameters",
    "actual_parameters",
    "avg_call_distance",
    "solve_time_s",
    "phase1_elapsed_s",
    "phase2_solve_time_s",
    "phase1_preview_iwc_gs_distance",
    "phase1_surrogate_cost",
    "status",
    "feasible",
    "batches_completed",
    "incumbent_elapsed_s",
    "extra",
]


def checkpoint_label(seconds: float) -> str:
    return f"{seconds / 60:g}min"


def finite(value: object) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def write_csv(path: Path, rows: list[dict]) -> None:
    rows.sort(key=lambda r: (int(r["llms"]), float(r["checkpoint_s"])))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def params_for_scale(llms: int, batch_time_s: float | None) -> dict:
    if llms <= 20:
        params = {"pop_size": 160, "local_iter": 6, "batch_time_s": 15.0}
    elif llms <= 40:
        params = {"pop_size": 144, "local_iter": 5, "batch_time_s": 20.0}
    elif llms <= 60:
        params = {"pop_size": 128, "local_iter": 4, "batch_time_s": 25.0}
    else:
        params = {"pop_size": 128, "local_iter": 4, "batch_time_s": 30.0}
    if batch_time_s is not None:
        params["batch_time_s"] = float(batch_time_s)
    params.update(
        {
            "max_gen": 100_000,
            "pc": 0.92,
            "pm": 0.14 if llms <= 60 else 0.16,
            "late_pm": 0.35,
            "late_mutation_start": 0.72,
        }
    )
    return params


def actual_parameters(inst, params: dict) -> str:
    h_count = tool_host_count(inst)
    q_terms = inst.llm_count * inst.llm_count * inst.g_count * max(0, inst.g_count - 1)
    return (
        f"phase1_x={inst.llm_count * inst.g_count};"
        f"phase1_q_terms={q_terms};"
        f"pop={params['pop_size']};batch_time={params['batch_time_s']};"
        f"local_iter={params['local_iter']};selection=best_iwc_gs_preview;"
        f"phase2_y={inst.tool_count * h_count};phase2_terms={inst.tool_count * h_count}"
    )


def snapshot_from_incumbent(
    *,
    checkpoint_s: float,
    best_llm: list[int],
    best_tool_preview: list[int],
    best_preview_distance: float,
    best_surrogate: float,
    best_elapsed_s: float,
    batches_completed: int,
    phase1_status: str,
    best_source: dict,
) -> dict:
    return {
        "checkpoint_s": float(checkpoint_s),
        "llm_place": list(best_llm),
        "iwc_gs_tool_place": list(best_tool_preview),
        "phase1_preview_iwc_gs_distance": float(best_preview_distance),
        "phase1_surrogate_cost": float(best_surrogate),
        "incumbent_elapsed_s": float(best_elapsed_s),
        "batches_completed": int(batches_completed),
        "phase1_status": phase1_status,
        "best_source": best_source,
    }


def run_hisc_ma_phase(inst, seed: int, checkpoints: list[float], params: dict) -> tuple[list[dict], list[dict]]:
    start = time.perf_counter()
    rng = random.Random(seed + 91)
    sim = llm_similarity(inst)

    initial_llm = random_llm_assignment(inst, rng)
    initial_tool = iwc_gs_tool_deployment(inst, initial_llm)
    best_llm = initial_llm
    best_tool_preview = initial_tool
    best_preview_distance = average_call_distance(inst, initial_llm, initial_tool)
    best_surrogate = llm_surrogate_cost(inst, initial_llm, sim)
    best_elapsed_s = 0.0
    best_source = {"source": "random_initial"}

    batch_records: list[dict] = []
    snapshots: list[dict] = []
    batches_completed = 0
    phase1_status = "RUNNING"

    for checkpoint_s in checkpoints:
        while True:
            elapsed = time.perf_counter() - start
            remaining_to_checkpoint = checkpoint_s - elapsed
            if remaining_to_checkpoint <= 0.15:
                break
            budget = min(float(params["batch_time_s"]), max(0.25, remaining_to_checkpoint))
            batch_seed = seed + 104_729 * (batches_completed + 1)
            llm_place, extra = hisc_ma_llm_deployment(
                inst,
                batch_seed,
                pop_size=int(params["pop_size"]),
                max_gen=int(params["max_gen"]),
                pc=float(params["pc"]),
                pm=float(params["pm"]),
                timeout_s=budget,
                local_iter=int(params["local_iter"]),
                late_pm=float(params["late_pm"]),
                late_mutation_start=float(params["late_mutation_start"]),
            )
            batches_completed += 1
            elapsed = time.perf_counter() - start
            if not is_llm_feasible(inst, llm_place):
                batch_records.append(
                    {
                        "batch": batches_completed,
                        "elapsed_s": elapsed,
                        "batch_budget_s": budget,
                        "status": "INFEASIBLE_LLM",
                        "solver_status": extra.get("solver_status"),
                    }
                )
                continue

            preview_tool = iwc_gs_tool_deployment(inst, llm_place)
            preview_feasible = is_tool_feasible(inst, preview_tool, llm_place)
            preview_distance = (
                average_call_distance(inst, llm_place, preview_tool) if preview_feasible else math.inf
            )
            surrogate = llm_surrogate_cost(inst, llm_place, sim)
            improved = preview_distance < best_preview_distance - 1e-12
            if improved:
                best_llm = list(llm_place)
                best_tool_preview = list(preview_tool)
                best_preview_distance = float(preview_distance)
                best_surrogate = float(surrogate)
                best_elapsed_s = float(elapsed)
                best_source = {
                    "source": "hisc_ma_batch",
                    "batch": batches_completed,
                    "batch_seed": batch_seed,
                    "batch_solver_status": extra.get("solver_status"),
                    "batch_solve_time_s": extra.get("solve_time_s"),
                    "batch_generations_completed": extra.get("generations_completed"),
                }
            batch_records.append(
                {
                    "batch": batches_completed,
                    "elapsed_s": elapsed,
                    "batch_budget_s": budget,
                    "batch_seed": batch_seed,
                    "solver_status": extra.get("solver_status"),
                    "generations_completed": extra.get("generations_completed"),
                    "candidate_preview_iwc_gs_distance": preview_distance,
                    "candidate_surrogate_cost": surrogate,
                    "best_preview_iwc_gs_distance": best_preview_distance,
                    "best_surrogate_cost": best_surrogate,
                    "improved": improved,
                }
            )

        phase1_status = "TIME_LIMIT" if time.perf_counter() - start >= checkpoint_s - 0.25 else "CHECKPOINT_REACHED"
        snapshots.append(
            snapshot_from_incumbent(
                checkpoint_s=checkpoint_s,
                best_llm=best_llm,
                best_tool_preview=best_tool_preview,
                best_preview_distance=best_preview_distance,
                best_surrogate=best_surrogate,
                best_elapsed_s=best_elapsed_s,
                batches_completed=batches_completed,
                phase1_status=phase1_status,
                best_source=best_source,
            )
        )
        print(
            f"    HISC checkpoint {checkpoint_label(checkpoint_s)}: "
            f"preview={best_preview_distance:.6f}, surrogate={best_surrogate:.6f}, "
            f"batches={batches_completed}, incumbent_at={best_elapsed_s:.2f}s",
            flush=True,
        )

    return snapshots, batch_records


def solve_tool_mip_rows(inst, params: dict, snapshots: list[dict], phase2_timeout_s: float) -> list[dict]:
    rows: list[dict] = []
    for snap in snapshots:
        checkpoint_s = float(snap["checkpoint_s"])
        llm_place = [int(x) for x in snap["llm_place"]]
        warm_tool = [int(x) for x in snap["iwc_gs_tool_place"]]
        phase2_start = time.perf_counter()
        tool_place, tool_extra = gurobi_tool_deployment(
            inst,
            llm_place,
            timeout_s=phase2_timeout_s,
            warm_start=warm_tool,
        )
        phase2_wall = time.perf_counter() - phase2_start
        avg = average_call_distance(inst, llm_place, tool_place)
        feasible = is_deployment_feasible(inst, llm_place, tool_place)
        phase2_time = float(tool_extra.get("phase2_solve_time_s", phase2_wall))
        rows.append(
            {
                "llms": inst.llm_count,
                "tools": inst.tool_count,
                "checkpoint_s": checkpoint_s,
                "checkpoint_label": checkpoint_label(checkpoint_s),
                "g_servers": inst.g_count,
                "c_servers": inst.c_count,
                "tool_hosts": tool_host_count(inst),
                "method": "HISC-MA + Tool-MIP",
                "decision_parameters": inst.llm_count,
                "actual_parameters": actual_parameters(inst, params),
                "avg_call_distance": avg if feasible else "NaN",
                "solve_time_s": checkpoint_s + phase2_time,
                "phase1_elapsed_s": checkpoint_s,
                "phase2_solve_time_s": phase2_time,
                "phase1_preview_iwc_gs_distance": snap["phase1_preview_iwc_gs_distance"],
                "phase1_surrogate_cost": snap["phase1_surrogate_cost"],
                "status": f"PHASE1_{snap['phase1_status']};PHASE2_{tool_extra.get('phase2_solver_status', 'UNKNOWN')}",
                "feasible": feasible,
                "batches_completed": snap["batches_completed"],
                "incumbent_elapsed_s": snap["incumbent_elapsed_s"],
                "extra": json.dumps(
                    {
                        "selection_rule": "best HISC-MA LLM incumbent by IWC-GS preview distance; Tool-MIP is final evaluator",
                        "phase1_snapshot": {
                            key: value
                            for key, value in snap.items()
                            if key not in {"llm_place", "iwc_gs_tool_place"}
                        },
                        "phase2_extra": tool_extra,
                        "phase2_wall_s": phase2_wall,
                        "iwc_gs_warm_start_distance": average_call_distance(inst, llm_place, warm_tool),
                    },
                    ensure_ascii=False,
                ),
            }
        )
        print(
            f"    Tool-MIP {checkpoint_label(checkpoint_s)}: "
            f"avg={avg:.6f}, status={tool_extra.get('phase2_solver_status')}, "
            f"phase2={phase2_time:.3f}s",
            flush=True,
        )
    return rows


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="HISC-MA first-stage checkpoints followed by Gurobi Tool-MIP.")
    parser.add_argument("--llm-list", default="20,40,60,80")
    parser.add_argument("--tool-ratio", type=int, default=3)
    parser.add_argument("--checkpoints-s", default="60,180,300,420,600")
    parser.add_argument("--seed-base", type=int, default=20260528)
    parser.add_argument("--capacity-mode", choices=["fixed_per_server", "g_only_fixed_per_server"], default="fixed_per_server")
    parser.add_argument("--phase2-timeout-s", type=float, default=60.0)
    parser.add_argument("--batch-time-s", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("jolt_hisc_ma_tool_mip_checkpoints_20_80_10min"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    llm_values = parse_int_list(args.llm_list)
    checkpoints = sorted(parse_float_list(args.checkpoints_s))
    all_rows: list[dict] = []
    summary_path = args.out_dir / "summary.csv"

    for llms in llm_values:
        tools = llms * args.tool_ratio
        seed = args.seed_base + llms * 1000
        inst = make_instance(
            seed,
            llm_count=llms,
            tool_count=tools,
            g_count=None,
            c_count=None,
            capacity_mode=args.capacity_mode,
        )
        params = params_for_scale(llms, args.batch_time_s)
        (args.out_dir / f"instance_L{llms}.json").write_text(
            json.dumps(instance_to_jsonable(inst), ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"\nScale L={llms}, T={tools}, G={inst.g_count}, C={inst.c_count}, "
            f"H={tool_host_count(inst)}, pop={params['pop_size']}, "
            f"batch={params['batch_time_s']:g}s",
            flush=True,
        )
        snapshots, batch_records = run_hisc_ma_phase(inst, seed + 10_000, checkpoints, params)
        trace = {
            "llms": llms,
            "tools": tools,
            "seed": seed,
            "params": params,
            "snapshots": snapshots,
            "batch_records": batch_records,
        }
        (args.out_dir / f"trace_L{llms}_hisc_ma_phase.json").write_text(
            json.dumps(trace, ensure_ascii=False),
            encoding="utf-8",
        )
        rows = solve_tool_mip_rows(inst, params, snapshots, args.phase2_timeout_s)
        all_rows.extend(rows)
        write_csv(summary_path, all_rows)
        print(
            "  scale results: "
            + ", ".join(
                f"{row['checkpoint_label']}={float(row['avg_call_distance']):.6f}"
                if finite(row["avg_call_distance"]) is not None
                else f"{row['checkpoint_label']}=NaN"
                for row in rows
            ),
            flush=True,
        )

    print(f"\nWrote {summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
