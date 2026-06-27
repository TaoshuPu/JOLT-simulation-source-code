from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

from jolt_small_scale_experiment import (
    AlgorithmResult,
    cpsat_alternating_heuristic_solver,
    cpsat_original_solver,
    gurobi_original_solver,
    hisc_ma_iwc_gs_solver,
    instance_to_jsonable,
    make_instance,
    proposed_two_stage_solver,
    scip_original_solver,
)


METHODS = [
    ("gurobi", "Gurobi MIP original"),
    ("scip", "SCIP MIQP original"),
    ("cpsat_orig", "OR-Tools CP-SAT original"),
    ("cpsat_alt", "OR-Tools CP-SAT alternating heuristic"),
    ("gqap_tool_mip", "JOLT"),
    ("hisc_iwc_gs", "HISC-MA + IWC-GS"),
]


FIELDS = [
    "llms",
    "tools",
    "g_servers",
    "c_servers",
    "tool_hosts",
    "method",
    "decision_parameters",
    "actual_parameters",
    "avg_call_distance",
    "solve_time_s",
    "status",
    "feasible",
    "extra",
]


def actual_parameters(result: AlgorithmResult) -> str:
    extra = result.extra
    x_vars = extra.get("x_variables")
    y_vars = extra.get("y_variables")
    q_terms = extra.get("quadratic_objective_terms")
    phase2_y = extra.get("phase2_y_variables")
    phase2_terms = extra.get("phase2_objective_terms")

    if result.name == "JOLT":
        return (
            f"phase1_x={extra.get('x_variables', 'NA')};"
            f"phase1_q_terms={q_terms};"
            f"phase2_y={phase2_y};"
            f"phase2_terms={phase2_terms}"
        )
    if result.name == "HISC-MA + IWC-GS":
        tuned = extra.get("tuned_params", {})
        return (
            f"phase1_x={extra.get('x_variables', 'NA')};"
            f"phase1_q_terms={q_terms};"
            f"pop={tuned.get('pop_size', 'NA')};"
            f"gen={tuned.get('max_gen', 'NA')};"
            f"restarts={extra.get('restarts_completed', tuned.get('restarts', 'NA'))};"
            f"phase2=iwc_gs"
        )
    if result.name == "OR-Tools CP-SAT alternating heuristic":
        infos = extra.get("phase_infos", [])
        max_tool_vars = max((p.get("variables", 0) for p in infos if p.get("phase") == "tool"), default="NA")
        max_llm_vars = max((p.get("variables", 0) for p in infos if p.get("phase") == "llm"), default="NA")
        return f"per_round_tool_vars={max_tool_vars};per_round_llm_vars={max_llm_vars};rounds={extra.get('rounds_completed')}"
    if result.name == "OR-Tools CP-SAT original":
        return (
            f"x={extra.get('x_variables', 'NA')};"
            f"y={extra.get('y_variables', 'NA')};"
            f"z={extra.get('z_variables', 'NA')};"
            f"top_tools={extra.get('top_tools_per_llm', 'full')}"
        )
    if x_vars is not None or y_vars is not None or q_terms is not None:
        return f"x={x_vars};y={y_vars};q_terms={q_terms}"
    return "NA"


def run_method_child(
    method_key: str,
    seed: int,
    llms: int,
    tools: int,
    solver_timeout_s: float,
    capacity_mode: str,
    queue: mp.Queue,
) -> None:
    start = time.perf_counter()
    try:
        inst = make_instance(
            seed,
            llm_count=llms,
            tool_count=tools,
            g_count=None,
            c_count=None,
            capacity_mode=capacity_mode,
        )
        if method_key == "gurobi":
            result = gurobi_original_solver(inst, timeout_s=solver_timeout_s)
        elif method_key == "scip":
            result = scip_original_solver(inst, timeout_s=solver_timeout_s)
        elif method_key == "cpsat_orig":
            result = cpsat_original_solver(
                inst,
                timeout_s=solver_timeout_s,
                warm_start=True,
                top_tools_per_llm=12,
            )
        elif method_key == "cpsat_alt":
            result = cpsat_alternating_heuristic_solver(inst, timeout_s=solver_timeout_s)
        elif method_key == "gqap_tool_mip":
            result = proposed_two_stage_solver(
                inst,
                seed + 10_000,
                phase1_timeout_s=solver_timeout_s,
                phase2_timeout_s=solver_timeout_s,
            )
        elif method_key == "hisc_iwc_gs":
            result = hisc_ma_iwc_gs_solver(inst, seed + 10_000, phase1_timeout_s=solver_timeout_s)
        else:
            raise ValueError(f"Unknown method: {method_key}")

        queue.put(
            {
                "ok": True,
                "result": {
                    "name": result.name,
                    "decision_count": result.decision_count,
                    "avg_call_distance": result.avg_call_distance,
                    "solve_time_s": result.solve_time_s,
                    "feasible": result.feasible,
                    "extra": result.extra,
                },
                "instance": {
                    "g_count": inst.g_count,
                    "c_count": inst.c_count,
                    "tool_hosts": inst.g_count + inst.c_count,
                },
            }
        )
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "error": repr(exc),
                "elapsed": time.perf_counter() - start,
            }
        )


def nan_row(llms: int, tools: int, inst_meta: dict, method_name: str, elapsed: float, status: str, extra: dict) -> dict:
    return {
        "llms": llms,
        "tools": tools,
        "g_servers": inst_meta["g_count"],
        "c_servers": inst_meta["c_count"],
        "tool_hosts": inst_meta["tool_hosts"],
        "method": method_name,
        "decision_parameters": llms + tools if method_name in {
            "Gurobi MIP original",
            "SCIP MIQP original",
            "OR-Tools CP-SAT original",
            "OR-Tools CP-SAT alternating heuristic",
        } else llms,
        "actual_parameters": "NA",
        "avg_call_distance": "NaN",
        "solve_time_s": elapsed,
        "status": status,
        "feasible": False,
        "extra": json.dumps(extra, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-list", default="20,40,60,80")
    parser.add_argument("--tool-ratio", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=20260528)
    parser.add_argument("--solver-timeout-s", type=float, default=180.0)
    parser.add_argument("--hard-timeout-s", type=float, default=240.0)
    parser.add_argument("--out-dir", type=Path, default=Path("jolt_mixed_scale_20_80_3min"))
    parser.add_argument("--methods", default=",".join(key for key, _ in METHODS))
    parser.add_argument(
        "--capacity-mode",
        choices=["fixed_per_server", "g_only_fixed_per_server"],
        default="fixed_per_server",
    )
    args = parser.parse_args()
    selected_methods = {x.strip() for x in args.methods.split(",") if x.strip()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "scale_summary.csv"
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as f:
            rows: list[dict] = list(csv.DictReader(f))
    else:
        rows = []

    ctx = mp.get_context("spawn")
    for llms in [int(x.strip()) for x in args.llm_list.split(",") if x.strip()]:
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
        inst_meta = {"g_count": inst.g_count, "c_count": inst.c_count, "tool_hosts": inst.g_count + inst.c_count}
        with (args.out_dir / f"instance_L{llms}.json").open("w", encoding="utf-8") as f:
            json.dump(instance_to_jsonable(inst), f)
        print(f"\nScale L={llms}, T={tools}, G={inst.g_count}, C={inst.c_count}, H={inst.g_count + inst.c_count}", flush=True)

        for method_key, method_name in METHODS:
            if method_key not in selected_methods:
                continue
            queue: mp.Queue = ctx.Queue()
            proc = ctx.Process(
                target=run_method_child,
                args=(method_key, seed, llms, tools, args.solver_timeout_s, args.capacity_mode, queue),
            )
            print(f"  running {method_name} ...", flush=True)
            start = time.perf_counter()
            proc.start()
            proc.join(args.hard_timeout_s)
            elapsed = time.perf_counter() - start
            if proc.is_alive():
                proc.terminate()
                proc.join(10)
                row = nan_row(
                    llms,
                    tools,
                    inst_meta,
                    method_name,
                    elapsed,
                    "HARD_TIMEOUT",
                    {"hard_timeout_s": args.hard_timeout_s, "solver_timeout_s": args.solver_timeout_s},
                )
            else:
                payload = queue.get() if not queue.empty() else {"ok": False, "error": "NO_RESULT", "elapsed": elapsed}
                if not payload.get("ok"):
                    row = nan_row(
                        llms,
                        tools,
                        inst_meta,
                        method_name,
                        float(payload.get("elapsed", elapsed)),
                        "ERROR",
                        {"error": payload.get("error"), "solver_timeout_s": args.solver_timeout_s},
                    )
                else:
                    result_data = payload["result"]
                    extra = result_data["extra"]
                    light_result = AlgorithmResult(
                        name=result_data["name"],
                        decision_count=int(result_data["decision_count"]),
                        avg_call_distance=float(result_data["avg_call_distance"]),
                        solve_time_s=float(result_data["solve_time_s"]),
                        feasible=bool(result_data["feasible"]),
                        assignment=[],
                        extra=extra,
                    )
                    row = {
                        "llms": llms,
                        "tools": tools,
                        "g_servers": inst_meta["g_count"],
                        "c_servers": inst_meta["c_count"],
                        "tool_hosts": inst_meta["tool_hosts"],
                        "method": result_data["name"],
                        "decision_parameters": result_data["decision_count"],
                        "actual_parameters": actual_parameters(light_result),
                        "avg_call_distance": result_data["avg_call_distance"] if result_data["feasible"] and math.isfinite(float(result_data["avg_call_distance"])) else "NaN",
                        "solve_time_s": result_data["solve_time_s"],
                        "status": extra.get("solver_status", ""),
                        "feasible": result_data["feasible"],
                        "extra": json.dumps(extra, ensure_ascii=False),
                    }
            rows.append(row)
            rows = [
                existing
                for existing in rows
                if not (
                    int(existing["llms"]) == llms
                    and existing["method"] == row["method"]
                )
            ] + [row]
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(
                f"    {row['status']} distance={row['avg_call_distance']} time={float(row['solve_time_s']):.3f}s",
                flush=True,
            )

    print(f"\nWrote {summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
