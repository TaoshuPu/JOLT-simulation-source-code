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
    cpsat_original_solver,
    gurobi_original_solver,
    hisc_ma_iwc_gs_solver,
    instance_to_jsonable,
    make_instance,
    proposed_two_stage_solver,
    scip_original_solver,
    tool_host_count,
)


METHODS = [
    ("gurobi", "Gurobi MIP original"),
    ("scip", "SCIP MIQP original"),
    ("cpsat", "OR-Tools CP-SAT original"),
    ("gqap_tool_mip", "JOLT"),
    ("hisc_iwc_gs", "HISC-MA + IWC-GS"),
]


FIELDS = [
    "llms",
    "tools",
    "timeout_s",
    "timeout_label",
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


def timeout_label(seconds: float) -> str:
    minutes = seconds / 60.0
    return f"{minutes:g}min"


def actual_parameters(result: AlgorithmResult) -> str:
    extra = result.extra
    if result.name == "OR-Tools CP-SAT original":
        return (
            f"x={extra.get('x_variables', 'NA')};"
            f"y={extra.get('y_variables', 'NA')};"
            f"z={extra.get('z_variables', 'NA')};"
            f"element_cost={extra.get('element_cost_variables', 'NA')};"
            f"formulation={extra.get('formulation', 'NA')}"
        )
    if result.name == "JOLT":
        return (
            f"phase1_x={extra.get('x_variables', 'NA')};"
            f"phase1_q_terms={extra.get('quadratic_objective_terms', 'NA')};"
            f"phase2_y={extra.get('phase2_y_variables', 'NA')};"
            f"phase2_terms={extra.get('phase2_objective_terms', 'NA')}"
        )
    if result.name == "HISC-MA + IWC-GS":
        tuned = extra.get("tuned_params", {})
        return (
            f"phase1_x={extra.get('x_variables', 'NA')};"
            f"phase1_q_terms={extra.get('quadratic_objective_terms', 'NA')};"
            f"pop={tuned.get('pop_size', 'NA')};"
            f"gen={tuned.get('max_gen', 'NA')};"
            f"restarts={extra.get('restarts_completed', tuned.get('restarts', 'NA'))};"
            f"phase2=iwc_gs"
        )
    x_vars = extra.get("x_variables")
    y_vars = extra.get("y_variables")
    q_terms = extra.get("quadratic_objective_terms")
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
        elif method_key == "cpsat":
            result = cpsat_original_solver(
                inst,
                timeout_s=solver_timeout_s,
                warm_start=True,
                formulation="element",
            )
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
                    "tool_hosts": tool_host_count(inst),
                },
            }
        )
    except Exception as exc:
        queue.put({"ok": False, "error": repr(exc), "elapsed": time.perf_counter() - start})


def nan_row(
    llms: int,
    tools: int,
    timeout_s: float,
    inst_meta: dict,
    method_name: str,
    elapsed: float,
    status: str,
    extra: dict,
) -> dict:
    return {
        "llms": llms,
        "tools": tools,
        "timeout_s": timeout_s,
        "timeout_label": timeout_label(timeout_s),
        "g_servers": inst_meta["g_count"],
        "c_servers": inst_meta["c_count"],
        "tool_hosts": inst_meta["tool_hosts"],
        "method": method_name,
        "decision_parameters": llms + tools if method_name in {
            "Gurobi MIP original",
            "SCIP MIQP original",
            "OR-Tools CP-SAT original",
        } else llms,
        "actual_parameters": "NA",
        "avg_call_distance": "NaN",
        "solve_time_s": elapsed,
        "status": status,
        "feasible": False,
        "extra": json.dumps(extra, ensure_ascii=False),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    rows.sort(key=lambda r: (int(r["llms"]), float(r["timeout_s"]), str(r["method"])))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def finite_distance(row: dict | None) -> float | None:
    if row is None:
        return None
    try:
        value = float(row["avg_call_distance"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def find_row(rows: list[dict], llms: int, timeout_s: float, method: str) -> dict | None:
    for row in rows:
        if (
            int(row["llms"]) == llms
            and abs(float(row["timeout_s"]) - timeout_s) < 1e-9
            and row["method"] == method
        ):
            return row
    return None


def find_previous_finite_row(
    rows: list[dict],
    llms: int,
    timeout_values: list[float],
    current_index: int,
    method: str,
) -> dict | None:
    for prev_index in range(current_index - 1, -1, -1):
        row = find_row(rows, llms, timeout_values[prev_index], method)
        if finite_distance(row) is not None:
            return row
    return None


def should_skip_after_stable(
    previous_row: dict | None,
    current_row: dict,
    rel_tol: float,
    abs_tol: float,
) -> tuple[bool, dict]:
    previous_distance = finite_distance(previous_row)
    current_distance = finite_distance(current_row)
    if previous_distance is None or current_distance is None:
        return False, {}
    improvement_abs = previous_distance - current_distance
    improvement_rel = improvement_abs / max(abs(previous_distance), 1e-12)
    stable = improvement_abs >= -1e-12 and (improvement_rel <= rel_tol or improvement_abs <= abs_tol)
    return stable, {
        "previous_timeout_s": float(previous_row["timeout_s"]) if previous_row else math.nan,
        "previous_avg_call_distance": previous_distance,
        "current_timeout_s": float(current_row["timeout_s"]),
        "current_avg_call_distance": current_distance,
        "improvement_abs": improvement_abs,
        "improvement_rel": improvement_rel,
        "stable_rel_tol": rel_tol,
        "stable_abs_tol": abs_tol,
    }


def should_skip_after_no_feasible(previous_row: dict | None, current_row: dict) -> tuple[bool, dict]:
    previous_distance = finite_distance(previous_row)
    current_distance = finite_distance(current_row)
    stable = previous_row is not None and previous_distance is None and current_distance is None
    return stable, {
        "previous_timeout_s": float(previous_row["timeout_s"]) if previous_row else math.nan,
        "previous_status": previous_row.get("status") if previous_row else None,
        "current_timeout_s": float(current_row["timeout_s"]),
        "current_status": current_row.get("status"),
    }


def make_stable_skip_row(base_row: dict, timeout_s: float, stable_extra: dict) -> dict:
    row = dict(base_row)
    row["timeout_s"] = timeout_s
    row["timeout_label"] = timeout_label(timeout_s)
    row["solve_time_s"] = 0.0
    row["status"] = f"SKIPPED_STABLE_FROM_{base_row['timeout_label']}"
    row["extra"] = json.dumps(
        {
            "skip_reason": "Previous two time budgets produced nearly unchanged objective.",
            "copied_from_timeout_s": float(base_row["timeout_s"]),
            **stable_extra,
        },
        ensure_ascii=False,
    )
    return row


def make_no_feasible_skip_row(base_row: dict, timeout_s: float, skip_extra: dict) -> dict:
    row = dict(base_row)
    row["timeout_s"] = timeout_s
    row["timeout_label"] = timeout_label(timeout_s)
    row["avg_call_distance"] = "NaN"
    row["solve_time_s"] = 0.0
    row["feasible"] = False
    row["status"] = f"SKIPPED_NO_FEASIBLE_AFTER_{base_row['timeout_label']}"
    row["extra"] = json.dumps(
        {
            "skip_reason": "Previous two time budgets both failed to produce a feasible solution.",
            "copied_from_timeout_s": float(base_row["timeout_s"]),
            **skip_extra,
        },
        ensure_ascii=False,
    )
    return row


def is_proven_done(row: dict) -> bool:
    status = str(row.get("status", ""))
    status_lower = status.lower()
    if status_lower == "optimal":
        return True
    return status == "OPTIMAL" or status == "PHASE1_OPTIMAL;PHASE2_OPTIMAL"


def make_proven_skip_row(base_row: dict, timeout_s: float) -> dict:
    row = dict(base_row)
    row["timeout_s"] = timeout_s
    row["timeout_label"] = timeout_label(timeout_s)
    row["solve_time_s"] = 0.0
    row["status"] = f"SKIPPED_PROVEN_FROM_{base_row['timeout_label']}"
    row["extra"] = json.dumps(
        {
            "skip_reason": "The method had already proven the relevant model optimal at a shorter time budget.",
            "copied_from_timeout_s": float(base_row["timeout_s"]),
            "copied_from_status": base_row.get("status"),
        },
        ensure_ascii=False,
    )
    return row


def add_future_skip_rows(
    rows: list[dict],
    llms: int,
    timeout_values: list[float],
    current_index: int,
    base_row: dict,
    make_row,
    skip_extra: dict | None,
    rerun_existing: bool,
) -> tuple[list[dict], list[dict]]:
    added: list[dict] = []
    for future_timeout in timeout_values[current_index + 1 :]:
        if find_row(rows, llms, future_timeout, base_row["method"]) and not rerun_existing:
            continue
        skip_row = make_row(base_row, future_timeout) if skip_extra is None else make_row(base_row, future_timeout, skip_extra)
        rows = [
            r
            for r in rows
            if not (
                int(r["llms"]) == llms
                and abs(float(r["timeout_s"]) - future_timeout) < 1e-9
                and r["method"] == skip_row["method"]
            )
        ] + [skip_row]
        added.append(skip_row)
    return rows, added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-list", default="20,40,60,80")
    parser.add_argument("--timeout-list", default="60,180,300,600")
    parser.add_argument("--tool-ratio", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=20260528)
    parser.add_argument("--capacity-mode", choices=["fixed_per_server", "g_only_fixed_per_server"], default="fixed_per_server")
    parser.add_argument("--methods", default=",".join(key for key, _ in METHODS))
    parser.add_argument("--hard-overhead-s", type=float, default=90.0)
    parser.add_argument("--out-dir", type=Path, default=Path("jolt_timeout_sweep_20_80"))
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--adaptive-skip", action="store_true")
    parser.add_argument("--skip-after-two-nan", action="store_true")
    parser.add_argument("--stable-rel-tol", type=float, default=0.01)
    parser.add_argument("--stable-abs-tol", type=float, default=0.005)
    parser.add_argument(
        "--stability-check-timeout-s",
        type=float,
        default=180.0,
        help="At this budget and every longer budget, skip remaining budgets if improvement over the previous budget is small.",
    )
    args = parser.parse_args()

    selected_methods = {x.strip() for x in args.methods.split(",") if x.strip()}
    llm_values = [int(x.strip()) for x in args.llm_list.split(",") if x.strip()]
    timeout_values = [float(x.strip()) for x in args.timeout_list.split(",") if x.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.csv"
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as f:
            rows: list[dict] = list(csv.DictReader(f))
    else:
        rows = []

    ctx = mp.get_context("spawn")
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
        inst_meta = {"g_count": inst.g_count, "c_count": inst.c_count, "tool_hosts": tool_host_count(inst)}
        with (args.out_dir / f"instance_L{llms}.json").open("w", encoding="utf-8") as f:
            json.dump(instance_to_jsonable(inst), f)
        print(
            f"\nScale L={llms}, T={tools}, G={inst.g_count}, C={inst.c_count}, H={tool_host_count(inst)}",
            flush=True,
        )

        for solver_timeout_s in timeout_values:
            print(f"  Time budget {timeout_label(solver_timeout_s)}", flush=True)
            for method_key, method_name in METHODS:
                if method_key not in selected_methods:
                    continue
                already_done = any(
                    int(r["llms"]) == llms
                    and abs(float(r["timeout_s"]) - solver_timeout_s) < 1e-9
                    and r["method"] == method_name
                    for r in rows
                )
                if already_done and not args.rerun_existing:
                    existing_row = find_row(rows, llms, solver_timeout_s, method_name)
                    if (
                        args.adaptive_skip
                        and existing_row is not None
                        and solver_timeout_s >= args.stability_check_timeout_s
                    ):
                        current_index = timeout_values.index(solver_timeout_s)
                        if is_proven_done(existing_row):
                            rows, added = add_future_skip_rows(
                                rows,
                                llms,
                                timeout_values,
                                current_index,
                                existing_row,
                                make_proven_skip_row,
                                None,
                                args.rerun_existing,
                            )
                            if added:
                                write_rows(summary_path, rows)
                        elif current_index > 0:
                            previous_row = find_previous_finite_row(
                                rows, llms, timeout_values, current_index, existing_row["method"]
                            )
                            stable, stable_extra = should_skip_after_stable(
                                previous_row,
                                existing_row,
                                rel_tol=args.stable_rel_tol,
                                abs_tol=args.stable_abs_tol,
                            )
                            if stable:
                                rows, added = add_future_skip_rows(
                                    rows,
                                    llms,
                                    timeout_values,
                                    current_index,
                                    existing_row,
                                    make_stable_skip_row,
                                    stable_extra,
                                    args.rerun_existing,
                                )
                                if added:
                                    write_rows(summary_path, rows)
                    print(f"    skip {method_name} (existing)", flush=True)
                    continue

                queue: mp.Queue = ctx.Queue()
                proc = ctx.Process(
                    target=run_method_child,
                    args=(method_key, seed, llms, tools, solver_timeout_s, args.capacity_mode, queue),
                )
                print(f"    running {method_name} ...", flush=True)
                start = time.perf_counter()
                proc.start()
                hard_timeout = solver_timeout_s + args.hard_overhead_s
                proc.join(hard_timeout)
                elapsed = time.perf_counter() - start

                if proc.is_alive():
                    proc.terminate()
                    proc.join(10)
                    row = nan_row(
                        llms,
                        tools,
                        solver_timeout_s,
                        inst_meta,
                        method_name,
                        elapsed,
                        "HARD_TIMEOUT",
                        {"hard_timeout_s": hard_timeout, "solver_timeout_s": solver_timeout_s},
                    )
                else:
                    payload = queue.get() if not queue.empty() else {"ok": False, "error": "NO_RESULT", "elapsed": elapsed}
                    if not payload.get("ok"):
                        row = nan_row(
                            llms,
                            tools,
                            solver_timeout_s,
                            inst_meta,
                            method_name,
                            float(payload.get("elapsed", elapsed)),
                            "ERROR",
                            {"error": payload.get("error"), "solver_timeout_s": solver_timeout_s},
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
                        feasible = bool(result_data["feasible"]) and math.isfinite(float(result_data["avg_call_distance"]))
                        row = {
                            "llms": llms,
                            "tools": tools,
                            "timeout_s": solver_timeout_s,
                            "timeout_label": timeout_label(solver_timeout_s),
                            "g_servers": inst_meta["g_count"],
                            "c_servers": inst_meta["c_count"],
                            "tool_hosts": inst_meta["tool_hosts"],
                            "method": result_data["name"],
                            "decision_parameters": result_data["decision_count"],
                            "actual_parameters": actual_parameters(light_result),
                            "avg_call_distance": result_data["avg_call_distance"] if feasible else "NaN",
                            "solve_time_s": result_data["solve_time_s"],
                            "status": extra.get("solver_status", extra.get("phase2_solver_status", "")),
                            "feasible": feasible,
                            "extra": json.dumps(extra, ensure_ascii=False),
                        }

                rows = [
                    r
                    for r in rows
                    if not (
                        int(r["llms"]) == llms
                        and abs(float(r["timeout_s"]) - solver_timeout_s) < 1e-9
                        and r["method"] == row["method"]
                    )
                ] + [row]
                write_rows(summary_path, rows)
                print(
                    f"      {row['status']} distance={row['avg_call_distance']} "
                    f"time={float(row['solve_time_s']):.3f}s",
                    flush=True,
                )

                if args.adaptive_skip and is_proven_done(row):
                    current_index = timeout_values.index(solver_timeout_s)
                    rows, added = add_future_skip_rows(
                        rows,
                        llms,
                        timeout_values,
                        current_index,
                        row,
                        make_proven_skip_row,
                        None,
                        args.rerun_existing,
                    )
                    for skip_row in added:
                        print(
                            f"      skip {row['method']} at {skip_row['timeout_label']} "
                            f"(already proven at {row['timeout_label']})",
                            flush=True,
                        )
                    if added:
                        write_rows(summary_path, rows)

                if (
                    args.adaptive_skip
                    and solver_timeout_s >= args.stability_check_timeout_s
                    and not is_proven_done(row)
                ):
                    current_index = timeout_values.index(solver_timeout_s)
                    if current_index > 0:
                        previous_row = find_previous_finite_row(rows, llms, timeout_values, current_index, row["method"])
                        stable, stable_extra = should_skip_after_stable(
                            previous_row,
                            row,
                            rel_tol=args.stable_rel_tol,
                            abs_tol=args.stable_abs_tol,
                        )
                        no_feasible_skip = False
                        if not stable and args.skip_after_two_nan:
                            immediate_previous = find_row(
                                rows, llms, timeout_values[current_index - 1], row["method"]
                            )
                            no_feasible_skip, stable_extra = should_skip_after_no_feasible(immediate_previous, row)
                        if stable or no_feasible_skip:
                            rows, added = add_future_skip_rows(
                                rows,
                                llms,
                                timeout_values,
                                current_index,
                                row,
                                make_no_feasible_skip_row if no_feasible_skip else make_stable_skip_row,
                                stable_extra,
                                args.rerun_existing,
                            )
                            for skip_row in added:
                                if no_feasible_skip:
                                    print(
                                        f"      skip {row['method']} at {skip_row['timeout_label']} "
                                        f"(no feasible solution at previous two budgets)",
                                        flush=True,
                                    )
                                else:
                                    print(
                                        f"      skip {row['method']} at {skip_row['timeout_label']} "
                                        f"(stable: rel_improve={stable_extra['improvement_rel']:.4g}, "
                                        f"abs_improve={stable_extra['improvement_abs']:.4g})",
                                        flush=True,
                                    )
                            if added:
                                write_rows(summary_path, rows)

    print(f"\nWrote {summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
