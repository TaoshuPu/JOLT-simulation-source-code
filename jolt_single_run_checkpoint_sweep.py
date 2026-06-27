from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty

from jolt_small_scale_experiment import (
    AlgorithmResult,
    average_call_distance,
    cpsat_original_solver,
    gurobi_gqap_llm_deployment,
    gurobi_original_solver,
    gurobi_tool_deployment,
    iwc_gs_tool_deployment,
    make_instance,
    random_llm_assignment,
    scip_original_solver,
    scip_gqap_llm_deployment,
    scip_tool_deployment,
    tool_host_count,
)


METHODS = [
    ("gurobi", "Gurobi MIP original"),
    ("scip", "SCIP MIQP original"),
    ("cpsat", "OR-Tools CP-SAT original"),
    ("gqap_tool_mip", "JOLT"),
]

METHOD_ALIASES = {
    "jolt": "gqap_tool_mip",
}

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
    "status",
    "feasible",
    "snapshot_elapsed_s",
    "stopped_early",
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


def normalize_method_key(method_key: str) -> str:
    key = method_key.strip().lower()
    return METHOD_ALIASES.get(key, key)


def method_display_name(method_key: str) -> str:
    method_key = normalize_method_key(method_key)
    return dict(METHODS)[method_key]


def static_actual_parameters(method_key: str, llms: int, tools: int, g_count: int, h_count: int) -> str:
    method_key = normalize_method_key(method_key)
    if method_key == "cpsat":
        return f"x={llms * g_count};y={tools * h_count};element_cost={llms * tools};formulation=element"
    if method_key in {"gurobi", "scip"}:
        return f"x={llms * g_count};y={tools * h_count};q_terms={llms * tools * g_count * h_count}"
    if method_key == "gqap_tool_mip":
        return (
            f"phase1_x={llms * g_count};"
            f"phase1_q_terms={llms * llms * g_count * max(0, g_count - 1)};"
            f"phase2_y={tools * h_count};phase2_terms={tools * h_count}"
        )
    return "NA"


def two_stage_actual_parameters(
    llms: int,
    tools: int,
    g_count: int,
    h_count: int,
    gqap_solver: str,
    tool_mip_solver: str,
) -> str:
    return (
        f"phase1_solver={gqap_solver};"
        f"phase1_x={llms * g_count};"
        f"phase1_q_terms={llms * llms * g_count * max(0, g_count - 1)};"
        f"phase2_solver={tool_mip_solver};"
        f"phase2_y={tools * h_count};phase2_terms={tools * h_count}"
    )


def decision_parameters(method_key: str, llms: int, tools: int) -> int:
    method_key = normalize_method_key(method_key)
    return llms if method_key == "gqap_tool_mip" else llms + tools


def solver_status(result: AlgorithmResult) -> str:
    return str(result.extra.get("solver_status", result.extra.get("phase2_solver_status", "")))


def emit_snapshot(
    queue: mp.Queue,
    elapsed_s: float,
    avg_call_distance: float,
    llm_place: list[int],
    tool_place: list[int],
    extra: dict,
) -> None:
    queue.put(
        {
            "type": "snapshot",
            "elapsed_s": float(elapsed_s),
            "avg_call_distance": float(avg_call_distance),
            "llm_place": list(llm_place),
            "tool_place": list(tool_place),
            "extra": extra,
        }
    )


def run_worker(
    method_key: str,
    seed: int,
    llms: int,
    tools: int,
    max_time_s: float,
    capacity_mode: str,
    queue: mp.Queue,
    gqap_solver: str = "gurobi",
    tool_mip_solver: str = "gurobi",
) -> None:
    method_key = normalize_method_key(method_key)
    start = time.perf_counter()
    inst = make_instance(
        seed,
        llm_count=llms,
        tool_count=tools,
        g_count=None,
        c_count=None,
        capacity_mode=capacity_mode,
    )

    def progress(elapsed: float, llm_place: list[int], tool_place: list[int], distance: float, extra: dict) -> None:
        emit_snapshot(queue, elapsed, distance, llm_place, tool_place, extra)

    try:
        if method_key == "gurobi":
            result = gurobi_original_solver(inst, timeout_s=max_time_s, progress_callback=progress)
            if result.feasible:
                progress(result.solve_time_s, result.assignment[:llms], result.assignment[llms:], result.avg_call_distance, {"source": "final"})
            queue.put({"type": "done", "elapsed_s": time.perf_counter() - start, "status": solver_status(result), "result": result.extra})
            return

        if method_key == "scip":
            result = scip_original_solver(inst, timeout_s=max_time_s, progress_callback=progress)
            if result.feasible:
                progress(result.solve_time_s, result.assignment[:llms], result.assignment[llms:], result.avg_call_distance, {"source": "final"})
            queue.put({"type": "done", "elapsed_s": time.perf_counter() - start, "status": solver_status(result), "result": result.extra})
            return

        if method_key == "cpsat":
            result = cpsat_original_solver(
                inst,
                timeout_s=max_time_s,
                warm_start=True,
                formulation="element",
                progress_callback=progress,
            )
            if result.feasible:
                progress(result.solve_time_s, result.assignment[:llms], result.assignment[llms:], result.avg_call_distance, {"source": "final"})
            queue.put({"type": "done", "elapsed_s": time.perf_counter() - start, "status": solver_status(result), "result": result.extra})
            return

        if method_key == "gqap_tool_mip":
            gqap_solver = gqap_solver.lower()
            gqap_solvers = {
                "gurobi": gurobi_gqap_llm_deployment,
                "scip": scip_gqap_llm_deployment,
            }
            if gqap_solver not in gqap_solvers:
                raise ValueError(f"Unsupported GQAP solver: {gqap_solver}")
            # A feasible fallback mirrors the solver's existing no-incumbent behavior and lets early checkpoints be meaningful.
            fallback_llm = random_llm_assignment(inst, __import__("random").Random(inst.seed + 79))
            fallback_tool = iwc_gs_tool_deployment(inst, fallback_llm)
            progress(
                time.perf_counter() - start,
                fallback_llm,
                fallback_tool,
                average_call_distance(inst, fallback_llm, fallback_tool),
                {"source": "fallback_iwc_gs", "phase": "phase1"},
            )

            def gqap_progress(elapsed: float, llm_place: list[int], surrogate_cost: float, extra: dict) -> None:
                tool_place = iwc_gs_tool_deployment(inst, llm_place)
                progress(
                    elapsed,
                    llm_place,
                    tool_place,
                    average_call_distance(inst, llm_place, tool_place),
                    {**extra, "phase": "phase1", "llm_surrogate_cost": surrogate_cost},
                )

            llm_place, phase1_extra = gqap_solvers[gqap_solver](
                inst,
                timeout_s=max_time_s,
                progress_callback=gqap_progress,
            )
            tool_place = iwc_gs_tool_deployment(inst, llm_place)
            progress(
                time.perf_counter() - start,
                llm_place,
                tool_place,
                average_call_distance(inst, llm_place, tool_place),
                {"source": "phase1_final_iwc_gs", "phase": "phase1"},
            )
            queue.put(
                {
                    "type": "done",
                    "elapsed_s": time.perf_counter() - start,
                    "status": f"PHASE1_{phase1_extra.get('solver_status', 'UNKNOWN')}",
                    "result": phase1_extra,
                }
            )
            return

        raise ValueError(f"Unknown method: {method_key}")
    except Exception as exc:
        queue.put({"type": "error", "elapsed_s": time.perf_counter() - start, "status": "ERROR", "error": repr(exc)})


def latest_snapshot_at(snapshots: list[dict], checkpoint_s: float, completion_s: float | None) -> dict | None:
    if completion_s is not None and checkpoint_s >= completion_s:
        candidates = snapshots
    else:
        candidates = [s for s in snapshots if float(s["elapsed_s"]) <= checkpoint_s + 1e-9]
    if not candidates:
        return None
    return max(candidates, key=lambda s: float(s["elapsed_s"]))


def stable_pair(prev_value: float | None, curr_value: float | None, rel_tol: float, abs_tol: float) -> tuple[bool, dict]:
    if prev_value is None or curr_value is None:
        return False, {}
    improvement_abs = prev_value - curr_value
    improvement_rel = improvement_abs / max(abs(prev_value), 1e-12)
    stable = improvement_abs >= -1e-12 and (improvement_abs <= abs_tol or improvement_rel <= rel_tol)
    return stable, {
        "previous_avg_call_distance": prev_value,
        "current_avg_call_distance": curr_value,
        "improvement_abs": improvement_abs,
        "improvement_rel": improvement_rel,
        "stable_rel_tol": rel_tol,
        "stable_abs_tol": abs_tol,
    }


def collect_process_messages(queue: mp.Queue, snapshots: list[dict]) -> tuple[dict | None, dict | None]:
    done = None
    error = None
    while True:
        try:
            message = queue.get(timeout=0.01)
        except Empty:
            break
        if message["type"] == "snapshot":
            snapshots.append(message)
        elif message["type"] == "done":
            done = message
        elif message["type"] == "error":
            error = message
    snapshots.sort(key=lambda s: float(s["elapsed_s"]))
    return done, error


def row_from_snapshot(
    *,
    method_key: str,
    llms: int,
    tools: int,
    checkpoint_s: float,
    inst_meta: dict,
    snapshot: dict | None,
    solve_time_s: float,
    status: str,
    stopped_early: bool,
    extra: dict,
) -> dict:
    value = finite(snapshot.get("avg_call_distance") if snapshot else None)
    return {
        "llms": llms,
        "tools": tools,
        "checkpoint_s": checkpoint_s,
        "checkpoint_label": checkpoint_label(checkpoint_s),
        "g_servers": inst_meta["g_count"],
        "c_servers": inst_meta["c_count"],
        "tool_hosts": inst_meta["tool_hosts"],
        "method": method_display_name(method_key),
        "decision_parameters": decision_parameters(method_key, llms, tools),
        "actual_parameters": static_actual_parameters(method_key, llms, tools, inst_meta["g_count"], inst_meta["tool_hosts"]),
        "avg_call_distance": value if value is not None else "NaN",
        "solve_time_s": solve_time_s,
        "status": status,
        "feasible": value is not None,
        "snapshot_elapsed_s": snapshot.get("elapsed_s") if snapshot else "NaN",
        "stopped_early": stopped_early,
        "extra": json.dumps(extra, ensure_ascii=False),
    }


def build_rows_from_trace(
    *,
    method_key: str,
    inst,
    llms: int,
    tools: int,
    checkpoints: list[float],
    snapshots: list[dict],
    done: dict | None,
    error: dict | None,
    stopped: dict | None,
    inst_meta: dict,
    phase2_timeout_s: float,
    tool_mip_solver: str = "gurobi",
    gqap_solver: str = "gurobi",
) -> list[dict]:
    method_key = normalize_method_key(method_key)
    completion_s = float(done["elapsed_s"]) if done else None
    stopped_s = float(stopped["elapsed_s"]) if stopped else None
    error_s = float(error["elapsed_s"]) if error else None
    terminal_s = completion_s if completion_s is not None else stopped_s if stopped_s is not None else error_s
    terminal_status = (done or stopped or error or {}).get("status", "RUNNING_INCUMBENT")
    rows: list[dict] = []
    tool_mip_cache: dict[tuple[int, ...], tuple[dict, dict]] = {}

    for checkpoint_s in checkpoints:
        snap = latest_snapshot_at(snapshots, checkpoint_s, completion_s)
        row_status = "RUNNING_INCUMBENT" if snap is not None else "NO_FEASIBLE_INCUMBENT"
        stopped_early = stopped is not None and checkpoint_s > stopped_s + 1e-9
        if completion_s is not None and checkpoint_s >= completion_s:
            row_status = terminal_status
        if error_s is not None and checkpoint_s >= error_s and snap is None:
            row_status = "ERROR"
        if stopped_early:
            row_status = terminal_status

        solve_time_s = min(checkpoint_s, terminal_s) if terminal_s is not None else checkpoint_s
        extra = {
            "single_run_checkpoint": True,
            "terminal_status": terminal_status,
            "terminal_elapsed_s": terminal_s,
            "snapshot_extra": snap.get("extra", {}) if snap else {},
        }

        if method_key == "gqap_tool_mip" and snap is not None:
            llm_key = tuple(int(x) for x in snap["llm_place"])
            if llm_key not in tool_mip_cache:
                phase2_start = time.perf_counter()
                tool_mip_solvers = {
                    "gurobi": gurobi_tool_deployment,
                    "scip": scip_tool_deployment,
                }
                solver_key = tool_mip_solver.lower()
                if solver_key not in tool_mip_solvers:
                    raise ValueError(f"Unsupported Tool-MIP solver: {tool_mip_solver}")
                tool_place, tool_extra = tool_mip_solvers[solver_key](
                    inst,
                    list(llm_key),
                    timeout_s=phase2_timeout_s,
                    warm_start=snap.get("tool_place"),
                )
                phase2_elapsed = time.perf_counter() - phase2_start
                evaluated_snapshot = dict(snap)
                evaluated_snapshot["tool_place"] = tool_place
                evaluated_snapshot["avg_call_distance"] = average_call_distance(inst, list(llm_key), tool_place)
                tool_mip_cache[llm_key] = (
                    evaluated_snapshot,
                    {**tool_extra, "phase2_post_eval_elapsed_s": phase2_elapsed},
                )
            snap, tool_extra = tool_mip_cache[llm_key]
            row_status = (
                f"{terminal_status};PHASE2_{tool_extra.get('phase2_solver_status', 'UNKNOWN')}"
                if terminal_status.startswith("PHASE1_") or terminal_status.startswith("STOPPED")
                else f"PHASE1_RUNNING;PHASE2_{tool_extra.get('phase2_solver_status', 'UNKNOWN')}"
            )
            solve_time_s += float(tool_extra.get("phase2_solve_time_s", 0.0))
            extra["phase2_extra"] = tool_extra
            extra["configured_gqap_solver"] = gqap_solver
            extra["configured_tool_mip_solver"] = tool_mip_solver

        if stopped is not None and checkpoint_s > stopped_s + 1e-9:
            extra["skip_reason"] = stopped.get("reason")
            extra["copied_from_checkpoint_s"] = stopped.get("checkpoint_s")

        rows.append(
            row_from_snapshot(
                method_key=method_key,
                llms=llms,
                tools=tools,
                checkpoint_s=checkpoint_s,
                inst_meta=inst_meta,
                snapshot=snap,
                solve_time_s=solve_time_s,
                status=row_status,
                stopped_early=stopped_early,
                extra=extra,
            )
        )
        if method_key == "gqap_tool_mip":
            rows[-1]["actual_parameters"] = two_stage_actual_parameters(
                llms,
                tools,
                inst_meta["g_count"],
                inst_meta["tool_hosts"],
                gqap_solver,
                tool_mip_solver,
            )
    return rows


def run_monitored_method(
    *,
    method_key: str,
    seed: int,
    llms: int,
    tools: int,
    checkpoints: list[float],
    max_time_s: float,
    capacity_mode: str,
    hard_overhead_s: float,
    rel_tol: float,
    abs_tol: float,
    skip_after_two_nan: bool,
    gqap_solver: str = "gurobi",
    tool_mip_solver: str = "gurobi",
) -> tuple[list[dict], list[dict], dict | None, dict | None, dict | None]:
    method_key = normalize_method_key(method_key)
    inst = make_instance(
        seed,
        llm_count=llms,
        tool_count=tools,
        g_count=None,
        c_count=None,
        capacity_mode=capacity_mode,
    )
    inst_meta = {"g_count": inst.g_count, "c_count": inst.c_count, "tool_hosts": tool_host_count(inst)}
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=run_worker,
        args=(method_key, seed, llms, tools, max_time_s, capacity_mode, queue, gqap_solver, tool_mip_solver),
    )
    proc.start()

    start = time.perf_counter()
    snapshots: list[dict] = []
    checkpoint_values: dict[float, float | None] = {}
    done = None
    error = None
    stopped = None

    while proc.is_alive():
        new_done, new_error = collect_process_messages(queue, snapshots)
        done = new_done or done
        error = new_error or error
        elapsed = time.perf_counter() - start

        for idx, checkpoint_s in enumerate(checkpoints):
            if checkpoint_s in checkpoint_values or elapsed < checkpoint_s:
                continue
            snap = latest_snapshot_at(snapshots, checkpoint_s, None)
            checkpoint_values[checkpoint_s] = finite(snap.get("avg_call_distance") if snap else None)
            if idx > 0:
                prev_cp = checkpoints[idx - 1]
                prev_value = checkpoint_values.get(prev_cp)
                curr_value = checkpoint_values[checkpoint_s]
                stable, stable_extra = stable_pair(prev_value, curr_value, rel_tol, abs_tol)
                no_feasible = skip_after_two_nan and prev_value is None and curr_value is None
                if stable or no_feasible:
                    stopped = {
                        "status": f"STOPPED_STABLE_FROM_{checkpoint_label(checkpoint_s)}" if stable else f"STOPPED_NO_FEASIBLE_FROM_{checkpoint_label(checkpoint_s)}",
                        "elapsed_s": elapsed,
                        "checkpoint_s": checkpoint_s,
                        "reason": "stable" if stable else "no_feasible_after_two_checkpoints",
                        **stable_extra,
                    }
                    proc.terminate()
                    proc.join(10)
                    break
        if stopped is not None:
            break
        if elapsed >= max_time_s + hard_overhead_s:
            stopped = {
                "status": "HARD_TIMEOUT",
                "elapsed_s": elapsed,
                "checkpoint_s": max(checkpoints),
                "reason": "hard_timeout",
            }
            proc.terminate()
            proc.join(10)
            break
        time.sleep(0.5)

    if proc.is_alive():
        proc.terminate()
        proc.join(10)
    else:
        proc.join()

    new_done, new_error = collect_process_messages(queue, snapshots)
    done = new_done or done
    error = new_error or error
    if error is not None and stopped is None:
        stopped = {"status": "ERROR", "elapsed_s": error.get("elapsed_s", time.perf_counter() - start), "reason": error.get("error")}

    rows = build_rows_from_trace(
        method_key=method_key,
        inst=inst,
        llms=llms,
        tools=tools,
        checkpoints=checkpoints,
        snapshots=snapshots,
        done=done,
        error=error,
        stopped=stopped,
        inst_meta=inst_meta,
        phase2_timeout_s=min(60.0, max_time_s),
        tool_mip_solver=tool_mip_solver,
        gqap_solver=gqap_solver,
    )
    return rows, snapshots, done, error, stopped


def write_rows(path: Path, rows: list[dict]) -> None:
    rows.sort(key=lambda r: (int(r["llms"]), float(r["checkpoint_s"]), str(r["method"])))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

