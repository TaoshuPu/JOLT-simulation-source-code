from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for deps_name in (".gurobi_deps", ".ortools_deps"):
    deps = ROOT / deps_name
    if deps.exists():
        sys.path.insert(0, str(deps))

from jolt_flp_iwc_gs_best_L80_T240 import plot_results, solve_flp_with_checkpoints, write_csv
from jolt_small_scale_experiment import (
    average_call_distance,
    hisc_ma_iwc_gs_solver,
    instance_to_jsonable,
    is_llm_feasible,
    is_tool_feasible,
    iwc_gs_tool_deployment,
    make_instance,
)


LLMS = 80
TOOLS = 1000
SEED = 20340528
PHASE1_TIMEOUT_S = 300.0
TIME_LIMIT_S = 600.0
CHECKPOINTS = [0, 1, 5, 10, 30, 60, 120, 180, 300, 420, 600]
OUT_DIR = ROOT / "jolt_flp_iwc_gs_best_L80_T1000"


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
    (OUT_DIR / "instance_L80_T1000.json").write_text(
        json.dumps(instance_to_jsonable(inst), ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"FLP scale-up experiment: LLM={LLMS}, tools={TOOLS}, "
        f"G={inst.g_count}, C={inst.c_count}, phase1_timeout={PHASE1_TIMEOUT_S:g}s",
        flush=True,
    )

    phase1_start = time.perf_counter()
    phase1_result = hisc_ma_iwc_gs_solver(inst, SEED + 10_000, phase1_timeout_s=PHASE1_TIMEOUT_S)
    phase1_wall = time.perf_counter() - phase1_start
    llm_place = [int(x) for x in phase1_result.assignment[: inst.llm_count]]
    if not is_llm_feasible(inst, llm_place):
        raise RuntimeError("Selected HISC-MA LLM deployment is infeasible.")

    iwc_tool_place = iwc_gs_tool_deployment(inst, llm_place)
    if not is_tool_feasible(inst, iwc_tool_place, llm_place):
        raise RuntimeError("IWC-GS warm start is infeasible.")

    source_meta = {
        "fixed_llm_source": "HISC-MA on the LLM=80, Tool=1000 instance",
        "phase1_timeout_s": PHASE1_TIMEOUT_S,
        "phase1_wall_s": phase1_wall,
        "phase1_result_avg_call_distance_with_iwc_gs": phase1_result.avg_call_distance,
        "phase1_solver_status": phase1_result.extra.get("solver_status"),
        "phase1_extra": phase1_result.extra,
        "llm_assignment": llm_place,
        "iwc_gs_tool_assignment": iwc_tool_place,
        "iwc_gs_avg_call_distance": average_call_distance(inst, llm_place, iwc_tool_place),
    }
    (OUT_DIR / "fixed_llm_and_iwc_gs_start.json").write_text(
        json.dumps(source_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Fixed LLM selected by HISC-MA in {phase1_wall:.2f}s; "
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
