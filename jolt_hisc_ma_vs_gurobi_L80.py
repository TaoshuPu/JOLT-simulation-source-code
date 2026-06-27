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

from jolt_algorithm1_convergence_L20 import (
    instance_to_jsonable,
    is_feasible,
    make_llm_instance,
    normalized_cost,
    preference_similarity,
    run_evolutionary,
)
from jolt_gqap_gurobi_knn_init_experiment import solve_with_checkpoints


OUT_DIR = ROOT / "jolt_hisc_ma_vs_gurobi_L80"
SEED = 20260529
LLMS = 80
TOOLS = 240
G_SERVERS = 27
GPU_CAP = 4
MEM_CAP = 48
POP_SIZE = 40
GENERATIONS = 200
PC = 0.88
PM_GENE = 0.14
LATE_PM_GENE = 0.35
LATE_MUTATION_START = 0.55
LOCAL_ITER = 4
HISC_TIME_LIMIT_S = 120
GUROBI_TIME_LIMIT_S = 120
CHECKPOINTS = [30, 60, 120]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inst = make_llm_instance(
        seed=SEED,
        llm_count=LLMS,
        tool_count=TOOLS,
        g_count=G_SERVERS,
        gpu_cap=GPU_CAP,
        mem_cap=MEM_CAP,
    )
    sim = preference_similarity(inst)
    (OUT_DIR / "instance_meta.json").write_text(
        json.dumps(instance_to_jsonable(inst), indent=2),
        encoding="utf-8",
    )
    print(
        f"Compare HISC-MA vs Gurobi: LLM={LLMS}, tools={TOOLS}, G={G_SERVERS}, "
        f"pop={POP_SIZE}, gen={GENERATIONS}, pc={PC}, "
        f"HISC={HISC_TIME_LIMIT_S}s, Gurobi={GUROBI_TIME_LIMIT_S}s",
        flush=True,
    )

    hisc_start = time.perf_counter()
    hisc = run_evolutionary(
        inst,
        sim,
        SEED + 404,
        method="HISC-MA",
        pop_size=POP_SIZE,
        max_gen=GENERATIONS,
        pc=PC,
        pm_gene=PM_GENE,
        late_pm_gene=LATE_PM_GENE,
        late_mutation_start=LATE_MUTATION_START,
        local_iter=LOCAL_ITER,
        hybrid_init=True,
        memetic=True,
        server_centric=True,
        timeout_s=HISC_TIME_LIMIT_S,
    )
    hisc_wall = time.perf_counter() - hisc_start
    hisc_status = "TIME_LIMIT" if hisc.elapsed_s >= HISC_TIME_LIMIT_S - 1e-6 and len(hisc.history) <= GENERATIONS else "HEURISTIC_DONE"
    print(
        f"[HISC-MA] initial={hisc.initial_cost:.6f}, final={hisc.best_cost:.6f}, "
        f"time={hisc.elapsed_s:.2f}s, status={hisc_status}, feasible={hisc.feasible}",
        flush=True,
    )
    hisc_row = {
        "method": "HISC-MA",
        "status": hisc_status,
        "initial_cost": hisc.initial_cost,
        "final_cost": hisc.best_cost,
        "best_bound_cost": math.nan,
        "gap_percent": math.nan,
        "elapsed_s": hisc.elapsed_s,
        "wall_s": hisc_wall,
        "feasible": hisc.feasible,
        "assignment": json.dumps(hisc.best_assignment),
        "history": json.dumps(hisc.history),
        "extra": json.dumps(
            {
                "pop_size": POP_SIZE,
                "generations": GENERATIONS,
                "generations_completed": len(hisc.history) - 1,
                "timeout_s": HISC_TIME_LIMIT_S,
                "pc": PC,
                "pm_gene": PM_GENE,
                "late_pm_gene": LATE_PM_GENE,
                "late_mutation_start": LATE_MUTATION_START,
                "local_iter": LOCAL_ITER,
            }
        ),
    }
    write_csv(OUT_DIR / "hisc_ma_result.csv", [hisc_row])

    g_rows, g_final = solve_with_checkpoints(
        inst,
        sim,
        method="Gurobi GQAP",
        mip_start=None,
        mip_start_cost=None,
        seed=SEED + 100,
        time_limit_s=GUROBI_TIME_LIMIT_S,
        checkpoints=CHECKPOINTS,
        output_flag=0,
    )
    g_cost = float(g_final["final_incumbent_cost"])
    g_assignment = json.loads(g_final["final_assignment"])
    g_feasible = bool(g_final["final_feasible"]) and is_feasible(inst, g_assignment)
    g_final["final_incumbent_cost_recomputed"] = normalized_cost(inst, g_assignment, sim) if g_feasible else math.nan
    write_csv(OUT_DIR / "gurobi_checkpoints.csv", g_rows)
    write_csv(OUT_DIR / "gurobi_final.csv", [g_final])

    winner = "HISC-MA" if hisc.best_cost < g_cost else "Gurobi GQAP" if g_cost < hisc.best_cost else "Tie"
    improvement = (g_cost - hisc.best_cost) / max(abs(g_cost), 1e-12) * 100.0
    comparison_rows = [
        {
            "method": "HISC-MA",
            "final_cost": hisc.best_cost,
            "elapsed_s": hisc.elapsed_s,
            "status": hisc_status,
            "feasible": hisc.feasible,
        },
        {
            "method": "Gurobi GQAP",
            "final_cost": g_cost,
            "elapsed_s": g_final["solver_runtime_s"],
            "status": g_final["status"],
            "feasible": g_feasible,
        },
    ]
    write_csv(OUT_DIR / "comparison_summary.csv", comparison_rows)
    (OUT_DIR / "winner.json").write_text(
        json.dumps(
            {
                "winner": winner,
                "hisc_cost": hisc.best_cost,
                "gurobi_cost": g_cost,
                "hisc_relative_improvement_vs_gurobi_percent": improvement,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Winner={winner}; HISC-MA={hisc.best_cost:.6f}, "
        f"Gurobi={g_cost:.6f}, HISC improvement={improvement:.2f}%",
        flush=True,
    )
    print(f"Wrote outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
