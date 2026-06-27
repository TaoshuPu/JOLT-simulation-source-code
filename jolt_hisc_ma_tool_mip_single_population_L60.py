from __future__ import annotations

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
    instance_to_jsonable,
    is_deployment_feasible,
    iwc_gs_tool_deployment,
    knn_init_llm,
    llm_similarity,
    llm_surrogate_cost,
    make_instance,
    random_llm_assignment,
    repair_llm_assignment,
    server_centric_crossover,
    tabu_search_llm,
    tool_host_count,
)


LLMS = 60
TOOLS = 180
SEED = 20260528 + LLMS * 1000
OUT_DIR = ROOT / "jolt_hisc_ma_tool_mip_single_population_L60_10min"
CHECKPOINTS = [60, 180, 300, 420, 600]
POP_SIZE = 256
MAX_GEN = 1_000_000
PC = 0.92
PM = 0.14
LATE_PM = 0.35
LATE_MUTATION_START = 0.55
LOCAL_ITER = 4
PHASE2_TIMEOUT_S = 60.0


SUMMARY_FIELDS = [
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
    "best_so_far_avg_call_distance",
    "solve_time_s",
    "phase1_recorded_elapsed_s",
    "phase2_solve_time_s",
    "phase1_surrogate_cost",
    "phase1_preview_iwc_gs_distance",
    "generations_completed",
    "status",
    "feasible",
    "extra",
]


def checkpoint_label(seconds: float) -> str:
    return f"{seconds / 60:g}min"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def actual_parameters(inst) -> str:
    q_terms = inst.llm_count * inst.llm_count * inst.g_count * max(0, inst.g_count - 1)
    h_count = tool_host_count(inst)
    return (
        f"phase1_x={inst.llm_count * inst.g_count};"
        f"phase1_q_terms={q_terms};"
        f"single_population={POP_SIZE};max_gen={MAX_GEN};"
        f"pc={PC};pm={PM};late_pm={LATE_PM};local_iter={LOCAL_ITER};"
        f"phase2_y={inst.tool_count * h_count};phase2_terms={inst.tool_count * h_count}"
    )


def tournament(population: list[list[int]], score, rng: random.Random, k: int = 3) -> list[int]:
    return min(rng.sample(population, min(k, len(population))), key=score).copy()


def run_hisc_ma_single_population(inst, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    start = time.perf_counter()
    sim = llm_similarity(inst)
    score_cache: dict[tuple[int, ...], float] = {}

    population: list[list[int]] = []
    knn_count = max(1, int(0.25 * POP_SIZE))
    for idx in range(POP_SIZE):
        if idx < knn_count:
            population.append(knn_init_llm(inst, sim, rng))
        else:
            population.append(random_llm_assignment(inst, rng))

    def score(place: list[int]) -> float:
        key = tuple(place)
        value = score_cache.get(key)
        if value is None:
            value = llm_surrogate_cost(inst, place, sim)
            score_cache[key] = value
        return value

    best_llm = min(population, key=score).copy()
    best_surrogate = score(best_llm)
    best_generation = 0
    history_rows: list[dict] = [
        {
            "generation": 0,
            "elapsed_s": 0.0,
            "best_surrogate_cost": best_surrogate,
        }
    ]
    snapshots: list[dict] = []
    next_checkpoint_idx = 0
    max_time_s = max(CHECKPOINTS)
    generations_completed = 0

    def record_snapshot(checkpoint_s: float, elapsed_s: float) -> None:
        tool_preview = iwc_gs_tool_deployment(inst, best_llm)
        preview_distance = average_call_distance(inst, best_llm, tool_preview)
        snapshots.append(
            {
                "checkpoint_s": float(checkpoint_s),
                "checkpoint_label": checkpoint_label(checkpoint_s),
                "recorded_elapsed_s": float(elapsed_s),
                "generations_completed": int(generations_completed),
                "best_generation": int(best_generation),
                "phase1_surrogate_cost": float(best_surrogate),
                "phase1_preview_iwc_gs_distance": float(preview_distance),
                "llm_place": list(best_llm),
                "iwc_gs_tool_place": list(tool_preview),
            }
        )
        print(
            f"    HISC-MA checkpoint {checkpoint_label(checkpoint_s)}: "
            f"surrogate={best_surrogate:.6f}, preview={preview_distance:.6f}, "
            f"gen={generations_completed}, elapsed={elapsed_s:.2f}s",
            flush=True,
        )

    while time.perf_counter() - start < max_time_s and generations_completed < MAX_GEN:
        elapsed = time.perf_counter() - start
        progress = min(1.0, elapsed / max_time_s)
        if progress >= LATE_MUTATION_START:
            span = max(1e-9, 1.0 - LATE_MUTATION_START)
            ratio = min(1.0, (progress - LATE_MUTATION_START) / span)
            effective_pm = PM + (LATE_PM - PM) * ratio
        else:
            effective_pm = PM

        offspring: list[list[int]] = []
        elite_count = max(2, POP_SIZE // 16)
        ranked = sorted(population, key=score)
        offspring.extend(candidate.copy() for candidate in ranked[:elite_count])

        while len(offspring) < POP_SIZE:
            if time.perf_counter() - start >= max_time_s:
                break
            p1 = tournament(population, score, rng)
            p2 = tournament(population, score, rng)
            if rng.random() < PC:
                c1, c2 = server_centric_crossover(inst, p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for child in (c1, c2):
                if rng.random() < effective_pm:
                    child[rng.randrange(inst.llm_count)] = rng.randrange(inst.g_count)
                    child = repair_llm_assignment(inst, child, rng)
                child = tabu_search_llm(inst, child, sim, rng, max_iter=LOCAL_ITER)
                offspring.append(child)
                if len(offspring) >= POP_SIZE:
                    break

        if not offspring:
            break
        population = sorted(population + offspring, key=score)[:POP_SIZE]
        generations_completed += 1
        if score(population[0]) < best_surrogate:
            best_llm = population[0].copy()
            best_surrogate = score(best_llm)
            best_generation = generations_completed

        elapsed = time.perf_counter() - start
        if generations_completed == 1 or generations_completed % 10 == 0:
            history_rows.append(
                {
                    "generation": generations_completed,
                    "elapsed_s": elapsed,
                    "best_surrogate_cost": best_surrogate,
                }
            )
        while next_checkpoint_idx < len(CHECKPOINTS) and elapsed >= CHECKPOINTS[next_checkpoint_idx]:
            record_snapshot(CHECKPOINTS[next_checkpoint_idx], elapsed)
            next_checkpoint_idx += 1

    elapsed = time.perf_counter() - start
    while next_checkpoint_idx < len(CHECKPOINTS):
        record_snapshot(CHECKPOINTS[next_checkpoint_idx], elapsed)
        next_checkpoint_idx += 1

    meta = {
        "phase1_solver": "HISC-MA single continuous population",
        "seed": seed,
        "pop_size": POP_SIZE,
        "max_gen": MAX_GEN,
        "pc": PC,
        "pm": PM,
        "late_pm": LATE_PM,
        "late_mutation_start": LATE_MUTATION_START,
        "local_iter": LOCAL_ITER,
        "generations_completed": generations_completed,
        "elapsed_s": elapsed,
        "best_generation": best_generation,
        "best_surrogate_cost": best_surrogate,
        "history_rows": history_rows,
    }
    return snapshots, meta


def solve_tool_mip_for_snapshots(inst, snapshots: list[dict], phase1_meta: dict) -> list[dict]:
    rows: list[dict] = []
    best_so_far = math.inf
    best_source = None
    for snap in snapshots:
        llm_place = [int(x) for x in snap["llm_place"]]
        warm_tool = [int(x) for x in snap["iwc_gs_tool_place"]]
        tool_place, tool_extra = gurobi_tool_deployment(
            inst,
            llm_place,
            timeout_s=PHASE2_TIMEOUT_S,
            warm_start=warm_tool,
        )
        avg = average_call_distance(inst, llm_place, tool_place)
        if avg < best_so_far:
            best_so_far = avg
            best_source = snap["checkpoint_label"]
        phase2_time = float(tool_extra.get("phase2_solve_time_s", math.nan))
        feasible = is_deployment_feasible(inst, llm_place, tool_place)
        row = {
            "llms": inst.llm_count,
            "tools": inst.tool_count,
            "checkpoint_s": snap["checkpoint_s"],
            "checkpoint_label": snap["checkpoint_label"],
            "g_servers": inst.g_count,
            "c_servers": inst.c_count,
            "tool_hosts": tool_host_count(inst),
            "method": "HISC-MA + Gurobi Tools",
            "decision_parameters": inst.llm_count,
            "actual_parameters": actual_parameters(inst),
            "avg_call_distance": avg if feasible else "NaN",
            "best_so_far_avg_call_distance": best_so_far if math.isfinite(best_so_far) else "NaN",
            "solve_time_s": float(snap["recorded_elapsed_s"]) + phase2_time,
            "phase1_recorded_elapsed_s": snap["recorded_elapsed_s"],
            "phase2_solve_time_s": phase2_time,
            "phase1_surrogate_cost": snap["phase1_surrogate_cost"],
            "phase1_preview_iwc_gs_distance": snap["phase1_preview_iwc_gs_distance"],
            "generations_completed": snap["generations_completed"],
            "status": f"PHASE1_TIME_CHECKPOINT;PHASE2_{tool_extra.get('phase2_solver_status', 'UNKNOWN')}",
            "feasible": feasible,
            "extra": json.dumps(
                {
                    "best_so_far_source_checkpoint": best_source,
                    "snapshot": {
                        key: value
                        for key, value in snap.items()
                        if key not in {"llm_place", "iwc_gs_tool_place"}
                    },
                    "phase1_meta": {
                        key: value
                        for key, value in phase1_meta.items()
                        if key != "history_rows"
                    },
                    "phase2_extra": tool_extra,
                    "iwc_gs_warm_start_distance": average_call_distance(inst, llm_place, warm_tool),
                },
                ensure_ascii=False,
            ),
        }
        rows.append(row)
        print(
            f"    Tool-MIP {snap['checkpoint_label']}: avg={avg:.6f}, "
            f"best_so_far={row['best_so_far_avg_call_distance']:.6f}, "
            f"status={tool_extra.get('phase2_solver_status')}, phase2={phase2_time:.3f}s",
            flush=True,
        )
    return rows


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
    (OUT_DIR / f"instance_L{LLMS}.json").write_text(
        json.dumps(instance_to_jsonable(inst), ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Single-population HISC-MA + Gurobi Tools: L={LLMS}, T={TOOLS}, "
        f"G={inst.g_count}, C={inst.c_count}, H={tool_host_count(inst)}, "
        f"pop={POP_SIZE}, checkpoints={CHECKPOINTS}",
        flush=True,
    )
    snapshots, phase1_meta = run_hisc_ma_single_population(inst, SEED + 10_000)
    (OUT_DIR / "phase1_trace.json").write_text(
        json.dumps({"snapshots": snapshots, **phase1_meta}, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(OUT_DIR / "phase1_history.csv", phase1_meta["history_rows"])
    rows = solve_tool_mip_for_snapshots(inst, snapshots, phase1_meta)
    write_csv(OUT_DIR / "summary.csv", rows, SUMMARY_FIELDS)
    print(f"Wrote {OUT_DIR / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
