from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
for deps_name in (".gurobi_deps", ".ortools_deps"):
    deps = ROOT / deps_name
    if deps.exists():
        sys.path.insert(0, str(deps))

from jolt_algorithm1_convergence_L20 import (
    build_population,
    instance_to_jsonable,
    is_feasible,
    make_llm_instance,
    mutate_assignment,
    normalized_cost,
    preference_similarity,
    select_parent,
    server_centric_crossover,
    tabu_local_search,
)
import gurobipy as gp
from gurobipy import GRB

from jolt_gqap_gurobi_knn_init_experiment import (
    build_gqap_model,
    gap_percent,
    norm_value,
    status_name,
)


OUT_DIR = ROOT / "jolt_hisc_ma_vs_gurobi_L80_10min_checkpoints"
SEED = 20260529
LLMS = 80
TOOLS = 240
G_SERVERS = 27
GPU_CAP = 4
MEM_CAP = 48
POP_SIZE = 40
MAX_GENERATIONS = 500
PC = 0.88
PM_GENE = 0.14
LATE_PM_GENE = 0.35
LATE_MUTATION_START = 0.55
LOCAL_ITER = 4
TIME_LIMIT_S = 600
CHECKPOINTS = [60 * minute for minute in range(1, 11)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_hisc_ma_with_time_checkpoints(inst, sim, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    start = time.perf_counter()
    population = build_population(inst, sim, rng, POP_SIZE, hybrid=True)
    score_cache: dict[tuple[int, ...], float] = {}

    def score(place: list[int]) -> float:
        key = tuple(place)
        value = score_cache.get(key)
        if value is None:
            value = normalized_cost(inst, place, sim)
            score_cache[key] = value
        return value

    best = min(population, key=score).copy()
    best_cost = score(best)
    initial_cost = best_cost
    elite_count = max(2, POP_SIZE // 12)
    records: dict[int, dict] = {}
    generation = 0

    def remember(elapsed_s: float, status: str) -> None:
        for checkpoint_s in CHECKPOINTS:
            if checkpoint_s not in records and elapsed_s >= checkpoint_s:
                records[checkpoint_s] = {
                    "method": "HISC-MA",
                    "checkpoint_s": checkpoint_s,
                    "checkpoint_label": f"{checkpoint_s // 60}min",
                    "recorded_elapsed_s": elapsed_s,
                    "incumbent_cost": best_cost,
                    "best_bound_cost": math.nan,
                    "gap_percent": math.nan,
                    "generations_completed": generation,
                    "status": status,
                    "feasible": is_feasible(inst, best),
                }
                print(
                    f"[HISC-MA] checkpoint {checkpoint_s}s: "
                    f"incumbent={best_cost:.6f}, gen={generation}, elapsed={elapsed_s:.2f}s",
                    flush=True,
                )

    timeout_hit = False
    for generation in range(MAX_GENERATIONS):
        elapsed = time.perf_counter() - start
        remember(elapsed, "RUNNING")
        if elapsed >= TIME_LIMIT_S:
            timeout_hit = True
            break

        progress = generation / max(1, MAX_GENERATIONS - 1)
        if LATE_PM_GENE > PM_GENE and progress >= LATE_MUTATION_START:
            span = max(1e-9, 1.0 - LATE_MUTATION_START)
            ratio = min(1.0, (progress - LATE_MUTATION_START) / span)
            effective_pm_gene = PM_GENE + (LATE_PM_GENE - PM_GENE) * ratio
        else:
            effective_pm_gene = PM_GENE

        ranked = sorted(population, key=score)
        offspring = [candidate.copy() for candidate in ranked[:elite_count]]
        while len(offspring) < POP_SIZE:
            elapsed = time.perf_counter() - start
            remember(elapsed, "RUNNING")
            if elapsed >= TIME_LIMIT_S:
                timeout_hit = True
                break

            p1 = select_parent(population, score, rng)
            p2 = select_parent(population, score, rng)
            if rng.random() < PC:
                c1, c2 = server_centric_crossover(inst, p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                elapsed = time.perf_counter() - start
                remember(elapsed, "RUNNING")
                if elapsed >= TIME_LIMIT_S:
                    timeout_hit = True
                    break
                child = mutate_assignment(inst, child, rng, pm_gene=effective_pm_gene, server_level=True)
                child = tabu_local_search(inst, child, sim, rng, max_iter=LOCAL_ITER)
                offspring.append(child)
                if len(offspring) >= POP_SIZE:
                    break
            if timeout_hit:
                break

        if timeout_hit:
            break

        population = sorted(population + offspring, key=score)[:POP_SIZE]
        current = population[0]
        current_cost = score(current)
        if current_cost < best_cost:
            best = current.copy()
            best_cost = current_cost

    elapsed = time.perf_counter() - start
    terminal_status = "TIME_LIMIT" if timeout_hit or elapsed >= TIME_LIMIT_S else "GENERATION_LIMIT"
    remember(elapsed, terminal_status)
    for checkpoint_s in CHECKPOINTS:
        if checkpoint_s not in records:
            records[checkpoint_s] = {
                "method": "HISC-MA",
                "checkpoint_s": checkpoint_s,
                "checkpoint_label": f"{checkpoint_s // 60}min",
                "recorded_elapsed_s": elapsed,
                "incumbent_cost": best_cost,
                "best_bound_cost": math.nan,
                "gap_percent": math.nan,
                "generations_completed": generation,
                "status": terminal_status,
                "feasible": is_feasible(inst, best),
            }

    final = {
        "method": "HISC-MA",
        "status": terminal_status,
        "initial_cost": initial_cost,
        "final_incumbent_cost": best_cost,
        "final_bound_cost": math.nan,
        "final_gap_percent": math.nan,
        "elapsed_s": elapsed,
        "generations_completed": generation,
        "feasible": is_feasible(inst, best),
        "assignment": json.dumps(best),
        "extra": json.dumps(
            {
                "pop_size": POP_SIZE,
                "max_generations": MAX_GENERATIONS,
                "pc": PC,
                "pm_gene": PM_GENE,
                "late_pm_gene": LATE_PM_GENE,
                "late_mutation_start": LATE_MUTATION_START,
                "local_iter": LOCAL_ITER,
                "hybrid_init": True,
                "memetic": True,
                "server_centric": True,
                "timeout_s": TIME_LIMIT_S,
            }
        ),
    }
    print(
        f"[HISC-MA] done: status={terminal_status}, incumbent={best_cost:.6f}, "
        f"gen={generation}, elapsed={elapsed:.2f}s",
        flush=True,
    )
    return [records[c] for c in CHECKPOINTS], final


def solve_gurobi_with_time_checkpoints(inst, sim, seed: int) -> tuple[list[dict], dict]:
    method = "Gurobi GQAP"
    denom = float(sim.sum())
    print(f"[{method}] building GQAP model...", flush=True)
    build_start = time.perf_counter()
    model, aux = build_gqap_model(inst, sim, mip_start=None, seed=seed, output_flag=0)
    build_time = time.perf_counter() - build_start
    model.Params.TimeLimit = float(TIME_LIMIT_S)
    print(
        f"[{method}] model built in {build_time:.2f}s, x={inst.llm_count * inst.g_count}, "
        f"quadratic_terms={aux['quadratic_terms']}, optimizing {TIME_LIMIT_S}s...",
        flush=True,
    )

    records: dict[int, dict] = {}

    def remember(runtime: float, best_raw: float, bound_raw: float, source: str) -> None:
        for checkpoint_s in CHECKPOINTS:
            if checkpoint_s not in records and runtime >= checkpoint_s:
                best_norm = norm_value(best_raw, denom)
                bound_norm = norm_value(bound_raw, denom)
                records[checkpoint_s] = {
                    "method": method,
                    "checkpoint_s": checkpoint_s,
                    "recorded_runtime_s": runtime,
                    "incumbent_cost": best_norm,
                    "best_bound_cost": bound_norm,
                    "gap_percent": gap_percent(best_norm, bound_norm),
                    "record_source": source,
                }
                print(
                    f"[{method}] checkpoint {checkpoint_s}s: incumbent={best_norm:.6f}, "
                    f"bound={bound_norm:.6f}, gap={records[checkpoint_s]['gap_percent']:.2f}%, "
                    f"source={source}",
                    flush=True,
                )

    optimize_start = time.perf_counter()
    model.optimizeAsync()
    while model.Status == GRB.INPROGRESS:
        elapsed = time.perf_counter() - optimize_start
        try:
            current_best_raw = float(model.ObjVal)
            current_bound_raw = float(model.ObjBound)
        except gp.GurobiError:
            current_best_raw = math.nan
            current_bound_raw = math.nan
        remember(elapsed, current_best_raw, current_bound_raw, source="async_poll")
        time.sleep(0.25)
    model.sync()
    optimize_wall = time.perf_counter() - optimize_start
    final_best_raw = float(model.ObjVal) if model.SolCount > 0 else math.nan
    final_bound_raw = float(model.ObjBound) if model.SolCount > 0 or math.isfinite(model.ObjBound) else math.nan
    final_best = norm_value(final_best_raw, denom)
    final_bound = norm_value(final_bound_raw, denom)
    final_gap = gap_percent(final_best, final_bound)
    runtime_for_fill = float(model.Runtime)

    for checkpoint_s in CHECKPOINTS:
        if checkpoint_s not in records:
            records[checkpoint_s] = {
                "method": method,
                "checkpoint_s": checkpoint_s,
                "recorded_runtime_s": runtime_for_fill,
                "incumbent_cost": final_best,
                "best_bound_cost": final_bound,
                "gap_percent": final_gap,
                "record_source": "async_final",
            }

    assignment = []
    feasible = False
    if model.SolCount > 0:
        x = aux["x"]
        assignment = [max(range(inst.g_count), key=lambda n: x[i, n].X) for i in range(inst.llm_count)]
        feasible = is_feasible(inst, assignment)
    final = {
        "method": method,
        "status": status_name(model.Status),
        "solver_runtime_s": float(model.Runtime),
        "optimize_wall_s": optimize_wall,
        "build_time_s": build_time,
        "sol_count": int(model.SolCount),
        "node_count": float(model.NodeCount),
        "final_incumbent_cost": final_best,
        "final_bound_cost": final_bound,
        "final_gap_percent": final_gap,
        "final_feasible": feasible,
        "final_assignment": json.dumps(assignment),
        "quadratic_terms": aux["quadratic_terms"],
    }
    print(
        f"[{method}] done: status={final['status']}, incumbent={final_best:.6f}, "
        f"bound={final_bound:.6f}, gap={final_gap:.2f}%, runtime={model.Runtime:.2f}s",
        flush=True,
    )
    return [records[c] for c in CHECKPOINTS], final


def draw_plot(combined_rows: list[dict]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int, bold: bool = False):
        names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
        for name in names:
            path = Path("C:/Windows/Fonts") / name
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        return ImageFont.load_default()

    width, height = 2580, 1470
    margin_left, margin_right, margin_top, margin_bottom = 250, 110, 160, 190
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = {"HISC-MA": (31, 119, 180), "Gurobi GQAP": (242, 142, 43)}
    labels = {"HISC-MA": "HISC-MA", "Gurobi GQAP": "Gurobi"}
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_title = font(50, True)
    f_axis = font(42, True)
    f_tick = font(34)
    f_legend = font(36, True)

    all_y = [float(row["incumbent_cost"]) for row in combined_rows if math.isfinite(float(row["incumbent_cost"]))]
    y_min = min(all_y)
    y_max = max(all_y)
    pad = max(1e-6, (y_max - y_min) * 0.08)
    y_min = max(0.0, y_min - pad)
    y_max += pad

    def x_map(minute: float) -> float:
        return margin_left + (minute - 1.0) / 9.0 * plot_w

    def y_map(value: float) -> float:
        return margin_top + (y_max - value) / max(1e-12, y_max - y_min) * plot_h

    # Grid and axes.
    for minute in range(1, 11):
        x = x_map(float(minute))
        draw.line((x, margin_top, x, margin_top + plot_h), fill=(235, 238, 242), width=2)
        label = str(minute)
        bbox = draw.textbbox((0, 0), label, font=f_tick)
        draw.text((x - (bbox[2] - bbox[0]) / 2, margin_top + plot_h + 22), label, fill=(30, 30, 30), font=f_tick)
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_map(value)
        draw.line((margin_left, y, margin_left + plot_w, y), fill=(220, 225, 230), width=2)
        label = f"{value:.3f}"
        bbox = draw.textbbox((0, 0), label, font=f_tick)
        draw.text((margin_left - 24 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), label, fill=(30, 30, 30), font=f_tick)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(30, 30, 30), width=4)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill=(30, 30, 30), width=4)

    # Lines.
    for method in ["HISC-MA", "Gurobi GQAP"]:
        rows = sorted([row for row in combined_rows if row["method"] == method], key=lambda row: float(row["checkpoint_s"]))
        points = [(x_map(float(row["checkpoint_s"]) / 60.0), y_map(float(row["incumbent_cost"]))) for row in rows]
        if len(points) > 1:
            draw.line(points, fill=colors[method], width=8, joint="curve")
        for x, y in points:
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=colors[method], outline="white", width=4)

    # Labels and legend.
    title = "LLM=80, G=27, Pop=40, Pc=0.88"
    bbox = draw.textbbox((0, 0), title, font=f_title)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, 48), title, fill=(20, 20, 20), font=f_title)
    xlabel = "Time (min)"
    bbox = draw.textbbox((0, 0), xlabel, font=f_axis)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, height - 82), xlabel, fill=(20, 20, 20), font=f_axis)
    ylabel = "Convergence Cost"
    label_img = Image.new("RGBA", (560, 80), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 0), ylabel, fill=(20, 20, 20), font=f_axis)
    rotated = label_img.rotate(90, expand=True)
    img.paste(rotated, (45, margin_top + plot_h // 2 - rotated.height // 2), rotated)

    legend_x = margin_left + plot_w - 520
    legend_y = 72
    for idx, method in enumerate(["HISC-MA", "Gurobi GQAP"]):
        y = legend_y + idx * 52
        draw.line((legend_x, y + 18, legend_x + 80, y + 18), fill=colors[method], width=8)
        draw.ellipse((legend_x + 32, y + 7, legend_x + 48, y + 23), fill=colors[method])
        draw.text((legend_x + 105, y), labels[method], fill=(20, 20, 20), font=f_legend)

    img.save(OUT_DIR / "hisc_ma_vs_gurobi_L80_10min_300dpi.png", dpi=(300, 300))
    img.save(OUT_DIR / "hisc_ma_vs_gurobi_L80_10min_300dpi.pdf", "PDF", resolution=300.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM=80 HISC-MA vs Gurobi 10-minute checkpoint comparison.")
    parser.add_argument("--gurobi-only", action="store_true", help="Reuse saved HISC-MA rows and rerun only Gurobi.")
    args = parser.parse_args()
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
        f"pop={POP_SIZE}, max_gen={MAX_GENERATIONS}, pc={PC}, "
        f"time_limit={TIME_LIMIT_S}s, checkpoints={CHECKPOINTS}",
        flush=True,
    )

    if args.gurobi_only:
        hisc_rows = read_csv(OUT_DIR / "hisc_ma_checkpoints.csv")
        hisc_final = read_csv(OUT_DIR / "hisc_ma_final.csv")[0]
        print("[HISC-MA] reusing saved 10-minute checkpoints.", flush=True)
    else:
        hisc_rows, hisc_final = run_hisc_ma_with_time_checkpoints(inst, sim, SEED + 404)
        write_csv(OUT_DIR / "hisc_ma_checkpoints.csv", hisc_rows)
        write_csv(OUT_DIR / "hisc_ma_final.csv", [hisc_final])

    gurobi_rows, gurobi_final = solve_gurobi_with_time_checkpoints(inst, sim, SEED + 100)
    write_csv(OUT_DIR / "gurobi_checkpoints.csv", gurobi_rows)
    write_csv(OUT_DIR / "gurobi_final.csv", [gurobi_final])

    combined_rows = hisc_rows + [
        {
            "method": row["method"],
            "checkpoint_s": row["checkpoint_s"],
            "checkpoint_label": f"{int(row['checkpoint_s']) // 60}min",
            "recorded_elapsed_s": row["recorded_runtime_s"],
            "incumbent_cost": row["incumbent_cost"],
            "best_bound_cost": row["best_bound_cost"],
            "gap_percent": row["gap_percent"],
            "generations_completed": "",
            "status": "TIME_LIMIT" if int(row["checkpoint_s"]) == TIME_LIMIT_S else "RUNNING",
            "feasible": "",
        }
        for row in gurobi_rows
    ]
    combined_rows.sort(key=lambda row: (float(row["checkpoint_s"]), row["method"]))
    write_csv(OUT_DIR / "checkpoint_comparison.csv", combined_rows)

    hisc_cost = float(hisc_final["final_incumbent_cost"])
    gurobi_cost = float(gurobi_final["final_incumbent_cost"])
    winner = "HISC-MA" if hisc_cost < gurobi_cost else "Gurobi GQAP" if gurobi_cost < hisc_cost else "Tie"
    improvement = (gurobi_cost - hisc_cost) / max(abs(gurobi_cost), 1e-12) * 100.0
    summary_rows = [
        {
            "method": "HISC-MA",
            "final_cost": hisc_cost,
            "elapsed_s": hisc_final["elapsed_s"],
            "status": hisc_final["status"],
            "feasible": hisc_final["feasible"],
        },
        {
            "method": "Gurobi GQAP",
            "final_cost": gurobi_cost,
            "elapsed_s": gurobi_final["solver_runtime_s"],
            "status": gurobi_final["status"],
            "feasible": gurobi_final["final_feasible"],
        },
    ]
    write_csv(OUT_DIR / "comparison_summary.csv", summary_rows)
    (OUT_DIR / "winner.json").write_text(
        json.dumps(
            {
                "winner": winner,
                "hisc_cost": hisc_cost,
                "gurobi_cost": gurobi_cost,
                "hisc_relative_improvement_vs_gurobi_percent": improvement,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    draw_plot(combined_rows)
    print(
        f"Winner={winner}; HISC-MA={hisc_cost:.6f}, "
        f"Gurobi={gurobi_cost:.6f}, HISC improvement={improvement:.2f}%",
        flush=True,
    )
    print(f"Wrote outputs to {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
