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

import gurobipy as gp
from gurobipy import GRB
from PIL import Image, ImageDraw, ImageFont

from jolt_small_scale_experiment import (
    hisc_ma_llm_deployment,
    instance_to_jsonable,
    is_llm_feasible,
    llm_similarity,
    llm_surrogate_cost,
    make_instance,
    random_llm_assignment,
)


OUT_DIR = ROOT / "jolt_full_c_type_gqap_L80_10min_compare"
SEED = 20260528 + 80 * 1000
LLMS = 80
TOOLS = 240
TIME_LIMIT_S = 600.0
CHECKPOINTS = [60.0 * minute for minute in range(1, 11)]
HISC_PARAMS = {
    "pop_size": 128,
    "max_gen": 100_000,
    "pc": 0.92,
    "pm": 0.16,
    "late_pm": 0.35,
    "late_mutation_start": 0.72,
    "local_iter": 4,
    "batch_time_s": 30.0,
}


def finite_cost(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.inf
    return result if math.isfinite(result) else math.inf


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_gurobi_gqap_model(inst, similarity, seed: int) -> tuple[gp.Model, dict]:
    model = gp.Model("jolt_phase1_llm_gqap_full_env")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2
    model.Params.Seed = seed
    model.Params.TimeLimit = TIME_LIMIT_S
    x = model.addVars(inst.llm_count, inst.g_count, vtype=GRB.BINARY, name="x")
    model.addConstrs((gp.quicksum(x[i, n] for n in range(inst.g_count)) == 1 for i in range(inst.llm_count)))
    model.addConstrs(
        (
            gp.quicksum(int(inst.llm_gpu[i]) * x[i, n] for i in range(inst.llm_count))
            <= int(inst.g_gpu_cap[n])
            for n in range(inst.g_count)
        )
    )
    model.addConstrs(
        (
            gp.quicksum(int(inst.llm_cpu[i]) * x[i, n] for i in range(inst.llm_count))
            <= int(inst.g_cpu_cap[n])
            for n in range(inst.g_count)
        )
    )
    model.addConstrs(
        (
            gp.quicksum(int(inst.llm_mem[i]) * x[i, n] for i in range(inst.llm_count))
            <= int(inst.g_mem_cap[n])
            for n in range(inst.g_count)
        )
    )

    objective = gp.QuadExpr()
    coeff_batch = []
    left_batch = []
    right_batch = []
    quadratic_terms = 0

    def flush() -> None:
        nonlocal coeff_batch, left_batch, right_batch
        if coeff_batch:
            objective.addTerms(coeff_batch, left_batch, right_batch)
            coeff_batch = []
            left_batch = []
            right_batch = []

    for i in range(inst.llm_count):
        for k in range(inst.llm_count):
            sim = float(similarity[i, k])
            if sim <= 1e-12:
                continue
            for n in range(inst.g_count):
                for q in range(inst.g_count):
                    dist = float(inst.d_gg[n, q])
                    if dist <= 1e-12:
                        continue
                    coeff_batch.append(sim * dist)
                    left_batch.append(x[i, n])
                    right_batch.append(x[k, q])
                    quadratic_terms += 1
                    if len(coeff_batch) >= 100_000:
                        flush()
    flush()
    model.setObjective(objective, GRB.MINIMIZE)
    model.update()
    return model, {"x": x, "quadratic_terms": quadratic_terms}


def run_hisc_gqap(inst, similarity) -> tuple[list[dict], dict, list[dict]]:
    start = time.perf_counter()
    rng = random.Random(SEED + 91)
    best = random_llm_assignment(inst, rng)
    best_cost = llm_surrogate_cost(inst, best, similarity)
    best_elapsed = 0.0
    best_source = {"source": "random_initial"}
    rows: list[dict] = []
    batches: list[dict] = []
    batches_completed = 0

    for checkpoint_s in CHECKPOINTS:
        while True:
            elapsed = time.perf_counter() - start
            remaining = checkpoint_s - elapsed
            if remaining <= 0.20:
                break
            budget = min(float(HISC_PARAMS["batch_time_s"]), max(0.25, remaining))
            batch_seed = SEED + 10_000 + 104_729 * (batches_completed + 1)
            candidate, extra = hisc_ma_llm_deployment(
                inst,
                batch_seed,
                pop_size=int(HISC_PARAMS["pop_size"]),
                max_gen=int(HISC_PARAMS["max_gen"]),
                pc=float(HISC_PARAMS["pc"]),
                pm=float(HISC_PARAMS["pm"]),
                timeout_s=budget,
                local_iter=int(HISC_PARAMS["local_iter"]),
                late_pm=float(HISC_PARAMS["late_pm"]),
                late_mutation_start=float(HISC_PARAMS["late_mutation_start"]),
            )
            batches_completed += 1
            elapsed = time.perf_counter() - start
            candidate_cost = llm_surrogate_cost(inst, candidate, similarity) if is_llm_feasible(inst, candidate) else math.inf
            improved = candidate_cost < best_cost - 1e-12
            if improved:
                best = list(candidate)
                best_cost = float(candidate_cost)
                best_elapsed = float(elapsed)
                best_source = {
                    "source": "hisc_ma_batch",
                    "batch": batches_completed,
                    "batch_seed": batch_seed,
                    "batch_solver_status": extra.get("solver_status"),
                    "batch_solve_time_s": extra.get("solve_time_s"),
                    "batch_generations_completed": extra.get("generations_completed"),
                }
            batches.append(
                {
                    "batch": batches_completed,
                    "elapsed_s": elapsed,
                    "batch_budget_s": budget,
                    "batch_seed": batch_seed,
                    "candidate_gqap_cost": candidate_cost,
                    "best_gqap_cost": best_cost,
                    "improved": improved,
                    "solver_status": extra.get("solver_status"),
                    "generations_completed": extra.get("generations_completed"),
                }
            )

        elapsed = time.perf_counter() - start
        rows.append(
            {
                "method": "HISC-MA",
                "checkpoint_s": checkpoint_s,
                "minute": int(checkpoint_s // 60),
                "recorded_elapsed_s": elapsed,
                "gqap_cost": best_cost,
                "best_bound": math.nan,
                "gap_percent": math.nan,
                "status": "TIME_LIMIT",
                "batches_completed": batches_completed,
                "incumbent_elapsed_s": best_elapsed,
                "extra": json.dumps({"best_source": best_source}, ensure_ascii=False),
            }
        )
        print(
            f"[HISC-MA] {int(checkpoint_s // 60)}min: cost={best_cost:.6f}, "
            f"batches={batches_completed}, incumbent_at={best_elapsed:.2f}s",
            flush=True,
        )

    final = {
        "method": "HISC-MA",
        "final_gqap_cost": best_cost,
        "elapsed_s": time.perf_counter() - start,
        "status": "TIME_LIMIT",
        "assignment": json.dumps(best),
        "batches_completed": batches_completed,
        "extra": json.dumps({"params": HISC_PARAMS, "best_source": best_source}, ensure_ascii=False),
    }
    return rows, final, batches


def run_gurobi_gqap(inst, similarity) -> tuple[list[dict], dict]:
    print("[Gurobi GQAP] building model...", flush=True)
    build_start = time.perf_counter()
    model, aux = build_gurobi_gqap_model(inst, similarity, seed=SEED + 100)
    build_time = time.perf_counter() - build_start
    print(
        f"[Gurobi GQAP] model built in {build_time:.2f}s, "
        f"x={inst.llm_count * inst.g_count}, q_terms={aux['quadratic_terms']}",
        flush=True,
    )
    records: dict[float, dict] = {}

    def remember(runtime: float, cost_raw: float, bound_raw: float, source: str) -> None:
        cost = finite_cost(cost_raw)
        bound = finite_cost(bound_raw)
        gap = math.nan
        if math.isfinite(cost) and math.isfinite(bound) and abs(cost) > 1e-12:
            gap = max(0.0, 100.0 * (cost - bound) / abs(cost))
        for checkpoint_s in CHECKPOINTS:
            if checkpoint_s not in records and runtime >= checkpoint_s:
                records[checkpoint_s] = {
                    "method": "Gurobi GQAP",
                    "checkpoint_s": checkpoint_s,
                    "minute": int(checkpoint_s // 60),
                    "recorded_elapsed_s": runtime,
                    "gqap_cost": cost,
                    "best_bound": bound if math.isfinite(bound) else math.nan,
                    "gap_percent": gap,
                    "status": "RUNNING",
                    "batches_completed": "",
                    "incumbent_elapsed_s": runtime,
                    "extra": json.dumps({"record_source": source}, ensure_ascii=False),
                }
                print(
                    f"[Gurobi GQAP] {int(checkpoint_s // 60)}min: "
                    f"cost={cost:.6f}, bound={bound if math.isfinite(bound) else math.nan:.6f}",
                    flush=True,
                )

    optimize_start = time.perf_counter()
    model.optimizeAsync()
    while model.Status == GRB.INPROGRESS:
        runtime = time.perf_counter() - optimize_start
        try:
            current_cost = float(model.ObjVal)
            current_bound = float(model.ObjBound)
        except gp.GurobiError:
            current_cost = math.inf
            current_bound = math.nan
        remember(runtime, current_cost, current_bound, "async_poll")
        time.sleep(0.25)
    model.sync()
    optimize_wall = time.perf_counter() - optimize_start

    if model.SolCount > 0:
        x = aux["x"]
        assignment = [max(range(inst.g_count), key=lambda n: x[i, n].X) for i in range(inst.llm_count)]
        final_cost = llm_surrogate_cost(inst, assignment, similarity)
    else:
        assignment = []
        final_cost = math.inf
    final_bound = float(model.ObjBound) if model.SolCount > 0 else math.nan
    final_gap = math.nan
    if math.isfinite(final_cost) and math.isfinite(final_bound) and abs(final_cost) > 1e-12:
        final_gap = max(0.0, 100.0 * (final_cost - final_bound) / abs(final_cost))
    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }.get(model.Status, str(model.Status))

    for checkpoint_s in CHECKPOINTS:
        if checkpoint_s not in records:
            records[checkpoint_s] = {
                "method": "Gurobi GQAP",
                "checkpoint_s": checkpoint_s,
                "minute": int(checkpoint_s // 60),
                "recorded_elapsed_s": float(model.Runtime),
                "gqap_cost": final_cost,
                "best_bound": final_bound,
                "gap_percent": final_gap,
                "status": status_name,
                "batches_completed": "",
                "incumbent_elapsed_s": float(model.Runtime),
                "extra": json.dumps({"record_source": "final_fill"}, ensure_ascii=False),
            }
    records[CHECKPOINTS[-1]]["status"] = status_name
    final = {
        "method": "Gurobi GQAP",
        "final_gqap_cost": final_cost,
        "best_bound": final_bound,
        "gap_percent": final_gap,
        "solver_runtime_s": float(model.Runtime),
        "optimize_wall_s": optimize_wall,
        "build_time_s": build_time,
        "status": status_name,
        "sol_count": int(model.SolCount),
        "node_count": float(model.NodeCount),
        "quadratic_terms": aux["quadratic_terms"],
        "assignment": json.dumps(assignment),
    }
    print(
        f"[Gurobi GQAP] done: status={status_name}, cost={final_cost:.6f}, "
        f"bound={final_bound:.6f}, runtime={model.Runtime:.2f}s",
        flush=True,
    )
    return [records[c] for c in CHECKPOINTS], final


def font(size: int, bold: bool = False):
    for name in ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_plot(rows: list[dict]) -> None:
    png_path = OUT_DIR / "gqap_hisc_ma_vs_gurobi_L80_10min_300dpi.png"
    pdf_path = OUT_DIR / "gqap_hisc_ma_vs_gurobi_L80_10min_300dpi.pdf"
    width, height = 2580, 1470
    ml, mr, mt, mb = 270, 110, 170, 190
    pw, ph = width - ml - mr, height - mt - mb
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_title, f_axis, f_tick, f_legend = font(50, True), font(42, True), font(34), font(36, True)
    colors = {"HISC-MA": (31, 119, 180), "Gurobi GQAP": (242, 142, 43)}
    labels = {"HISC-MA": "HISC-MA", "Gurobi GQAP": "Gurobi GQAP"}
    values = [float(row["gqap_cost"]) for row in rows if math.isfinite(float(row["gqap_cost"]))]
    ymin, ymax = min(values), max(values)
    pad = max(1e-6, (ymax - ymin) * 0.08)
    ymin = max(0.0, ymin - pad)
    ymax += pad

    def x_map(minute: float) -> float:
        return ml + (minute - 1.0) / 9.0 * pw

    def y_map(value: float) -> float:
        return mt + (ymax - value) / max(1e-12, ymax - ymin) * ph

    for minute in range(1, 11):
        x = x_map(float(minute))
        draw.line((x, mt, x, mt + ph), fill=(235, 238, 242), width=2)
        label = str(minute)
        bbox = draw.textbbox((0, 0), label, font=f_tick)
        draw.text((x - (bbox[2] - bbox[0]) / 2, mt + ph + 22), label, fill=(30, 30, 30), font=f_tick)
    for idx in range(6):
        value = ymin + (ymax - ymin) * idx / 5
        y = y_map(value)
        draw.line((ml, y, ml + pw, y), fill=(220, 225, 230), width=2)
        label = f"{value:.2f}"
        bbox = draw.textbbox((0, 0), label, font=f_tick)
        draw.text((ml - 24 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), label, fill=(30, 30, 30), font=f_tick)
    draw.line((ml, mt, ml, mt + ph), fill=(30, 30, 30), width=4)
    draw.line((ml, mt + ph, ml + pw, mt + ph), fill=(30, 30, 30), width=4)

    for method in ["HISC-MA", "Gurobi GQAP"]:
        points = [
            (x_map(float(row["minute"])), y_map(float(row["gqap_cost"])))
            for row in rows
            if row["method"] == method
        ]
        if len(points) > 1:
            draw.line(points, fill=colors[method], width=8, joint="curve")
        for x, y in points:
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=colors[method], outline="white", width=4)

    title = "GQAP Phase: LLM=80, Tools=240, G=20, C=32"
    bbox = draw.textbbox((0, 0), title, font=f_title)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, 48), title, fill=(20, 20, 20), font=f_title)
    xlabel = "Time (min)"
    bbox = draw.textbbox((0, 0), xlabel, font=f_axis)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, height - 82), xlabel, fill=(20, 20, 20), font=f_axis)
    ylabel = "GQAP Objective Cost"
    label_img = Image.new("RGBA", (650, 90), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 0), ylabel, fill=(20, 20, 20), font=f_axis)
    rotated = label_img.rotate(90, expand=True)
    img.paste(rotated, (45, mt + ph // 2 - rotated.height // 2), rotated)

    legend_x, legend_y = ml + pw - 680, 70
    for idx, method in enumerate(["HISC-MA", "Gurobi GQAP"]):
        y = legend_y + idx * 56
        draw.line((legend_x, y + 20, legend_x + 80, y + 20), fill=colors[method], width=8)
        draw.ellipse((legend_x + 32, y + 9, legend_x + 48, y + 25), fill=colors[method])
        draw.text((legend_x + 105, y), labels[method], fill=(20, 20, 20), font=f_legend)

    img.save(png_path, dpi=(300, 300))
    img.save(pdf_path, "PDF", resolution=300.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inst = make_instance(SEED, llm_count=LLMS, tool_count=TOOLS, g_count=None, c_count=None, capacity_mode="fixed_per_server")
    similarity = llm_similarity(inst)
    (OUT_DIR / "instance_L80.json").write_text(json.dumps(instance_to_jsonable(inst), ensure_ascii=False), encoding="utf-8")
    print(
        f"Full environment for GQAP comparison: LLM={inst.llm_count}, Tools={inst.tool_count}, "
        f"G={inst.g_count}, C={inst.c_count}, seed={SEED}",
        flush=True,
    )

    hisc_rows, hisc_final, hisc_batches = run_hisc_gqap(inst, similarity)
    write_csv(OUT_DIR / "hisc_ma_gqap_checkpoints.csv", hisc_rows)
    write_csv(OUT_DIR / "hisc_ma_gqap_batches.csv", hisc_batches)
    write_csv(OUT_DIR / "hisc_ma_gqap_final.csv", [hisc_final])

    gurobi_rows, gurobi_final = run_gurobi_gqap(inst, similarity)
    write_csv(OUT_DIR / "gurobi_gqap_checkpoints.csv", gurobi_rows)
    write_csv(OUT_DIR / "gurobi_gqap_final.csv", [gurobi_final])

    combined = sorted(hisc_rows + gurobi_rows, key=lambda row: (int(row["minute"]), row["method"]))
    write_csv(OUT_DIR / "gqap_checkpoint_comparison.csv", combined)
    h_cost = float(hisc_final["final_gqap_cost"])
    g_cost = float(gurobi_final["final_gqap_cost"])
    winner = "HISC-MA" if h_cost < g_cost else "Gurobi GQAP" if g_cost < h_cost else "Tie"
    improvement = (g_cost - h_cost) / max(abs(g_cost), 1e-12) * 100.0
    write_csv(
        OUT_DIR / "gqap_comparison_summary.csv",
        [
            {"method": "HISC-MA", "final_gqap_cost": h_cost, "status": hisc_final["status"]},
            {"method": "Gurobi GQAP", "final_gqap_cost": g_cost, "status": gurobi_final["status"]},
        ],
    )
    (OUT_DIR / "winner.json").write_text(
        json.dumps(
            {
                "winner": winner,
                "hisc_gqap_cost": h_cost,
                "gurobi_gqap_cost": g_cost,
                "hisc_relative_improvement_vs_gurobi_percent": improvement,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    draw_plot(combined)
    print(
        f"Winner={winner}; HISC-MA={h_cost:.6f}, Gurobi={g_cost:.6f}, "
        f"HISC improvement={improvement:.2f}%",
        flush=True,
    )
    print(f"Wrote outputs to {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
