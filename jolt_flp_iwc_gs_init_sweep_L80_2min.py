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

from jolt_flp_iwc_gs_best_L80_T240 import (
    SOURCE_TRACE,
    load_best_llm_assignment,
    solve_flp_with_checkpoints,
    write_csv,
)
from jolt_small_scale_experiment import (
    average_call_distance,
    instance_to_jsonable,
    is_llm_feasible,
    is_tool_feasible,
    iwc_gs_tool_deployment,
    make_instance,
    tool_host_count,
)


LLMS = 80
TOOL_COUNTS = [500, 1000, 1500, 2000]
SEED = 20340528
TIME_LIMIT_S = 120.0
CHECKPOINTS = [0, 1, 5, 10, 30, 60, 120]
OUT_DIR = ROOT / "jolt_flp_iwc_gs_init_sweep_L80_2min"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def percent_gap(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(candidate) or abs(reference) < 1e-12:
        return math.nan
    return 100.0 * (candidate - reference) / abs(reference)


def to_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def make_svg(summary_rows: list[dict]) -> None:
    svg_path = OUT_DIR / "flp_iwc_gs_init_sweep_L80_2min.svg"
    width, height = 980, 560
    ml, mr, mt, mb = 105, 40, 70, 105
    pw, ph = width - ml - mr, height - mt - mb
    tools = [int(row["tools"]) for row in summary_rows]
    series = {
        "Gurobi FLP direct": [float(row["direct_final_avg_call_distance"]) for row in summary_rows],
        "Gurobi FLP + IWC-GS init": [float(row["warm_final_avg_call_distance"]) for row in summary_rows],
        "IWC-GS initial": [float(row["iwc_gs_initial_avg_call_distance"]) for row in summary_rows],
    }
    values = [v for ys in series.values() for v in ys if math.isfinite(v)]
    ymin = min(values)
    ymax = max(values)
    pad = (ymax - ymin) * 0.12 if ymax > ymin else max(0.01, ymax * 0.1)
    ymin = max(0.0, ymin - pad)
    ymax += pad

    def x_map(tool_count: int) -> float:
        if len(tools) == 1:
            return ml + pw / 2
        return ml + tools.index(tool_count) / (len(tools) - 1) * pw

    def y_map(value: float) -> float:
        return mt + (ymax - value) / max(1e-12, ymax - ymin) * ph

    colors = {
        "Gurobi FLP direct": "#f28e2b",
        "Gurobi FLP + IWC-GS init": "#1f77b4",
        "IWC-GS initial": "#59a14f",
    }
    out: list[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    out.append('<rect width="100%" height="100%" fill="white"/>')
    out.append(
        '<text x="490" y="34" text-anchor="middle" font-family="Arial" '
        'font-size="24" font-weight="700">IWC-GS Warm Start vs Gurobi Direct (LLM=80, 2min)</text>'
    )
    out.append(
        '<text x="490" y="59" text-anchor="middle" font-family="Arial" '
        'font-size="15" font-weight="700">Fixed LLM placement, C-Type servers included; lower is better</text>'
    )

    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = y_map(value)
        out.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{ml + pw}" y2="{y:.2f}" stroke="#dfe5ec" stroke-width="1"/>')
        out.append(
            f'<text x="{ml - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="Arial" '
            f'font-size="14">{value:.3f}</text>'
        )
    for tool_count in tools:
        x = x_map(tool_count)
        out.append(f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{mt + ph}" stroke="#eef2f6" stroke-width="1"/>')
        out.append(
            f'<text x="{x:.2f}" y="{mt + ph + 30}" text-anchor="middle" font-family="Arial" '
            f'font-size="15">{tool_count}</text>'
        )
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#222" stroke-width="2"/>')
    out.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#222" stroke-width="2"/>')

    for name, ys in series.items():
        points = [(x_map(tool_count), y_map(value)) for tool_count, value in zip(tools, ys)]
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        out.append(f'<polyline points="{path}" fill="none" stroke="{colors[name]}" stroke-width="3.4"/>')
        for x, y in points:
            out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{colors[name]}" stroke="white" stroke-width="2"/>')

    out.append(
        f'<text x="{ml + pw / 2:.2f}" y="{height - 36}" text-anchor="middle" '
        'font-family="Arial" font-size="17" font-weight="700">Number of Tools</text>'
    )
    out.append(
        f'<text transform="translate(32 {mt + ph / 2:.2f}) rotate(-90)" text-anchor="middle" '
        'font-family="Arial" font-size="17" font-weight="700">Average Call Distance</text>'
    )

    legend_x, legend_y = ml + 35, height - 80
    for idx, name in enumerate(["Gurobi FLP + IWC-GS init", "Gurobi FLP direct", "IWC-GS initial"]):
        x = legend_x + idx * 285
        out.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 46}" y2="{legend_y}" stroke="{colors[name]}" stroke-width="4"/>')
        out.append(f'<circle cx="{x + 23}" cy="{legend_y}" r="5" fill="{colors[name]}"/>')
        out.append(
            f'<text x="{x + 58}" y="{legend_y + 5}" font-family="Arial" font-size="15" '
            f'font-weight="700">{name}</text>'
        )
    out.append("</svg>")
    svg_path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    llm_place, source_snapshot = load_best_llm_assignment(SOURCE_TRACE)
    source_meta = {
        "llm_count": LLMS,
        "tool_counts": TOOL_COUNTS,
        "seed": SEED,
        "time_limit_s": TIME_LIMIT_S,
        "checkpoints_s": CHECKPOINTS,
        "fixed_llm_source": str(SOURCE_TRACE),
        "fixed_llm_selection_rule": "minimum phase-1 LLM surrogate cost among saved LLM=80 JOLT snapshots",
        "source_snapshot_elapsed_s": source_snapshot.get("elapsed_s"),
        "source_snapshot_avg_call_distance": source_snapshot.get("avg_call_distance"),
        "source_snapshot_extra": source_snapshot.get("extra", {}),
        "llm_assignment": llm_place,
    }
    (OUT_DIR / "fixed_llm_source.json").write_text(json.dumps(source_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    all_checkpoint_rows: list[dict] = []
    all_final_rows: list[dict] = []
    iwc_rows: list[dict] = []
    summary_rows: list[dict] = []

    for tools in TOOL_COUNTS:
        scale_dir = OUT_DIR / f"L80_T{tools}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        inst = make_instance(
            SEED,
            llm_count=LLMS,
            tool_count=tools,
            g_count=None,
            c_count=None,
            capacity_mode="fixed_per_server",
        )
        if not is_llm_feasible(inst, llm_place):
            raise RuntimeError(f"Fixed LLM assignment is infeasible for Tools={tools}.")
        (scale_dir / f"instance_L80_T{tools}.json").write_text(
            json.dumps(instance_to_jsonable(inst), ensure_ascii=False),
            encoding="utf-8",
        )

        iwc_start = time.perf_counter()
        iwc_tool_place = iwc_gs_tool_deployment(inst, llm_place)
        iwc_wall = time.perf_counter() - iwc_start
        if not is_tool_feasible(inst, iwc_tool_place, llm_place):
            raise RuntimeError(f"IWC-GS warm start is infeasible for Tools={tools}.")
        iwc_avg = average_call_distance(inst, llm_place, iwc_tool_place)
        iwc_row = {
            "llms": LLMS,
            "tools": tools,
            "g_servers": inst.g_count,
            "c_servers": inst.c_count,
            "tool_hosts": tool_host_count(inst),
            "iwc_gs_runtime_s": iwc_wall,
            "iwc_gs_initial_avg_call_distance": iwc_avg,
            "tool_feasible": True,
            "iwc_gs_tool_assignment": json.dumps(iwc_tool_place),
        }
        iwc_rows.append(iwc_row)
        write_csv(scale_dir / "iwc_gs_initial_result.csv", [iwc_row])

        print(
            f"\n[Scale] LLM={LLMS}, Tools={tools}, G={inst.g_count}, C={inst.c_count}, "
            f"H={tool_host_count(inst)}, IWC-GS={iwc_avg:.6f} in {iwc_wall:.3f}s",
            flush=True,
        )

        direct_rows, direct_final = solve_flp_with_checkpoints(
            inst=inst,
            llm_place=llm_place,
            method="Gurobi FLP direct",
            warm_start=None,
            seed=SEED + 700,
            time_limit_s=TIME_LIMIT_S,
            checkpoints=CHECKPOINTS,
        )
        warm_rows, warm_final = solve_flp_with_checkpoints(
            inst=inst,
            llm_place=llm_place,
            method="Gurobi FLP + IWC-GS init",
            warm_start=iwc_tool_place,
            seed=SEED + 700,
            time_limit_s=TIME_LIMIT_S,
            checkpoints=CHECKPOINTS,
        )

        for row in direct_rows + warm_rows:
            row = dict(row)
            row["llms"] = LLMS
            row["tools"] = tools
            row["g_servers"] = inst.g_count
            row["c_servers"] = inst.c_count
            row["tool_hosts"] = tool_host_count(inst)
            all_checkpoint_rows.append(row)
        for row in [direct_final, warm_final]:
            row = dict(row)
            row["iwc_gs_runtime_s"] = iwc_wall
            row["iwc_gs_initial_avg_call_distance"] = iwc_avg
            all_final_rows.append(row)

        direct_avg = to_float(direct_final["final_avg_call_distance"])
        warm_avg = to_float(warm_final["final_avg_call_distance"])
        summary = {
            "llms": LLMS,
            "tools": tools,
            "g_servers": inst.g_count,
            "c_servers": inst.c_count,
            "tool_hosts": tool_host_count(inst),
            "time_limit_s": TIME_LIMIT_S,
            "iwc_gs_runtime_s": iwc_wall,
            "iwc_gs_initial_avg_call_distance": iwc_avg,
            "direct_status": direct_final["status"],
            "direct_final_avg_call_distance": direct_avg,
            "direct_gap_percent": to_float(direct_final["final_gap_percent"]),
            "direct_runtime_s": to_float(direct_final["solver_runtime_s"]),
            "warm_status": warm_final["status"],
            "warm_final_avg_call_distance": warm_avg,
            "warm_gap_percent": to_float(warm_final["final_gap_percent"]),
            "warm_runtime_s": to_float(warm_final["solver_runtime_s"]),
            "warm_minus_direct_avg": warm_avg - direct_avg,
            "warm_vs_direct_percent": percent_gap(direct_avg, warm_avg),
            "iwc_initial_minus_direct_avg": iwc_avg - direct_avg,
            "iwc_initial_vs_direct_percent": percent_gap(direct_avg, iwc_avg),
        }
        summary_rows.append(summary)

        write_csv(scale_dir / "flp_checkpoint_results.csv", direct_rows + warm_rows)
        write_csv(scale_dir / "flp_final_results.csv", [direct_final, warm_final])
        write_csv(scale_dir / "summary.csv", [summary])
        write_csv(OUT_DIR / "flp_checkpoint_results_all.csv", all_checkpoint_rows)
        write_csv(OUT_DIR / "flp_final_results_all.csv", all_final_rows)
        write_csv(OUT_DIR / "iwc_gs_initial_results.csv", iwc_rows)
        write_csv(OUT_DIR / "summary.csv", summary_rows)

    make_svg(summary_rows)
    print(f"\nWrote outputs to {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
