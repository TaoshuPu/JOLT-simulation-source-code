from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from jolt_full_c_type_gqap_hisc_vs_gurobi_L80_10min import (
    CHECKPOINTS,
    LLMS,
    OUT_DIR,
    SEED,
    TOOLS,
    draw_plot,
    run_gurobi_gqap,
    write_csv,
)
from jolt_small_scale_experiment import instance_to_jsonable, llm_similarity, make_instance


HISC_TRACE = Path("jolt_full_c_type_L80_10min_compare/hisc_tool_mip/trace_L80_hisc_ma_phase.json")


def hisc_rows_from_trace(trace_path: Path) -> tuple[list[dict], dict, list[dict]]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    batch_records = trace["batch_records"]
    rows = []
    best_cost = math.inf
    best_elapsed = 0.0
    best_source = {"source": "none"}
    for checkpoint_s in CHECKPOINTS:
        for batch in batch_records:
            elapsed = float(batch["elapsed_s"])
            if elapsed > checkpoint_s + 0.50:
                continue
            cost = float(batch["candidate_surrogate_cost"])
            if math.isfinite(cost) and cost < best_cost - 1e-12:
                best_cost = cost
                best_elapsed = elapsed
                best_source = {
                    "source": "hisc_ma_batch_trace",
                    "batch": int(batch["batch"]),
                    "batch_seed": int(batch["batch_seed"]),
                    "batch_solver_status": batch.get("solver_status"),
                    "batch_generations_completed": batch.get("generations_completed"),
                }
        rows.append(
            {
                "method": "HISC-MA",
                "checkpoint_s": checkpoint_s,
                "minute": int(checkpoint_s // 60),
                "recorded_elapsed_s": checkpoint_s,
                "gqap_cost": best_cost,
                "best_bound": math.nan,
                "gap_percent": math.nan,
                "status": "TRACE_REUSED",
                "batches_completed": len([b for b in batch_records if float(b["elapsed_s"]) <= checkpoint_s + 0.50]),
                "incumbent_elapsed_s": best_elapsed,
                "extra": json.dumps({"best_source": best_source, "trace_path": str(trace_path)}, ensure_ascii=False),
            }
        )
    final = {
        "method": "HISC-MA",
        "final_gqap_cost": rows[-1]["gqap_cost"],
        "elapsed_s": CHECKPOINTS[-1],
        "status": "TRACE_REUSED",
        "assignment": "",
        "batches_completed": rows[-1]["batches_completed"],
        "extra": json.dumps({"trace_path": str(trace_path), "best_source": best_source}, ensure_ascii=False),
    }
    return rows, final, batch_records


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

    hisc_rows, hisc_final, hisc_batches = hisc_rows_from_trace(HISC_TRACE)
    write_csv(OUT_DIR / "hisc_ma_gqap_checkpoints.csv", hisc_rows)
    write_csv(OUT_DIR / "hisc_ma_gqap_batches_from_trace.csv", hisc_batches)
    write_csv(OUT_DIR / "hisc_ma_gqap_final.csv", [hisc_final])
    print("Reused HISC-MA GQAP costs from saved full C-Type trace.", flush=True)

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
