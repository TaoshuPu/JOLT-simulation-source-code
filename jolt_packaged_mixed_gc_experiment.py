from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from jolt_single_run_checkpoint_sweep import (
    finite,
    run_monitored_method,
    write_rows,
)
from jolt_small_scale_experiment import instance_to_jsonable, make_instance, tool_host_count


PACKAGED_METHODS = [
    ("gurobi", "Gurobi MIP original"),
    ("scip", "SCIP MIQP original"),
    ("cpsat", "OR-Tools CP-SAT original"),
    ("gqap_tool_mip", "JOLT"),
]

METHOD_ALIASES = {
    "jolt": "gqap_tool_mip",
}


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def method_name(method_key: str) -> str:
    method_key = METHOD_ALIASES.get(method_key.strip().lower(), method_key)
    names = dict(PACKAGED_METHODS)
    if method_key not in names:
        supported = sorted([*names, *METHOD_ALIASES])
        raise ValueError(f"Unsupported method '{method_key}'. Supported: {', '.join(supported)}")
    return names[method_key]


def combo_complete(rows: list[dict], llms: int, method: str, checkpoints: list[float]) -> bool:
    seen = {
        float(row["checkpoint_s"])
        for row in rows
        if int(row["llms"]) == llms and str(row["method"]) == method
    }
    return all(any(abs(item - checkpoint) < 1e-9 for item in seen) for checkpoint in checkpoints)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def all_snapshots_are_warm_start(snapshots: list[dict]) -> bool:
    if not snapshots:
        return False
    sources = [str(snap.get("extra", {}).get("source", "")) for snap in snapshots]
    useful = [source for source in sources if source]
    return bool(useful) and all(source in {"warm_start", "fallback_iwc_gs"} for source in useful)


def mark_nan_warm_start_only(rows: list[dict], method_key: str, snapshots: list[dict]) -> list[dict]:
    if method_key not in {"scip", "cpsat"} or not all_snapshots_are_warm_start(snapshots):
        return rows
    updated: list[dict] = []
    for row in rows:
        new_row = dict(row)
        new_row["avg_call_distance"] = "NaN"
        new_row["feasible"] = "False"
        new_row["status"] = "NAN_WARM_START_ONLY"
        extra = {}
        try:
            extra = json.loads(str(new_row.get("extra", "{}")))
        except json.JSONDecodeError:
            extra = {}
        extra.update(
            {
                "reported_as_nan": True,
                "reason": "Only a deterministic warm-start incumbent was available; no direct-solver incumbent was reported.",
            }
        )
        new_row["extra"] = json.dumps(extra, ensure_ascii=False)
        updated.append(new_row)
    return updated


def time_sort_key(label: str) -> float:
    label = str(label).strip()
    if label.endswith("min"):
        try:
            return float(label[:-3])
        except ValueError:
            return math.inf
    return math.inf


def write_wide_distance_table(path: Path, rows: list[dict]) -> None:
    methods = [name for _, name in PACKAGED_METHODS]
    by_key: dict[tuple[int, int, str], dict] = {}
    for row in rows:
        key = (int(row["llms"]), int(row["tools"]), str(row["checkpoint_label"]))
        item = by_key.setdefault(
            key,
            {
                "LLM": int(row["llms"]),
                "Tools": int(row["tools"]),
                "Time": str(row["checkpoint_label"]),
            },
        )
        item[str(row["method"])] = row["avg_call_distance"]

    wide_rows = []
    for key in sorted(by_key, key=lambda x: (x[0], time_sort_key(x[2]))):
        item = by_key[key]
        wide_rows.append(
            {
                "LLM": item["LLM"],
                "Tools": item["Tools"],
                "Time": item["Time"],
                **{method: item.get(method, "NaN") for method in methods},
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["LLM", "Tools", "Time", *methods])
        writer.writeheader()
        writer.writerows(wide_rows)


def write_wide_status_table(path: Path, rows: list[dict]) -> None:
    methods = [name for _, name in PACKAGED_METHODS]
    by_key: dict[tuple[int, int, str], dict] = {}
    for row in rows:
        key = (int(row["llms"]), int(row["tools"]), str(row["checkpoint_label"]))
        item = by_key.setdefault(
            key,
            {
                "LLM": int(row["llms"]),
                "Tools": int(row["tools"]),
                "Time": str(row["checkpoint_label"]),
            },
        )
        item[str(row["method"])] = row["status"]

    wide_rows = []
    for key in sorted(by_key, key=lambda x: (x[0], time_sort_key(x[2]))):
        item = by_key[key]
        wide_rows.append(
            {
                "LLM": item["LLM"],
                "Tools": item["Tools"],
                "Time": item["Time"],
                **{method: item.get(method, "NA") for method in methods},
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["LLM", "Tools", "Time", *methods])
        writer.writeheader()
        writer.writerows(wide_rows)


def print_distance_summary(rows: list[dict]) -> None:
    latest = sorted(rows, key=lambda r: (int(r["llms"]), time_sort_key(r["checkpoint_label"]), str(r["method"])))
    print("\nAverage call distance summary:", flush=True)
    for row in latest:
        value = finite(row["avg_call_distance"])
        text = "NaN" if value is None else f"{value:.6f}"
        print(
            f"  L={int(row['llms']):>3} T={int(row['tools']):>3} "
            f"{row['checkpoint_label']:>5} | {row['method']:<26} {text:<10} {row['status']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Packaged mixed G/C JOLT comparison: direct Gurobi, direct SCIP, "
            "direct OR-Tools CP-SAT, and JOLT."
        )
    )
    parser.add_argument("--llm-list", default="20,40,60,80")
    parser.add_argument("--tool-ratio", type=int, default=3)
    parser.add_argument("--checkpoints-s", default="60,180,300,420,600")
    parser.add_argument("--max-time-s", type=float, default=600.0)
    parser.add_argument("--seed-base", type=int, default=20260528)
    parser.add_argument("--capacity-mode", choices=["fixed_per_server", "g_only_fixed_per_server"], default="fixed_per_server")
    parser.add_argument("--methods", default=",".join(key for key, _ in PACKAGED_METHODS))
    parser.add_argument("--hard-overhead-s", type=float, default=180.0)
    parser.add_argument("--stable-rel-tol", type=float, default=0.01)
    parser.add_argument("--stable-abs-tol", type=float, default=0.005)
    parser.add_argument("--skip-after-two-nan", action="store_true")
    parser.add_argument("--keep-warm-start-only", action="store_true")
    parser.add_argument("--gqap-solver", choices=["gurobi", "scip"], default="gurobi")
    parser.add_argument("--tool-mip-solver", choices=["gurobi", "scip"], default="gurobi")
    parser.add_argument("--two-stage-solver", choices=["gurobi", "scip"], default=None)
    parser.add_argument("--restart", action="store_true", help="Ignore existing summary rows and rerun selected methods.")
    parser.add_argument("--out-dir", type=Path, default=Path("jolt_packaged_mixed_gc_20_80"))
    args = parser.parse_args()

    if args.two_stage_solver is not None:
        args.gqap_solver = args.two_stage_solver
        args.tool_mip_solver = args.two_stage_solver

    selected_keys = [METHOD_ALIASES.get(item.strip().lower(), item.strip()) for item in args.methods.split(",") if item.strip()]
    selected = [(key, method_name(key)) for key in selected_keys]
    llm_values = parse_int_list(args.llm_list)
    checkpoints = sorted(c for c in parse_float_list(args.checkpoints_s) if 0 < c <= args.max_time_s)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary_long.csv"
    all_rows: list[dict] = [] if args.restart else load_rows(summary_path)

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
        instance_path = args.out_dir / f"instance_L{llms}.json"
        instance_path.write_text(json.dumps(instance_to_jsonable(inst), ensure_ascii=False), encoding="utf-8")
        print(
            f"\nScale L={llms}, T={tools}, G={inst.g_count}, C={inst.c_count}, H={tool_host_count(inst)}",
            flush=True,
        )

        for method_key, display_name in selected:
            if combo_complete(all_rows, llms, display_name, checkpoints):
                print(f"  skip {display_name} (existing complete)", flush=True)
                continue
            print(f"  running {display_name} ...", flush=True)
            rows, snapshots, done, error, stopped = run_monitored_method(
                method_key=method_key,
                seed=seed,
                llms=llms,
                tools=tools,
                checkpoints=checkpoints,
                max_time_s=args.max_time_s,
                capacity_mode=args.capacity_mode,
                hard_overhead_s=args.hard_overhead_s,
                rel_tol=args.stable_rel_tol,
                abs_tol=args.stable_abs_tol,
                skip_after_two_nan=args.skip_after_two_nan,
                gqap_solver=args.gqap_solver,
                tool_mip_solver=args.tool_mip_solver,
            )
            if not args.keep_warm_start_only:
                rows = mark_nan_warm_start_only(rows, method_key, snapshots)

            all_rows = [
                row
                for row in all_rows
                if not (int(row["llms"]) == llms and str(row["method"]) == display_name)
            ]
            all_rows.extend(rows)
            write_rows(summary_path, all_rows)
            trace_path = args.out_dir / f"trace_L{llms}_{method_key}.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "snapshots": snapshots,
                        "done": done,
                        "error": error,
                        "stopped": stopped,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            values = []
            for row in rows:
                value = finite(row["avg_call_distance"])
                values.append("NaN" if value is None else f"{value:.6f}")
            terminal = (done or stopped or error or {}).get("status", "")
            print(f"    checkpoints: {', '.join(values)}; terminal={terminal}", flush=True)

    all_rows = load_rows(summary_path)
    write_wide_distance_table(args.out_dir / "avg_call_distance_wide.csv", all_rows)
    write_wide_status_table(args.out_dir / "status_wide.csv", all_rows)
    print_distance_summary(all_rows)
    print(f"\nWrote outputs to {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
