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


METHODS = [
    ("gurobi", "Gurobi MIP original", "gurobipy + Gurobi license"),
    ("scip", "SCIP MIQP original", "pyscipopt"),
    ("cpsat", "OR-Tools CP-SAT original", "ortools"),
    ("gqap_tool_mip", "JOLT", "gurobipy or pyscipopt, depending on solver options"),
]

METHOD_ALIASES = {
    "jolt": "gqap_tool_mip",
}

METHOD_CLI_NAMES = {
    "gqap_tool_mip": "jolt",
}


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def method_name(method_key: str) -> str:
    method_key = METHOD_ALIASES.get(method_key.strip().lower(), method_key)
    names = {key: name for key, name, _ in METHODS}
    if method_key not in names:
        supported = sorted([*names, *METHOD_ALIASES])
        raise ValueError(f"Unsupported method '{method_key}'. Supported: {', '.join(supported)}")
    return names[method_key]


def normalize_method_key(method_key: str) -> str:
    key = method_key.strip().lower()
    return METHOD_ALIASES.get(key, key)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def combo_complete(rows: list[dict], llms: int, method: str, checkpoints: list[float]) -> bool:
    seen = {
        float(row["checkpoint_s"])
        for row in rows
        if int(row["llms"]) == llms and str(row["method"]) == method
    }
    return all(any(abs(item - checkpoint) < 1e-9 for item in seen) for checkpoint in checkpoints)


def time_sort_key(label: str) -> float:
    label = str(label).strip()
    if label.endswith("min"):
        try:
            return float(label[:-3])
        except ValueError:
            return math.inf
    return math.inf


def write_wide_table(path: Path, rows: list[dict], value_field: str, missing: str) -> None:
    methods = [name for _, name, _ in METHODS]
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
        item[str(row["method"])] = row[value_field]

    wide_rows = []
    for key in sorted(by_key, key=lambda x: (x[0], time_sort_key(x[2]))):
        item = by_key[key]
        wide_rows.append(
            {
                "LLM": item["LLM"],
                "Tools": item["Tools"],
                "Time": item["Time"],
                **{method: item.get(method, missing) for method in methods},
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["LLM", "Tools", "Time", *methods])
        writer.writeheader()
        writer.writerows(wide_rows)


def print_summary(rows: list[dict]) -> None:
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


def resolve_two_stage_solvers(args: argparse.Namespace) -> tuple[str, str]:
    gqap_solver = args.gqap_solver
    tool_mip_solver = args.tool_mip_solver
    if args.two_stage_solver is not None:
        gqap_solver = args.two_stage_solver
        tool_mip_solver = args.two_stage_solver
    return gqap_solver, tool_mip_solver


def run_experiment(args: argparse.Namespace, llm_values: list[int]) -> None:
    gqap_solver, tool_mip_solver = resolve_two_stage_solvers(args)
    selected_keys = [normalize_method_key(item) for item in args.methods.split(",") if item.strip()]
    selected = [(key, method_name(key)) for key in selected_keys]
    checkpoints = sorted(c for c in parse_float_list(args.checkpoints_s) if 0 < c <= args.time_limit_s)

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
                max_time_s=args.time_limit_s,
                capacity_mode=args.capacity_mode,
                hard_overhead_s=args.hard_overhead_s,
                rel_tol=args.stable_rel_tol,
                abs_tol=args.stable_abs_tol,
                skip_after_two_nan=args.skip_after_two_nan,
                gqap_solver=gqap_solver,
                tool_mip_solver=tool_mip_solver,
            )

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
    write_wide_table(args.out_dir / "avg_call_distance_wide.csv", all_rows, "avg_call_distance", "NaN")
    write_wide_table(args.out_dir / "status_wide.csv", all_rows, "status", "NA")
    print_summary(all_rows)
    print(f"\nWrote outputs to {args.out_dir.resolve()}", flush=True)


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tool-ratio", type=int, default=3)
    parser.add_argument("--time-limit-s", type=float, default=600.0)
    parser.add_argument("--checkpoints-s", default="60,180,300,420,600")
    parser.add_argument("--seed-base", type=int, default=20260528)
    parser.add_argument("--capacity-mode", choices=["fixed_per_server", "g_only_fixed_per_server"], default="fixed_per_server")
    parser.add_argument("--methods", default=",".join(key for key, _, _ in METHODS))
    parser.add_argument("--gqap-solver", choices=["gurobi", "scip"], default="gurobi")
    parser.add_argument("--tool-mip-solver", choices=["gurobi", "scip"], default="gurobi")
    parser.add_argument("--two-stage-solver", choices=["gurobi", "scip"], default=None)
    parser.add_argument("--hard-overhead-s", type=float, default=180.0)
    parser.add_argument("--stable-rel-tol", type=float, default=0.01)
    parser.add_argument("--stable-abs-tol", type=float, default=0.005)
    parser.add_argument("--skip-after-two-nan", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)


def list_methods(_: argparse.Namespace) -> None:
    for key, name, requirements in METHODS:
        cli_key = METHOD_CLI_NAMES.get(key, key)
        alias_text = f" (alias: {key})" if cli_key != key else ""
        print(f"{cli_key:<14} {name:<28} requires: {requirements}{alias_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Command-line runner for JOLT deployment experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="Run a tiny JOLT correctness test.")
    smoke_parser.add_argument("--llms", type=int, default=5)
    smoke_parser.add_argument("--tool-ratio", type=int, default=3)
    smoke_parser.add_argument("--time-limit-s", type=float, default=4.0)
    smoke_parser.add_argument("--checkpoints-s", default="2,4")
    smoke_parser.add_argument("--seed-base", type=int, default=20260528)
    smoke_parser.add_argument("--capacity-mode", choices=["fixed_per_server", "g_only_fixed_per_server"], default="fixed_per_server")
    smoke_parser.add_argument("--methods", default="jolt")
    smoke_parser.add_argument("--gqap-solver", choices=["gurobi", "scip"], default="gurobi")
    smoke_parser.add_argument("--tool-mip-solver", choices=["gurobi", "scip"], default="gurobi")
    smoke_parser.add_argument("--two-stage-solver", choices=["gurobi", "scip"], default=None)
    smoke_parser.add_argument("--hard-overhead-s", type=float, default=30.0)
    smoke_parser.add_argument("--stable-rel-tol", type=float, default=0.01)
    smoke_parser.add_argument("--stable-abs-tol", type=float, default=0.005)
    smoke_parser.add_argument("--skip-after-two-nan", action="store_true")
    smoke_parser.add_argument("--restart", action="store_true", default=True)
    smoke_parser.add_argument("--out-dir", type=Path, default=Path("results/smoke"))
    smoke_parser.set_defaults(func=lambda args: run_experiment(args, [args.llms]))

    run_parser = subparsers.add_parser("run", help="Run one LLM scale.")
    run_parser.add_argument("--llms", type=int, required=True)
    add_common_run_args(run_parser)
    run_parser.set_defaults(func=lambda args: run_experiment(args, [args.llms]))

    sweep_parser = subparsers.add_parser("sweep", help="Run multiple LLM scales.")
    sweep_parser.add_argument("--llm-list", default="20,40,60,80")
    add_common_run_args(sweep_parser)
    sweep_parser.set_defaults(func=lambda args: run_experiment(args, parse_int_list(args.llm_list)))

    list_parser = subparsers.add_parser("list-methods", help="Print available methods and solver requirements.")
    list_parser.set_defaults(func=list_methods)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
