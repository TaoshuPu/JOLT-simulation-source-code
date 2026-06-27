from __future__ import annotations

import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for deps_name in ('.gurobi_deps', '.ortools_deps'):
    deps = ROOT / deps_name
    if deps.exists():
        sys.path.insert(0, str(deps))

from jolt_flp_iwc_gs_best_L80_T240 import SOURCE_TRACE, load_best_llm_assignment, solve_flp_with_checkpoints, write_csv
from jolt_small_scale_experiment import average_call_distance, instance_to_jsonable, is_llm_feasible, is_tool_feasible, iwc_gs_tool_deployment, make_instance, tool_host_count

LLMS = 80
TOOLS = 10000
SEED = 20340528
TIME_LIMIT_S = 120.0
CHECKPOINTS = [0, 1, 5, 10, 30, 60, 120]
OUT_DIR = ROOT / 'jolt_flp_iwc_gs_init_L80_T10000_2min'

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_path = OUT_DIR / 'stage_status.json'
    def status(data):
        stage_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(data, ensure_ascii=False), flush=True)

    llm_place, source_snapshot = load_best_llm_assignment(SOURCE_TRACE)
    status({'stage': 'make_instance_start', 'llms': LLMS, 'tools': TOOLS})
    t0 = time.perf_counter()
    inst = make_instance(SEED, llm_count=LLMS, tool_count=TOOLS, g_count=None, c_count=None, capacity_mode='fixed_per_server')
    make_time = time.perf_counter() - t0
    h_count = tool_host_count(inst)
    meta = {
        'llms': LLMS,
        'tools': TOOLS,
        'g_servers': inst.g_count,
        'c_servers': inst.c_count,
        'tool_hosts': h_count,
        'y_variables': TOOLS * h_count,
        'tool_cpu_sum': int(inst.tool_cpu.sum()),
        'tool_mem_sum': int(inst.tool_mem.sum()),
        'make_instance_s': make_time,
        'llm_feasible': is_llm_feasible(inst, llm_place),
        'fixed_llm_source': str(SOURCE_TRACE),
        'source_snapshot_elapsed_s': source_snapshot.get('elapsed_s'),
    }
    status({'stage': 'make_instance_done', **meta})
    (OUT_DIR / 'instance_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    # Keep the huge full instance optional; metadata and assignments are enough for this stress test.

    status({'stage': 'iwc_gs_start', **meta})
    t1 = time.perf_counter()
    iwc_tool = iwc_gs_tool_deployment(inst, llm_place)
    iwc_time = time.perf_counter() - t1
    iwc_avg = average_call_distance(inst, llm_place, iwc_tool)
    iwc_feasible = is_tool_feasible(inst, iwc_tool, llm_place)
    iwc_row = {
        **meta,
        'iwc_gs_runtime_s': iwc_time,
        'iwc_gs_initial_avg_call_distance': iwc_avg,
        'iwc_gs_feasible': iwc_feasible,
        'iwc_gs_tool_assignment': json.dumps(iwc_tool),
    }
    write_csv(OUT_DIR / 'iwc_gs_initial_result.csv', [iwc_row])
    status({'stage': 'iwc_gs_done', 'iwc_gs_runtime_s': iwc_time, 'iwc_gs_avg_call_distance': iwc_avg, 'iwc_gs_feasible': iwc_feasible, **meta})

    final_rows = []
    checkpoint_rows = []
    try:
        status({'stage': 'gurobi_direct_start', **meta})
        direct_rows, direct_final = solve_flp_with_checkpoints(
            inst=inst,
            llm_place=llm_place,
            method='Gurobi FLP direct',
            warm_start=None,
            seed=SEED + 700,
            time_limit_s=TIME_LIMIT_S,
            checkpoints=CHECKPOINTS,
        )
        checkpoint_rows.extend(direct_rows)
        final_rows.append(direct_final)
        write_csv(OUT_DIR / 'flp_checkpoint_results.csv', checkpoint_rows)
        write_csv(OUT_DIR / 'flp_final_results.csv', final_rows)
        status({'stage': 'gurobi_direct_done', 'status': direct_final.get('status'), 'avg': direct_final.get('final_avg_call_distance'), **meta})
    except Exception as exc:
        err = {'stage': 'gurobi_direct_failed', 'error': repr(exc), 'traceback': traceback.format_exc(), **meta}
        (OUT_DIR / 'gurobi_direct_error.json').write_text(json.dumps(err, indent=2, ensure_ascii=False), encoding='utf-8')
        status(err)

    try:
        status({'stage': 'gurobi_warm_start', **meta})
        warm_rows, warm_final = solve_flp_with_checkpoints(
            inst=inst,
            llm_place=llm_place,
            method='Gurobi FLP + IWC-GS init',
            warm_start=iwc_tool,
            seed=SEED + 700,
            time_limit_s=TIME_LIMIT_S,
            checkpoints=CHECKPOINTS,
        )
        checkpoint_rows.extend(warm_rows)
        final_rows.append(warm_final)
        write_csv(OUT_DIR / 'flp_checkpoint_results.csv', checkpoint_rows)
        write_csv(OUT_DIR / 'flp_final_results.csv', final_rows)
        status({'stage': 'gurobi_warm_done', 'status': warm_final.get('status'), 'avg': warm_final.get('final_avg_call_distance'), **meta})
    except Exception as exc:
        err = {'stage': 'gurobi_warm_failed', 'error': repr(exc), 'traceback': traceback.format_exc(), **meta}
        (OUT_DIR / 'gurobi_warm_error.json').write_text(json.dumps(err, indent=2, ensure_ascii=False), encoding='utf-8')
        status(err)

    if final_rows:
        summary = []
        for row in final_rows:
            r = dict(row)
            r['iwc_gs_runtime_s'] = iwc_time
            r['iwc_gs_initial_avg_call_distance'] = iwc_avg
            summary.append(r)
        write_csv(OUT_DIR / 'summary.csv', summary)
    status({'stage': 'done', **meta})

if __name__ == '__main__':
    main()
