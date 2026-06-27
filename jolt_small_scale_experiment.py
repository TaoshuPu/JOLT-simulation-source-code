from __future__ import annotations

import math
import random
import sys
import time
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


def _add_local_dependency_dir(name: str) -> None:
    for root in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
        local_deps = root / name
        if not local_deps.exists():
            continue
        local_deps_text = str(local_deps)
        if local_deps_text not in sys.path:
            sys.path.insert(0, local_deps_text)
        if name == ".scip_deps" and hasattr(os, "add_dll_directory"):
            for dll_dir in ("bin", "pyscipopt.libs", "numpy.libs"):
                dll_path = local_deps / dll_dir
                if dll_path.exists():
                    os.add_dll_directory(str(dll_path))


for local_deps_name in (".ortools_deps", ".gurobi_deps", ".scip_deps"):
    _add_local_dependency_dir(local_deps_name)

import numpy as np


@dataclass(frozen=True)
class Instance:
    seed: int
    g_count: int
    c_count: int
    llm_count: int
    tool_count: int
    g_coords: np.ndarray
    c_coords: np.ndarray
    d_gg: np.ndarray
    d_gc: np.ndarray
    pref: np.ndarray
    arrival: np.ndarray
    llm_cpu: np.ndarray
    llm_gpu: np.ndarray
    llm_mem: np.ndarray
    g_cpu_cap: np.ndarray
    g_gpu_cap: np.ndarray
    g_mem_cap: np.ndarray
    tool_cpu: np.ndarray
    tool_mem: np.ndarray
    c_cpu_cap: np.ndarray
    c_mem_cap: np.ndarray


@dataclass
class AlgorithmResult:
    name: str
    decision_count: int
    avg_call_distance: float
    solve_time_s: float
    feasible: bool
    assignment: list[int]
    extra: dict


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def make_instance(
    seed: int,
    llm_count: int = 5,
    tool_count: int = 15,
    g_count: int | None = 4,
    c_count: int | None = 6,
    capacity_mode: str = "scaled_capacity",
    fixed_g_gpu_cap: int = 4,
    fixed_g_cpu_cap: int = 32,
    fixed_g_mem_cap: int = 56,
    fixed_c_cpu_cap: int = 16,
    fixed_c_mem_cap: int = 32,
) -> Instance:
    rng = np.random.default_rng(seed)
    if capacity_mode == "fixed_per_server":
        g_count = max(1, math.ceil(llm_count / fixed_g_gpu_cap)) if g_count is None else g_count
        # This provisional value is refined after tool demands are sampled.
        c_count = 1 if c_count is None else c_count
    elif capacity_mode == "g_only_fixed_per_server":
        c_count = 0
    else:
        g_count = 4 if g_count is None else g_count
        c_count = 6 if c_count is None else c_count

    g_base = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.2],
            [0.2, 3.0],
            [3.1, 3.1],
        ]
    )
    c_base = np.array(
        [
            [0.1, 0.4],
            [1.5, 0.1],
            [3.2, 0.4],
            [0.2, 2.7],
            [1.7, 3.1],
            [3.3, 2.8],
        ]
    )
    alpha = np.full((llm_count, tool_count), 0.08)
    cluster_count = max(1, min(llm_count, max(3, llm_count // 2), max(1, tool_count // 4)))
    tool_clusters = np.array_split(np.arange(tool_count), cluster_count)
    for i in range(llm_count):
        cluster_id = i % cluster_count
        primary_tools = tool_clusters[cluster_id]
        alpha[i, primary_tools] += 4.5
        neighbor_tools = tool_clusters[(cluster_id + 1) % cluster_count]
        if len(neighbor_tools) > 0:
            alpha[i, rng.choice(neighbor_tools, size=min(2, len(neighbor_tools)), replace=False)] += 1.0
        global_tools = rng.choice(tool_count, size=min(2, tool_count), replace=False)
        alpha[i, global_tools] += 0.5
    pref = np.vstack([rng.dirichlet(alpha[i]) for i in range(llm_count)])
    arrival = rng.uniform(0.8, 1.4, size=llm_count)

    llm_cpu = rng.integers(1, 3, size=llm_count)
    llm_gpu = np.ones(llm_count, dtype=int)
    llm_mem = rng.integers(7, 12, size=llm_count)

    tool_cpu = rng.integers(1, 4, size=tool_count)
    tool_mem = rng.integers(2, 6, size=tool_count)

    if capacity_mode in {"fixed_per_server", "g_only_fixed_per_server"}:
        if capacity_mode == "g_only_fixed_per_server":
            g_count = max(
                math.ceil(int(llm_gpu.sum()) / fixed_g_gpu_cap),
                math.ceil(int(llm_cpu.sum() + tool_cpu.sum()) / fixed_g_cpu_cap),
                math.ceil(int(llm_mem.sum() + tool_mem.sum()) / fixed_g_mem_cap),
            ) if g_count is None else g_count
            c_count = 0
        elif g_count is None:
            g_count = max(
                math.ceil(int(llm_gpu.sum()) / fixed_g_gpu_cap),
                math.ceil(int(llm_mem.sum()) / fixed_g_mem_cap),
            )
        if c_count is None or c_count == 1:
            c_count = max(
                math.ceil(int(tool_cpu.sum()) / fixed_c_cpu_cap),
                math.ceil(int(tool_mem.sum()) / fixed_c_mem_cap),
            )
        g_gpu_cap = np.full(g_count, fixed_g_gpu_cap, dtype=int)
        g_cpu_cap = np.full(g_count, fixed_g_cpu_cap, dtype=int)
        g_mem_cap = np.full(g_count, fixed_g_mem_cap, dtype=int)
        c_cpu_cap = np.full(c_count, fixed_c_cpu_cap, dtype=int)
        c_mem_cap = np.full(c_count, fixed_c_mem_cap, dtype=int)
    else:
        g_gpu_cap = np.full(g_count, math.ceil(llm_count / g_count), dtype=int)
        g_cpu_cap = np.full(g_count, fixed_g_cpu_cap, dtype=int)
        g_mem_cap = g_gpu_cap * 14
        c_cpu_cap = np.full(c_count, math.ceil(int(tool_cpu.sum()) / c_count) + 3, dtype=int)
        c_mem_cap = np.full(c_count, math.ceil(int(tool_mem.sum()) / c_count) + 6, dtype=int)

    if g_count > len(g_base):
        extra_g = rng.uniform(0.0, 3.4, size=(g_count - len(g_base), 2))
        g_base = np.vstack([g_base, extra_g])
    if c_count > len(c_base):
        extra_c = rng.uniform(0.0, 3.4, size=(c_count - len(c_base), 2))
        c_base = np.vstack([c_base, extra_c])
    g_base = g_base[:g_count]
    c_base = c_base[:c_count]

    g_coords = g_base + rng.normal(0.0, 0.08, size=g_base.shape)
    c_coords = c_base + rng.normal(0.0, 0.08, size=c_base.shape)
    d_gg = euclidean_dist(g_coords, g_coords)
    d_gc = euclidean_dist(g_coords, c_coords)

    return Instance(
        seed=seed,
        g_count=g_count,
        c_count=c_count,
        llm_count=llm_count,
        tool_count=tool_count,
        g_coords=g_coords,
        c_coords=c_coords,
        d_gg=d_gg,
        d_gc=d_gc,
        pref=pref,
        arrival=arrival,
        llm_cpu=llm_cpu,
        llm_gpu=llm_gpu,
        llm_mem=llm_mem,
        g_cpu_cap=g_cpu_cap,
        g_gpu_cap=g_gpu_cap,
        g_mem_cap=g_mem_cap,
        tool_cpu=tool_cpu,
        tool_mem=tool_mem,
        c_cpu_cap=c_cpu_cap,
        c_mem_cap=c_mem_cap,
    )


def weighted_calls(inst: Instance) -> np.ndarray:
    return inst.arrival[:, None] * inst.pref


def article_objective_weights(inst: Instance) -> np.ndarray:
    base = weighted_calls(inst)
    tool_frequency = base.sum(axis=0)
    return base * tool_frequency[None, :]


def tool_host_count(inst: Instance) -> int:
    return inst.g_count + inst.c_count


def tool_host_coords(inst: Instance) -> np.ndarray:
    return np.vstack([inst.g_coords, inst.c_coords])


def tool_host_capacity(inst: Instance, llm_place: Iterable[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    cpu = np.concatenate([inst.g_cpu_cap.astype(int), inst.c_cpu_cap.astype(int)])
    mem = np.concatenate([inst.g_mem_cap.astype(int), inst.c_mem_cap.astype(int)])
    if llm_place is not None:
        for i, server in enumerate(llm_place):
            cpu[server] -= int(inst.llm_cpu[i])
            mem[server] -= int(inst.llm_mem[i])
    return cpu, mem


def tool_distance(inst: Instance, llm_server: int, tool_host: int) -> float:
    if tool_host < inst.g_count:
        return float(inst.d_gg[llm_server, tool_host])
    return float(inst.d_gc[llm_server, tool_host - inst.g_count])


def average_call_distance(inst: Instance, llm_place: Iterable[int], tool_place: Iterable[int]) -> float:
    llm_place_arr = np.array(list(llm_place), dtype=int)
    tool_place_arr = np.array(list(tool_place), dtype=int)
    w = article_objective_weights(inst)
    total = float(weighted_calls(inst).sum())
    cost = 0.0
    for i in range(inst.llm_count):
        for j in range(inst.tool_count):
            cost += float(w[i, j]) * tool_distance(inst, int(llm_place_arr[i]), int(tool_place_arr[j]))
    return cost / total


def is_llm_feasible(inst: Instance, llm_place: Iterable[int]) -> bool:
    cpu = np.zeros(inst.g_count, dtype=int)
    gpu = np.zeros(inst.g_count, dtype=int)
    mem = np.zeros(inst.g_count, dtype=int)
    for i, server in enumerate(llm_place):
        cpu[server] += inst.llm_cpu[i]
        gpu[server] += inst.llm_gpu[i]
        mem[server] += inst.llm_mem[i]
    return bool(np.all(cpu <= inst.g_cpu_cap) and np.all(gpu <= inst.g_gpu_cap) and np.all(mem <= inst.g_mem_cap))


def is_tool_feasible(inst: Instance, tool_place: Iterable[int], llm_place: Iterable[int] | None = None) -> bool:
    rem_cpu, rem_mem = tool_host_capacity(inst, llm_place)
    used_cpu = np.zeros(tool_host_count(inst), dtype=int)
    used_mem = np.zeros(tool_host_count(inst), dtype=int)
    for j, server in enumerate(tool_place):
        used_cpu[server] += inst.tool_cpu[j]
        used_mem[server] += inst.tool_mem[j]
    return bool(np.all(used_cpu <= rem_cpu) and np.all(used_mem <= rem_mem))


def is_deployment_feasible(inst: Instance, llm_place: Iterable[int], tool_place: Iterable[int]) -> bool:
    llm_list = list(llm_place)
    return is_llm_feasible(inst, llm_list) and is_tool_feasible(inst, tool_place, llm_list)


def random_llm_assignment(inst: Instance, rng: random.Random) -> list[int]:
    order = list(range(inst.llm_count))
    rng.shuffle(order)
    place = [-1] * inst.llm_count
    rem_cpu = inst.g_cpu_cap.astype(int).tolist()
    rem_gpu = inst.g_gpu_cap.astype(int).tolist()
    rem_mem = inst.g_mem_cap.astype(int).tolist()
    for i in order:
        servers = [
            n
            for n in range(inst.g_count)
            if rem_cpu[n] >= inst.llm_cpu[i] and rem_gpu[n] >= inst.llm_gpu[i] and rem_mem[n] >= inst.llm_mem[i]
        ]
        if not servers:
            raise RuntimeError("No feasible G-Type server found.")
        n = rng.choice(servers)
        place[i] = n
        rem_cpu[n] -= int(inst.llm_cpu[i])
        rem_gpu[n] -= int(inst.llm_gpu[i])
        rem_mem[n] -= int(inst.llm_mem[i])
    return place


def random_tool_assignment(inst: Instance, rng: random.Random) -> list[int]:
    best_fit = best_fit_tool_assignment(inst, rng=rng)
    if best_fit is not None:
        return best_fit
    order = list(range(inst.tool_count))
    rng.shuffle(order)
    order.sort(key=lambda j: (int(inst.tool_cpu[j]) + int(inst.tool_mem[j]), int(inst.tool_cpu[j])), reverse=True)
    place = [-1] * inst.tool_count
    rem_cpu, rem_mem = (arr.astype(int).tolist() for arr in tool_host_capacity(inst))

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        j = order[pos]
        servers = list(range(tool_host_count(inst)))
        rng.shuffle(servers)
        servers.sort(key=lambda m: rem_cpu[m] + rem_mem[m], reverse=True)
        for m in servers:
            if rem_cpu[m] >= inst.tool_cpu[j] and rem_mem[m] >= inst.tool_mem[j]:
                place[j] = m
                rem_cpu[m] -= int(inst.tool_cpu[j])
                rem_mem[m] -= int(inst.tool_mem[j])
                if rec(pos + 1):
                    return True
                rem_cpu[m] += int(inst.tool_cpu[j])
                rem_mem[m] += int(inst.tool_mem[j])
                place[j] = -1
        return False

    if not rec(0):
        raise RuntimeError("No feasible C-Type server found.")
    return place


def resource_greedy_llm_assignment(inst: Instance) -> list[int]:
    order = sorted(
        range(inst.llm_count),
        key=lambda i: (
            int(inst.llm_gpu[i]) / max(1, int(inst.g_gpu_cap.max()))
            + int(inst.llm_cpu[i]) / max(1, int(inst.g_cpu_cap.max()))
            + int(inst.llm_mem[i]) / max(1, int(inst.g_mem_cap.max())),
            int(inst.llm_mem[i]),
        ),
        reverse=True,
    )
    place = [-1] * inst.llm_count
    rem_cpu = inst.g_cpu_cap.astype(int).tolist()
    rem_gpu = inst.g_gpu_cap.astype(int).tolist()
    rem_mem = inst.g_mem_cap.astype(int).tolist()
    for i in order:
        feasible = [
            n
            for n in range(inst.g_count)
            if rem_cpu[n] >= inst.llm_cpu[i] and rem_gpu[n] >= inst.llm_gpu[i] and rem_mem[n] >= inst.llm_mem[i]
        ]
        if not feasible:
            return random_llm_assignment(inst, random.Random(inst.seed + 17))
        n = max(
            feasible,
            key=lambda server: (
                rem_gpu[server] - int(inst.llm_gpu[i]),
                rem_cpu[server] - int(inst.llm_cpu[i]),
                rem_mem[server] - int(inst.llm_mem[i]),
            ),
        )
        place[i] = n
        rem_cpu[n] -= int(inst.llm_cpu[i])
        rem_gpu[n] -= int(inst.llm_gpu[i])
        rem_mem[n] -= int(inst.llm_mem[i])
    return place


def best_fit_tool_assignment(
    inst: Instance,
    tool_cost: np.ndarray | None = None,
    llm_place: list[int] | None = None,
    rng: random.Random | None = None,
    attempts: int = 16,
) -> list[int] | None:
    base_orders = [
        sorted(
            range(inst.tool_count),
            key=lambda j: (int(inst.tool_cpu[j]) + int(inst.tool_mem[j]), int(inst.tool_cpu[j])),
            reverse=True,
        ),
        sorted(range(inst.tool_count), key=lambda j: (int(inst.tool_cpu[j]), int(inst.tool_mem[j])), reverse=True),
        sorted(range(inst.tool_count), key=lambda j: (int(inst.tool_mem[j]), int(inst.tool_cpu[j])), reverse=True),
    ]
    if tool_cost is not None:
        base_orders.append(
            sorted(
                range(inst.tool_count),
                key=lambda j: float(np.max(tool_cost[j]) - np.min(tool_cost[j])),
                reverse=True,
            )
        )
    if rng is not None:
        for _ in range(attempts):
            order = base_orders[0].copy()
            rng.shuffle(order)
            order.sort(
                key=lambda j: int(inst.tool_cpu[j]) + int(inst.tool_mem[j]) + rng.random() * 0.01,
                reverse=True,
            )
            base_orders.append(order)

    for order in base_orders:
        place = [-1] * inst.tool_count
        rem_cpu_arr, rem_mem_arr = tool_host_capacity(inst, llm_place)
        rem_cpu = rem_cpu_arr.astype(int).tolist()
        rem_mem = rem_mem_arr.astype(int).tolist()
        failed = False
        for j in order:
            feasible = [
                m
                for m in range(tool_host_count(inst))
                if rem_cpu[m] >= inst.tool_cpu[j] and rem_mem[m] >= inst.tool_mem[j]
            ]
            if not feasible:
                failed = True
                break
            if tool_cost is None:
                chosen = min(
                    feasible,
                    key=lambda m: (
                        rem_cpu[m] - int(inst.tool_cpu[j]) + rem_mem[m] - int(inst.tool_mem[j]),
                        max(rem_cpu[m] - int(inst.tool_cpu[j]), rem_mem[m] - int(inst.tool_mem[j])),
                    ),
                )
            else:
                chosen = min(
                    feasible,
                    key=lambda m: (
                        float(tool_cost[j, m]),
                        rem_cpu[m] - int(inst.tool_cpu[j]) + rem_mem[m] - int(inst.tool_mem[j]),
                    ),
                )
            place[j] = chosen
            rem_cpu[chosen] -= int(inst.tool_cpu[j])
            rem_mem[chosen] -= int(inst.tool_mem[j])
        if not failed:
            return place
    return None


def greedy_tool_assignment_for_cost(
    inst: Instance,
    tool_cost: np.ndarray,
    llm_place: list[int] | None = None,
) -> tuple[float, list[int] | None]:
    freq = weighted_calls(inst).sum(axis=0)
    order = sorted(range(inst.tool_count), key=lambda j: (freq[j], max(tool_cost[j]) - min(tool_cost[j])), reverse=True)
    rem_cpu_arr, rem_mem_arr = tool_host_capacity(inst, llm_place)
    rem_cpu = rem_cpu_arr.astype(int).tolist()
    rem_mem = rem_mem_arr.astype(int).tolist()
    place = [-1] * inst.tool_count
    total = 0.0
    for j in order:
        servers = sorted(range(tool_cost.shape[1]), key=lambda m: tool_cost[j, m])
        chosen = None
        for m in servers:
            if rem_cpu[m] >= inst.tool_cpu[j] and rem_mem[m] >= inst.tool_mem[j]:
                chosen = m
                break
        if chosen is None:
            return math.inf, None
        place[j] = chosen
        rem_cpu[chosen] -= int(inst.tool_cpu[j])
        rem_mem[chosen] -= int(inst.tool_mem[j])
        total += float(tool_cost[j, chosen])
    return total, place


def tool_cost_matrix(inst: Instance, llm_place: list[int]) -> np.ndarray:
    tool_cost = np.zeros((inst.tool_count, tool_host_count(inst)))
    w = article_objective_weights(inst)
    for j in range(inst.tool_count):
        for m in range(tool_host_count(inst)):
            tool_cost[j, m] = sum(float(w[i, j]) * tool_distance(inst, llm_place[i], m) for i in range(inst.llm_count))
    return tool_cost


def cpsat_original_solver(
    inst: Instance,
    timeout_s: float | None = None,
    objective_scale: int = 1_000_000,
    warm_start: bool = False,
    hint_search: bool = False,
    warm_samples: int = 25,
    hint_z_variables: bool = False,
    top_tools_per_llm: int | None = None,
    formulation: str = "linearized",
    progress_callback: Callable[[float, list[int], list[int], float, dict], None] | None = None,
) -> AlgorithmResult:
    start = time.perf_counter()
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        elapsed = time.perf_counter() - start
        return AlgorithmResult(
            name="OR-Tools CP-SAT original",
            decision_count=inst.llm_count + inst.tool_count,
            avg_call_distance=math.inf,
            solve_time_s=elapsed,
            feasible=False,
            assignment=[],
            extra={
                "solver_status": "IMPORT_ERROR",
                "error": str(exc),
                "proven_optimal": False,
                "is_exact_baseline": True,
            },
        )

    model = cp_model.CpModel()
    h_count = tool_host_count(inst)
    x = {
        (i, n): model.NewBoolVar(f"x_{i}_{n}")
        for i in range(inst.llm_count)
        for n in range(inst.g_count)
    }
    y = {
        (j, h): model.NewBoolVar(f"y_{j}_{h}")
        for j in range(inst.tool_count)
        for h in range(h_count)
    }
    llm_host_vars = [
        model.NewIntVar(0, inst.g_count - 1, f"llm_host_{i}") for i in range(inst.llm_count)
    ]
    tool_host_vars = [
        model.NewIntVar(0, h_count - 1, f"tool_host_{j}") for j in range(inst.tool_count)
    ]

    for i in range(inst.llm_count):
        model.Add(sum(x[i, n] for n in range(inst.g_count)) == 1)
        model.Add(llm_host_vars[i] == sum(n * x[i, n] for n in range(inst.g_count)))
    for j in range(inst.tool_count):
        model.Add(sum(y[j, h] for h in range(h_count)) == 1)
        model.Add(tool_host_vars[j] == sum(h * y[j, h] for h in range(h_count)))

    for n in range(inst.g_count):
        model.Add(sum(int(inst.llm_gpu[i]) * x[i, n] for i in range(inst.llm_count)) <= int(inst.g_gpu_cap[n]))
        model.Add(
            sum(int(inst.llm_cpu[i]) * x[i, n] for i in range(inst.llm_count))
            + sum(int(inst.tool_cpu[j]) * y[j, n] for j in range(inst.tool_count))
            <= int(inst.g_cpu_cap[n])
        )
        model.Add(
            sum(int(inst.llm_mem[i]) * x[i, n] for i in range(inst.llm_count))
            + sum(int(inst.tool_mem[j]) * y[j, n] for j in range(inst.tool_count))
            <= int(inst.g_mem_cap[n])
        )
    for m in range(inst.c_count):
        h = inst.g_count + m
        model.Add(sum(int(inst.tool_cpu[j]) * y[j, h] for j in range(inst.tool_count)) <= int(inst.c_cpu_cap[m]))
        model.Add(sum(int(inst.tool_mem[j]) * y[j, h] for j in range(inst.tool_count)) <= int(inst.c_mem_cap[m]))

    w = article_objective_weights(inst)
    active_pairs: set[tuple[int, int]] | None = None
    retained_weight = float(w.sum())
    total_weight = float(w.sum())
    if (
        formulation != "element"
        and top_tools_per_llm is not None
        and top_tools_per_llm > 0
        and top_tools_per_llm < inst.tool_count
    ):
        active_pairs = set()
        retained_weight = 0.0
        for i in range(inst.llm_count):
            order = sorted(range(inst.tool_count), key=lambda j: float(w[i, j]), reverse=True)
            for j in order[:top_tools_per_llm]:
                if w[i, j] > 1e-12:
                    active_pairs.add((i, j))
                    retained_weight += float(w[i, j])
    objective_terms = []
    z_count = 0
    z_vars = {}
    cost_var_count = 0
    if formulation == "element":
        for i in range(inst.llm_count):
            for j in range(inst.tool_count):
                if w[i, j] <= 1e-12:
                    continue
                table = [
                    int(round(float(w[i, j]) * tool_distance(inst, n, h) * objective_scale))
                    for n in range(inst.g_count)
                    for h in range(h_count)
                ]
                max_cost = max(table)
                if max_cost == 0:
                    continue
                idx = model.NewIntVar(0, inst.g_count * h_count - 1, f"pair_host_{i}_{j}")
                cost = model.NewIntVar(0, max_cost, f"pair_cost_{i}_{j}")
                model.Add(idx == h_count * llm_host_vars[i] + tool_host_vars[j])
                model.AddElement(idx, table, cost)
                objective_terms.append(cost)
                cost_var_count += 1
    else:
        for i in range(inst.llm_count):
            for j in range(inst.tool_count):
                if active_pairs is not None and (i, j) not in active_pairs:
                    continue
                if w[i, j] <= 1e-12:
                    continue
                for n in range(inst.g_count):
                    for h in range(h_count):
                        coeff = int(round(float(w[i, j]) * tool_distance(inst, n, h) * objective_scale))
                        if coeff == 0:
                            continue
                        z = model.NewBoolVar(f"z_{i}_{j}_{n}_{h}")
                        model.Add(z <= x[i, n])
                        model.Add(z <= y[j, h])
                        model.Add(z >= x[i, n] + y[j, h] - 1)
                        objective_terms.append(coeff * z)
                        z_vars[(i, j, n, h)] = z
                        z_count += 1
    model.Minimize(sum(objective_terms))

    hint_llm: list[int] = []
    hint_tool: list[int] = []
    hint_distance = math.inf
    warm_extra: dict = {}
    if warm_start:
        hint_llm, hint_tool, warm_extra = greedy_original_warm_start(inst, samples=warm_samples)
        hint_distance = average_call_distance(inst, hint_llm, hint_tool)
        if progress_callback is not None:
            progress_callback(
                time.perf_counter() - start,
                hint_llm.copy(),
                hint_tool.copy(),
                hint_distance,
                {"source": "warm_start"},
            )
        for i in range(inst.llm_count):
            model.AddHint(llm_host_vars[i], hint_llm[i])
            for n in range(inst.g_count):
                model.AddHint(x[i, n], 1 if hint_llm[i] == n else 0)
        for j in range(inst.tool_count):
            model.AddHint(tool_host_vars[j], hint_tool[j])
            for h in range(h_count):
                model.AddHint(y[j, h], 1 if hint_tool[j] == h else 0)
        if hint_z_variables:
            for key, z in z_vars.items():
                i, j, n, h = key
                model.AddHint(z, 1 if hint_llm[i] == n and hint_tool[j] == h else 0)

    solver = cp_model.CpSolver()
    if timeout_s is not None and timeout_s > 0:
        solver.parameters.max_time_in_seconds = float(timeout_s)
    solver.parameters.num_search_workers = 8
    if warm_start:
        solver.parameters.use_optimization_hints = formulation != "element"
        if formulation != "element":
            solver.parameters.repair_hint = True
            solver.parameters.hint_conflict_limit = 100
        if hint_search:
            solver.parameters.search_branching = cp_model.HINT_SEARCH
    if progress_callback is None:
        status = solver.Solve(model)
    else:
        class IncumbentCallback(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self) -> None:
                llm_place = [
                    max(range(inst.g_count), key=lambda n: self.Value(x[i, n]))
                    for i in range(inst.llm_count)
                ]
                tool_place = [
                    max(range(h_count), key=lambda h: self.Value(y[j, h]))
                    for j in range(inst.tool_count)
                ]
                progress_callback(
                    time.perf_counter() - start,
                    llm_place,
                    tool_place,
                    average_call_distance(inst, llm_place, tool_place),
                    {
                        "source": "solver_incumbent",
                        "solver_wall_time_s": float(self.WallTime()),
                        "objective_value_scaled": float(self.ObjectiveValue()),
                        "best_objective_bound_scaled": float(self.BestObjectiveBound()),
                    },
                )

        status = solver.Solve(model, IncumbentCallback())
    elapsed = time.perf_counter() - start
    status_name = solver.StatusName(status)
    solver_feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    feasible = solver_feasible

    llm_place: list[int] = []
    tool_place: list[int] = []
    avg = math.inf
    relative_gap = math.nan
    if solver_feasible:
        for i in range(inst.llm_count):
            llm_place.append(max(range(inst.g_count), key=lambda n: solver.Value(x[i, n])))
        for j in range(inst.tool_count):
            tool_place.append(max(range(h_count), key=lambda h: solver.Value(y[j, h])))
        avg = average_call_distance(inst, llm_place, tool_place)
        objective = float(solver.ObjectiveValue())
        bound = float(solver.BestObjectiveBound())
        if abs(objective) > 1e-12:
            relative_gap = max(0.0, (objective - bound) / abs(objective))
    elif warm_start and is_deployment_feasible(inst, hint_llm, hint_tool):
        feasible = True
        llm_place = hint_llm
        tool_place = hint_tool
        avg = hint_distance
        status_name = "HINT_FALLBACK"

    return AlgorithmResult(
        name="OR-Tools CP-SAT original",
        decision_count=inst.llm_count + inst.tool_count,
        avg_call_distance=avg,
        solve_time_s=elapsed,
        feasible=feasible,
        assignment=llm_place + tool_place,
        extra={
            "solver_status": status_name,
            "proven_optimal": status == cp_model.OPTIMAL,
            "is_exact_baseline": True,
            "objective_scale": objective_scale,
            "best_objective_bound_scaled": float(solver.BestObjectiveBound()) if solver_feasible else math.nan,
            "objective_value_scaled": float(solver.ObjectiveValue()) if solver_feasible else math.nan,
            "relative_gap": relative_gap,
            "solver_found_solution": solver_feasible,
            "warm_start": warm_start,
            "hint_search": hint_search,
            "warm_start_method": warm_extra.get("warm_start"),
            "warm_start_candidates": warm_extra.get("warm_start_candidates"),
            "hint_avg_call_distance": hint_distance,
            "hint_z_variables": hint_z_variables,
            "top_tools_per_llm": top_tools_per_llm,
            "formulation": formulation,
            "objective_active_pairs": len(active_pairs) if active_pairs is not None else inst.llm_count * inst.tool_count,
            "objective_retained_weight_ratio": retained_weight / total_weight if total_weight > 1e-12 else 1.0,
            "wall_time_s": float(solver.WallTime()),
            "num_branches": int(solver.NumBranches()),
            "num_conflicts": int(solver.NumConflicts()),
            "x_variables": inst.llm_count * inst.g_count,
            "y_variables": inst.tool_count * h_count,
            "z_variables": z_count,
            "element_cost_variables": cost_var_count,
            "timeout_s": timeout_s,
        },
    )


def greedy_original_warm_start(inst: Instance, samples: int = 24) -> tuple[list[int], list[int], dict]:
    rng = random.Random(inst.seed + 530)
    candidates: list[tuple[float, list[int], list[int], str]] = []

    def add_candidate(llm_place: list[int], label: str) -> None:
        if not is_llm_feasible(inst, llm_place):
            return
        tool_cost = tool_cost_matrix(inst, llm_place)
        _, tool_place = greedy_tool_assignment_for_cost(inst, tool_cost, llm_place)
        if tool_place is None:
            tool_place = best_fit_tool_assignment(inst, tool_cost=tool_cost, llm_place=llm_place, rng=rng)
        if tool_place is None:
            tool_place = best_fit_tool_assignment(inst, llm_place=llm_place, rng=rng)
        if tool_place is None:
            return
        if is_tool_feasible(inst, tool_place, llm_place):
            candidates.append((average_call_distance(inst, llm_place, tool_place), llm_place, tool_place, label))

    add_candidate(resource_greedy_llm_assignment(inst), "resource_greedy")
    for idx in range(samples):
        try:
            add_candidate(random_llm_assignment(inst, rng), f"random_{idx}")
        except RuntimeError:
            continue

    if not candidates:
        llm_place = resource_greedy_llm_assignment(inst)
        tool_place = best_fit_tool_assignment(inst, llm_place=llm_place, rng=rng)
        if tool_place is None:
            raise RuntimeError("No fast feasible tool warm-start found.")
        return llm_place, tool_place, {"warm_start": "resource_fallback", "warm_start_distance": average_call_distance(inst, llm_place, tool_place)}

    distance, llm_place, tool_place, label = min(candidates, key=lambda item: item[0])
    return llm_place, tool_place, {
        "warm_start": label,
        "warm_start_distance": distance,
        "warm_start_candidates": len(candidates),
    }


def gurobi_original_solver(
    inst: Instance,
    timeout_s: float | None = None,
    progress_callback: Callable[[float, list[int], list[int], float, dict], None] | None = None,
) -> AlgorithmResult:
    start = time.perf_counter()
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        elapsed = time.perf_counter() - start
        return AlgorithmResult(
            name="Gurobi MIP original",
            decision_count=inst.llm_count + inst.tool_count,
            avg_call_distance=math.inf,
            solve_time_s=elapsed,
            feasible=False,
            assignment=[],
            extra={
                "solver_status": "IMPORT_ERROR",
                "error": str(exc),
                "proven_optimal": False,
                "is_exact_baseline": True,
            },
        )

    try:
        model = gp.Model("jolt_original_deployment")
        model.Params.OutputFlag = 0
        if timeout_s is not None and timeout_s > 0:
            model.Params.TimeLimit = float(timeout_s)

        h_count = tool_host_count(inst)
        x = model.addVars(inst.llm_count, inst.g_count, vtype=GRB.BINARY, name="x")
        y = model.addVars(inst.tool_count, h_count, vtype=GRB.BINARY, name="y")

        model.addConstrs((gp.quicksum(x[i, n] for n in range(inst.g_count)) == 1 for i in range(inst.llm_count)))
        model.addConstrs((gp.quicksum(y[j, h] for h in range(h_count)) == 1 for j in range(inst.tool_count)))

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
                + gp.quicksum(int(inst.tool_cpu[j]) * y[j, n] for j in range(inst.tool_count))
                <= int(inst.g_cpu_cap[n])
                for n in range(inst.g_count)
            )
        )
        model.addConstrs(
            (
                gp.quicksum(int(inst.llm_mem[i]) * x[i, n] for i in range(inst.llm_count))
                + gp.quicksum(int(inst.tool_mem[j]) * y[j, n] for j in range(inst.tool_count))
                <= int(inst.g_mem_cap[n])
                for n in range(inst.g_count)
            )
        )
        model.addConstrs(
            (
                gp.quicksum(int(inst.tool_cpu[j]) * y[j, inst.g_count + m] for j in range(inst.tool_count))
                <= int(inst.c_cpu_cap[m])
                for m in range(inst.c_count)
            )
        )
        model.addConstrs(
            (
                gp.quicksum(int(inst.tool_mem[j]) * y[j, inst.g_count + m] for j in range(inst.tool_count))
                <= int(inst.c_mem_cap[m])
                for m in range(inst.c_count)
            )
        )

        w = article_objective_weights(inst)
        objective = gp.QuadExpr()
        quadratic_terms = 0
        coeff_batch = []
        x_batch = []
        y_batch = []

        def flush_quadratic_terms() -> None:
            nonlocal coeff_batch, x_batch, y_batch
            if coeff_batch:
                objective.addTerms(coeff_batch, x_batch, y_batch)
                coeff_batch = []
                x_batch = []
                y_batch = []

        for i in range(inst.llm_count):
            for j in range(inst.tool_count):
                if w[i, j] <= 1e-12:
                    continue
                for n in range(inst.g_count):
                    for h in range(h_count):
                        coeff_batch.append(float(w[i, j]) * tool_distance(inst, n, h))
                        x_batch.append(x[i, n])
                        y_batch.append(y[j, h])
                        quadratic_terms += 1
                        if len(coeff_batch) >= 100_000:
                            flush_quadratic_terms()
        flush_quadratic_terms()
        model.setObjective(objective, GRB.MINIMIZE)

        def incumbent_callback(model_cb: gp.Model, where: int) -> None:
            if progress_callback is None or where != GRB.Callback.MIPSOL:
                return
            llm_place = [
                max(range(inst.g_count), key=lambda n: model_cb.cbGetSolution(x[i, n]))
                for i in range(inst.llm_count)
            ]
            tool_place = [
                max(range(h_count), key=lambda h: model_cb.cbGetSolution(y[j, h]))
                for j in range(inst.tool_count)
            ]
            progress_callback(
                time.perf_counter() - start,
                llm_place,
                tool_place,
                average_call_distance(inst, llm_place, tool_place),
                {
                    "source": "solver_incumbent",
                    "objective_value": float(model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)),
                },
            )

        if progress_callback is not None:
            model.optimize(incumbent_callback)
        else:
            model.optimize()
        elapsed = time.perf_counter() - start

        feasible = model.SolCount > 0
        llm_place: list[int] = []
        tool_place: list[int] = []
        avg = math.inf
        relative_gap = math.nan
        if feasible:
            for i in range(inst.llm_count):
                llm_place.append(max(range(inst.g_count), key=lambda n: x[i, n].X))
            for j in range(inst.tool_count):
                tool_place.append(max(range(h_count), key=lambda h: y[j, h].X))
            avg = average_call_distance(inst, llm_place, tool_place)
            if math.isfinite(model.ObjVal) and abs(float(model.ObjVal)) > 1e-12:
                relative_gap = max(0.0, float(model.MIPGap))

        status_name = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(model.Status, str(model.Status))
        return AlgorithmResult(
            name="Gurobi MIP original",
            decision_count=inst.llm_count + inst.tool_count,
            avg_call_distance=avg,
            solve_time_s=elapsed,
            feasible=feasible,
            assignment=llm_place + tool_place,
            extra={
                "solver_status": status_name,
                "proven_optimal": model.Status == GRB.OPTIMAL,
                "is_exact_baseline": True,
                "objective_value": float(model.ObjVal) if feasible else math.nan,
                "best_objective_bound": float(model.ObjBound) if feasible else math.nan,
                "relative_gap": relative_gap,
                "runtime_s": float(model.Runtime),
                "node_count": float(model.NodeCount),
                "x_variables": inst.llm_count * inst.g_count,
                "y_variables": inst.tool_count * h_count,
                "quadratic_objective_terms": quadratic_terms,
                "timeout_s": timeout_s,
            },
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return AlgorithmResult(
            name="Gurobi MIP original",
            decision_count=inst.llm_count + inst.tool_count,
            avg_call_distance=math.inf,
            solve_time_s=elapsed,
            feasible=False,
            assignment=[],
            extra={
                "solver_status": "ERROR",
                "error": str(exc),
                "proven_optimal": False,
                "is_exact_baseline": True,
            },
        )


def scip_original_solver(
    inst: Instance,
    timeout_s: float | None = None,
    progress_callback: Callable[[float, list[int], list[int], float, dict], None] | None = None,
) -> AlgorithmResult:
    start = time.perf_counter()
    scip_deps = Path(__file__).resolve().parent / ".scip_deps"
    if scip_deps.exists():
        sys.path.insert(0, str(scip_deps))
        if hasattr(os, "add_dll_directory"):
            for dll_dir in ("bin", "pyscipopt.libs", "numpy.libs"):
                path = scip_deps / dll_dir
                if path.exists():
                    os.add_dll_directory(str(path))
    try:
        from pyscipopt import Model, quicksum
        from pyscipopt.recipes.nonlinear import set_nonlinear_objective
    except ImportError as exc:
        elapsed = time.perf_counter() - start
        return AlgorithmResult(
            name="SCIP MIQP original",
            decision_count=inst.llm_count + inst.tool_count,
            avg_call_distance=math.inf,
            solve_time_s=elapsed,
            feasible=False,
            assignment=[],
            extra={
                "solver_status": "IMPORT_ERROR",
                "error": str(exc),
                "proven_optimal": False,
                "is_exact_baseline": True,
            },
        )

    try:
        model = Model("jolt_original_deployment_scip")
        model.hideOutput()
        if timeout_s is not None and timeout_s > 0:
            model.setParam("limits/time", float(timeout_s))

        h_count = tool_host_count(inst)
        x = {
            (i, n): model.addVar(vtype="B", name=f"x_{i}_{n}")
            for i in range(inst.llm_count)
            for n in range(inst.g_count)
        }
        y = {
            (j, h): model.addVar(vtype="B", name=f"y_{j}_{h}")
            for j in range(inst.tool_count)
            for h in range(h_count)
        }

        for i in range(inst.llm_count):
            model.addCons(quicksum(x[i, n] for n in range(inst.g_count)) == 1)
        for j in range(inst.tool_count):
            model.addCons(quicksum(y[j, h] for h in range(h_count)) == 1)

        for n in range(inst.g_count):
            model.addCons(quicksum(int(inst.llm_gpu[i]) * x[i, n] for i in range(inst.llm_count)) <= int(inst.g_gpu_cap[n]))
            model.addCons(
                quicksum(int(inst.llm_cpu[i]) * x[i, n] for i in range(inst.llm_count))
                + quicksum(int(inst.tool_cpu[j]) * y[j, n] for j in range(inst.tool_count))
                <= int(inst.g_cpu_cap[n])
            )
            model.addCons(
                quicksum(int(inst.llm_mem[i]) * x[i, n] for i in range(inst.llm_count))
                + quicksum(int(inst.tool_mem[j]) * y[j, n] for j in range(inst.tool_count))
                <= int(inst.g_mem_cap[n])
            )
        for m in range(inst.c_count):
            h = inst.g_count + m
            model.addCons(quicksum(int(inst.tool_cpu[j]) * y[j, h] for j in range(inst.tool_count)) <= int(inst.c_cpu_cap[m]))
            model.addCons(quicksum(int(inst.tool_mem[j]) * y[j, h] for j in range(inst.tool_count)) <= int(inst.c_mem_cap[m]))

        w = article_objective_weights(inst)
        quadratic_terms = 0
        for i in range(inst.llm_count):
            for j in range(inst.tool_count):
                if w[i, j] <= 1e-12:
                    continue
                for n in range(inst.g_count):
                    for h in range(h_count):
                        coeff = float(w[i, j]) * tool_distance(inst, n, h)
                        if coeff <= 1e-12:
                            continue
                        quadratic_terms += 1

        objective = quicksum(
            float(w[i, j]) * tool_distance(inst, n, h) * x[i, n] * y[j, h]
            for i in range(inst.llm_count)
            for j in range(inst.tool_count)
            if w[i, j] > 1e-12
            for n in range(inst.g_count)
            for h in range(h_count)
            if float(w[i, j]) * tool_distance(inst, n, h) > 1e-12
        )
        set_nonlinear_objective(model, objective, sense="minimize")

        warm_llm, warm_tool, warm_extra = greedy_original_warm_start(inst, samples=16)
        if progress_callback is not None:
            progress_callback(
                time.perf_counter() - start,
                warm_llm.copy(),
                warm_tool.copy(),
                average_call_distance(inst, warm_llm, warm_tool),
                {"source": "warm_start"},
            )
        sol = model.createSol()
        for i in range(inst.llm_count):
            for n in range(inst.g_count):
                model.setSolVal(sol, x[i, n], 1.0 if warm_llm[i] == n else 0.0)
        for j in range(inst.tool_count):
            for h in range(h_count):
                model.setSolVal(sol, y[j, h], 1.0 if warm_tool[j] == h else 0.0)
        accepted_warm_start = bool(model.addSol(sol))

        if progress_callback is not None:
            from pyscipopt import Eventhdlr, SCIP_EVENTTYPE

            class BestSolutionHandler(Eventhdlr):
                def eventinit(self) -> None:
                    self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

                def eventexit(self) -> None:
                    self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

                def eventexec(self, event) -> None:
                    best_sol = self.model.getBestSol()
                    if best_sol is None:
                        return
                    llm_place = [
                        max(range(inst.g_count), key=lambda n: self.model.getSolVal(best_sol, x[i, n]))
                        for i in range(inst.llm_count)
                    ]
                    tool_place = [
                        max(range(h_count), key=lambda h: self.model.getSolVal(best_sol, y[j, h]))
                        for j in range(inst.tool_count)
                    ]
                    progress_callback(
                        time.perf_counter() - start,
                        llm_place,
                        tool_place,
                        average_call_distance(inst, llm_place, tool_place),
                        {"source": "solver_incumbent"},
                    )

            model.includeEventhdlr(BestSolutionHandler(), "best_solution_trace", "Record incumbent deployments.")

        model.optimize()
        elapsed = time.perf_counter() - start

        status = str(model.getStatus())
        feasible = model.getNSols() > 0
        llm_place: list[int] = []
        tool_place: list[int] = []
        avg = math.inf
        objective_value = math.nan
        best_bound = math.nan
        relative_gap = math.nan
        if feasible:
            for i in range(inst.llm_count):
                llm_place.append(max(range(inst.g_count), key=lambda n: model.getVal(x[i, n])))
            for j in range(inst.tool_count):
                tool_place.append(max(range(h_count), key=lambda h: model.getVal(y[j, h])))
            avg = average_call_distance(inst, llm_place, tool_place)
            objective_value = float(model.getObjVal())
            try:
                best_bound = float(model.getDualbound())
                if abs(objective_value) > 1e-12:
                    relative_gap = max(0.0, (objective_value - best_bound) / abs(objective_value))
            except Exception:
                best_bound = math.nan

        return AlgorithmResult(
            name="SCIP MIQP original",
            decision_count=inst.llm_count + inst.tool_count,
            avg_call_distance=avg,
            solve_time_s=elapsed,
            feasible=feasible,
            assignment=llm_place + tool_place,
            extra={
                "solver_status": status,
                "proven_optimal": status.lower() == "optimal",
                "is_exact_baseline": True,
                "objective_value": objective_value,
                "best_objective_bound": best_bound,
                "relative_gap": relative_gap,
                "x_variables": inst.llm_count * inst.g_count,
                "y_variables": inst.tool_count * h_count,
                "quadratic_objective_terms": quadratic_terms,
                "timeout_s": timeout_s,
                "warm_start_accepted": accepted_warm_start,
                **warm_extra,
            },
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return AlgorithmResult(
            name="SCIP MIQP original",
            decision_count=inst.llm_count + inst.tool_count,
            avg_call_distance=math.inf,
            solve_time_s=elapsed,
            feasible=False,
            assignment=[],
            extra={
                "solver_status": "ERROR",
                "error": str(exc),
                "proven_optimal": False,
                "is_exact_baseline": True,
            },
        )


def llm_similarity(inst: Instance) -> np.ndarray:
    v = weighted_calls(inst)
    return v @ v.T


def llm_surrogate_cost(inst: Instance, llm_place: list[int], similarity: np.ndarray) -> float:
    cost = 0.0
    for i in range(inst.llm_count):
        for k in range(inst.llm_count):
            cost += float(similarity[i, k]) * float(inst.d_gg[llm_place[i], llm_place[k]])
    return cost


def gurobi_gqap_llm_deployment(
    inst: Instance,
    timeout_s: float | None = None,
    progress_callback: Callable[[float, list[int], float, dict], None] | None = None,
) -> tuple[list[int], dict]:
    start = time.perf_counter()
    similarity = llm_similarity(inst)
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        fallback = random_llm_assignment(inst, random.Random(inst.seed + 73))
        return fallback, {
            "phase1_solver": "gurobi_gqap",
            "solver_status": "IMPORT_ERROR",
            "error": str(exc),
            "llm_surrogate_cost": llm_surrogate_cost(inst, fallback, similarity),
            "solve_time_s": time.perf_counter() - start,
            "proven_optimal": False,
        }

    try:
        model = gp.Model("jolt_llm_gqap")
        model.Params.OutputFlag = 0
        model.Params.NonConvex = 2
        if timeout_s is not None and timeout_s > 0:
            model.Params.TimeLimit = float(timeout_s)

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
        quadratic_terms = 0
        coeff_batch = []
        left_batch = []
        right_batch = []

        def flush_quadratic_terms() -> None:
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
                            flush_quadratic_terms()
        flush_quadratic_terms()
        model.setObjective(objective, GRB.MINIMIZE)

        def incumbent_callback(model_cb: gp.Model, where: int) -> None:
            if progress_callback is None or where != GRB.Callback.MIPSOL:
                return
            place = [
                max(range(inst.g_count), key=lambda n: model_cb.cbGetSolution(x[i, n]))
                for i in range(inst.llm_count)
            ]
            progress_callback(
                time.perf_counter() - start,
                place,
                llm_surrogate_cost(inst, place, similarity),
                {
                    "source": "solver_incumbent",
                    "objective_value": float(model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)),
                },
            )

        if progress_callback is not None:
            model.optimize(incumbent_callback)
        else:
            model.optimize()

        if model.SolCount > 0:
            place = [max(range(inst.g_count), key=lambda n: x[i, n].X) for i in range(inst.llm_count)]
        else:
            place = random_llm_assignment(inst, random.Random(inst.seed + 79))
        status_name = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(model.Status, str(model.Status))
        return place, {
            "phase1_solver": "gurobi_gqap",
            "solver_status": status_name,
            "llm_surrogate_cost": llm_surrogate_cost(inst, place, similarity),
            "objective_value": float(model.ObjVal) if model.SolCount > 0 else math.nan,
            "best_objective_bound": float(model.ObjBound) if model.SolCount > 0 else math.nan,
            "relative_gap": float(model.MIPGap) if model.SolCount > 0 and math.isfinite(model.MIPGap) else math.nan,
            "solve_time_s": time.perf_counter() - start,
            "runtime_s": float(model.Runtime),
            "node_count": float(model.NodeCount),
            "x_variables": inst.llm_count * inst.g_count,
            "quadratic_objective_terms": quadratic_terms,
            "proven_optimal": model.Status == GRB.OPTIMAL,
            "timeout_s": timeout_s,
        }
    except Exception as exc:
        fallback = random_llm_assignment(inst, random.Random(inst.seed + 83))
        return fallback, {
            "phase1_solver": "gurobi_gqap",
            "solver_status": "ERROR",
            "error": str(exc),
            "llm_surrogate_cost": llm_surrogate_cost(inst, fallback, similarity),
            "solve_time_s": time.perf_counter() - start,
            "proven_optimal": False,
        }


def scip_gqap_llm_deployment(
    inst: Instance,
    timeout_s: float | None = None,
    progress_callback: Callable[[float, list[int], float, dict], None] | None = None,
) -> tuple[list[int], dict]:
    start = time.perf_counter()
    similarity = llm_similarity(inst)
    _add_local_dependency_dir(".scip_deps")
    try:
        from pyscipopt import Model, quicksum
        from pyscipopt.recipes.nonlinear import set_nonlinear_objective
    except ImportError as exc:
        fallback = random_llm_assignment(inst, random.Random(inst.seed + 73))
        return fallback, {
            "phase1_solver": "scip_gqap",
            "solver_status": "IMPORT_ERROR",
            "error": str(exc),
            "llm_surrogate_cost": llm_surrogate_cost(inst, fallback, similarity),
            "solve_time_s": time.perf_counter() - start,
            "proven_optimal": False,
        }

    try:
        model = Model("jolt_llm_gqap_scip")
        model.hideOutput(True)
        if timeout_s is not None and timeout_s > 0:
            model.setParam("limits/time", float(timeout_s))

        x = {
            (i, n): model.addVar(vtype="B", name=f"x_{i}_{n}")
            for i in range(inst.llm_count)
            for n in range(inst.g_count)
        }
        for i in range(inst.llm_count):
            model.addCons(quicksum(x[i, n] for n in range(inst.g_count)) == 1)
        for n in range(inst.g_count):
            model.addCons(quicksum(int(inst.llm_gpu[i]) * x[i, n] for i in range(inst.llm_count)) <= int(inst.g_gpu_cap[n]))
            model.addCons(quicksum(int(inst.llm_cpu[i]) * x[i, n] for i in range(inst.llm_count)) <= int(inst.g_cpu_cap[n]))
            model.addCons(quicksum(int(inst.llm_mem[i]) * x[i, n] for i in range(inst.llm_count)) <= int(inst.g_mem_cap[n]))

        objective_terms = []
        quadratic_terms = 0
        for i in range(inst.llm_count):
            for k in range(inst.llm_count):
                sim = float(similarity[i, k])
                if sim <= 1e-12:
                    continue
                for n in range(inst.g_count):
                    for q in range(inst.g_count):
                        coeff = sim * float(inst.d_gg[n, q])
                        if coeff <= 1e-12:
                            continue
                        objective_terms.append(coeff * x[i, n] * x[k, q])
                        quadratic_terms += 1
        objective = quicksum(objective_terms) if objective_terms else 0.0 * x[0, 0]
        if quadratic_terms:
            set_nonlinear_objective(model, objective, sense="minimize")
        else:
            model.setObjective(objective, sense="minimize")

        warm_llm = random_llm_assignment(inst, random.Random(inst.seed + 73))
        sol = model.createSol()
        for i in range(inst.llm_count):
            for n in range(inst.g_count):
                model.setSolVal(sol, x[i, n], 1.0 if warm_llm[i] == n else 0.0)
        accepted_warm_start = bool(model.addSol(sol))

        if progress_callback is not None:
            progress_callback(
                time.perf_counter() - start,
                warm_llm.copy(),
                llm_surrogate_cost(inst, warm_llm, similarity),
                {"source": "warm_start", "solver": "scip_gqap"},
            )

            from pyscipopt import Eventhdlr, SCIP_EVENTTYPE

            class BestSolutionHandler(Eventhdlr):
                def eventinit(self) -> None:
                    self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

                def eventexit(self) -> None:
                    self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

                def eventexec(self, event) -> None:
                    best_sol = self.model.getBestSol()
                    if best_sol is None:
                        return
                    place = [
                        max(range(inst.g_count), key=lambda n: self.model.getSolVal(best_sol, x[i, n]))
                        for i in range(inst.llm_count)
                    ]
                    progress_callback(
                        time.perf_counter() - start,
                        place,
                        llm_surrogate_cost(inst, place, similarity),
                        {"source": "solver_incumbent", "solver": "scip_gqap"},
                    )

            model.includeEventhdlr(BestSolutionHandler(), "best_solution_trace", "Record incumbent LLM deployments.")

        model.optimize()
        elapsed = time.perf_counter() - start
        status = str(model.getStatus())
        n_sols = int(model.getNSols())
        if n_sols > 0:
            place = [max(range(inst.g_count), key=lambda n: model.getVal(x[i, n])) for i in range(inst.llm_count)]
        else:
            place = warm_llm

        objective_value = math.nan
        best_bound = math.nan
        relative_gap = math.nan
        try:
            if n_sols > 0:
                objective_value = float(model.getObjVal())
                best_bound = float(model.getDualbound())
                relative_gap = float(model.getGap())
        except Exception:
            pass

        return place, {
            "phase1_solver": "scip_gqap",
            "solver_status": status.upper(),
            "llm_surrogate_cost": llm_surrogate_cost(inst, place, similarity),
            "objective_value": objective_value,
            "best_objective_bound": best_bound,
            "relative_gap": relative_gap,
            "solve_time_s": elapsed,
            "runtime_s": elapsed,
            "node_count": math.nan,
            "x_variables": inst.llm_count * inst.g_count,
            "quadratic_objective_terms": quadratic_terms,
            "proven_optimal": status.lower() == "optimal",
            "timeout_s": timeout_s,
            "warm_start_accepted": accepted_warm_start,
            "solution_count": n_sols,
        }
    except Exception as exc:
        fallback = random_llm_assignment(inst, random.Random(inst.seed + 83))
        return fallback, {
            "phase1_solver": "scip_gqap",
            "solver_status": "ERROR",
            "error": str(exc),
            "llm_surrogate_cost": llm_surrogate_cost(inst, fallback, similarity),
            "solve_time_s": time.perf_counter() - start,
            "proven_optimal": False,
        }


def iwc_gs_tool_deployment(inst: Instance, llm_place: list[int]) -> list[int]:
    w = weighted_calls(inst)
    freq = w.sum(axis=0)
    h_count = tool_host_count(inst)
    host_coords = tool_host_coords(inst)
    server_orders: dict[int, list[int]] = {}
    for j in range(inst.tool_count):
        if freq[j] <= 1e-12:
            centroid = inst.g_coords[llm_place].mean(axis=0)
        else:
            centroid = sum(float(w[i, j]) * inst.g_coords[llm_place[i]] for i in range(inst.llm_count)) / float(freq[j])
        server_orders[j] = sorted(range(h_count), key=lambda h: float(np.linalg.norm(host_coords[h] - centroid)))

    def try_order(order: list[int]) -> list[int] | None:
        rem_cpu_arr, rem_mem_arr = tool_host_capacity(inst, llm_place)
        rem_cpu = rem_cpu_arr.astype(int).tolist()
        rem_mem = rem_mem_arr.astype(int).tolist()
        place = [-1] * inst.tool_count
        for j in order:
            assigned = False
            for m in server_orders[j]:
                if rem_cpu[m] >= inst.tool_cpu[j] and rem_mem[m] >= inst.tool_mem[j]:
                    place[j] = m
                    rem_cpu[m] -= int(inst.tool_cpu[j])
                    rem_mem[m] -= int(inst.tool_mem[j])
                    assigned = True
                    break
            if not assigned:
                return None
        return place

    freq_order = sorted(range(inst.tool_count), key=lambda j: freq[j], reverse=True)
    resource_order = sorted(
        range(inst.tool_count),
        key=lambda j: (int(inst.tool_cpu[j]) + int(inst.tool_mem[j]), int(inst.tool_cpu[j]), freq[j]),
        reverse=True,
    )
    combined_order = sorted(
        range(inst.tool_count),
        key=lambda j: (freq[j], int(inst.tool_cpu[j]) + int(inst.tool_mem[j])),
        reverse=True,
    )
    candidates = [
        place
        for place in (try_order(freq_order), try_order(resource_order), try_order(combined_order))
        if place is not None
    ]
    if candidates:
        return min(candidates, key=lambda place: average_call_distance(inst, llm_place, place))
    fallback = best_fit_tool_assignment(
        inst,
        tool_cost=tool_cost_matrix(inst, llm_place),
        llm_place=llm_place,
        rng=random.Random(inst.seed + 41),
    )
    if fallback is not None:
        return fallback
    return random_tool_assignment(inst, random.Random(inst.seed + 41))


def gurobi_tool_deployment(
    inst: Instance,
    llm_place: list[int],
    timeout_s: float | None = None,
    warm_start: list[int] | None = None,
) -> tuple[list[int], dict]:
    start = time.perf_counter()
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        fallback = iwc_gs_tool_deployment(inst, llm_place)
        return fallback, {
            "phase2_solver": "gurobi_tool_mip",
            "phase2_solver_status": "IMPORT_ERROR",
            "phase2_error": str(exc),
            "phase2_solve_time_s": time.perf_counter() - start,
            "phase2_proven_optimal": False,
            "phase2_fallback": "iwc_gs",
        }

    tool_cost = tool_cost_matrix(inst, llm_place)
    h_count = tool_host_count(inst)
    rem_cpu, rem_mem = tool_host_capacity(inst, llm_place)
    if warm_start is None:
        warm_start = iwc_gs_tool_deployment(inst, llm_place)
    try:
        model = gp.Model("jolt_tool_assignment_fixed_llm")
        model.Params.OutputFlag = 0
        if timeout_s is not None and timeout_s > 0:
            model.Params.TimeLimit = float(timeout_s)

        y = model.addVars(inst.tool_count, h_count, vtype=GRB.BINARY, name="y")
        model.addConstrs((gp.quicksum(y[j, h] for h in range(h_count)) == 1 for j in range(inst.tool_count)))
        model.addConstrs(
            (
                gp.quicksum(int(inst.tool_cpu[j]) * y[j, h] for j in range(inst.tool_count))
                <= int(rem_cpu[h])
                for h in range(h_count)
            )
        )
        model.addConstrs(
            (
                gp.quicksum(int(inst.tool_mem[j]) * y[j, h] for j in range(inst.tool_count))
                <= int(rem_mem[h])
                for h in range(h_count)
            )
        )

        for j in range(inst.tool_count):
            for h in range(h_count):
                y[j, h].Start = 1.0 if warm_start and warm_start[j] == h else 0.0

        model.setObjective(
            gp.quicksum(float(tool_cost[j, h]) * y[j, h] for j in range(inst.tool_count) for h in range(h_count)),
            GRB.MINIMIZE,
        )
        model.optimize()
        elapsed = time.perf_counter() - start

        if model.SolCount > 0:
            place = [max(range(h_count), key=lambda h: y[j, h].X) for j in range(inst.tool_count)]
        elif warm_start and is_tool_feasible(inst, warm_start, llm_place):
            place = warm_start.copy()
        else:
            fallback = best_fit_tool_assignment(inst, tool_cost=tool_cost, llm_place=llm_place, rng=random.Random(inst.seed + 47))
            place = fallback if fallback is not None else random_tool_assignment(inst, random.Random(inst.seed + 47))

        status_name = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(model.Status, str(model.Status))
        objective_value = float(model.ObjVal) if model.SolCount > 0 else math.nan
        best_bound = float(model.ObjBound) if model.SolCount > 0 else math.nan
        relative_gap = float(model.MIPGap) if model.SolCount > 0 and math.isfinite(model.MIPGap) else math.nan
        return place, {
            "phase2_solver": "gurobi_tool_mip",
            "phase2_solver_status": status_name,
            "phase2_proven_optimal": model.Status == GRB.OPTIMAL,
            "phase2_objective_value": objective_value,
            "phase2_best_objective_bound": best_bound,
            "phase2_relative_gap": relative_gap,
            "phase2_runtime_s": float(model.Runtime),
            "phase2_solve_time_s": elapsed,
            "phase2_node_count": float(model.NodeCount),
            "phase2_y_variables": inst.tool_count * h_count,
            "phase2_objective_terms": inst.tool_count * h_count,
            "phase2_timeout_s": timeout_s,
            "phase2_warm_start_distance": average_call_distance(inst, llm_place, warm_start) if warm_start else math.nan,
        }
    except Exception as exc:
        fallback = warm_start if warm_start and is_tool_feasible(inst, warm_start, llm_place) else iwc_gs_tool_deployment(inst, llm_place)
        return fallback, {
            "phase2_solver": "gurobi_tool_mip",
            "phase2_solver_status": "ERROR",
            "phase2_error": str(exc),
            "phase2_solve_time_s": time.perf_counter() - start,
            "phase2_proven_optimal": False,
            "phase2_fallback": "warm_start_or_iwc_gs",
        }


def scip_tool_deployment(
    inst: Instance,
    llm_place: list[int],
    timeout_s: float | None = None,
    warm_start: list[int] | None = None,
) -> tuple[list[int], dict]:
    start = time.perf_counter()
    _add_local_dependency_dir(".scip_deps")
    try:
        from pyscipopt import Model, quicksum
    except ImportError as exc:
        fallback = iwc_gs_tool_deployment(inst, llm_place)
        return fallback, {
            "phase2_solver": "scip_tool_mip",
            "phase2_solver_status": "IMPORT_ERROR",
            "phase2_error": str(exc),
            "phase2_solve_time_s": time.perf_counter() - start,
            "phase2_proven_optimal": False,
            "phase2_fallback": "iwc_gs",
        }

    tool_cost = tool_cost_matrix(inst, llm_place)
    h_count = tool_host_count(inst)
    rem_cpu, rem_mem = tool_host_capacity(inst, llm_place)
    if warm_start is None:
        warm_start = iwc_gs_tool_deployment(inst, llm_place)

    try:
        model = Model("jolt_tool_assignment_fixed_llm_scip")
        model.hideOutput(True)
        if timeout_s is not None and timeout_s > 0:
            model.setParam("limits/time", float(timeout_s))

        y = {
            (j, h): model.addVar(vtype="B", name=f"y_{j}_{h}")
            for j in range(inst.tool_count)
            for h in range(h_count)
        }
        for j in range(inst.tool_count):
            model.addCons(quicksum(y[j, h] for h in range(h_count)) == 1)
        for h in range(h_count):
            model.addCons(quicksum(int(inst.tool_cpu[j]) * y[j, h] for j in range(inst.tool_count)) <= int(rem_cpu[h]))
            model.addCons(quicksum(int(inst.tool_mem[j]) * y[j, h] for j in range(inst.tool_count)) <= int(rem_mem[h]))

        accepted_warm_start = False
        if warm_start and is_tool_feasible(inst, warm_start, llm_place):
            sol = model.createSol()
            for j in range(inst.tool_count):
                for h in range(h_count):
                    model.setSolVal(sol, y[j, h], 1.0 if warm_start[j] == h else 0.0)
            accepted_warm_start = bool(model.addSol(sol))

        model.setObjective(
            quicksum(float(tool_cost[j, h]) * y[j, h] for j in range(inst.tool_count) for h in range(h_count)),
            "minimize",
        )
        model.optimize()
        elapsed = time.perf_counter() - start
        status = str(model.getStatus())
        n_sols = int(model.getNSols())

        if n_sols > 0:
            place = [max(range(h_count), key=lambda h: model.getVal(y[j, h])) for j in range(inst.tool_count)]
        elif warm_start and is_tool_feasible(inst, warm_start, llm_place):
            place = warm_start.copy()
        else:
            fallback = best_fit_tool_assignment(inst, tool_cost=tool_cost, llm_place=llm_place, rng=random.Random(inst.seed + 47))
            place = fallback if fallback is not None else random_tool_assignment(inst, random.Random(inst.seed + 47))

        objective_value = math.nan
        best_bound = math.nan
        relative_gap = math.nan
        try:
            if n_sols > 0:
                objective_value = float(model.getObjVal())
                best_bound = float(model.getDualbound())
                relative_gap = float(model.getGap())
        except Exception:
            pass

        return place, {
            "phase2_solver": "scip_tool_mip",
            "phase2_solver_status": status.upper(),
            "phase2_proven_optimal": status.lower() == "optimal",
            "phase2_objective_value": objective_value,
            "phase2_best_objective_bound": best_bound,
            "phase2_relative_gap": relative_gap,
            "phase2_runtime_s": elapsed,
            "phase2_solve_time_s": elapsed,
            "phase2_node_count": math.nan,
            "phase2_y_variables": inst.tool_count * h_count,
            "phase2_objective_terms": inst.tool_count * h_count,
            "phase2_timeout_s": timeout_s,
            "phase2_warm_start_accepted": accepted_warm_start,
            "phase2_warm_start_distance": average_call_distance(inst, llm_place, warm_start) if warm_start else math.nan,
            "phase2_solution_count": n_sols,
        }
    except Exception as exc:
        fallback = warm_start if warm_start and is_tool_feasible(inst, warm_start, llm_place) else iwc_gs_tool_deployment(inst, llm_place)
        return fallback, {
            "phase2_solver": "scip_tool_mip",
            "phase2_solver_status": "ERROR",
            "phase2_error": str(exc),
            "phase2_solve_time_s": time.perf_counter() - start,
            "phase2_proven_optimal": False,
            "phase2_fallback": "warm_start_or_iwc_gs",
        }


def instance_to_jsonable(inst: Instance) -> dict:
    data = asdict(inst)
    for key, value in list(data.items()):
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
    return data
