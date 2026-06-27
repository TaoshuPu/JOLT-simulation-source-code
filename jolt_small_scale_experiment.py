from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
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


class SolverTimeout(Exception):
    pass


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


def tool_distance_matrix(inst: Instance) -> np.ndarray:
    return np.hstack([inst.d_gg, inst.d_gc])


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


def resource_greedy_tool_assignment(inst: Instance) -> list[int]:
    order = sorted(
        range(inst.tool_count),
        key=lambda j: (
            int(inst.tool_cpu[j]) / max(1, int(max(inst.g_cpu_cap.max(), inst.c_cpu_cap.max())))
            + int(inst.tool_mem[j]) / max(1, int(max(inst.g_mem_cap.max(), inst.c_mem_cap.max()))),
            int(inst.tool_cpu[j]),
        ),
        reverse=True,
    )
    place = [-1] * inst.tool_count
    rem_cpu, rem_mem = (arr.astype(int).tolist() for arr in tool_host_capacity(inst))
    for j in order:
        feasible = [
            m
            for m in range(tool_host_count(inst))
            if rem_cpu[m] >= inst.tool_cpu[j] and rem_mem[m] >= inst.tool_mem[j]
        ]
        if not feasible:
            return random_tool_assignment(inst, random.Random(inst.seed + 29))
        m = max(feasible, key=lambda server: (rem_cpu[server] - int(inst.tool_cpu[j]), rem_mem[server] - int(inst.tool_mem[j])))
        place[j] = m
        rem_cpu[m] -= int(inst.tool_cpu[j])
        rem_mem[m] -= int(inst.tool_mem[j])
    return place


def resource_greedy_original_assignment(inst: Instance) -> tuple[list[int], list[int]]:
    llm_place = resource_greedy_llm_assignment(inst)
    tool_place = best_fit_tool_assignment(inst, tool_cost=tool_cost_matrix(inst, llm_place), llm_place=llm_place)
    if tool_place is None:
        tool_place = best_fit_tool_assignment(inst, llm_place=llm_place, rng=random.Random(inst.seed + 31))
    if tool_place is None:
        tool_place = resource_greedy_tool_assignment(inst)
    return llm_place, tool_place


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


def iter_feasible_llm_assignments(inst: Instance):
    place = [-1] * inst.llm_count
    rem_cpu = inst.g_cpu_cap.astype(int).tolist()
    rem_gpu = inst.g_gpu_cap.astype(int).tolist()
    rem_mem = inst.g_mem_cap.astype(int).tolist()
    order = sorted(range(inst.llm_count), key=lambda i: (inst.llm_gpu[i], inst.llm_cpu[i], inst.llm_mem[i]), reverse=True)

    def rec(pos: int):
        if pos == len(order):
            yield place.copy()
            return
        i = order[pos]
        for n in range(inst.g_count):
            if rem_cpu[n] >= inst.llm_cpu[i] and rem_gpu[n] >= inst.llm_gpu[i] and rem_mem[n] >= inst.llm_mem[i]:
                place[i] = n
                rem_cpu[n] -= int(inst.llm_cpu[i])
                rem_gpu[n] -= int(inst.llm_gpu[i])
                rem_mem[n] -= int(inst.llm_mem[i])
                yield from rec(pos + 1)
                rem_cpu[n] += int(inst.llm_cpu[i])
                rem_gpu[n] += int(inst.llm_gpu[i])
                rem_mem[n] += int(inst.llm_mem[i])
                place[i] = -1

    yield from rec(0)


def all_feasible_llm_assignments(inst: Instance) -> list[list[int]]:
    return list(iter_feasible_llm_assignments(inst))


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


def exact_tool_bnb(
    inst: Instance,
    tool_cost: np.ndarray,
    incumbent_cost: float,
    deadline: float | None = None,
    llm_place: list[int] | None = None,
) -> tuple[float, list[int] | None, int]:
    freq = weighted_calls(inst).sum(axis=0)
    order = sorted(
        range(inst.tool_count),
        key=lambda j: (freq[j] * (float(np.max(tool_cost[j])) - float(np.min(tool_cost[j]))), freq[j]),
        reverse=True,
    )
    min_remaining = [0.0] * (inst.tool_count + 1)
    for pos in range(inst.tool_count - 1, -1, -1):
        j = order[pos]
        min_remaining[pos] = min_remaining[pos + 1] + float(np.min(tool_cost[j]))

    greedy_cost, greedy_place = greedy_tool_assignment_for_cost(inst, tool_cost, llm_place)
    best_cost = min(incumbent_cost, greedy_cost)
    best_place = greedy_place if greedy_cost <= incumbent_cost else None
    place = [-1] * inst.tool_count
    rem_cpu_arr, rem_mem_arr = tool_host_capacity(inst, llm_place)
    rem_cpu = rem_cpu_arr.astype(int).tolist()
    rem_mem = rem_mem_arr.astype(int).tolist()
    seen: dict[tuple[int, tuple[int, ...], tuple[int, ...]], float] = {}
    nodes = 0

    def rec(pos: int, current_cost: float) -> None:
        nonlocal best_cost, best_place, nodes
        nodes += 1
        if deadline is not None and nodes % 256 == 0 and time.perf_counter() > deadline:
            raise SolverTimeout
        if current_cost + min_remaining[pos] >= best_cost - 1e-12:
            return
        state = (pos, tuple(rem_cpu), tuple(rem_mem))
        prev = seen.get(state)
        if prev is not None and prev <= current_cost + 1e-12:
            return
        seen[state] = current_cost
        if pos == inst.tool_count:
            best_cost = current_cost
            best_place = place.copy()
            return
        j = order[pos]
        servers = sorted(range(tool_cost.shape[1]), key=lambda m: tool_cost[j, m])
        for m in servers:
            if rem_cpu[m] >= inst.tool_cpu[j] and rem_mem[m] >= inst.tool_mem[j]:
                next_cost = current_cost + float(tool_cost[j, m])
                if next_cost + min_remaining[pos + 1] >= best_cost - 1e-12:
                    continue
                place[j] = m
                rem_cpu[m] -= int(inst.tool_cpu[j])
                rem_mem[m] -= int(inst.tool_mem[j])
                rec(pos + 1, next_cost)
                rem_cpu[m] += int(inst.tool_cpu[j])
                rem_mem[m] += int(inst.tool_mem[j])
                place[j] = -1

    rec(0, 0.0)
    return best_cost, best_place, nodes


def exact_original_solver(inst: Instance, timeout_s: float | None = None) -> AlgorithmResult:
    start = time.perf_counter()
    deadline = start + timeout_s if timeout_s is not None and timeout_s > 0 else None
    total_call_weight = float(weighted_calls(inst).sum())
    best_cost = math.inf
    best_llm: list[int] | None = None
    best_tool: list[int] | None = None
    llm_checked = 0
    bnb_nodes = 0
    proven_optimal = True

    warm_rng = random.Random(inst.seed + 999)
    try:
        warm_llm = random_llm_assignment(inst, warm_rng)
        warm_tool_cost = tool_cost_matrix(inst, warm_llm)
        warm_cost, warm_tool = greedy_tool_assignment_for_cost(inst, warm_tool_cost, warm_llm)
        if warm_tool is not None:
            best_cost = warm_cost
            best_llm = warm_llm
            best_tool = warm_tool
    except RuntimeError:
        pass

    try:
        for llm_place in iter_feasible_llm_assignments(inst):
            if deadline is not None and time.perf_counter() > deadline:
                raise SolverTimeout
            llm_checked += 1
            tool_cost = tool_cost_matrix(inst, llm_place)
            lower = float(np.min(tool_cost, axis=1).sum())
            if lower >= best_cost - 1e-12:
                continue
            cost, tool_place, nodes = exact_tool_bnb(inst, tool_cost, best_cost, deadline, llm_place)
            bnb_nodes += nodes
            if tool_place is not None and cost < best_cost:
                best_cost = cost
                best_llm = llm_place.copy()
                best_tool = tool_place.copy()
    except SolverTimeout:
        proven_optimal = False

    elapsed = time.perf_counter() - start
    feasible = best_llm is not None and best_tool is not None
    avg = best_cost / total_call_weight if feasible else math.inf
    assignment = (best_llm or []) + (best_tool or [])
    return AlgorithmResult(
        name="Exact B&B original",
        decision_count=inst.llm_count + inst.tool_count,
        avg_call_distance=avg,
        solve_time_s=elapsed,
        feasible=feasible,
        assignment=assignment,
        extra={
            "feasible_llm_assignments_checked": llm_checked,
            "bnb_nodes": bnb_nodes,
            "proven_optimal": proven_optimal,
            "timeout_s": timeout_s,
        },
    )


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


def cpsat_heuristic_solver(inst: Instance, timeout_s: float = 5.0) -> AlgorithmResult:
    result = cpsat_original_solver(inst, timeout_s=timeout_s, warm_start=True, hint_search=False)
    result.name = "OR-Tools CP-SAT heuristic"
    result.extra["is_exact_baseline"] = False
    result.extra["heuristic_timeout_s"] = timeout_s
    return result


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


def cpsat_tool_subproblem(
    inst: Instance,
    llm_place: list[int],
    hint_tool: list[int],
    timeout_s: float,
    objective_scale: int = 1_000_000,
) -> tuple[list[int], dict]:
    from ortools.sat.python import cp_model

    phase_start = time.perf_counter()
    model = cp_model.CpModel()
    h_count = tool_host_count(inst)
    rem_cpu, rem_mem = tool_host_capacity(inst, llm_place)
    y = {
        (j, h): model.NewBoolVar(f"y_{j}_{h}")
        for j in range(inst.tool_count)
        for h in range(h_count)
    }
    for j in range(inst.tool_count):
        model.Add(sum(y[j, h] for h in range(h_count)) == 1)
    for h in range(h_count):
        model.Add(sum(int(inst.tool_cpu[j]) * y[j, h] for j in range(inst.tool_count)) <= int(rem_cpu[h]))
        model.Add(sum(int(inst.tool_mem[j]) * y[j, h] for j in range(inst.tool_count)) <= int(rem_mem[h]))

    tool_cost = tool_cost_matrix(inst, llm_place)
    terms = []
    for j in range(inst.tool_count):
        for h in range(h_count):
            coeff = int(round(float(tool_cost[j, h]) * objective_scale))
            if coeff:
                terms.append(coeff * y[j, h])
            if hint_tool and hint_tool[j] == h:
                model.AddHint(y[j, h], 1)
            elif hint_tool:
                model.AddHint(y[j, h], 0)
    model.Minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.1, float(timeout_s))
    solver.parameters.num_search_workers = 8
    solver.parameters.use_optimization_hints = True
    status = solver.Solve(model)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if feasible:
        place = [max(range(h_count), key=lambda h: solver.Value(y[j, h])) for j in range(inst.tool_count)]
    else:
        place = hint_tool.copy()
    objective = float(solver.ObjectiveValue()) if feasible else math.nan
    bound = float(solver.BestObjectiveBound()) if feasible else math.nan
    relative_gap = max(0.0, (objective - bound) / abs(objective)) if feasible and abs(objective) > 1e-12 else math.nan
    return place, {
        "phase": "tool",
        "solver_status": solver.StatusName(status),
        "proven_optimal": status == cp_model.OPTIMAL,
        "relative_gap": relative_gap,
        "wall_time_s": float(solver.WallTime()),
        "elapsed_s": time.perf_counter() - phase_start,
        "variables": inst.tool_count * h_count,
        "objective_terms": len(terms),
    }


def cpsat_llm_subproblem(
    inst: Instance,
    tool_place: list[int],
    hint_llm: list[int],
    timeout_s: float,
    objective_scale: int = 1_000_000,
) -> tuple[list[int], dict]:
    from ortools.sat.python import cp_model

    phase_start = time.perf_counter()
    model = cp_model.CpModel()
    fixed_tool_cpu_on_g = np.zeros(inst.g_count, dtype=int)
    fixed_tool_mem_on_g = np.zeros(inst.g_count, dtype=int)
    for j, host in enumerate(tool_place):
        if host < inst.g_count:
            fixed_tool_cpu_on_g[host] += int(inst.tool_cpu[j])
            fixed_tool_mem_on_g[host] += int(inst.tool_mem[j])
    x = {
        (i, n): model.NewBoolVar(f"x_{i}_{n}")
        for i in range(inst.llm_count)
        for n in range(inst.g_count)
    }
    for i in range(inst.llm_count):
        model.Add(sum(x[i, n] for n in range(inst.g_count)) == 1)
    for n in range(inst.g_count):
        model.Add(sum(int(inst.llm_gpu[i]) * x[i, n] for i in range(inst.llm_count)) <= int(inst.g_gpu_cap[n]))
        model.Add(
            sum(int(inst.llm_cpu[i]) * x[i, n] for i in range(inst.llm_count)) + int(fixed_tool_cpu_on_g[n])
            <= int(inst.g_cpu_cap[n])
        )
        model.Add(
            sum(int(inst.llm_mem[i]) * x[i, n] for i in range(inst.llm_count)) + int(fixed_tool_mem_on_g[n])
            <= int(inst.g_mem_cap[n])
        )

    w = article_objective_weights(inst)
    terms = []
    for i in range(inst.llm_count):
        for n in range(inst.g_count):
            cost = sum(float(w[i, j]) * tool_distance(inst, n, tool_place[j]) for j in range(inst.tool_count))
            coeff = int(round(cost * objective_scale))
            if coeff:
                terms.append(coeff * x[i, n])
            if hint_llm and hint_llm[i] == n:
                model.AddHint(x[i, n], 1)
            elif hint_llm:
                model.AddHint(x[i, n], 0)
    model.Minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.1, float(timeout_s))
    solver.parameters.num_search_workers = 8
    solver.parameters.use_optimization_hints = True
    status = solver.Solve(model)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if feasible:
        place = [max(range(inst.g_count), key=lambda n: solver.Value(x[i, n])) for i in range(inst.llm_count)]
    else:
        place = hint_llm.copy()
    objective = float(solver.ObjectiveValue()) if feasible else math.nan
    bound = float(solver.BestObjectiveBound()) if feasible else math.nan
    relative_gap = max(0.0, (objective - bound) / abs(objective)) if feasible and abs(objective) > 1e-12 else math.nan
    return place, {
        "phase": "llm",
        "solver_status": solver.StatusName(status),
        "proven_optimal": status == cp_model.OPTIMAL,
        "relative_gap": relative_gap,
        "wall_time_s": float(solver.WallTime()),
        "elapsed_s": time.perf_counter() - phase_start,
        "variables": inst.llm_count * inst.g_count,
        "objective_terms": len(terms),
    }


def cpsat_alternating_heuristic_solver(
    inst: Instance,
    timeout_s: float = 180.0,
    max_rounds: int = 5,
    phase_timeout_s: float = 20.0,
    objective_scale: int = 1_000_000,
    warm_samples: int = 24,
) -> AlgorithmResult:
    start = time.perf_counter()
    try:
        llm_place, tool_place, warm_extra = greedy_original_warm_start(inst, samples=warm_samples)
        best_llm = llm_place.copy()
        best_tool = tool_place.copy()
        best_distance = average_call_distance(inst, best_llm, best_tool)
        history = [best_distance]
        phase_infos: list[dict] = []
        deadline = start + timeout_s if timeout_s and timeout_s > 0 else math.inf

        for round_idx in range(max_rounds):
            remaining = deadline - time.perf_counter()
            if remaining <= 3.0:
                break
            phase_budget = min(phase_timeout_s, max(1.0, remaining * 0.45))
            tool_place, info = cpsat_tool_subproblem(inst, llm_place, tool_place, phase_budget, objective_scale)
            info["round"] = round_idx
            phase_infos.append(info)
            distance = average_call_distance(inst, llm_place, tool_place)
            if distance < best_distance and is_deployment_feasible(inst, llm_place, tool_place):
                best_distance = distance
                best_llm = llm_place.copy()
                best_tool = tool_place.copy()
            history.append(best_distance)

            remaining = deadline - time.perf_counter()
            if remaining <= 3.0:
                break
            phase_budget = min(phase_timeout_s, max(1.0, remaining * 0.45))
            llm_place, info = cpsat_llm_subproblem(inst, tool_place, llm_place, phase_budget, objective_scale)
            info["round"] = round_idx
            phase_infos.append(info)
            distance = average_call_distance(inst, llm_place, tool_place)
            if distance < best_distance and is_deployment_feasible(inst, llm_place, tool_place):
                best_distance = distance
                best_llm = llm_place.copy()
                best_tool = tool_place.copy()
            history.append(best_distance)

        elapsed = time.perf_counter() - start
        return AlgorithmResult(
            name="OR-Tools CP-SAT alternating heuristic",
            decision_count=inst.llm_count + inst.tool_count,
            avg_call_distance=best_distance,
            solve_time_s=elapsed,
            feasible=is_deployment_feasible(inst, best_llm, best_tool),
            assignment=best_llm + best_tool,
            extra={
                "solver_status": "FEASIBLE_HEURISTIC",
                "proven_optimal": False,
                "is_exact_baseline": False,
                "timeout_s": timeout_s,
                "max_rounds": max_rounds,
                "phase_timeout_s": phase_timeout_s,
                "objective_scale": objective_scale,
                **warm_extra,
                "history_head": history[:5],
                "history_tail": history[-5:],
                "rounds_completed": len(phase_infos) // 2,
                "phase_infos": phase_infos,
            },
        )
    except ImportError as exc:
        elapsed = time.perf_counter() - start
        return AlgorithmResult(
            name="OR-Tools CP-SAT alternating heuristic",
            decision_count=inst.llm_count + inst.tool_count,
            avg_call_distance=math.inf,
            solve_time_s=elapsed,
            feasible=False,
            assignment=[],
            extra={"solver_status": "IMPORT_ERROR", "error": str(exc), "proven_optimal": False},
        )


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


def repair_llm_assignment(inst: Instance, place: list[int], rng: random.Random) -> list[int]:
    fixed = place.copy()
    for _ in range(100):
        cpu = np.zeros(inst.g_count, dtype=int)
        gpu = np.zeros(inst.g_count, dtype=int)
        mem = np.zeros(inst.g_count, dtype=int)
        by_server = [[] for _ in range(inst.g_count)]
        for i, n in enumerate(fixed):
            cpu[n] += inst.llm_cpu[i]
            gpu[n] += inst.llm_gpu[i]
            mem[n] += inst.llm_mem[i]
            by_server[n].append(i)
        bad = [
            n
            for n in range(inst.g_count)
            if cpu[n] > inst.g_cpu_cap[n] or gpu[n] > inst.g_gpu_cap[n] or mem[n] > inst.g_mem_cap[n]
        ]
        if not bad:
            return fixed
        n = rng.choice(bad)
        rng.shuffle(by_server[n])
        moved = False
        for i in by_server[n]:
            targets = list(range(inst.g_count))
            rng.shuffle(targets)
            for target in targets:
                if target == n:
                    continue
                if (
                    cpu[target] + inst.llm_cpu[i] <= inst.g_cpu_cap[target]
                    and gpu[target] + inst.llm_gpu[i] <= inst.g_gpu_cap[target]
                    and mem[target] + inst.llm_mem[i] <= inst.g_mem_cap[target]
                ):
                    fixed[i] = target
                    moved = True
                    break
            if moved:
                break
        if not moved:
            return random_llm_assignment(inst, rng)
    return random_llm_assignment(inst, rng)


def repair_tool_assignment(
    inst: Instance,
    place: list[int],
    rng: random.Random,
    llm_place: list[int] | None = None,
) -> list[int]:
    fixed = place.copy()
    for _ in range(300):
        h_count = tool_host_count(inst)
        rem_cpu, rem_mem = tool_host_capacity(inst, llm_place)
        cpu = np.zeros(h_count, dtype=int)
        mem = np.zeros(h_count, dtype=int)
        by_server = [[] for _ in range(h_count)]
        for j, m in enumerate(fixed):
            cpu[m] += inst.tool_cpu[j]
            mem[m] += inst.tool_mem[j]
            by_server[m].append(j)
        bad = [m for m in range(h_count) if cpu[m] > rem_cpu[m] or mem[m] > rem_mem[m]]
        if not bad:
            return fixed
        m_bad = rng.choice(bad)
        candidates = by_server[m_bad].copy()
        rng.shuffle(candidates)
        moved = False
        for j in candidates:
            targets = list(range(h_count))
            if llm_place is not None:
                w = article_objective_weights(inst)
                targets.sort(
                    key=lambda m: sum(float(w[i, j]) * tool_distance(inst, llm_place[i], m) for i in range(inst.llm_count))
                )
            else:
                rng.shuffle(targets)
            for target in targets:
                if target == m_bad:
                    continue
                if (
                    cpu[target] + inst.tool_cpu[j] <= rem_cpu[target]
                    and mem[target] + inst.tool_mem[j] <= rem_mem[target]
                ):
                    fixed[j] = target
                    moved = True
                    break
            if moved:
                break
        if not moved:
            fallback = best_fit_tool_assignment(inst, llm_place=llm_place, rng=rng)
            return fallback if fallback is not None else random_tool_assignment(inst, rng)
    fallback = best_fit_tool_assignment(inst, llm_place=llm_place, rng=rng)
    return fallback if fallback is not None else random_tool_assignment(inst, rng)


def direct_ga_original(
    inst: Instance,
    seed: int,
    pop_size: int = 90,
    max_gen: int = 220,
    pc: float = 0.9,
    pm_gene: float = 0.08,
) -> AlgorithmResult:
    rng = random.Random(seed)
    start = time.perf_counter()

    def random_chromosome() -> list[int]:
        return random_llm_assignment(inst, rng) + random_tool_assignment(inst, rng)

    def repair(chrom: list[int]) -> list[int]:
        llm = repair_llm_assignment(inst, chrom[: inst.llm_count], rng)
        tool = repair_tool_assignment(inst, chrom[inst.llm_count :], rng, llm)
        return llm + tool

    def score(chrom: list[int]) -> float:
        llm = chrom[: inst.llm_count]
        tool = chrom[inst.llm_count :]
        if not is_deployment_feasible(inst, llm, tool):
            return 1e9
        return average_call_distance(inst, llm, tool)

    def tournament(pop: list[list[int]], k: int = 3) -> list[int]:
        sample = rng.sample(pop, k)
        return min(sample, key=score).copy()

    population = [random_chromosome() for _ in range(pop_size)]
    best = min(population, key=score).copy()
    best_score = score(best)
    history = [best_score]

    for _ in range(max_gen):
        ranked = sorted(population, key=score)
        next_pop = [ranked[0].copy(), ranked[1].copy()]
        while len(next_pop) < pop_size:
            p1 = tournament(population)
            p2 = tournament(population)
            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < pc:
                for idx in range(inst.llm_count + inst.tool_count):
                    if rng.random() < 0.5:
                        c1[idx], c2[idx] = c2[idx], c1[idx]
            for child in (c1, c2):
                for idx in range(inst.llm_count):
                    if rng.random() < pm_gene:
                        child[idx] = rng.randrange(inst.g_count)
                for idx in range(inst.llm_count, inst.llm_count + inst.tool_count):
                    if rng.random() < pm_gene:
                        child[idx] = rng.randrange(tool_host_count(inst))
                next_pop.append(repair(child))
                if len(next_pop) >= pop_size:
                    break
        population = next_pop
        current = min(population, key=score).copy()
        current_score = score(current)
        if current_score < best_score:
            best = current
            best_score = current_score
        history.append(best_score)

    elapsed = time.perf_counter() - start
    return AlgorithmResult(
        name="GA original",
        decision_count=inst.llm_count + inst.tool_count,
        avg_call_distance=best_score,
        solve_time_s=elapsed,
        feasible=is_deployment_feasible(inst, best[: inst.llm_count], best[inst.llm_count :]),
        assignment=best,
        extra={"generations": max_gen, "pop_size": pop_size, "history_head": history[:5], "history_tail": history[-5:]},
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


def knn_init_llm(inst: Instance, similarity: np.ndarray, rng: random.Random) -> list[int]:
    centers = rng.sample(range(inst.llm_count), min(inst.g_count, inst.llm_count))
    clusters: dict[int, list[int]] = {c: [c] for c in centers}
    for i in range(inst.llm_count):
        if i in clusters:
            continue
        best_center = max(centers, key=lambda c: similarity[i, c])
        clusters[best_center].append(i)
    cluster_items = list(clusters.items())

    def cluster_sim(item: tuple[int, list[int]]) -> float:
        _, members = item
        return sum(float(similarity[a, b]) for a in members for b in members)

    cluster_items.sort(key=cluster_sim, reverse=True)
    centrality = [(n, float(inst.d_gg[n].sum())) for n in range(inst.g_count)]
    centrality.sort(key=lambda x: x[1])
    place = [-1] * inst.llm_count
    for idx, (_, members) in enumerate(cluster_items):
        n = centrality[idx % inst.g_count][0]
        for i in members:
            place[i] = n
    for i, n in enumerate(place):
        if n < 0:
            place[i] = rng.randrange(inst.g_count)
    return repair_llm_assignment(inst, place, rng)


def server_centric_crossover(
    inst: Instance,
    p1: list[int],
    p2: list[int],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    c1, c2 = p1.copy(), p2.copy()
    server = rng.randrange(inst.g_count)
    for i in range(inst.llm_count):
        if p2[i] == server:
            c1[i] = server
        if p1[i] == server:
            c2[i] = server
    return repair_llm_assignment(inst, c1, rng), repair_llm_assignment(inst, c2, rng)


def tabu_search_llm(
    inst: Instance,
    place: list[int],
    similarity: np.ndarray,
    rng: random.Random,
    max_iter: int = 6,
    tenure: int = 5,
) -> list[int]:
    current = place.copy()
    best = current.copy()
    current_cost = llm_surrogate_cost(inst, current, similarity)
    best_cost = current_cost
    tabu: dict[tuple[int, int], int] = {}
    cpu = np.zeros(inst.g_count, dtype=int)
    gpu = np.zeros(inst.g_count, dtype=int)
    mem = np.zeros(inst.g_count, dtype=int)
    for i, n in enumerate(current):
        cpu[n] += int(inst.llm_cpu[i])
        gpu[n] += int(inst.llm_gpu[i])
        mem[n] += int(inst.llm_mem[i])

    def move_delta(i: int, old: int, new: int) -> float:
        delta = 0.0
        for k, server_k in enumerate(current):
            delta += float(similarity[i, k]) * (float(inst.d_gg[new, server_k]) - float(inst.d_gg[old, server_k]))
            delta += float(similarity[k, i]) * (float(inst.d_gg[server_k, new]) - float(inst.d_gg[server_k, old]))
        return delta

    for it in range(max_iter):
        moves = []
        for i in range(inst.llm_count):
            old = current[i]
            for new in range(inst.g_count):
                if new == old:
                    continue
                if (
                    cpu[new] + int(inst.llm_cpu[i]) > int(inst.g_cpu_cap[new])
                    or gpu[new] + int(inst.llm_gpu[i]) > int(inst.g_gpu_cap[new])
                    or mem[new] + int(inst.llm_mem[i]) > int(inst.g_mem_cap[new])
                ):
                    continue
                c = current_cost + move_delta(i, old, new)
                tabu_key = (i, old)
                if tabu.get(tabu_key, -1) > it and c >= best_cost:
                    continue
                moves.append((c, i, old, new))
        if not moves:
            break
        moves.sort(key=lambda x: x[0])
        chosen = moves[0]
        chosen_cost, i, old, new = chosen
        current[i] = new
        cpu[old] -= int(inst.llm_cpu[i])
        gpu[old] -= int(inst.llm_gpu[i])
        mem[old] -= int(inst.llm_mem[i])
        cpu[new] += int(inst.llm_cpu[i])
        gpu[new] += int(inst.llm_gpu[i])
        mem[new] += int(inst.llm_mem[i])
        current_cost = chosen_cost
        tabu[(i, old)] = it + tenure + rng.randrange(3)
        if chosen_cost < best_cost:
            best = current.copy()
            best_cost = chosen_cost
        elif rng.random() < 0.05:
            current = random_llm_assignment(inst, rng)
            current_cost = llm_surrogate_cost(inst, current, similarity)
            cpu[:] = 0
            gpu[:] = 0
            mem[:] = 0
            for item, server in enumerate(current):
                cpu[server] += int(inst.llm_cpu[item])
                gpu[server] += int(inst.llm_gpu[item])
                mem[server] += int(inst.llm_mem[item])
    return best


def hisc_ma_llm_deployment(
    inst: Instance,
    seed: int,
    pop_size: int = 24,
    max_gen: int = 45,
    pc: float = 0.85,
    pm: float = 0.25,
    timeout_s: float | None = None,
    local_iter: int = 6,
    late_pm: float | None = None,
    late_mutation_start: float = 0.72,
) -> tuple[list[int], dict]:
    start = time.perf_counter()
    rng = random.Random(seed)
    similarity = llm_similarity(inst)
    population: list[list[int]] = []
    score_cache: dict[tuple[int, ...], float] = {}
    knn_count = max(1, int(0.2 * pop_size))
    for idx in range(pop_size):
        if idx < knn_count:
            population.append(knn_init_llm(inst, similarity, rng))
        else:
            population.append(random_llm_assignment(inst, rng))

    def score(place: list[int]) -> float:
        key = tuple(place)
        value = score_cache.get(key)
        if value is None:
            value = llm_surrogate_cost(inst, place, similarity)
            score_cache[key] = value
        return value

    def tournament() -> list[int]:
        sample = rng.sample(population, 3)
        return min(sample, key=score).copy()

    best = min(population, key=score).copy()
    history = [score(best)]
    generations_completed = 0
    for gen in range(max_gen):
        if timeout_s is not None and time.perf_counter() - start >= timeout_s:
            break
        progress = gen / max(1, max_gen - 1)
        if late_pm is not None and progress >= late_mutation_start:
            span = max(1e-9, 1.0 - late_mutation_start)
            ratio = min(1.0, (progress - late_mutation_start) / span)
            effective_pm = pm + (late_pm - pm) * ratio
        else:
            effective_pm = pm
        offspring: list[list[int]] = []
        while len(offspring) < pop_size:
            if timeout_s is not None and time.perf_counter() - start >= timeout_s:
                break
            p1 = tournament()
            p2 = tournament()
            if rng.random() < pc:
                c1, c2 = server_centric_crossover(inst, p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for child in (c1, c2):
                if rng.random() < effective_pm:
                    i = rng.randrange(inst.llm_count)
                    child[i] = rng.randrange(inst.g_count)
                    child = repair_llm_assignment(inst, child, rng)
                child = tabu_search_llm(inst, child, similarity, rng, max_iter=local_iter)
                offspring.append(child)
                if len(offspring) >= pop_size:
                    break
        if not offspring:
            break
        population = sorted(population + offspring, key=score)[:pop_size]
        if score(population[0]) < score(best):
            best = population[0].copy()
        history.append(score(best))
        generations_completed += 1
    return best, {
        "phase1_solver": "hisc_ma",
        "solver_status": "TIME_LIMIT" if timeout_s is not None and time.perf_counter() - start >= timeout_s else "HEURISTIC_DONE",
        "llm_surrogate_cost": score(best),
        "solve_time_s": time.perf_counter() - start,
        "x_variables": inst.llm_count * inst.g_count,
        "quadratic_objective_terms": inst.llm_count * inst.llm_count * inst.g_count * max(0, inst.g_count - 1),
        "pop_size": pop_size,
        "max_generations": max_gen,
        "generations_completed": generations_completed,
        "local_iter": local_iter,
        "mutation_probability": pm,
        "late_mutation_probability": late_pm,
        "history_head": history[:5],
        "history_tail": history[-5:],
        "proven_optimal": False,
        "timeout_s": timeout_s,
    }


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
        set_nonlinear_objective(model, quicksum(objective_terms), sense="minimize")

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


def proposed_two_stage(
    inst: Instance,
    seed: int,
    split_pop_size: int = 24,
    split_generations: int = 45,
    phase1_timeout_s: float | None = 30.0,
) -> AlgorithmResult:
    start = time.perf_counter()
    llm_place, llm_extra = gurobi_gqap_llm_deployment(inst, timeout_s=phase1_timeout_s)
    tool_place = iwc_gs_tool_deployment(inst, llm_place)
    elapsed = time.perf_counter() - start
    return AlgorithmResult(
        name="GQAP+IWC-GS",
        decision_count=inst.llm_count,
        avg_call_distance=average_call_distance(inst, llm_place, tool_place),
        solve_time_s=elapsed,
        feasible=is_deployment_feasible(inst, llm_place, tool_place),
        assignment=llm_place + tool_place,
        extra={
            **llm_extra,
            "solver_status": f"PHASE1_{llm_extra.get('solver_status', 'UNKNOWN')}",
            "phase1_proven_optimal": bool(llm_extra.get("proven_optimal", False)),
            "proven_optimal": False,
            "phase2_solver": "iwc_gs",
            "display_name_note": "Phase 1 solved by Gurobi GQAP; method label kept as GQAP+IWC-GS.",
        },
    )


def hisc_ma_iwc_gs_solver(
    inst: Instance,
    seed: int,
    phase1_timeout_s: float | None = 180.0,
    progress_callback: Callable[[float, list[int], list[int], float, dict], None] | None = None,
) -> AlgorithmResult:
    start = time.perf_counter()
    if inst.llm_count <= 20:
        params = {"pop_size": 96, "max_gen": 180, "pc": 0.92, "pm": 0.14, "local_iter": 6, "late_pm": 0.35, "restarts": 6}
    elif inst.llm_count <= 40:
        params = {"pop_size": 96, "max_gen": 180, "pc": 0.92, "pm": 0.14, "local_iter": 5, "late_pm": 0.35, "restarts": 5}
    elif inst.llm_count <= 60:
        params = {"pop_size": 96, "max_gen": 180, "pc": 0.92, "pm": 0.14, "local_iter": 4, "late_pm": 0.35, "restarts": 3}
    else:
        params = {"pop_size": 64, "max_gen": 140, "pc": 0.92, "pm": 0.16, "local_iter": 4, "late_pm": 0.35, "restarts": 4}

    best_llm: list[int] | None = None
    best_tool: list[int] | None = None
    best_distance = math.inf
    restart_summaries: list[dict] = []
    timeout_hit = False

    for restart in range(int(params["restarts"])):
        elapsed = time.perf_counter() - start
        if phase1_timeout_s is not None and elapsed >= phase1_timeout_s:
            timeout_hit = True
            break
        remaining = None if phase1_timeout_s is None else max(1e-3, phase1_timeout_s - elapsed)
        llm_place, llm_extra = hisc_ma_llm_deployment(
            inst,
            seed + 104729 * restart,
            pop_size=int(params["pop_size"]),
            max_gen=int(params["max_gen"]),
            pc=float(params["pc"]),
            pm=float(params["pm"]),
            timeout_s=remaining,
            local_iter=int(params["local_iter"]),
            late_pm=float(params["late_pm"]),
        )
        tool_place = iwc_gs_tool_deployment(inst, llm_place)
        distance = average_call_distance(inst, llm_place, tool_place)
        if distance < best_distance and is_deployment_feasible(inst, llm_place, tool_place):
            best_llm = llm_place.copy()
            best_tool = tool_place.copy()
            best_distance = distance
            if progress_callback is not None:
                progress_callback(
                    time.perf_counter() - start,
                    best_llm.copy(),
                    best_tool.copy(),
                    best_distance,
                    {"source": "restart_incumbent", "restart": restart},
                )
        restart_summaries.append(
            {
                "restart": restart,
                "avg_call_distance": distance,
                "llm_surrogate_cost": llm_extra.get("llm_surrogate_cost"),
                "solve_time_s": llm_extra.get("solve_time_s"),
                "generations_completed": llm_extra.get("generations_completed"),
                "solver_status": llm_extra.get("solver_status"),
            }
        )
        if llm_extra.get("solver_status") == "TIME_LIMIT":
            timeout_hit = True
            break

    if best_llm is None or best_tool is None:
        best_llm = random_llm_assignment(inst, random.Random(seed + 91))
        best_tool = iwc_gs_tool_deployment(inst, best_llm)
        best_distance = average_call_distance(inst, best_llm, best_tool)

    elapsed_total = time.perf_counter() - start
    similarity = llm_similarity(inst)
    q_terms = inst.llm_count * inst.llm_count * inst.g_count * max(0, inst.g_count - 1)
    return AlgorithmResult(
        name="HISC-MA + IWC-GS",
        decision_count=inst.llm_count,
        avg_call_distance=best_distance,
        solve_time_s=elapsed_total,
        feasible=is_deployment_feasible(inst, best_llm, best_tool),
        assignment=best_llm + best_tool,
        extra={
            "phase1_solver": "hisc_ma",
            "solver_status": "PHASE1_TIME_LIMIT" if timeout_hit else "PHASE1_HEURISTIC_DONE",
            "phase1_proven_optimal": False,
            "proven_optimal": False,
            "phase2_solver": "iwc_gs",
            "llm_surrogate_cost": llm_surrogate_cost(inst, best_llm, similarity),
            "solve_time_s": elapsed_total,
            "x_variables": inst.llm_count * inst.g_count,
            "quadratic_objective_terms": q_terms,
            "timeout_s": phase1_timeout_s,
            "tuned_params": params,
            "restarts_completed": len(restart_summaries),
            "restart_summaries": restart_summaries,
            "history_head": restart_summaries[:2],
            "history_tail": restart_summaries[-2:],
            "display_name_note": "Phase 1 solved by HISC-MA; Phase 2 solved by IWC-GS.",
        },
    )


def proposed_two_stage_solver(
    inst: Instance,
    seed: int,
    phase1_timeout_s: float | None = 30.0,
    phase2_timeout_s: float | None = 30.0,
    gqap_solver: str = "gurobi",
    tool_mip_solver: str = "gurobi",
) -> AlgorithmResult:
    start = time.perf_counter()
    gqap_solver = gqap_solver.lower()
    tool_mip_solver = tool_mip_solver.lower()
    gqap_solvers = {
        "gurobi": gurobi_gqap_llm_deployment,
        "scip": scip_gqap_llm_deployment,
    }
    tool_mip_solvers = {
        "gurobi": gurobi_tool_deployment,
        "scip": scip_tool_deployment,
    }
    if gqap_solver not in gqap_solvers:
        raise ValueError(f"Unsupported GQAP solver: {gqap_solver}")
    if tool_mip_solver not in tool_mip_solvers:
        raise ValueError(f"Unsupported Tool-MIP solver: {tool_mip_solver}")

    llm_place, llm_extra = gqap_solvers[gqap_solver](inst, timeout_s=phase1_timeout_s)
    warm_tool = iwc_gs_tool_deployment(inst, llm_place)
    tool_place, tool_extra = tool_mip_solvers[tool_mip_solver](
        inst,
        llm_place,
        timeout_s=phase2_timeout_s,
        warm_start=warm_tool,
    )
    elapsed = time.perf_counter() - start
    phase1_status = llm_extra.get("solver_status", "UNKNOWN")
    phase2_status = tool_extra.get("phase2_solver_status", "UNKNOWN")
    return AlgorithmResult(
        name="JOLT",
        decision_count=inst.llm_count,
        avg_call_distance=average_call_distance(inst, llm_place, tool_place),
        solve_time_s=elapsed,
        feasible=is_deployment_feasible(inst, llm_place, tool_place),
        assignment=llm_place + tool_place,
        extra={
            **llm_extra,
            **tool_extra,
            "solver_status": f"PHASE1_{phase1_status};PHASE2_{phase2_status}",
            "phase1_proven_optimal": bool(llm_extra.get("proven_optimal", False)),
            "proven_optimal": bool(llm_extra.get("proven_optimal", False))
            and bool(tool_extra.get("phase2_proven_optimal", False)),
            "phase1_solver": llm_extra.get("phase1_solver", f"{gqap_solver}_gqap"),
            "configured_gqap_solver": gqap_solver,
            "configured_tool_mip_solver": tool_mip_solver,
            "display_name_note": (
                f"Phase 1 solves the LLM GQAP with {gqap_solver}; "
                f"Phase 2 solves fixed-LLM tool placement with {tool_mip_solver}."
            ),
            "unused_seed": seed,
        },
    )


def run_gqap_tool_mip(
    inst: Instance,
    gqap_solver: str = "gurobi",
    tool_mip_solver: str = "gurobi",
    phase1_timeout_s: float | None = 30.0,
    phase2_timeout_s: float | None = 30.0,
    seed: int = 0,
) -> AlgorithmResult:
    return proposed_two_stage_solver(
        inst,
        seed=seed,
        phase1_timeout_s=phase1_timeout_s,
        phase2_timeout_s=phase2_timeout_s,
        gqap_solver=gqap_solver,
        tool_mip_solver=tool_mip_solver,
    )


def instance_to_jsonable(inst: Instance) -> dict:
    data = asdict(inst)
    for key, value in list(data.items()):
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
    return data


def run_trial(
    seed: int,
    llm_count: int,
    tool_count: int,
    g_count: int,
    c_count: int,
    exact_timeout_s: float | None,
    exact_solver: str,
    ga_pop_size: int,
    ga_generations: int,
    split_pop_size: int,
    split_generations: int,
    capacity_mode: str = "scaled_capacity",
) -> tuple[Instance, list[AlgorithmResult]]:
    inst = make_instance(
        seed,
        llm_count=llm_count,
        tool_count=tool_count,
        g_count=None if capacity_mode == "fixed_per_server" else g_count,
        c_count=None if capacity_mode == "fixed_per_server" else c_count,
        capacity_mode=capacity_mode,
    )
    split = proposed_two_stage(inst, seed + 10_000, split_pop_size=split_pop_size, split_generations=split_generations)
    if exact_solver == "gurobi":
        exact = gurobi_original_solver(inst, timeout_s=exact_timeout_s)
    elif exact_solver == "bnb":
        exact = exact_original_solver(inst, timeout_s=exact_timeout_s)
    else:
        exact = cpsat_original_solver(inst, timeout_s=exact_timeout_s)
    ga = direct_ga_original(inst, seed + 20_000, pop_size=ga_pop_size, max_gen=ga_generations)
    return inst, [exact, ga, split]


def summarize(rows: list[dict]) -> list[dict]:
    names = sorted({row["method"] for row in rows})
    summary = []
    for name in names:
        group = [row for row in rows if row["method"] == name]
        distances = [float(row["avg_call_distance"]) for row in group]
        times = [float(row["solve_time_s"]) for row in group]
        gaps = [float(row["gap_to_exact_percent"]) for row in group if row["gap_to_exact_percent"] != ""]
        summary.append(
            {
                "method": name,
                "decision_count": group[0]["decision_count"],
                "trials": len(group),
                "mean_avg_call_distance": statistics.mean(distances),
                "std_avg_call_distance": statistics.pstdev(distances) if len(distances) > 1 else 0.0,
                "mean_solve_time_s": statistics.mean(times),
                "std_solve_time_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
                "mean_gap_to_exact_percent": statistics.mean(gaps) if gaps else math.nan,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict]) -> None:
    print("\nSummary")
    print(
        f"{'Method':<28} {'Decisions':>9} {'Distance mean':>15} {'Distance std':>14} "
        f"{'Time mean(s)':>13} {'Time std(s)':>12} {'Gap %':>9}"
    )
    for row in summary:
        gap = row["mean_gap_to_exact_percent"]
        gap_text = "NA" if isinstance(gap, float) and math.isnan(gap) else f"{gap:>9.3f}"
        print(
            f"{row['method']:<28} {row['decision_count']:>9} "
            f"{row['mean_avg_call_distance']:>15.6f} {row['std_avg_call_distance']:>14.6f} "
            f"{row['mean_solve_time_s']:>13.6f} {row['std_solve_time_s']:>12.6f} "
            f"{gap_text:>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-scale JOLT deployment simulation.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--llms", type=int, default=5)
    parser.add_argument("--tools", type=int, default=15)
    parser.add_argument("--g-servers", type=int, default=4)
    parser.add_argument("--c-servers", type=int, default=6)
    parser.add_argument("--capacity-mode", choices=["scaled_capacity", "fixed_per_server", "g_only_fixed_per_server"], default="scaled_capacity")
    parser.add_argument("--exact-timeout-s", type=float, default=30.0)
    parser.add_argument("--exact-solver", choices=["cpsat", "gurobi", "bnb"], default="cpsat")
    parser.add_argument("--ga-pop-size", type=int, default=90)
    parser.add_argument("--ga-generations", type=int, default=220)
    parser.add_argument("--split-pop-size", type=int, default=24)
    parser.add_argument("--split-generations", type=int, default=45)
    parser.add_argument("--out-dir", type=Path, default=Path("jolt_small_scale_outputs"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    first_instance: Instance | None = None

    for trial in range(args.trials):
        seed = args.seed + trial
        inst, results = run_trial(
            seed,
            llm_count=args.llms,
            tool_count=args.tools,
            g_count=args.g_servers,
            c_count=args.c_servers,
            exact_timeout_s=args.exact_timeout_s,
            exact_solver=args.exact_solver,
            ga_pop_size=args.ga_pop_size,
            ga_generations=args.ga_generations,
            split_pop_size=args.split_pop_size,
            split_generations=args.split_generations,
            capacity_mode=args.capacity_mode,
        )
        if first_instance is None:
            first_instance = inst
        exact_result = next(r for r in results if r.extra.get("is_exact_baseline"))
        exact_distance = exact_result.avg_call_distance
        exact_is_proven = bool(exact_result.extra.get("proven_optimal", False))
        print(f"Trial {trial + 1}/{args.trials} seed={seed}")
        for result in results:
            gap = ""
            if exact_is_proven and math.isfinite(exact_distance):
                gap = 100.0 * (result.avg_call_distance - exact_distance) / exact_distance
            rows.append(
                {
                    "trial": trial,
                    "seed": seed,
                    "method": result.name,
                    "decision_count": result.decision_count,
                    "avg_call_distance": result.avg_call_distance,
                    "solve_time_s": result.solve_time_s,
                    "gap_to_exact_percent": gap,
                    "feasible": result.feasible,
                    "assignment": json.dumps(result.assignment),
                    "extra": json.dumps(result.extra),
                }
            )
            optimal_note = ""
            if result.extra.get("is_exact_baseline") and not result.extra.get("proven_optimal", True):
                optimal_note = " best-so-far"
            print(
                f"  {result.name:<28} distance={result.avg_call_distance:.6f} "
                f"time={result.solve_time_s:.6f}s gap={gap if gap == '' else f'{gap:.3f}%'}{optimal_note}"
            )

    summary = summarize(rows)
    result_fields = [
        "trial",
        "seed",
        "method",
        "decision_count",
        "avg_call_distance",
        "solve_time_s",
        "gap_to_exact_percent",
        "feasible",
        "assignment",
        "extra",
    ]
    summary_fields = [
        "method",
        "decision_count",
        "trials",
        "mean_avg_call_distance",
        "std_avg_call_distance",
        "mean_solve_time_s",
        "std_solve_time_s",
        "mean_gap_to_exact_percent",
    ]
    write_csv(args.out_dir / "trial_results.csv", rows, result_fields)
    write_csv(args.out_dir / "summary.csv", summary, summary_fields)
    if first_instance is not None:
        (args.out_dir / "first_instance.json").write_text(
            json.dumps(instance_to_jsonable(first_instance), indent=2),
            encoding="utf-8",
        )
    print_summary(summary)
    print(f"\nWrote results to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
