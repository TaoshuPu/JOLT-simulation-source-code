from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for deps_name in (".ortools_deps", ".gurobi_deps"):
    deps = ROOT / deps_name
    if deps.exists():
        sys.path.insert(0, str(deps))

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class LlmInstance:
    seed: int
    llm_count: int
    tool_count: int
    g_count: int
    g_coords: np.ndarray
    d_gg: np.ndarray
    pref: np.ndarray
    arrival: np.ndarray
    llm_gpu: np.ndarray
    llm_mem: np.ndarray
    g_gpu_cap: np.ndarray
    g_mem_cap: np.ndarray


@dataclass
class RunResult:
    method: str
    seed: int
    history: list[float]
    best_assignment: list[int]
    best_cost: float
    initial_cost: float
    elapsed_s: float
    feasible: bool


def make_llm_instance(
    seed: int,
    llm_count: int = 20,
    tool_count: int = 60,
    g_count: int = 6,
    gpu_cap: int = 4,
    mem_cap: int = 48,
) -> LlmInstance:
    rng = np.random.default_rng(seed)
    if llm_count > g_count * gpu_cap:
        raise ValueError("Infeasible GPU capacity: increase g_count or gpu_cap.")

    base_coords = np.array(
        [
            [0.0, 0.0],
            [1.7, 0.1],
            [3.4, 0.0],
            [0.2, 2.0],
            [1.8, 2.1],
            [3.5, 1.9],
        ],
        dtype=float,
    )
    if g_count > len(base_coords):
        side = math.ceil(math.sqrt(g_count))
        grid = np.array([(x * 1.7, y * 1.7) for y in range(side) for x in range(side)], dtype=float)
        base_coords = grid[:g_count]
    g_coords = base_coords[:g_count] + rng.normal(0.0, 0.06, size=(g_count, 2))
    d_gg = np.linalg.norm(g_coords[:, None, :] - g_coords[None, :, :], axis=2)

    group_count = max(2, min(g_count, llm_count // 4))
    tool_axis = np.arange(tool_count, dtype=float)
    centers = np.linspace(0.12 * tool_count, 0.88 * tool_count, group_count)
    sigma = max(2.5, tool_count / (group_count * 3.2))
    pref_rows = []
    for i in range(llm_count):
        group = i % group_count
        primary = centers[group] + rng.normal(0.0, 1.2)
        secondary = centers[(group + 1) % group_count] + rng.normal(0.0, 1.8)
        primary_bump = np.exp(-0.5 * ((tool_axis - primary) / sigma) ** 2)
        secondary_bump = 0.32 * np.exp(-0.5 * ((tool_axis - secondary) / (sigma * 1.35)) ** 2)
        background = rng.lognormal(mean=-2.45, sigma=0.55, size=tool_count)
        raw = primary_bump + secondary_bump + background
        pref_rows.append(raw / raw.sum())

    pref = np.vstack(pref_rows)
    arrival = rng.uniform(0.8, 1.4, size=llm_count)
    llm_gpu = np.ones(llm_count, dtype=int)
    llm_mem = rng.integers(8, 13, size=llm_count)
    g_gpu_cap = np.full(g_count, gpu_cap, dtype=int)
    g_mem_cap = np.full(g_count, mem_cap, dtype=int)
    if int(llm_mem.sum()) > int(g_mem_cap.sum()):
        raise ValueError("Infeasible memory capacity: increase g_count or mem_cap.")

    return LlmInstance(
        seed=seed,
        llm_count=llm_count,
        tool_count=tool_count,
        g_count=g_count,
        g_coords=g_coords,
        d_gg=d_gg,
        pref=pref,
        arrival=arrival,
        llm_gpu=llm_gpu,
        llm_mem=llm_mem,
        g_gpu_cap=g_gpu_cap,
        g_mem_cap=g_mem_cap,
    )


def instance_to_jsonable(inst: LlmInstance) -> dict:
    data = asdict(inst)
    for key, value in list(data.items()):
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
    return data


def preference_similarity(inst: LlmInstance) -> np.ndarray:
    weighted_pref = inst.arrival[:, None] * inst.pref
    return weighted_pref @ weighted_pref.T


def usage(inst: LlmInstance, place: list[int]) -> tuple[np.ndarray, np.ndarray]:
    gpu = np.zeros(inst.g_count, dtype=int)
    mem = np.zeros(inst.g_count, dtype=int)
    for i, server in enumerate(place):
        gpu[server] += int(inst.llm_gpu[i])
        mem[server] += int(inst.llm_mem[i])
    return gpu, mem


def is_feasible(inst: LlmInstance, place: list[int]) -> bool:
    if len(place) != inst.llm_count:
        return False
    if any(server < 0 or server >= inst.g_count for server in place):
        return False
    gpu, mem = usage(inst, place)
    return bool(np.all(gpu <= inst.g_gpu_cap) and np.all(mem <= inst.g_mem_cap))


def normalized_cost(inst: LlmInstance, place: list[int], sim: np.ndarray) -> float:
    if not is_feasible(inst, place):
        return math.inf
    servers = np.asarray(place, dtype=int)
    distance = inst.d_gg[np.ix_(servers, servers)]
    denom = max(1e-12, float(sim.sum()))
    return float(np.sum(sim * distance) / denom)


def random_feasible_assignment(inst: LlmInstance, rng: random.Random) -> list[int]:
    for _ in range(300):
        order = list(range(inst.llm_count))
        rng.shuffle(order)
        order.sort(key=lambda i: (int(inst.llm_gpu[i]), int(inst.llm_mem[i]), rng.random()), reverse=True)
        rem_gpu = inst.g_gpu_cap.astype(int).tolist()
        rem_mem = inst.g_mem_cap.astype(int).tolist()
        place = [-1] * inst.llm_count
        ok = True
        for i in order:
            feasible_servers = [
                n
                for n in range(inst.g_count)
                if rem_gpu[n] >= inst.llm_gpu[i] and rem_mem[n] >= inst.llm_mem[i]
            ]
            if not feasible_servers:
                ok = False
                break
            n = rng.choice(feasible_servers)
            place[i] = n
            rem_gpu[n] -= int(inst.llm_gpu[i])
            rem_mem[n] -= int(inst.llm_mem[i])
        if ok and is_feasible(inst, place):
            return place
    raise RuntimeError("Failed to build a feasible LLM placement.")


def repair_assignment(inst: LlmInstance, place: list[int], rng: random.Random) -> list[int]:
    fixed = [min(inst.g_count - 1, max(0, int(server))) for server in place]
    for _ in range(220):
        gpu, mem = usage(inst, fixed)
        bad_servers = [
            n
            for n in range(inst.g_count)
            if gpu[n] > inst.g_gpu_cap[n] or mem[n] > inst.g_mem_cap[n]
        ]
        if not bad_servers:
            return fixed
        bad = max(
            bad_servers,
            key=lambda n: max(gpu[n] - inst.g_gpu_cap[n], 0) + max(mem[n] - inst.g_mem_cap[n], 0),
        )
        members = [i for i, server in enumerate(fixed) if server == bad]
        rng.shuffle(members)
        members.sort(key=lambda i: (int(inst.llm_gpu[i]), int(inst.llm_mem[i])), reverse=True)
        moved = False
        for i in members:
            targets = list(range(inst.g_count))
            rng.shuffle(targets)
            targets.sort(
                key=lambda n: (
                    gpu[n] + int(inst.llm_gpu[i]) <= inst.g_gpu_cap[n],
                    mem[n] + int(inst.llm_mem[i]) <= inst.g_mem_cap[n],
                    int(inst.g_gpu_cap[n] - gpu[n]),
                    int(inst.g_mem_cap[n] - mem[n]),
                ),
                reverse=True,
            )
            for target in targets:
                if target == bad:
                    continue
                if (
                    gpu[target] + inst.llm_gpu[i] <= inst.g_gpu_cap[target]
                    and mem[target] + inst.llm_mem[i] <= inst.g_mem_cap[target]
                ):
                    fixed[i] = target
                    moved = True
                    break
            if moved:
                break
        if not moved:
            return random_feasible_assignment(inst, rng)
    return random_feasible_assignment(inst, rng)


def knn_diverse_initialization(inst: LlmInstance, sim: np.ndarray, rng: random.Random) -> list[int]:
    centers = rng.sample(range(inst.llm_count), min(inst.g_count, inst.llm_count))
    clusters: dict[int, list[int]] = {center: [center] for center in centers}
    for i in range(inst.llm_count):
        if i in clusters:
            continue
        center = max(centers, key=lambda c: float(sim[i, c]))
        clusters[center].append(i)

    def cluster_strength(members: list[int]) -> float:
        return float(sum(sim[a, b] for a in members for b in members))

    cluster_items = sorted(clusters.values(), key=cluster_strength, reverse=True)
    server_centrality = sorted(range(inst.g_count), key=lambda n: float(inst.d_gg[n].sum()))
    place = [-1] * inst.llm_count
    for idx, members in enumerate(cluster_items):
        server = server_centrality[idx % inst.g_count]
        for i in members:
            place[i] = server
    for i, server in enumerate(place):
        if server < 0:
            place[i] = rng.randrange(inst.g_count)
    return repair_assignment(inst, place, rng)


def build_population(
    inst: LlmInstance,
    sim: np.ndarray,
    rng: random.Random,
    pop_size: int,
    hybrid: bool,
) -> list[list[int]]:
    population = []
    knn_count = max(1, int(round(0.2 * pop_size))) if hybrid else 0
    for idx in range(pop_size):
        if idx < knn_count:
            population.append(knn_diverse_initialization(inst, sim, rng))
        else:
            population.append(random_feasible_assignment(inst, rng))
    return population


def uniform_crossover(
    inst: LlmInstance,
    p1: list[int],
    p2: list[int],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    c1 = p1.copy()
    c2 = p2.copy()
    for i in range(inst.llm_count):
        if rng.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return repair_assignment(inst, c1, rng), repair_assignment(inst, c2, rng)


def server_centric_crossover(
    inst: LlmInstance,
    p1: list[int],
    p2: list[int],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    c1 = p1.copy()
    c2 = p2.copy()
    selected_servers = set(rng.sample(range(inst.g_count), rng.randint(1, max(1, inst.g_count // 2))))
    for i in range(inst.llm_count):
        if p2[i] in selected_servers:
            c1[i] = p2[i]
        if p1[i] in selected_servers:
            c2[i] = p1[i]
    return repair_assignment(inst, c1, rng), repair_assignment(inst, c2, rng)


def mutate_assignment(
    inst: LlmInstance,
    place: list[int],
    rng: random.Random,
    pm_gene: float,
    server_level: bool,
) -> list[int]:
    child = place.copy()
    if server_level and rng.random() < 0.5:
        a, b = rng.sample(range(inst.g_count), 2)
        for i, server in enumerate(child):
            if server == a:
                child[i] = b
            elif server == b:
                child[i] = a
    else:
        for i in range(inst.llm_count):
            if rng.random() < pm_gene:
                child[i] = rng.randrange(inst.g_count)
    return repair_assignment(inst, child, rng)


def tabu_local_search(
    inst: LlmInstance,
    start: list[int],
    sim: np.ndarray,
    rng: random.Random,
    max_iter: int = 5,
    tenure: int = 7,
) -> list[int]:
    current = start.copy()
    best = current.copy()
    denom = max(1e-12, float(sim.sum()))
    best_cost = normalized_cost(inst, best, sim)
    tabu: dict[tuple[int, int], int] = {}

    for it in range(max_iter):
        moves = []
        gpu, mem = usage(inst, current)
        current_cost = normalized_cost(inst, current, sim)
        pair_weight = sim + sim.T
        pair_weight = pair_weight.copy()
        np.fill_diagonal(pair_weight, 0.0)
        current_servers = np.asarray(current, dtype=int)
        pair_move_cost = pair_weight @ inst.d_gg[:, current_servers].T
        for i in range(inst.llm_count):
            old_server = current[i]
            for new_server in range(inst.g_count):
                if new_server == old_server:
                    continue
                if (
                    gpu[new_server] + inst.llm_gpu[i] > inst.g_gpu_cap[new_server]
                    or mem[new_server] + inst.llm_mem[i] > inst.g_mem_cap[new_server]
                ):
                    continue
                delta = float(pair_move_cost[i, new_server] - pair_move_cost[i, old_server])
                cost = current_cost + delta / denom
                tabu_key = (i, old_server)
                if tabu.get(tabu_key, -1) > it and cost >= best_cost:
                    continue
                candidate = current.copy()
                candidate[i] = new_server
                moves.append((cost, candidate, tabu_key))

        swap_pairs = [(a, b) for a in range(inst.llm_count) for b in range(a + 1, inst.llm_count) if current[a] != current[b]]
        rng.shuffle(swap_pairs)
        for a, b in swap_pairs[: min(35, len(swap_pairs))]:
            server_a, server_b = current[a], current[b]
            if (
                mem[server_a] - inst.llm_mem[a] + inst.llm_mem[b] > inst.g_mem_cap[server_a]
                or mem[server_b] - inst.llm_mem[b] + inst.llm_mem[a] > inst.g_mem_cap[server_b]
            ):
                continue
            candidate = current.copy()
            candidate[a], candidate[b] = candidate[b], candidate[a]
            cost = normalized_cost(inst, candidate, sim)
            tabu_key = (a, server_a)
            if tabu.get(tabu_key, -1) > it and cost >= best_cost:
                continue
            moves.append((cost, candidate, tabu_key))

        if not moves:
            break
        moves.sort(key=lambda item: item[0])
        top = moves[: min(5, len(moves))]
        cost, current, tabu_key = top[0] if rng.random() > 0.08 else rng.choice(top)
        tabu[tabu_key] = it + tenure + rng.randrange(3)
        if cost < best_cost:
            best = current.copy()
            best_cost = cost
    return best


def select_parent(
    population: list[list[int]],
    score,
    rng: random.Random,
    tournament_size: int = 3,
) -> list[int]:
    sample = rng.sample(population, min(tournament_size, len(population)))
    return min(sample, key=score).copy()


def run_random_search(
    inst: LlmInstance,
    sim: np.ndarray,
    seed: int,
    pop_size: int,
    max_gen: int,
) -> RunResult:
    rng = random.Random(seed)
    start = time.perf_counter()
    best = random_feasible_assignment(inst, rng)
    best_cost = normalized_cost(inst, best, sim)
    initial_cost = best_cost
    history = [best_cost]
    for _ in range(max_gen):
        for _ in range(pop_size):
            candidate = random_feasible_assignment(inst, rng)
            cost = normalized_cost(inst, candidate, sim)
            if cost < best_cost:
                best = candidate
                best_cost = cost
        history.append(best_cost)
    return RunResult(
        method="Random Search",
        seed=seed,
        history=history,
        best_assignment=best,
        best_cost=best_cost,
        initial_cost=initial_cost,
        elapsed_s=time.perf_counter() - start,
        feasible=is_feasible(inst, best),
    )


def run_evolutionary(
    inst: LlmInstance,
    sim: np.ndarray,
    seed: int,
    method: str,
    pop_size: int,
    max_gen: int,
    pc: float,
    pm_gene: float,
    late_pm_gene: float,
    late_mutation_start: float,
    local_iter: int,
    hybrid_init: bool,
    memetic: bool,
    server_centric: bool,
    timeout_s: float | None = None,
) -> RunResult:
    rng = random.Random(seed)
    start = time.perf_counter()
    population = build_population(inst, sim, rng, pop_size, hybrid=hybrid_init)
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
    history = [best_cost]
    elite_count = max(2, pop_size // 12)

    timeout_hit = False
    for gen in range(max_gen):
        if timeout_s is not None and time.perf_counter() - start >= timeout_s:
            timeout_hit = True
            break
        progress = gen / max(1, max_gen - 1)
        if late_pm_gene > pm_gene and progress >= late_mutation_start:
            span = max(1e-9, 1.0 - late_mutation_start)
            ratio = min(1.0, (progress - late_mutation_start) / span)
            effective_pm_gene = pm_gene + (late_pm_gene - pm_gene) * ratio
        else:
            effective_pm_gene = pm_gene
        ranked = sorted(population, key=score)
        offspring = [candidate.copy() for candidate in ranked[:elite_count]]
        while len(offspring) < pop_size:
            if timeout_s is not None and time.perf_counter() - start >= timeout_s:
                timeout_hit = True
                break
            p1 = select_parent(population, score, rng)
            p2 = select_parent(population, score, rng)
            if rng.random() < pc:
                if server_centric:
                    c1, c2 = server_centric_crossover(inst, p1, p2, rng)
                else:
                    c1, c2 = uniform_crossover(inst, p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                if timeout_s is not None and time.perf_counter() - start >= timeout_s:
                    timeout_hit = True
                    break
                child = mutate_assignment(inst, child, rng, pm_gene=effective_pm_gene, server_level=server_centric)
                if memetic:
                    child = tabu_local_search(inst, child, sim, rng, max_iter=local_iter)
                offspring.append(child)
                if len(offspring) >= pop_size:
                    break
            if timeout_hit:
                break
        if timeout_hit:
            break

        population = sorted(population + offspring, key=score)[:pop_size]
        current = population[0]
        current_cost = score(current)
        if current_cost < best_cost:
            best = current.copy()
            best_cost = current_cost
        history.append(best_cost)

    return RunResult(
        method=method,
        seed=seed,
        history=history,
        best_assignment=best,
        best_cost=best_cost,
        initial_cost=initial_cost,
        elapsed_s=time.perf_counter() - start,
        feasible=is_feasible(inst, best),
    )


def aggregate_histories(results: list[RunResult], max_gen: int) -> list[dict]:
    methods = list(dict.fromkeys(result.method for result in results))
    rows = []
    for gen in range(max_gen + 1):
        row = {"generation": gen}
        for method in methods:
            values = [result.history[gen] for result in results if result.method == method]
            row[f"{method}_mean"] = statistics.mean(values)
            row[f"{method}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
    for name in font_names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_convergence_png(
    path: Path,
    aggregate_rows: list[dict],
    methods: list[str],
    title: str = "Algorithm 1 Convergence",
) -> None:
    width, height = 1400, 900
    margin_left, margin_right, margin_top, margin_bottom = 180, 60, 90, 130
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = {
        "Random Search": (112, 112, 112),
        "Standard GA": (35, 105, 190),
        "Standard MA": (215, 128, 40),
        "Proposed HISC-MA": (25, 145, 105),
    }
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_title = load_font(34, bold=True)
    font_label = load_font(24, bold=False)
    font_tick = load_font(20, bold=False)
    font_legend = load_font(22, bold=False)

    all_values = [
        float(row[f"{method}_mean"])
        for row in aggregate_rows
        for method in methods
        if math.isfinite(float(row[f"{method}_mean"]))
    ]
    y_min = min(all_values)
    y_max = max(all_values)
    pad = max(1e-6, (y_max - y_min) * 0.08)
    y_min = max(0.0, y_min - pad)
    y_max = y_max + pad
    max_gen = int(aggregate_rows[-1]["generation"])

    def x_of(gen: int) -> float:
        return margin_left + plot_w * gen / max(1, max_gen)

    def y_of(value: float) -> float:
        return margin_top + plot_h * (y_max - value) / max(1e-12, y_max - y_min)

    draw.text((margin_left, 28), title, fill=(20, 20, 20), font=font_title)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(30, 30, 30), width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill=(30, 30, 30), width=2)

    for idx in range(6):
        value = y_min + (y_max - y_min) * idx / 5
        y = y_of(value)
        draw.line((margin_left, y, margin_left + plot_w, y), fill=(228, 232, 236), width=1)
        draw.text((72, y - 12), f"{value:.3f}", fill=(50, 50, 50), font=font_tick)

    tick_step = max(1, max_gen // 6)
    for gen in range(0, max_gen + 1, tick_step):
        x = x_of(gen)
        draw.line((x, margin_top + plot_h, x, margin_top + plot_h + 8), fill=(30, 30, 30), width=2)
        draw.text((x - 12, margin_top + plot_h + 16), str(gen), fill=(50, 50, 50), font=font_tick)

    for method in methods:
        pts = [
            (x_of(int(row["generation"])), y_of(float(row[f"{method}_mean"])))
            for row in aggregate_rows
        ]
        draw.line(pts, fill=colors.get(method, (0, 0, 0)), width=4, joint="curve")

    draw.text((margin_left + plot_w // 2 - 60, height - 58), "Generation", fill=(30, 30, 30), font=font_label)
    label = "Best normalized weighted network distance"
    label_box = Image.new("RGBA", (560, 42), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_box)
    label_draw.text((0, 4), label, fill=(30, 30, 30), font=font_label)
    rotated_label = label_box.rotate(90, expand=True)
    img.paste(rotated_label, (16, margin_top + plot_h // 2 - rotated_label.height // 2), rotated_label)

    legend_x = margin_left + plot_w - 330
    legend_y = margin_top + 20
    for idx, method in enumerate(methods):
        y = legend_y + idx * 34
        draw.line((legend_x, y + 12, legend_x + 42, y + 12), fill=colors.get(method, (0, 0, 0)), width=5)
        draw.text((legend_x + 54, y), method, fill=(35, 35, 35), font=font_legend)

    img.save(path)


def write_convergence_svg(
    path: Path,
    aggregate_rows: list[dict],
    methods: list[str],
    title: str = "Algorithm 1 Convergence",
) -> None:
    width, height = 1000, 640
    left, right, top, bottom = 88, 40, 68, 88
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {
        "Random Search": "#707070",
        "Standard GA": "#2369be",
        "Standard MA": "#d78028",
        "Proposed HISC-MA": "#199169",
    }
    all_values = [float(row[f"{method}_mean"]) for row in aggregate_rows for method in methods]
    y_min = min(all_values)
    y_max = max(all_values)
    pad = max(1e-6, (y_max - y_min) * 0.08)
    y_min = max(0.0, y_min - pad)
    y_max = y_max + pad
    max_gen = int(aggregate_rows[-1]["generation"])

    def x_of(gen: int) -> float:
        return left + plot_w * gen / max(1, max_gen)

    def y_of(value: float) -> float:
        return top + plot_h * (y_max - value) / max(1e-12, y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="36" font-family="Arial" font-size="24" font-weight="700">{title}</text>',
    ]
    for idx in range(6):
        value = y_min + (y_max - y_min) * idx / 5
        y = y_of(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e9ee" stroke-width="1"/>')
        parts.append(f'<text x="16" y="{y + 5:.2f}" font-family="Arial" font-size="14" fill="#333">{value:.3f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>')

    tick_step = max(1, max_gen // 6)
    for gen in range(0, max_gen + 1, tick_step):
        x = x_of(gen)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 7}" stroke="#222" stroke-width="1.5"/>')
        parts.append(f'<text x="{x - 8:.2f}" y="{top + plot_h + 25}" font-family="Arial" font-size="14" fill="#333">{gen}</text>')

    for method in methods:
        points = " ".join(
            f'{x_of(int(row["generation"])):.2f},{y_of(float(row[f"{method}_mean"])):.2f}'
            for row in aggregate_rows
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors.get(method, "#000")}" stroke-width="3"/>')

    parts.append(f'<text x="{left + plot_w / 2 - 36:.2f}" y="{height - 28}" font-family="Arial" font-size="17">Generation</text>')
    parts.append(f'<text x="{left}" y="{height - 56}" font-family="Arial" font-size="17">Best normalized weighted network distance</text>')

    legend_x = left + plot_w - 250
    legend_y = top + 12
    for idx, method in enumerate(methods):
        y = legend_y + idx * 25
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 34}" y2="{y}" stroke="{colors.get(method, "#000")}" stroke-width="4"/>')
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Arial" font-size="15" fill="#222">{method}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 1 convergence simulation for JOLT HISC-MA.")
    parser.add_argument("--llms", type=int, default=20)
    parser.add_argument("--tools", type=int, default=60)
    parser.add_argument("--g-servers", type=int, default=6)
    parser.add_argument("--gpu-cap", type=int, default=4)
    parser.add_argument("--mem-cap", type=int, default=48)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=48)
    parser.add_argument("--pm-gene", type=float, default=0.06)
    parser.add_argument("--late-pm-gene", type=float, default=0.16)
    parser.add_argument("--late-mutation-start", type=float, default=0.55)
    parser.add_argument("--local-iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--out-dir", type=Path, default=Path("jolt_algorithm1_convergence_L20"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[RunResult] = []
    first_instance: LlmInstance | None = None

    for trial in range(args.trials):
        instance_seed = args.seed + trial
        inst = make_llm_instance(
            seed=instance_seed,
            llm_count=args.llms,
            tool_count=args.tools,
            g_count=args.g_servers,
            gpu_cap=args.gpu_cap,
            mem_cap=args.mem_cap,
        )
        if first_instance is None:
            first_instance = inst
        sim = preference_similarity(inst)
        print(
            f"Trial {trial + 1}/{args.trials}: LLM={args.llms}, tools={args.tools}, "
            f"G={args.g_servers}, GPU cap={args.gpu_cap}, MEM cap={args.mem_cap}",
            flush=True,
        )

        trial_results = [
            run_random_search(inst, sim, instance_seed + 101, args.pop_size, args.generations),
            run_evolutionary(
                inst,
                sim,
                instance_seed + 202,
                method="Standard GA",
                pop_size=args.pop_size,
                max_gen=args.generations,
                pc=0.88,
                pm_gene=args.pm_gene,
                late_pm_gene=args.late_pm_gene,
                late_mutation_start=args.late_mutation_start,
                local_iter=args.local_iter,
                hybrid_init=False,
                memetic=False,
                server_centric=False,
            ),
            run_evolutionary(
                inst,
                sim,
                instance_seed + 303,
                method="Standard MA",
                pop_size=args.pop_size,
                max_gen=args.generations,
                pc=0.88,
                pm_gene=args.pm_gene,
                late_pm_gene=args.late_pm_gene,
                late_mutation_start=args.late_mutation_start,
                local_iter=args.local_iter,
                hybrid_init=False,
                memetic=True,
                server_centric=False,
            ),
            run_evolutionary(
                inst,
                sim,
                instance_seed + 404,
                method="Proposed HISC-MA",
                pop_size=args.pop_size,
                max_gen=args.generations,
                pc=0.88,
                pm_gene=args.pm_gene,
                late_pm_gene=args.late_pm_gene,
                late_mutation_start=args.late_mutation_start,
                local_iter=args.local_iter,
                hybrid_init=True,
                memetic=True,
                server_centric=True,
            ),
        ]
        for result in trial_results:
            print(
                f"  {result.method:<17} initial={result.initial_cost:.6f} "
                f"final={result.best_cost:.6f} time={result.elapsed_s:.2f}s feasible={result.feasible}",
                flush=True,
            )
        all_results.extend(trial_results)

    methods = ["Random Search", "Standard GA", "Standard MA", "Proposed HISC-MA"]
    aggregate_rows = aggregate_histories(all_results, args.generations)
    write_csv(args.out_dir / "convergence_mean.csv", aggregate_rows)

    trial_rows = []
    for result in all_results:
        trial_rows.append(
            {
                "method": result.method,
                "seed": result.seed,
                "initial_cost": result.initial_cost,
                "best_cost": result.best_cost,
                "elapsed_s": result.elapsed_s,
                "feasible": result.feasible,
                "best_assignment": json.dumps(result.best_assignment),
                "history": json.dumps(result.history),
            }
        )
    write_csv(args.out_dir / "trial_results.csv", trial_rows)

    summary_rows = []
    for method in methods:
        group = [result for result in all_results if result.method == method]
        finals = [result.best_cost for result in group]
        initials = [result.initial_cost for result in group]
        times = [result.elapsed_s for result in group]
        summary_rows.append(
            {
                "method": method,
                "trials": len(group),
                "mean_initial_cost": statistics.mean(initials),
                "mean_final_cost": statistics.mean(finals),
                "std_final_cost": statistics.pstdev(finals) if len(finals) > 1 else 0.0,
                "mean_elapsed_s": statistics.mean(times),
                "feasible_trials": sum(1 for result in group if result.feasible),
            }
        )
    write_csv(args.out_dir / "summary.csv", summary_rows)

    if first_instance is not None:
        meta = instance_to_jsonable(first_instance)
        meta["experiment"] = {
            "objective": "minimize sum_i,k div(L_i,L_k) * D(P_i,P_k), normalized by total similarity",
            "llm_count": args.llms,
            "tool_count_for_preferences": args.tools,
            "g_type_servers": args.g_servers,
            "gpu_capacity_per_g_server": args.gpu_cap,
            "memory_capacity_gib_per_g_server": args.mem_cap,
            "population_size": args.pop_size,
            "generations": args.generations,
            "trials": args.trials,
            "pm_gene": args.pm_gene,
            "late_pm_gene": args.late_pm_gene,
            "late_mutation_start": args.late_mutation_start,
        }
        (args.out_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mutation_note = f", late Pm={args.late_pm_gene:g}" if args.late_pm_gene > args.pm_gene else ""
    title = f"Algorithm 1 Convergence (LLM={args.llms}, G={args.g_servers}{mutation_note})"
    draw_convergence_png(args.out_dir / "algorithm1_convergence.png", aggregate_rows, methods, title=title)
    write_convergence_svg(args.out_dir / "algorithm1_convergence.svg", aggregate_rows, methods, title=title)

    print("\nSummary")
    print(f"{'Method':<18} {'Initial':>12} {'Final':>12} {'Std':>10} {'Time(s)':>10}")
    for row in summary_rows:
        print(
            f"{row['method']:<18} {row['mean_initial_cost']:>12.6f} "
            f"{row['mean_final_cost']:>12.6f} {row['std_final_cost']:>10.6f} "
            f"{row['mean_elapsed_s']:>10.2f}"
        )
    print(f"\nWrote outputs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
