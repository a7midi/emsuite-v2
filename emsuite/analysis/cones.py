from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple
import math
import random

from emsuite.core.graph import DiGraph, Vertex


# ---------- Directed distance (Def. 10.1 style) ----------

def directed_distance_to_target(dag: DiGraph, target: Vertex, r_max: int) -> Dict[Vertex, int]:
    """
    Shortest directed path length d(u, target) on a DAG, computed by BFS
    on reversed edges (predecessors), truncated at r_max.
    """
    dist: Dict[Vertex, int] = {target: 0}
    q = deque([target])

    while q:
        v = q.popleft()
        dv = dist[v]
        if dv >= r_max:
            continue
        for u in dag.predecessors(v):
            if u not in dist:
                dist[u] = dv + 1
                q.append(u)

    return dist


# ---------- Proxy cone volume (node counts) ----------

def node_past_cone_counts(dag: DiGraph, target: Vertex, r_max: int) -> List[int]:
    """
    Proxy cone "volume": node counts in the past cone:
        N_r(v) = #{ u : d(u,v) <= r }.
    Returns [N_0, ..., N_r_max].
    """
    dist = directed_distance_to_target(dag, target, r_max=r_max)
    exact = [0] * (r_max + 1)
    for d in dist.values():
        if 0 <= d <= r_max:
            exact[d] += 1
    out = []
    running = 0
    for r in range(r_max + 1):
        running += exact[r]
        out.append(running)
    return out


def estimate_cone_dimension_proxy(
    dag: DiGraph,
    r_values: List[int] = [1, 2, 3, 4, 5, 6],
    n_targets: int = 12,
    seed: int = 0,
) -> Tuple[Optional[float], Optional[float], List[float]]:
    """
    Fit log N_r(v) ~ D log r using node-count cone proxy.
    """
    if not r_values:
        return None, None, []

    r_max = max(r_values)
    depths = dag.depth_map()
    verts = list(dag.vertices())
    if not verts:
        return None, None, []

    eligible = [v for v in verts if depths.get(v, 0) >= r_max] or verts
    rng = random.Random(seed)
    targets = rng.sample(eligible, k=min(n_targets, len(eligible)))

    slopes: List[float] = []
    for v in targets:
        counts = node_past_cone_counts(dag, v, r_max=r_max)
        xs, ys = [], []
        for r in r_values:
            Nr = counts[r]
            if r > 0 and Nr > 0:
                xs.append(math.log(r))
                ys.append(math.log(Nr))
        if len(xs) < 2:
            continue
        x_bar = sum(xs) / len(xs)
        y_bar = sum(ys) / len(ys)
        num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
        den = sum((x - x_bar) ** 2 for x in xs)
        if den > 0:
            slopes.append(num / den)

    if not slopes:
        return None, None, []
    mean = sum(slopes) / len(slopes)
    var = sum((s - mean) ** 2 for s in slopes) / max(1, len(slopes) - 1)
    return mean, math.sqrt(var), slopes


# ---------- Paper cone volume (Def. 10.3 path counts) ----------

def _log_int(x: int) -> float:
    """
    Robust log for huge integers: avoids float overflow.
    """
    if x <= 0:
        return float("-inf")
    try:
        return math.log(x)
    except OverflowError:
        # approximate via top bits
        b = x.bit_length()
        k = 53  # mantissa bits of double
        shift = max(0, b - k)
        mant = x >> shift
        return math.log(mant) + shift * math.log(2.0)


def path_cone_volumes_all(dag: DiGraph, r_max: int) -> List[Dict[Vertex, int]]:
    """
    Compute V_r(v) for all vertices v and all r <= r_max on a DAG, where per Def. 10.3:

        V_r(v) := #{ γ : γ is a directed path in G↓, |γ| <= r, target(γ) = v }.

    Includes the length-0 trivial path, so V_0(v)=1.

    DP:
      exact_0(v)=1
      exact_{ℓ}(v) = sum_{u in Pred(v)} exact_{ℓ-1}(u)
      V_r(v) = sum_{ℓ=0..r} exact_{ℓ}(v)
    """
    verts = list(dag.vertices())
    if r_max < 0:
        raise ValueError("r_max must be >= 0")

    # Ensure DAG (will throw if cyclic)
    dag.topological_order()

    exact_prev: Dict[Vertex, int] = {v: 1 for v in verts}  # length 0
    cumulative: Dict[Vertex, int] = {v: 1 for v in verts}  # V_0

    vols: List[Dict[Vertex, int]] = [dict(cumulative)]
    for _ell in range(1, r_max + 1):
        exact: Dict[Vertex, int] = {}
        for v in verts:
            s = 0
            for u in dag.predecessors(v):
                s += exact_prev.get(u, 0)
            exact[v] = s

        for v in verts:
            cumulative[v] = cumulative.get(v, 0) + exact[v]

        vols.append(dict(cumulative))
        exact_prev = exact

    return vols


def estimate_cone_dimension_paths(
    dag: DiGraph,
    r_values: List[int] = [1, 2, 3, 4, 5, 6],
    n_targets: int = 12,
    seed: int = 0,
) -> Tuple[Optional[float], Optional[float], List[float]]:
    """
    Paper-faithful cone dimension proxy using Def. 10.3 path-count cones:

        V_r(v) = # of directed paths γ with |γ| <= r and target(γ)=v.

    Fit log V_r(v) ~ D log r + const across r in r_values and average slopes over targets.
    """
    if not r_values:
        return None, None, []

    r_max = max(r_values)
    verts = list(dag.vertices())
    if not verts:
        return None, None, []

    # Precompute V_r(v) for all v once for the chosen r_max
    vols = path_cone_volumes_all(dag, r_max=r_max)

    depths = dag.depth_map()
    eligible = [v for v in verts if depths.get(v, 0) >= r_max] or verts

    rng = random.Random(seed)
    targets = rng.sample(eligible, k=min(n_targets, len(eligible)))

    slopes: List[float] = []
    for v in targets:
        xs, ys = [], []
        for r in r_values:
            Vr = vols[r].get(v, 0)
            if r > 0 and Vr > 0:
                xs.append(math.log(r))
                ys.append(_log_int(int(Vr)))
        if len(xs) < 2:
            continue

        x_bar = sum(xs) / len(xs)
        y_bar = sum(ys) / len(ys)
        num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
        den = sum((x - x_bar) ** 2 for x in xs)
        if den > 0:
            slopes.append(num / den)

    if not slopes:
        return None, None, []

    mean = sum(slopes) / len(slopes)
    var = sum((s - mean) ** 2 for s in slopes) / max(1, len(slopes) - 1)
    return mean, math.sqrt(var), slopes
