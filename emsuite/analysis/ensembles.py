# emsuite/analysis/ensembles.py
from __future__ import annotations

from typing import Any, Dict, List
import random
import math

from emsuite.core.graph import DiGraph
from emsuite.analysis.micro_profile import MicroProfileLite
from emsuite.analysis.reviewer_stats import sample_from_pmf


def _poisson_knuth(rng: random.Random, lam: float) -> int:
    if lam <= 0.0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def macro_dag_from_micro_stats(micro: Any, N: int, seed: int = 0) -> DiGraph:
    """
    Macro DAG generator informed by micro stats. Tries to match:
      - depth distribution (depth_pmf)
      - in-degree distribution (in_degree_pmf)
      - edge span distribution (edge_span_pmf)

    Falls back gracefully to means if PMFs absent.
    """
    rng = random.Random(seed)
    g = DiGraph()
    if N <= 0:
        return g

    for v in range(N):
        g.add_vertex(v)

    # Pull distributions with robust naming (MicroProfileLite provides aliases)
    depth_pmf: Dict[int, float] = getattr(micro, "depth_pmf", {}) or {0: 0.5, 1: 0.5}
    in_pmf: Dict[int, float] = (
        getattr(micro, "in_degree_pmf", None)
        or getattr(micro, "in_deg_pmf", None)
        or {}
    )
    span_pmf: Dict[int, float] = getattr(micro, "edge_span_pmf", {}) or {1: 1.0}

    lam_in = float(getattr(micro, "avg_in_degree", 1.0) or 1.0)

    # Sample depths per node
    depths: List[int] = [int(sample_from_pmf(rng, depth_pmf, default=0)) for _ in range(N)]
    if 0 not in depths:
        depths[rng.randrange(N)] = 0

    by_depth: Dict[int, List[int]] = {}
    for v, d in enumerate(depths):
        by_depth.setdefault(int(d), []).append(v)

    depth_levels = sorted(by_depth.keys())

    # Build prefix lists of earlier nodes
    earlier: Dict[int, List[int]] = {}
    running: List[int] = []
    for d in depth_levels:
        earlier[d] = list(running)
        running.extend(by_depth[d])

    for d in depth_levels:
        if d == 0:
            continue
        preds_all = earlier[d]
        if not preds_all:
            continue

        for v in by_depth[d]:
            if in_pmf:
                kin = int(sample_from_pmf(rng, in_pmf, default=int(round(lam_in))))
            else:
                kin = int(_poisson_knuth(rng, lam_in))
            kin = max(0, min(kin, len(preds_all)))
            if kin == 0:
                continue

            chosen = set()
            for _ in range(kin):
                span = int(sample_from_pmf(rng, span_pmf, default=1))
                span = max(1, span)
                src_depth = max(0, d - span)

                candidates = by_depth.get(src_depth, [])
                if not candidates:
                    candidates = preds_all

                for _try in range(50):
                    u = rng.choice(candidates)
                    if u != v and u not in chosen and depths[u] < depths[v]:
                        chosen.add(u)
                        g.add_edge(u, v)
                        break

    return g


def macro_dag_from_micro_profile(profile: MicroProfileLite, N: int, seed: int = 0) -> DiGraph:
    """Backward-compatible name: profile is just a richer micro stats object."""
    return macro_dag_from_micro_stats(profile, N=N, seed=seed)
