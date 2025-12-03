# emsuite/analysis/microstats.py
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple
import random

from emsuite.core.graph import DiGraph
from emsuite.core.projection import find_diamonds
from emsuite.analysis.micro_profile import MicroProfileLite, normalize_pmf


def _pmf_from_counts(counts: Counter[int]) -> Dict[int, float]:
    return normalize_pmf({k: float(v) for k, v in counts.items()})


def _safe_condensation(graph: DiGraph) -> Tuple[DiGraph, Dict]:
    """
    Handles both styles:
      - condensation() -> (dag, comp_map)
      - condensation() -> dag
    """
    res = graph.condensation()
    if isinstance(res, tuple) and len(res) == 2:
        dag, comp = res
        return dag, comp
    return res, {}


def infer_micro_stats_from_graphs(
    graphs: Sequence[DiGraph],
    *,
    seed: int = 0,
    r_max: int = 6,  # accepted for API compatibility; not required for the PMFs
    n_targets_per_graph: int = 10,  # accepted for API compatibility
) -> MicroProfileLite:
    """
    Infer micro statistics used to parameterize macro ensembles.

    Notes:
      - Degree/depth/span/diamond stats are computed on the condensation DAG (G↓),
        since the macro generators are DAG-based.
      - cycle_fraction is computed on the *raw* graph as “fraction of vertices in SCCs of size>1”.
    """
    _ = r_max, n_targets_per_graph  # reserved for future curve-fitting extensions
    rng = random.Random(seed)

    in_deg = Counter()
    out_deg = Counter()
    depth_hist = Counter()
    span_hist = Counter()

    diamond_nodes_total = 0
    dag_nodes_total = 0

    raw_nodes_total = 0
    nodes_in_cycles_total = 0
    n_scc_total = 0
    n_graphs_used = 0

    for g in graphs:
        raw_verts = list(g.vertices())
        n_raw = len(raw_verts)
        if n_raw == 0:
            continue
        raw_nodes_total += n_raw

        dag, comp = _safe_condensation(g)
        dag_verts = list(dag.vertices())
        if not dag_verts:
            # All collapsed? Still count SCCs / cycles from the raw mapping.
            if comp:
                sizes = Counter(comp.values())
                n_scc_total += len(sizes)
                nodes_in_cycles_total += sum(sz for sz in sizes.values() if sz > 1)
                n_graphs_used += 1
            continue

        # DAG degree distributions
        for v in dag_verts:
            in_deg[int(dag.in_degree(v))] += 1
            out_deg[int(dag.out_degree(v))] += 1

        # Depth and edge-span distributions
        depths = dag.depth_map() if hasattr(dag, "depth_map") else {}
        if depths:
            for v in dag_verts:
                depth_hist[int(depths.get(v, 0))] += 1
            for u in dag_verts:
                du = int(depths.get(u, 0))
                for v in dag.successors(u):
                    dv = int(depths.get(v, 0))
                    span = max(1, dv - du)
                    span_hist[int(span)] += 1

        # Diamond participation on DAG
        diamonds = find_diamonds(dag)
        dn = set()
        for d in diamonds:
            dn.update([d.i, d.v, d.w, d.x])
        diamond_nodes_total += len(dn)
        dag_nodes_total += len(dag_verts)

        # Cycle fraction + SCC count on raw
        if comp:
            sizes = Counter(comp.values())
            n_scc_total += len(sizes)
            nodes_in_cycles_total += sum(sz for sz in sizes.values() if sz > 1)

        n_graphs_used += 1

    def _weighted_mean_from_counter(c: Counter[int]) -> float:
        total = sum(c.values())
        if total <= 0:
            return 0.0
        return float(sum(k * v for k, v in c.items()) / total)

    avg_in = _weighted_mean_from_counter(in_deg)
    avg_out = _weighted_mean_from_counter(out_deg)

    diamond_fraction = (diamond_nodes_total / dag_nodes_total) if dag_nodes_total > 0 else 0.0
    cycle_fraction = (nodes_in_cycles_total / raw_nodes_total) if raw_nodes_total > 0 else 0.0
    n_scc = (n_scc_total / n_graphs_used) if n_graphs_used > 0 else None

    return MicroProfileLite(
        avg_out_degree=float(avg_out),
        avg_in_degree=float(avg_in),
        diamond_fraction=float(diamond_fraction),
        cycle_fraction=float(cycle_fraction),
        n_scc=(None if n_scc is None else float(n_scc)),
        in_degree_pmf=_pmf_from_counts(in_deg),
        out_degree_pmf=_pmf_from_counts(out_deg),
        depth_pmf=_pmf_from_counts(depth_hist),
        edge_span_pmf=_pmf_from_counts(span_hist) or {1: 1.0},
    )
