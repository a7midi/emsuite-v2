# emsuite/scripts/run_scaling_sweep.py
from __future__ import annotations

import json
import os
import time
import math
from dataclasses import asdict
from typing import Dict, Any, List
import random

from emsuite.evolution.selection import EvolutionConfig, evolve_population
from emsuite.analysis.microstats import infer_micro_profile_from_graphs
from emsuite.analysis.ensembles import macro_dag_from_micro_stats

from emsuite.physics.geometry import estimate_g_star, alpha_from_g_star

try:
    from emsuite.physics.cones import estimate_cone_dimension_paths
except Exception:
    estimate_cone_dimension_paths = None


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    out_dir = os.path.join("results", "scaling_sweep")
    _mkdir(out_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_jsonl = os.path.join(out_dir, f"sweep_{stamp}.jsonl")
    out_summary = os.path.join(out_dir, f"sweep_{stamp}.summary.json")

    # --- micro evolution (small N) ---
    evo_cfg = EvolutionConfig(
        n_nodes=8,
        population_size=64,
        generations=40,
        elite_fraction=0.25,
        edge_prob=0.25,
        mutations_per_child=2,
        projection_mode="coarsen",
        entropy_method="local",
        observers_per_eval=3,
        random_seed=0,
        use_condensation_for_fitness=True,
        scc_density_normalize=True,
    )
    pop, micro_topo = evolve_population(evo_cfg)
    graphs = [ind.graph for ind in pop]
    micro_profile = infer_micro_profile_from_graphs(graphs, seed=0)

    # --- macro sweep ---
    N_values = [100, 200, 400, 800]
    seeds_per_N = 20
    scales = [1, 2, 4, 8]

    rows: List[Dict[str, Any]] = []

    for N in N_values:
        for s in range(seeds_per_N):
            g = macro_dag_from_micro_stats(micro_profile, N=N, seed=s)
            dag, _ = g.condensation()

            g_res = estimate_g_star(dag, scales=scales)
            g_star = g_res.get("g_star", None)
            status = g_res.get("status", None)

            alpha = None
            if g_star is not None and g_star > 0:
                alpha = float(alpha_from_g_star(g_star))

            d_est = None
            slopes = None
            if estimate_cone_dimension_paths is not None:
                try:
                    d_est, slopes, _meta = estimate_cone_dimension_paths(dag, r_max=8, n_targets=16, seed=s)
                except TypeError:
                    d_est, slopes = estimate_cone_dimension_paths(dag, r_max=8, n_targets=16)

            row = {
                "N": N,
                "seed": s,
                "micro_topology": micro_topo,
                "g_star": g_star,
                "g_flow": g_res.get("flow"),
                "g_status": status,
                "alpha": alpha,
                "D_est": d_est,
                "slopes": slopes,
            }
            rows.append(row)
            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    # Summaries per N
    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs)

    def _stderr(xs: List[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = _mean(xs)
        v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return math.sqrt(v / len(xs))

    summary: Dict[str, Any] = {
        "timestamp": stamp,
        "evolution_config": asdict(evo_cfg),
        "micro_topology": micro_topo,
        "N_values": N_values,
        "seeds_per_N": seeds_per_N,
        "per_N": {},
    }

    for N in N_values:
        subset = [r for r in rows if r["N"] == N]
        g_ok = [r["g_star"] for r in subset if isinstance(r["g_star"], (int, float))]
        a_ok = [r["alpha"] for r in subset if isinstance(r["alpha"], (int, float))]
        d_ok = [r["D_est"] for r in subset if isinstance(r["D_est"], (int, float))]

        stable = sum(1 for r in subset if r.get("g_status") == "CONVERGED")
        summary["per_N"][str(N)] = {
            "n": len(subset),
            "stable_frac": stable / len(subset) if subset else 0.0,
            "g_star_mean": _mean(g_ok) if g_ok else None,
            "g_star_stderr": _stderr(g_ok) if g_ok else None,
            "alpha_mean": _mean(a_ok) if a_ok else None,
            "alpha_stderr": _stderr(a_ok) if a_ok else None,
            "D_mean": _mean(d_ok) if d_ok else None,
            "D_stderr": _stderr(d_ok) if d_ok else None,
        }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved JSONL: {out_jsonl}")
    print(f"Saved summary: {out_summary}")


if __name__ == "__main__":
    main()
