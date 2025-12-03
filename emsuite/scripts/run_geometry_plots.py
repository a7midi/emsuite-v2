# emsuite/scripts/run_geometry_plots.py
from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from emsuite.evolution.selection import EvolutionConfig, evolve_population
from emsuite.analysis.microstats import infer_micro_stats_from_graphs
from emsuite.analysis.ensembles import macro_dag_from_micro_stats
from emsuite.physics.geometry import estimate_g_star
from emsuite.analysis.cones import estimate_cone_dimension_paths


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _std(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)


def _stderr(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    s = _std(xs)
    assert s is not None
    return s / math.sqrt(len(xs))


def _call_with_supported_kwargs(fn, *args, **kwargs):
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(*args, **filtered)


def _safe_condensation(G) -> Tuple[Any, Any]:
    res = G.condensation()
    if isinstance(res, tuple) and len(res) == 2:
        return res
    return res, {}


def _make_evo_cfg(wanted: Dict[str, Any]) -> EvolutionConfig:
    """
    Build EvolutionConfig robustly across schema drift:
    - only pass what the dataclass accepts
    - apply common aliases, including seed -> random_seed
    """
    sig = inspect.signature(EvolutionConfig)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in wanted.items() if k in allowed}

    aliases = {
        # population aliases
        "population_size": wanted.get("pop_size") or wanted.get("population_size"),
        "pop_size": wanted.get("pop_size") or wanted.get("population_size"),
        # generations aliases
        "generations": wanted.get("generations") or wanted.get("n_generations"),
        "n_generations": wanted.get("generations") or wanted.get("n_generations"),
        # elite fraction aliases
        "elite_fraction": wanted.get("elite_fraction") or wanted.get("selection_fraction"),
        "selection_fraction": wanted.get("elite_fraction") or wanted.get("selection_fraction"),
        # seed aliases
        "random_seed": wanted.get("seed") if wanted.get("seed") is not None else wanted.get("random_seed"),
        "seed": wanted.get("seed"),
        # size aliases
        "n_nodes": wanted.get("n_nodes"),
    }

    for k, v in aliases.items():
        if k in allowed and k not in filtered and v is not None:
            filtered[k] = v

    return EvolutionConfig(**filtered)


def main() -> None:
    outdir = Path("results") / "geometry_plots"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    wanted_cfg = dict(
        n_nodes=10,
        pop_size=64,
        generations=25,
        elite_fraction=0.25,
        seed=0,
        # add these if your EvolutionConfig supports them (filtered otherwise):
        projection_mode="coarsen",
        observers_per_eval=3,
        observer_pocket_size=1,
        steps=3,
        entropy_method="local",
    )
    evo_cfg = _make_evo_cfg(wanted_cfg)

    population, avg_topology = evolve_population(evo_cfg)

    graphs = []
    for ind in population:
        if hasattr(ind, "graph"):
            graphs.append(ind.graph)
        elif hasattr(ind, "grammar"):
            graphs.append(ind.grammar.graph)
        else:
            raise TypeError("Population individual has neither .graph nor .grammar")

    micro = _call_with_supported_kwargs(
        infer_micro_stats_from_graphs,
        graphs,
        seed=0,
        r_max=6,
        n_targets_per_graph=10,
    )

    N_macro = 200
    n_samples = 30
    scales = [1, 2, 4, 8]

    gstars: List[float] = []
    dims: List[float] = []
    alphas: List[float] = []
    statuses: List[str] = []
    flows: List[Dict[int, float]] = []

    # optional helper
    try:
        from emsuite.physics.geometry import alpha_from_g_star  # type: ignore
    except Exception:
        alpha_from_g_star = None  # type: ignore

    for s in range(n_samples):
        G = macro_dag_from_micro_stats(micro, N=N_macro, seed=s)
        dag, _ = _safe_condensation(G)

        res = estimate_g_star(dag, scales=scales)
        statuses.append(res.get("status", "UNKNOWN"))

        flow_raw = res.get("flow", {}) or {}
        flow = {int(k): float(v) for k, v in flow_raw.items() if v is not None}
        flows.append(flow)

        g = res.get("g_star")
        if g is not None:
            gstars.append(float(g))
            if alpha_from_g_star is not None:
                a = alpha_from_g_star(float(g))
                if a is not None:
                    alphas.append(float(a))

        # call cone-dimension estimator defensively
        d_mean, slopes, meta = _call_with_supported_kwargs(
            estimate_cone_dimension_paths,
            dag,
            r_values=[1, 2, 3, 4, 5, 6],
            n_targets=10,
            seed=s,
        )
        if d_mean is not None:
            dims.append(float(d_mean))

    stable_frac = sum(1 for st in statuses if st == "CONVERGED") / len(statuses) if statuses else 0.0

    # include PMFs if present (reviewer-grade traceability)
    in_pmf = getattr(micro, "in_degree_pmf", None) or getattr(micro, "in_deg_pmf", None) or {}
    out_pmf = getattr(micro, "out_degree_pmf", None) or getattr(micro, "out_deg_pmf", None) or {}
    depth_pmf = getattr(micro, "depth_pmf", None) or {}
    span_pmf = getattr(micro, "edge_span_pmf", None) or {}

    out = {
        "timestamp": stamp,
        "evolution_config": wanted_cfg,
        "micro_stats": {
            "avg_out_degree": float(getattr(micro, "avg_out_degree", 0.0)),
            "avg_in_degree": float(getattr(micro, "avg_in_degree", 0.0)),
            "diamond_fraction": float(getattr(micro, "diamond_fraction", getattr(micro, "diamond_density", 0.0))),
            "cycle_fraction": float(getattr(micro, "cycle_fraction", 0.0)),
        },
        "micro_pmfs": {
            "in_degree_pmf": in_pmf,
            "out_degree_pmf": out_pmf,
            "depth_pmf": depth_pmf,
            "edge_span_pmf": span_pmf,
        },
        "N_macro": N_macro,
        "n_samples": n_samples,
        "scales": scales,
        "g_star_mean": _mean(gstars),
        "g_star_stderr": _stderr(gstars),
        "alpha_mean": _mean(alphas),
        "alpha_stderr": _stderr(alphas),
        "dim_mean": _mean(dims),
        "dim_stderr": _stderr(dims),
        "stable_frac": stable_frac,
        "statuses": statuses,
        "flows": flows,
    }

    json_path = outdir / f"run_{stamp}.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # plots
    if alphas:
        plt.figure()
        plt.hist(alphas, bins=12)
        plt.title("Distribution of α_est across macro ensemble")
        plt.xlabel("α_est")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / f"alpha_hist_{stamp}.png", dpi=160)

    if dims:
        plt.figure()
        plt.hist(dims, bins=12)
        plt.title("Path-cone dimension proxy (Def. 10.3) across macro ensemble")
        plt.xlabel("D_est")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / f"dimension_hist_{stamp}.png", dpi=160)

    # plot mean RG flow with stderr
    if flows:
        by_R: Dict[int, List[float]] = {R: [] for R in scales}
        for f in flows:
            for R in scales:
                if R in f:
                    by_R[R].append(float(f[R]))

        xs: List[int] = []
        ys: List[float] = []
        es: List[float] = []
        for R in scales:
            m = _mean(by_R[R])
            se = _stderr(by_R[R]) or 0.0
            if m is not None:
                xs.append(R)
                ys.append(float(m))
                es.append(float(se))

        if xs:
            plt.figure()
            plt.errorbar(xs, ys, yerr=es, fmt="o-")
            plt.xscale("log", base=2)
            plt.title("Ensemble RG flow: mean g_R(R) ± stderr")
            plt.xlabel("Block scale R")
            plt.ylabel("g_R")
            plt.tight_layout()
            plt.savefig(outdir / f"rg_flow_mean_{stamp}.png", dpi=160)

    print(json.dumps(out, indent=2))
    print("Saved:", str(json_path))


if __name__ == "__main__":
    main()
