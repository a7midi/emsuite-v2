# emsuite/analysis/micro_profile.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

PMF = Dict[int, float]


def normalize_pmf(pmf: Dict[int, float]) -> PMF:
    items = {int(k): float(v) for k, v in pmf.items() if float(v) > 0.0}
    total = sum(items.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in items.items()}


@dataclass(frozen=True)
class MicroProfileLite:
    """
    Minimal “micro -> macro” profile.

    Field names are chosen to be compatible with older code paths:
      - diamond_fraction is the canonical name (run_geometry_plots expects it)
      - diamond_density is provided as an alias for backwards compatibility
      - in_deg_pmf is provided as an alias for older ensemble code
    """
    avg_out_degree: float = 0.0
    avg_in_degree: float = 0.0
    diamond_fraction: float = 0.0
    cycle_fraction: float = 0.0
    n_scc: Optional[float] = None

    # Distributional bridges (Upgrade 5 style)
    in_degree_pmf: PMF = field(default_factory=dict)
    out_degree_pmf: PMF = field(default_factory=dict)
    depth_pmf: PMF = field(default_factory=dict)
    edge_span_pmf: PMF = field(default_factory=dict)

    @property
    def diamond_density(self) -> float:
        return float(self.diamond_fraction)

    @property
    def in_deg_pmf(self) -> PMF:
        # Older ensemble function expects this name
        return self.in_degree_pmf

    @property
    def out_deg_pmf(self) -> PMF:
        return self.out_degree_pmf

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_out_degree": self.avg_out_degree,
            "avg_in_degree": self.avg_in_degree,
            "diamond_fraction": self.diamond_fraction,
            "cycle_fraction": self.cycle_fraction,
            "n_scc": self.n_scc,
            "in_degree_pmf": dict(self.in_degree_pmf),
            "out_degree_pmf": dict(self.out_degree_pmf),
            "depth_pmf": dict(self.depth_pmf),
            "edge_span_pmf": dict(self.edge_span_pmf),
        }


@dataclass(frozen=True)
class MicroProfile(MicroProfileLite):
    """
    Extended profile for reviewer-grade reporting (Upgrade 7).
    Keep optional extra curves/metadata here so MicroProfileLite stays stable.
    """
    meta: Dict[str, Any] = field(default_factory=dict)
