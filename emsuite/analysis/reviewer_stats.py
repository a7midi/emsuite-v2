# emsuite/analysis/reviewer_stats.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def stdev(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return 0.0 if xs else None
    m = mean(xs)
    assert m is not None
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def stderr(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    s = stdev(xs)
    assert s is not None
    return s / math.sqrt(len(xs))


def ci95(xs: List[float]) -> Optional[Tuple[float, float]]:
    if not xs:
        return None
    m = mean(xs)
    se = stderr(xs)
    assert m is not None and se is not None
    z = 1.96
    return (m - z * se, m + z * se)


def normalize_counts(counts: Dict[int, int]) -> Dict[int, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def sample_from_pmf(rng, pmf: Dict[int, float], default: int = 0) -> int:
    if not pmf:
        return default
    items = sorted(pmf.items(), key=lambda kv: kv[0])
    r = rng.random()
    c = 0.0
    last_k = items[-1][0]
    for k, p in items:
        c += float(p)
        if r <= c:
            return k
    return last_k
