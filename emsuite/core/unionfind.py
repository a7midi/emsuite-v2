# emsuite/core/unionfind.py
from __future__ import annotations

from typing import Dict, List


class UnionFindInt:
    """
    Union-Find for integer indices [0..n-1], deterministic.
    Tie-break: if ranks equal, the smaller root wins.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.merges = 0

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False

        # union by rank, deterministic tie-break by root index
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        elif self.rank[ra] == self.rank[rb] and rb < ra:
            ra, rb = rb, ra

        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.merges += 1
        return True

    def groups(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            out.setdefault(r, []).append(i)
        return out
