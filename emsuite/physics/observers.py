from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Set
from collections import deque
import random

from emsuite.core.graph import DiGraph, Vertex
from emsuite.core.grammar import Grammar, LocalRule
from emsuite.core.projection import project_to_consistent


@dataclass(frozen=True)
class Observer:
    """
    Paper-style observer:
      - pocket P (Def. 4.14)
      - visible set O_P = Cl_{Λ*}(P)
      - hidden set H_P = V \\ O_P
    """
    pocket: FrozenSet[Vertex]
    visible: FrozenSet[Vertex]
    hidden: FrozenSet[Vertex]


def undirected_neighbors(graph: DiGraph, v: Vertex) -> Set[Vertex]:
    return set(graph.successors(v)) | set(graph.predecessors(v))


def is_connected_undirected(graph: DiGraph, nodes: Set[Vertex]) -> bool:
    if not nodes:
        return False
    start = next(iter(nodes))
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for w in undirected_neighbors(graph, u):
            if w in nodes and w not in seen:
                seen.add(w)
                q.append(w)
    return seen == nodes


def predecessor_closure(graph: DiGraph, seeds: Set[Vertex]) -> Set[Vertex]:
    """
    PredCl(S): smallest set containing S that is closed under predecessors in G.
    """
    cl = set(seeds)
    q = deque(seeds)
    while q:
        v = q.popleft()
        for u in graph.predecessors(v):
            if u not in cl:
                cl.add(u)
                q.append(u)
    return cl


def _essential_indices(rule: LocalRule) -> Set[int]:
    """
    Indices j such that output depends on the j-th input coordinate:
    there exist two inputs differing only at j that map to different outputs.
    """
    preds = rule.preds
    if not preds:
        return set()

    essential: Set[int] = set()
    table = rule.table

    for j in range(len(preds)):
        # group by all coordinates except j; check whether output varies across values at j
        groups: Dict[tuple, Dict[object, object]] = {}
        for inp, out in table.items():
            key = inp[:j] + inp[j + 1 :]
            groups.setdefault(key, {})
            groups[key][inp[j]] = out

        for mapping in groups.values():
            if len(set(mapping.values())) > 1:
                essential.add(j)
                break

    return essential


def essential_predecessor_sets(grammar: Grammar) -> Dict[Vertex, FrozenSet[Vertex]]:
    """
    Ess(v) as a *set of predecessor vertices* (not indices), derived from λ_v table.
    """
    out: Dict[Vertex, FrozenSet[Vertex]] = {}
    for v, rule in grammar.rules.items():
        idxs = _essential_indices(rule)
        ess = {rule.preds[j] for j in idxs}
        out[v] = frozenset(ess)
    return out


class ForcedClosureComputer:
    """
    Paper-faithful closure via PredCl + forcing fixed point:

      X0 = PredCl(S)
      X_{k+1} = PredCl( X_k ∪ { v : Ess(v) ⊆ X_k } )

    Iterate to the least fixed point. This is monotone, stabilizes, and is minimal.
    """
    def __init__(self, grammar_star: Grammar):
        self.G = grammar_star.graph
        self.ess = essential_predecessor_sets(grammar_star)
        self.vertices = list(self.G.vertices())

    def closure(self, S: Iterable[Vertex], max_iter: int = 10_000) -> Set[Vertex]:
        X = predecessor_closure(self.G, set(S))

        # Fixed-point iteration on a finite set; at worst we add vertices until saturation.
        for _ in range(max_iter):
            before = set(X)

            # Force step: add any v whose essential preds already live in X
            for v in self.vertices:
                if v in X:
                    continue
                ess_v = self.ess.get(v, frozenset())
                if ess_v.issubset(X):
                    X.add(v)

            # PredCl step
            X = predecessor_closure(self.G, X)

            if X == before:
                return X

        raise RuntimeError("ForcedClosureComputer.closure did not stabilize (unexpected).")


def minimal_depth_pocket(grammar_star: Grammar) -> Set[Vertex]:
    """
    Reasonable fallback pocket: choose a vertex from a minimal-depth SCC
    in the condensation DAG (Prop. 4.15 spirit).
    """
    G = grammar_star.graph
    dag, comp = G.condensation()
    depths = dag.depth_map()
    verts = list(G.vertices())
    if not verts:
        return set()
    if not depths:
        return {verts[0]}

    min_depth = min(depths.values())
    source_comps = [c for c, d in depths.items() if d == min_depth]
    source_comp = sorted(source_comps, key=lambda x: repr(x))[0]

    for v, c in comp.items():
        if c == source_comp:
            return {v}
    return {verts[0]}


def random_connected_pocket(graph: DiGraph, rng: random.Random, size: int) -> Set[Vertex]:
    """
    Sample an undirected-connected pocket by random walk expansion.
    """
    verts = list(graph.vertices())
    if not verts:
        return set()
    size = max(1, size)
    start = rng.choice(verts)
    pocket = {start}
    frontier = [start]

    while len(pocket) < size and frontier:
        u = rng.choice(frontier)
        nbrs = list(undirected_neighbors(graph, u) - pocket)
        if not nbrs:
            frontier.remove(u)
            continue
        w = rng.choice(nbrs)
        pocket.add(w)
        frontier.append(w)

    return pocket


def make_observer(grammar: Grammar, pocket: Set[Vertex], *, grammar_is_projected: bool = False) -> Observer:
    """
    Build Observer(pocket, visible=Cl_{Λ*}(pocket), hidden=complement).
    """
    grammar_star = grammar if grammar_is_projected else project_to_consistent(grammar, mode="coarsen")
    P = set(pocket) if pocket else minimal_depth_pocket(grammar_star)

    cc = ForcedClosureComputer(grammar_star)
    O = cc.closure(P)

    V = set(grammar_star.graph.vertices())
    H = V - O
    return Observer(pocket=frozenset(P), visible=frozenset(O), hidden=frozenset(H))


def sample_persistent_observer(
    grammar: Grammar,
    rng: random.Random,
    pocket_size: int = 1,
    max_tries: int = 200,
    *,
    grammar_is_projected: bool = False,
) -> Observer:
    """
    Sample a connected pocket P and accept if its closure O_P is connected (Def. 4.14).
    Falls back to the minimal-depth pocket.
    """
    grammar_star = grammar if grammar_is_projected else project_to_consistent(grammar, mode="coarsen")
    G = grammar_star.graph
    cc = ForcedClosureComputer(grammar_star)

    for _ in range(max_tries):
        P = random_connected_pocket(G, rng, size=pocket_size)
        O = cc.closure(P)
        if O and is_connected_undirected(G, O):
            V = set(G.vertices())
            return Observer(pocket=frozenset(P), visible=frozenset(O), hidden=frozenset(V - O))

    # fallback
    P0 = minimal_depth_pocket(grammar_star)
    O0 = cc.closure(P0)
    V0 = set(G.vertices())
    return Observer(pocket=frozenset(P0), visible=frozenset(O0), hidden=frozenset(V0 - O0))
