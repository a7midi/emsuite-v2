# emsuite/verification/test_projection_coarsen_properties.py
import copy

from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar
from emsuite.core.projection import project_to_consistent, find_diamonds


def _small_symmetry_graph():
    # 0 -> 1 and 0 -> 2 ; (star-isomorphic 1 and 2)
    # 1,2 -> 3 (diamond-like)
    g = DiGraph()
    for v in range(4):
        g.add_vertex(v)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    return g


def test_coarsen_idempotent():
    g = _small_symmetry_graph()
    base = Grammar.random_binary(g, seed=0)
    p1 = project_to_consistent(copy.deepcopy(base), mode="coarsen")
    p2 = project_to_consistent(copy.deepcopy(p1), mode="coarsen")
    assert p1.alphabets == p2.alphabets
    for v in p1.rules.keys():
        assert p1.rules[v].preds == p2.rules[v].preds
        assert p1.rules[v].table == p2.rules[v].table


def test_coarsen_never_increases_alphabet_sizes():
    g = _small_symmetry_graph()
    base = Grammar.random_binary(g, seed=1)
    proj = project_to_consistent(copy.deepcopy(base), mode="coarsen")
    for v in base.alphabets.keys():
        assert len(proj.alphabets[v]) <= len(base.alphabets[v])


def test_star_equivariance_conjugacy_binary_outputs():
    g = _small_symmetry_graph()
    base = Grammar.random_binary(g, seed=2)
    proj = project_to_consistent(copy.deepcopy(base), mode="coarsen")

    # vertices 1 and 2 have same in-star type in this graph
    r1 = proj.rules[1].table
    r2 = proj.rules[2].table

    # If alphabet collapsed, must match trivially
    A1 = proj.alphabets[1]
    A2 = proj.alphabets[2]
    assert len(A1) == len(A2)

    if len(A1) == 1:
        assert r1 == r2
        return

    # binary: allow identity or swap conjugacy
    def swap(sym):
        return 1 if sym == 0 else 0

    r2_swapped = {k: swap(v) for k, v in r2.items()}
    assert (r1 == r2) or (r1 == r2_swapped)
