import copy
import random

from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar, LocalRule
from emsuite.core.projection import project_to_consistent


def relabel_graph(g: DiGraph, perm: dict) -> DiGraph:
    g2 = DiGraph()
    for v in g.vertices():
        g2.add_vertex(perm[v])
    for u in g.vertices():
        for v in g.successors(u):
            g2.add_edge(perm[u], perm[v])
    return g2


def relabel_grammar(gr: Grammar, perm: dict) -> Grammar:
    g2 = relabel_graph(gr.graph, perm)

    alph2 = {perm[v]: gr.alphabets[v] for v in gr.alphabets.keys()}
    rules2 = {}
    for v, rule in gr.rules.items():
        v2 = perm[v]
        preds2 = tuple(sorted((perm[p] for p in rule.preds), key=lambda x: (type(x).__name__, str(x))))
        table2 = dict(rule.table)  # inputs are symbols, unchanged
        rules2[v2] = LocalRule(preds=preds2, table=table2)

    return Grammar(graph=g2, rules=rules2, alphabets=alph2)


def canonical_dump(gr: Grammar) -> str:
    parts = []
    for v in sorted(gr.rules.keys()):
        A = gr.alphabets[v]
        rule = gr.rules[v]
        items = sorted(rule.table.items(), key=lambda kv: kv[0])
        parts.append(f"{v}|A={list(A)}|" + ",".join(f"{inp}->{out}" for inp, out in items))
    return "\n".join(parts)


def test_projection_coarsen_equivariant_under_vertex_relabel():
    # Small graph with a diamond and a context predecessor so E2 actually bites
    g = DiGraph()
    for node in [0, 1, 2, 3, 4, 5, 6]:
        g.add_vertex(node)
    # diamond i=0, v=3, w=4, x=5
    g.add_edge(0, 3)
    g.add_edge(0, 4)
    g.add_edge(3, 5)
    g.add_edge(4, 5)
    # extra preds to vary inputs
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    g.add_edge(6, 5)

    base = Grammar.random_binary(graph=g, seed=123)

    # random permutation of vertex labels
    rng = random.Random(0)
    verts = list(g.vertices())
    shuffled = verts[:]
    rng.shuffle(shuffled)
    perm = {v: shuffled[i] for i, v in enumerate(verts)}

    # Check Cons(perm·Λ) == perm·Cons(Λ)
    proj = project_to_consistent(copy.deepcopy(base), mode="coarsen")
    lhs = project_to_consistent(copy.deepcopy(relabel_grammar(base, perm)), mode="coarsen")
    rhs = relabel_grammar(proj, perm)

    assert canonical_dump(lhs) == canonical_dump(rhs)
