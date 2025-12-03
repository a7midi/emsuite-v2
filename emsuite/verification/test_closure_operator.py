from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar
from emsuite.core.projection import project_to_consistent
from emsuite.physics.observers import ForcedClosureComputer, predecessor_closure


def small_graph():
    g = DiGraph()
    for v in range(5):
        g.add_vertex(v)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    return g


def test_forced_closure_is_pred_closed_idempotent_monotone():
    g = small_graph()
    base = Grammar.random_binary(g, seed=0)
    gs = project_to_consistent(base)

    cc = ForcedClosureComputer(gs)

    S = {2}
    T = {2, 3}

    ClS = cc.closure(S)
    ClT = cc.closure(T)

    # contains PredCl(S)
    assert predecessor_closure(gs.graph, S).issubset(ClS)

    # predecessor-closed
    for v in ClS:
        for u in gs.graph.predecessors(v):
            assert u in ClS

    # idempotent
    assert cc.closure(ClS) == ClS

    # monotone
    assert ClS.issubset(ClT)
