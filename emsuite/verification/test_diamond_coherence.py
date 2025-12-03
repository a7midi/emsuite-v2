import copy
import itertools

from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar
from emsuite.core.projection import find_diamonds, project_to_consistent


def diamond_with_context_graph() -> DiGraph:
    """
    Build a graph containing a diamond i→v, i→w, v→x, w→x,
    and also give x an extra predecessor c to play the role of 'context' in:

        λ_x(c, λ_v(a), λ_w(b)) = λ_x(c, λ_w(b), λ_v(a))

    Also give v and w extra predecessors a and b so their inputs can vary
    independently (a and b in the paper equation need not be identical).
    """
    g = DiGraph()
    for node in ["i", "a", "b", "c", "v", "w", "x"]:
        g.add_vertex(node)

    # Diamond skeleton
    g.add_edge("i", "v")
    g.add_edge("i", "w")
    g.add_edge("v", "x")
    g.add_edge("w", "x")

    # Extra context / degrees of freedom
    g.add_edge("a", "v")   # v depends on (a, i)
    g.add_edge("b", "w")   # w depends on (b, i)
    g.add_edge("c", "x")   # x depends on (c, v, w)

    return g


def test_projection_enforces_diamond_commutativity_equation():
    """
    After projection, for every diamond (i,v,w,x) and every admissible choice
    of inputs, the projected rule tables satisfy the commutativity equation:

        λ_x(c, λ_v(a), λ_w(b)) = λ_x(c, λ_w(b), λ_v(a))

    We implement this in table form by:
      - enumerating all input tuples to v and w,
      - pushing them through λ_v and λ_w to get outputs,
      - plugging those outputs into λ_x (along with all c values),
      - and asserting swapping the v/w slots leaves λ_x unchanged.
    """
    g = diamond_with_context_graph()
    base = Grammar.random_binary(graph=g, seed=0)
    proj = project_to_consistent(copy.deepcopy(base), mode="coarsen")

    diamonds = find_diamonds(g)
    assert any(d.i == "i" and d.x == "x" and {d.v, d.w} == {"v", "w"} for d in diamonds)

    rule_v = proj.rules["v"]
    rule_w = proj.rules["w"]
    rule_x = proj.rules["x"]

    # Indices of v,w,c in x's predecessor ordering (grammar decides the order)
    idx_v = rule_x.preds.index("v")
    idx_w = rule_x.preds.index("w")
    idx_c = rule_x.preds.index("c")

    # Default filler for other predecessor slots, if any
    default_by_pred = {p: proj.alphabets[p][0] for p in rule_x.preds}

    # Enumerate all possible inputs to v and w (table keys are the full domain)
    for inp_v, out_v in rule_v.table.items():
        for inp_w, out_w in rule_w.table.items():
            # c ranges over its alphabet
            for c_val in proj.alphabets["c"]:
                x_in = [default_by_pred[p] for p in rule_x.preds]
                x_in[idx_c] = c_val
                x_in[idx_v] = out_v
                x_in[idx_w] = out_w
                x_in = tuple(x_in)

                x_sw = list(x_in)
                x_sw[idx_v], x_sw[idx_w] = x_sw[idx_w], x_sw[idx_v]
                x_sw = tuple(x_sw)

                assert rule_x.table[x_in] == rule_x.table[x_sw]
