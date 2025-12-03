# emsuite/scripts/run_cons_reference_check.py
from __future__ import annotations

import copy
import json
import random

from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar
from emsuite.core.projection import project_to_consistent


def _diamond_graph():
    g = DiGraph()
    for v in ["i", "v", "w", "x"]:
        g.add_vertex(v)
    g.add_edge("i", "v")
    g.add_edge("i", "w")
    g.add_edge("v", "x")
    g.add_edge("w", "x")
    return g


def reference_cons(grammar: Grammar, max_iter: int = 50) -> Grammar:
    """
    Slow, independent closure that coarsens output symbols by enforcing:
      - well-definedness under predecessor output congruences
      - diamond swap invariance (E2-like)

    This is not optimized; intended only for tiny graphs.
    """
    g = grammar.graph
    # maintain output congruence per vertex as dict sym->rep (updated monotonically)
    reps = {v: {s: s for s in grammar.alphabets[v]} for v in g.vertices()}

    def rep(v, s):
        return reps[v][s]

    def canonize(v):
        # path compression (two-symbol case still fine)
        changed = True
        while changed:
            changed = False
            for a in list(reps[v].keys()):
                b = reps[v][a]
                c = reps[v].get(b, b)
                if c != b:
                    reps[v][a] = c
                    changed = True

    diamonds = []
    for i in g.vertices():
        for v in g.successors(i):
            for w in g.successors(i):
                if v == w:
                    continue
                for x in g.successors(v):
                    if x in g.successors(w):
                        diamonds.append((i, v, w, x))

    for _ in range(max_iter):
        any_change = False

        # (1) pullback/pushforward well-definedness:
        # if two inputs to v are equivalent under predecessor reps, their outputs must be equivalent
        for v in g.vertices():
            rule = grammar.rules[v]
            # bucket domain points by predecessor representative tuple
            buckets = {}
            for inp, out in rule.table.items():
                key = tuple(rep(p, a) for p, a in zip(rule.preds, inp))
                buckets.setdefault(key, []).append(out)

            for outs in buckets.values():
                outs = [rep(v, o) for o in outs]
                base = outs[0]
                for o in outs[1:]:
                    if rep(v, o) != rep(v, base):
                        # union by choosing smaller int if ints
                        a, b = rep(v, o), rep(v, base)
                        new = min(a, b) if isinstance(a, int) and isinstance(b, int) else a
                        old = max(a, b) if isinstance(a, int) and isinstance(b, int) else b
                        # merge old -> new
                        for s in reps[v]:
                            if reps[v][s] == old:
                                reps[v][s] = new
                        any_change = True
            canonize(v)

        # (2) diamond swap: enforce x output invariance under swapping v,w slots
        for (_i, v, w, x) in diamonds:
            rx = grammar.rules[x]
            if v not in rx.preds or w not in rx.preds:
                continue
            iv = rx.preds.index(v)
            iw = rx.preds.index(w)

            for inp, out in rx.table.items():
                inp2 = list(inp)
                inp2[iv], inp2[iw] = inp2[iw], inp2[iv]
                inp2 = tuple(inp2)
                if inp2 not in rx.table:
                    continue
                o1 = rep(x, out)
                o2 = rep(x, rx.table[inp2])
                if rep(x, o1) != rep(x, o2):
                    a, b = rep(x, o1), rep(x, o2)
                    new = min(a, b) if isinstance(a, int) and isinstance(b, int) else a
                    old = max(a, b) if isinstance(a, int) and isinstance(b, int) else b
                    for s in reps[x]:
                        if reps[x][s] == old:
                            reps[x][s] = new
                    any_change = True
            canonize(x)

        if not any_change:
            break

    # build quotient alphabets and rewrite tables
    new_alph = {}
    for v in g.vertices():
        repset = sorted(set(reps[v][s] for s in grammar.alphabets[v]), key=lambda z: (type(z).__name__, str(z)))
        new_alph[v] = tuple(repset)

    new_rules = {}
    for v in g.vertices():
        rule = grammar.rules[v]
        table = {}
        for inp in rule.table:
            out = reps[v][rule.table[inp]]
            # inp still uses old predecessor symbols; but reps on preds don't change input symbols here
            table[inp] = out
        new_rules[v] = type(rule)(preds=rule.preds, table=table)

    return Grammar(graph=g, rules=new_rules, alphabets=new_alph)


def main():
    g = _diamond_graph()
    fails = 0
    trials = 50
    for s in range(trials):
        gr = Grammar.random_binary(g, seed=s)
        fast = project_to_consistent(copy.deepcopy(gr), mode="coarsen")
        ref = reference_cons(copy.deepcopy(gr))
        if fast.alphabets != ref.alphabets:
            fails += 1
    print(json.dumps({"trials": trials, "fails_alphabet_mismatch": fails}, indent=2))


if __name__ == "__main__":
    main()
