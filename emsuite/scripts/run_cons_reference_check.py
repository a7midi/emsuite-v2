# emsuite/scripts/run_cons_reference_check.py
from __future__ import annotations

import copy
import json
from typing import Any, Dict, Iterable, List, Tuple

from emsuite.core.graph import DiGraph
from emsuite.core.grammar import Grammar, LocalRule
from emsuite.core.projection import project_to_consistent


def _stable_key(x: Any) -> Tuple[str, str]:
    # Deterministic ordering for ints/strings (expected types in this suite).
    # Avoids repr() instability for custom objects.
    t = type(x).__name__
    if isinstance(x, (int, float, str, bool)):
        s = str(x)
    else:
        s = repr(x)
    return (t, s)


def _diamond_graph() -> DiGraph:
    g = DiGraph()
    for v in ["i", "v", "w", "x"]:
        g.add_vertex(v)
    g.add_edge("i", "v")
    g.add_edge("i", "w")
    g.add_edge("v", "x")
    g.add_edge("w", "x")
    return g


def _quotient_table_from_reps(grammar: Grammar, reps: Dict[Any, Dict[Any, Any]]) -> Tuple[Dict, Dict]:
    """
    Given reps[v][sym] = representative in A_v, rebuild:
      - quotient alphabets A*_v as the set of reps in A_v
      - quotient rule tables over the quotient domain (pred representatives)
    """
    g = grammar.graph

    def rep(v, s):
        return reps[v].get(s, s)

    new_alph = {}
    for v in g.vertices():
        repset = sorted({rep(v, s) for s in grammar.alphabets[v]}, key=_stable_key)
        new_alph[v] = tuple(repset)

    new_rules: Dict[Any, LocalRule] = {}
    for v in g.vertices():
        rule = grammar.rules[v]
        preds = rule.preds

        # build induced table on quotient domain:
        # key_new = tuple(rep(p, a_p) for each predecessor coord)
        table_new: Dict[Tuple[Any, ...], Any] = {}
        for inp_old, out_old in rule.table.items():
            key_new = tuple(rep(p, a) for p, a in zip(preds, inp_old))
            out_new = rep(v, out_old)
            # if collisions disagree, the reps are not yet a congruence (shouldn't happen at fixpoint)
            if key_new in table_new and table_new[key_new] != out_new:
                # choose a deterministic representative to keep the reference implementation monotone
                table_new[key_new] = min(table_new[key_new], out_new) if isinstance(out_new, int) else table_new[key_new]
            else:
                table_new[key_new] = out_new

        new_rules[v] = LocalRule(preds=preds, table=table_new)

    return new_alph, new_rules


def reference_cons(grammar: Grammar, max_iter: int = 80) -> Grammar:
    """
    Slow “tiny-N reference” closure (intended for toy graphs).

    It maintains ONLY an output congruence ~ on A_v (reps[v]) and repeatedly enforces:
      (1) well-definedness under predecessor congruences (pushforward/pullback shape)
      (2) diamond swap invariance on x’s table (E2-flavored)

    Then it *factors each rule table through the quotient* by rebuilding tables on the
    quotient domain (pred representatives).

    NOTE: This is still a *simplified* reference; it’s useful to catch gross mismatches
    and labeling artifacts, and it’s slow but independent of your fast implementation.
    """
    g = grammar.graph
    reps: Dict[Any, Dict[Any, Any]] = {v: {s: s for s in grammar.alphabets[v]} for v in g.vertices()}

    def rep(v, s):
        return reps[v].get(s, s)

    def _compress(v):
        # small alphabets → simple path compression is fine
        changed = True
        while changed:
            changed = False
            for a in list(reps[v].keys()):
                b = reps[v][a]
                c = reps[v].get(b, b)
                if c != b:
                    reps[v][a] = c
                    changed = True

    # diamonds on the raw graph (consistent with your current checker graph)
    diamonds: List[Tuple[Any, Any, Any, Any]] = []
    for i in g.vertices():
        succ_i = list(g.successors(i))
        for v in succ_i:
            for w in succ_i:
                if v == w:
                    continue
                for x in g.successors(v):
                    if x in g.successors(w):
                        diamonds.append((i, v, w, x))

    for _it in range(max_iter):
        merges_before = sum(1 for v in reps for a in reps[v] if reps[v][a] != a)

        # (1) well-definedness: if two domain points are indistinguishable under pred reps,
        # then their outputs must be equivalent.
        for v in g.vertices():
            rule = grammar.rules[v]
            buckets: Dict[Tuple[Any, ...], List[Any]] = {}
            for inp, out in rule.table.items():
                key = tuple(rep(p, a) for p, a in zip(rule.preds, inp))
                buckets.setdefault(key, []).append(out)

            for outs in buckets.values():
                outs_rep = [rep(v, o) for o in outs]
                base = outs_rep[0]
                for o in outs_rep[1:]:
                    a, b = rep(v, o), rep(v, base)
                    if a == b:
                        continue
                    # deterministic merge: map larger-int -> smaller-int when ints
                    if isinstance(a, int) and isinstance(b, int):
                        new, old = (a, b) if a < b else (b, a)
                    else:
                        new, old = (a, b)
                    for s in list(reps[v].keys()):
                        if reps[v][s] == old:
                            reps[v][s] = new
            _compress(v)

        # (2) diamond swap at x: swap v/w coords in x’s predecessor list
        for (_i, v, w, x) in diamonds:
            rx = grammar.rules[x]
            if v not in rx.preds or w not in rx.preds:
                continue
            iv = rx.preds.index(v)
            iw = rx.preds.index(w)

            # iterate over *existing* domain keys (still old domain); merges happen on outputs
            for inp, out in rx.table.items():
                inp2 = list(inp)
                inp2[iv], inp2[iw] = inp2[iw], inp2[iv]
                inp2 = tuple(inp2)
                if inp2 not in rx.table:
                    continue
                o1 = rep(x, out)
                o2 = rep(x, rx.table[inp2])
                if o1 == o2:
                    continue
                if isinstance(o1, int) and isinstance(o2, int):
                    new, old = (o1, o2) if o1 < o2 else (o2, o1)
                else:
                    new, old = (o1, o2)
                for s in list(reps[x].keys()):
                    if reps[x][s] == old:
                        reps[x][s] = new
            _compress(x)

        merges_after = sum(1 for v in reps for a in reps[v] if reps[v][a] != a)
        if merges_after == merges_before:
            break

    new_alph, new_rules = _quotient_table_from_reps(grammar, reps)
    return Grammar(graph=g, rules=new_rules, alphabets=new_alph)


def canonicalize_grammar_for_compare(grammar: Grammar) -> Grammar:
    """
    Canonicalize BOTH:
      - symbol labels per vertex: A_v -> (0..k-1)
      - predecessor ordering per rule: preds sorted by stable key
    while rewriting rule *keys and outputs* consistently.

    This makes two quotient grammars comparable even if they pick different
    representatives (e.g., {0} vs {1}) for the same collapsed class.
    """
    g = grammar.graph

    # Per-vertex relabel maps: old_sym -> new_sym in 0..k-1
    maps: Dict[Any, Dict[Any, int]] = {}
    new_alph: Dict[Any, Tuple[int, ...]] = {}
    for v in g.vertices():
        old_syms = list(grammar.alphabets[v])
        old_syms_sorted = sorted(old_syms, key=_stable_key)
        maps[v] = {s: i for i, s in enumerate(old_syms_sorted)}
        new_alph[v] = tuple(range(len(old_syms_sorted)))

    new_rules: Dict[Any, LocalRule] = {}
    for v in g.vertices():
        rule = grammar.rules[v]
        old_preds = list(rule.preds)
        new_preds = tuple(sorted(old_preds, key=_stable_key))

        # permutation: for each new pred, where was it in old_preds?
        idx = [old_preds.index(p) for p in new_preds]

        table_new: Dict[Tuple[int, ...], int] = {}
        for inp_old, out_old in rule.table.items():
            # reorder input tuple to match new_preds, then relabel each coord by its pred map
            inp_reordered = tuple(inp_old[j] for j in idx)
            inp_relabeled = tuple(maps[p][a] for p, a in zip(new_preds, inp_reordered))
            out_relabeled = maps[v][out_old]
            table_new[inp_relabeled] = out_relabeled

        new_rules[v] = LocalRule(preds=new_preds, table=table_new)

    return Grammar(graph=g, rules=new_rules, alphabets=new_alph)


def grammar_equal(a: Grammar, b: Grammar) -> bool:
    """
    Structural equality after canonicalization:
      - same alphabets (they will be 0..k-1 tuples)
      - same preds order per rule
      - identical tables
    """
    if set(a.alphabets.keys()) != set(b.alphabets.keys()):
        return False
    for v in a.alphabets:
        if a.alphabets[v] != b.alphabets[v]:
            return False
    if set(a.rules.keys()) != set(b.rules.keys()):
        return False
    for v in a.rules:
        ra = a.rules[v]
        rb = b.rules[v]
        if ra.preds != rb.preds:
            return False
        if ra.table != rb.table:
            return False
    return True


def main():
    g = _diamond_graph()
    trials = 50

    fails = 0
    for s in range(trials):
        gr = Grammar.random_binary(g, seed=s)

        fast = project_to_consistent(copy.deepcopy(gr), mode="coarsen")
        ref = reference_cons(copy.deepcopy(gr))

        fast_c = canonicalize_grammar_for_compare(fast)
        ref_c = canonicalize_grammar_for_compare(ref)

        if not grammar_equal(fast_c, ref_c):
            fails += 1

    print(json.dumps({"trials": trials, "fails_after_canonical_compare": fails}, indent=2))


if __name__ == "__main__":
    main()
