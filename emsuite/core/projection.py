# emsuite/core/projection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Set, Optional
import itertools
import copy
import hashlib

from emsuite.core.graph import DiGraph, Vertex
from emsuite.core.grammar import Grammar, LocalRule, InputTuple, Symbol
from emsuite.core.unionfind import UnionFindInt


def _stable_key(x: object) -> Tuple[str, str]:
    return (type(x).__name__, str(x))


@dataclass(frozen=True)
class Diamond:
    i: Vertex
    v: Vertex
    w: Vertex
    x: Vertex


def find_diamonds(g: DiGraph) -> List[Diamond]:
    out: List[Diamond] = []
    for i in g.vertices():
        succ = sorted(list(g.successors(i)), key=_stable_key)
        if len(succ) < 2:
            continue
        succ_sets = {u: set(g.successors(u)) for u in succ}
        for a in range(len(succ)):
            v = succ[a]
            for b in range(a + 1, len(succ)):
                w = succ[b]
                common = succ_sets[v] & succ_sets[w]
                for x in sorted(common, key=_stable_key):
                    vv, ww = (v, w) if _stable_key(v) <= _stable_key(w) else (w, v)
                    out.append(Diamond(i=i, v=vv, w=ww, x=x))
    return out


def _full_domain(pred_alphs: List[Tuple[Symbol, ...]]) -> List[InputTuple]:
    if not pred_alphs:
        return [tuple()]
    return list(itertools.product(*pred_alphs))


def _num_classes(uf: UnionFindInt, n: int) -> int:
    return len({uf.find(i) for i in range(n)})


def _rep_index_by_root(uf: UnionFindInt, n: int) -> Dict[int, int]:
    reps: Dict[int, int] = {}
    for i in range(n):
        r = uf.find(i)
        if r not in reps or i < reps[r]:
            reps[r] = i
    return reps


def _symbol_rep_map(out_syms: List[Symbol], uf_out: UnionFindInt) -> Dict[Symbol, Symbol]:
    rep_idx = _rep_index_by_root(uf_out, len(out_syms))
    out: Dict[Symbol, Symbol] = {}
    for i, sym in enumerate(out_syms):
        r = uf_out.find(i)
        out[sym] = out_syms[rep_idx[r]]
    return out


def _kernel_seed_input_partition(
    dom: List[InputTuple],
    table: Dict[InputTuple, Symbol],
    out_index: Dict[Symbol, int],
) -> UnionFindInt:
    uf = UnionFindInt(len(dom))
    buckets: Dict[int, List[int]] = {}
    for idx, inp in enumerate(dom):
        j = out_index[table[inp]]
        buckets.setdefault(j, []).append(idx)
    for idxs in buckets.values():
        if len(idxs) >= 2:
            b = idxs[0]
            for k in idxs[1:]:
                uf.union(b, k)
    return uf


def _push_forward_outputs(
    dom: List[InputTuple],
    table: Dict[InputTuple, Symbol],
    out_index: Dict[Symbol, int],
    uf_in: UnionFindInt,
    uf_out: UnionFindInt,
) -> None:
    groups: Dict[int, List[int]] = {}
    for idx in range(len(dom)):
        groups.setdefault(uf_in.find(idx), []).append(idx)
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        outs = [out_index[table[dom[i]]] for i in idxs]
        base = outs[0]
        for j in outs[1:]:
            uf_out.union(base, j)


def _pull_back_inputs_from_outputs(
    dom: List[InputTuple],
    table: Dict[InputTuple, Symbol],
    out_index: Dict[Symbol, int],
    uf_in: UnionFindInt,
    uf_out: UnionFindInt,
) -> None:
    buckets: Dict[int, List[int]] = {}
    for idx, inp in enumerate(dom):
        oj = out_index[table[inp]]
        r = uf_out.find(oj)
        buckets.setdefault(r, []).append(idx)
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        b = idxs[0]
        for j in idxs[1:]:
            uf_in.union(b, j)


def _merge_unused_output_symbols(
    dom: List[InputTuple],
    table: Dict[InputTuple, Symbol],
    out_index: Dict[Symbol, int],
    uf_out: UnionFindInt,
    n_out: int,
) -> None:
    """
    Critical fix: if some output symbols are never produced by λ_v, they are
    observationally irrelevant and should be merged into the used class.

    This is monotone (only unions) and eliminates the “constant rule but 2-symbol alphabet”
    pathology that breaks star-type conjugacy tests.
    """
    used: Set[int] = set()
    for inp in dom:
        used.add(out_index[table[inp]])
    if not used:
        return
    rep = min(used)  # deterministic
    for j in range(n_out):
        if j not in used:
            uf_out.union(rep, j)


def _pred_pullback_coarsening(
    rules: Dict[Vertex, LocalRule],
    doms: Dict[Vertex, List[InputTuple]],
    dom_index: Dict[Vertex, Dict[InputTuple, int]],
    uf_in: Dict[Vertex, UnionFindInt],
    sym_rep: Dict[Vertex, Dict[Symbol, Symbol]],
) -> None:
    for v, rule in rules.items():
        if not rule.preds:
            continue
        dom = doms[v]
        idx_of = dom_index[v]
        for inp in dom:
            norm = list(inp)
            changed = False
            for j, p in enumerate(rule.preds):
                r = sym_rep[p].get(norm[j], norm[j])
                if r != norm[j]:
                    norm[j] = r
                    changed = True
            if changed:
                uf_in[v].union(idx_of[inp], idx_of[tuple(norm)])


def _diamond_exchange_coarsening(
    diamonds: List[Diamond],
    rules: Dict[Vertex, LocalRule],
    doms: Dict[Vertex, List[InputTuple]],
    dom_index: Dict[Vertex, Dict[InputTuple, int]],
    uf_in: Dict[Vertex, UnionFindInt],
) -> None:
    for d in diamonds:
        x = d.x
        if x not in rules:
            continue
        preds = rules[x].preds
        if d.v not in preds or d.w not in preds:
            continue
        iv = preds.index(d.v)
        iw = preds.index(d.w)

        dom = doms[x]
        idx_of = dom_index[x]
        for inp in dom:
            if inp[iv] == inp[iw]:
                continue
            sw = list(inp)
            sw[iv], sw[iw] = sw[iw], sw[iv]
            uf_in[x].union(idx_of[inp], idx_of[tuple(sw)])


def _star_type_key(g: DiGraph, rules: Dict[Vertex, LocalRule], v: Vertex) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    preds = rules[v].preds
    pred_sigs = sorted([(g.in_degree(p), g.out_degree(p)) for p in preds])
    return (len(preds), tuple(pred_sigs))


def _equalize_star_type_output_class_counts(
    g: DiGraph,
    rules: Dict[Vertex, LocalRule],
    uf_out: Dict[Vertex, UnionFindInt],
    out_syms: Dict[Vertex, List[Symbol]],
) -> None:
    groups: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], List[Vertex]] = {}
    for v in rules.keys():
        groups.setdefault(_star_type_key(g, rules, v), []).append(v)

    for _, vs in groups.items():
        if len(vs) < 2:
            continue
        mins = min(_num_classes(uf_out[v], len(out_syms[v])) for v in vs)
        for v in vs:
            n = len(out_syms[v])
            while _num_classes(uf_out[v], n) > mins:
                roots = sorted({uf_out[v].find(i) for i in range(n)})
                if len(roots) < 2:
                    break
                uf_out[v].union(roots[0], roots[1])


def project_to_consistent(
    grammar: Grammar,
    *,
    mode: str = "coarsen",
    max_iter: int = 50,
    star_equivariance: bool = True,
) -> Grammar:
    if mode not in {"canonical", "coarsen"}:
        raise ValueError("mode must be 'canonical' or 'coarsen'")

    if mode == "canonical":
        # debug-only deterministic attractor
        g = grammar.graph
        new_rules: Dict[Vertex, LocalRule] = {}
        for v in sorted(list(g.vertices()), key=_stable_key):
            preds = tuple(sorted(list(g.predecessors(v)), key=_stable_key))
            dom = _full_domain([grammar.alphabets[p] for p in preds])
            table: Dict[InputTuple, Symbol] = {}
            for inp in dom:
                h = int.from_bytes(hashlib.sha256(repr((v, preds, inp)).encode("utf-8")).digest(), "big")
                table[inp] = grammar.alphabets[v][h % len(grammar.alphabets[v])]
            new_rules[v] = LocalRule(preds=preds, table=table)
        return Grammar(graph=g, rules=new_rules, alphabets=dict(grammar.alphabets))

    # --- coarsen mode ---
    base = copy.deepcopy(grammar)
    g = base.graph
    rules = base.rules

    # stable predecessor ordering
    for v, r in list(rules.items()):
        rules[v] = LocalRule(preds=tuple(sorted(r.preds, key=_stable_key)), table=dict(r.table))

    doms: Dict[Vertex, List[InputTuple]] = {}
    dom_index: Dict[Vertex, Dict[InputTuple, int]] = {}
    out_syms: Dict[Vertex, List[Symbol]] = {}
    out_index: Dict[Vertex, Dict[Symbol, int]] = {}

    for v in rules.keys():
        preds = rules[v].preds
        dom = _full_domain([base.alphabets[p] for p in preds])
        doms[v] = dom
        dom_index[v] = {inp: i for i, inp in enumerate(dom)}

        os = list(base.alphabets[v])
        out_syms[v] = os
        out_index[v] = {s: i for i, s in enumerate(os)}

    uf_in: Dict[Vertex, UnionFindInt] = {}
    uf_out: Dict[Vertex, UnionFindInt] = {}

    for v in rules.keys():
        uf_in[v] = _kernel_seed_input_partition(doms[v], rules[v].table, out_index[v])
        uf_out[v] = UnionFindInt(len(out_syms[v]))

    diamonds = find_diamonds(g)

    for _ in range(max_iter):
        merges_before = sum(getattr(uf_in[v], "merges", 0) + getattr(uf_out[v], "merges", 0) for v in rules.keys())

        sym_rep: Dict[Vertex, Dict[Symbol, Symbol]] = {v: _symbol_rep_map(out_syms[v], uf_out[v]) for v in rules.keys()}

        _diamond_exchange_coarsening(diamonds, rules, doms, dom_index, uf_in)
        _pred_pullback_coarsening(rules, doms, dom_index, uf_in, sym_rep)

        for v in rules.keys():
            _push_forward_outputs(doms[v], rules[v].table, out_index[v], uf_in[v], uf_out[v])
        for v in rules.keys():
            _pull_back_inputs_from_outputs(doms[v], rules[v].table, out_index[v], uf_in[v], uf_out[v])

        # *** fix: collapse unused outputs so constant maps actually become 1-symbol quotients ***
        for v in rules.keys():
            _merge_unused_output_symbols(doms[v], rules[v].table, out_index[v], uf_out[v], len(out_syms[v]))

        if star_equivariance:
            _equalize_star_type_output_class_counts(g, rules, uf_out, out_syms)

        merges_after = sum(getattr(uf_in[v], "merges", 0) + getattr(uf_out[v], "merges", 0) for v in rules.keys())
        if merges_after == merges_before:
            break

    # build quotient alphabets + output representative map
    new_alph: Dict[Vertex, Tuple[Symbol, ...]] = {}
    out_rep_map: Dict[Vertex, Dict[Symbol, Symbol]] = {}
    for v in rules.keys():
        rep_map = _symbol_rep_map(out_syms[v], uf_out[v])
        out_rep_map[v] = rep_map
        rep_idx = _rep_index_by_root(uf_out[v], len(out_syms[v]))
        reps = sorted(set(rep_idx.values()))
        new_alph[v] = tuple(out_syms[v][i] for i in reps)

    # rebuild tables on NEW predecessor alphabets, and rewrite outputs to reps
    new_rules: Dict[Vertex, LocalRule] = {}
    for v, r in rules.items():
        preds = r.preds
        dom_new = _full_domain([new_alph[p] for p in preds])
        table_new: Dict[InputTuple, Symbol] = {}
        for inp in dom_new:
            base_out = base.rules[v].table[inp]
            table_new[inp] = out_rep_map[v][base_out]
        new_rules[v] = LocalRule(preds=preds, table=table_new)

    return Grammar(graph=g, rules=new_rules, alphabets=new_alph)
