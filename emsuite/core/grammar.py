# emsuite/core/grammar.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, TypeAlias
import itertools
import random

from emsuite.core.graph import DiGraph, Vertex

Symbol: TypeAlias = int
InputTuple: TypeAlias = Tuple[Symbol, ...]
State: TypeAlias = Dict[Vertex, Symbol]


@dataclass
class LocalRule:
    preds: Tuple[Vertex, ...]
    table: Dict[InputTuple, Symbol]


@dataclass
class Grammar:
    graph: DiGraph
    rules: Dict[Vertex, LocalRule]
    alphabets: Dict[Vertex, Tuple[Symbol, ...]]

    @staticmethod
    def _full_domain(alphabets: List[Tuple[Symbol, ...]]) -> List[InputTuple]:
        if not alphabets:
            return [tuple()]
        return list(itertools.product(*alphabets))

    @classmethod
    def random_binary(
        cls,
        graph: DiGraph,
        seed: int = 0,
        *,
        enforce_surjective: bool = False,
        surjective_non_sources_only: bool = True,
    ) -> "Grammar":
        """
        Random binary grammar.

        If enforce_surjective=True, make each (eligible) local rule λ_v surjective
        onto {0,1} by construction (no rejection loops).

        Notes:
          - If Pred(v) is empty, |D_v|=1 so surjectivity to {0,1} is impossible.
            By default we do NOT enforce surjectivity on sources.
          - This is a model-class option, not tuning.
        """
        rng = random.Random(seed)

        alphabets: Dict[Vertex, Tuple[Symbol, ...]] = {v: (0, 1) for v in graph.vertices()}
        rules: Dict[Vertex, LocalRule] = {}

        def stable_vertex_key(x: Vertex) -> Tuple[str, str]:
            return (type(x).__name__, str(x))

        for v in graph.vertices():
            preds = tuple(sorted(graph.predecessors(v), key=stable_vertex_key))
            pred_alphs = [alphabets[p] for p in preds]
            domain = cls._full_domain(pred_alphs)

            want_surj = enforce_surjective
            if surjective_non_sources_only and len(preds) == 0:
                want_surj = False
            if want_surj and len(domain) < 2:
                want_surj = False

            table = {inp: rng.choice(alphabets[v]) for inp in domain}

            if want_surj:
                image = set(table.values())
                if len(image) < 2:
                    # flip one random entry to the missing symbol
                    present = next(iter(image))
                    missing = 1 if present == 0 else 0
                    inp_flip = rng.choice(domain)
                    table[inp_flip] = missing

            rules[v] = LocalRule(preds=preds, table=table)

        return cls(graph=graph, rules=rules, alphabets=alphabets)

    def random_state(self, seed: int = 0) -> State:
        rng = random.Random(seed)
        return {v: rng.choice(self.alphabets[v]) for v in self.graph.vertices()}

    def update(self, state: State) -> State:
        """
        Synchronous update: apply every λ_v using the previous state's values.
        """
        for v in self.graph.vertices():
            if v not in state:
                raise KeyError(f"State missing vertex {v!r}")

        new_state: State = {}
        for v, rule in self.rules.items():
            inp = tuple(state[p] for p in rule.preds)
            new_state[v] = table_lookup(rule.table, inp, v)
        return new_state

    def orbit(self, init: State, steps: int) -> Tuple[State, List[State]]:
        """
        Return (final_state, history) for a synchronous orbit.

        history has length steps+1 and includes the initial state at index 0.
        """
        if steps < 0:
            raise ValueError("steps must be >= 0")

        state: State = dict(init)
        history: List[State] = [dict(state)]

        for _ in range(steps):
            state = self.update(state)
            history.append(dict(state))

        return state, history

    def surjectivity_report(self) -> Dict[Vertex, bool]:
        out: Dict[Vertex, bool] = {}
        for v, rule in self.rules.items():
            A = set(self.alphabets[v])
            image = set(rule.table.values())
            out[v] = (image == A)
        return out

    def surjectivity_ratio(self) -> float:
        rep = self.surjectivity_report()
        if not rep:
            return 0.0
        return sum(1 for ok in rep.values() if ok) / float(len(rep))


def table_lookup(table: Dict[InputTuple, Symbol], inp: InputTuple, v: Vertex) -> Symbol:
    try:
        return table[inp]
    except KeyError as e:
        raise KeyError(f"Rule table missing input {inp} at vertex {v!r}") from e
