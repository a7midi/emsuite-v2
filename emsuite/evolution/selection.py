# emsuite/evolution/selection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random

from emsuite.core.graph import DiGraph, Vertex
from emsuite.core.grammar import Grammar
from emsuite.core.projection import find_diamonds


def measure_topology(graph: DiGraph) -> Dict[str, float]:
    """
    Topology stats on the RAW graph (cycles live here).
    Diamonds are computed on the condensation DAG by default (more paper-aligned
    and avoids SCC-induced diamond explosions).

    Returns:
      - avg_out_degree (raw)
      - avg_in_degree (raw)
      - diamond_fraction (condensation DAG)
      - cycle_fraction (raw SCC > 1)
      - n_scc (raw -> condensation node count)
    """
    verts = list(graph.vertices())
    n = len(verts)
    if n == 0:
        return {
            "avg_out_degree": 0.0,
            "avg_in_degree": 0.0,
            "diamond_fraction": 0.0,
            "cycle_fraction": 0.0,
            "n_scc": 0.0,
        }

    total_out = sum(graph.out_degree(v) for v in verts)
    total_in = sum(graph.in_degree(v) for v in verts)
    avg_out = total_out / n
    avg_in = total_in / n

    # SCC sizes
    dag, comp = graph.condensation()
    comp_sizes: Dict[int, int] = {}
    for v, c in comp.items():
        comp_sizes[c] = comp_sizes.get(c, 0) + 1

    nodes_in_cycles = sum(sz for sz in comp_sizes.values() if sz > 1)
    cycle_fraction = nodes_in_cycles / n
    n_scc = len({c for c in comp.values()})

    # Diamonds on condensation DAG (paper-causal object)
    diamonds = find_diamonds(dag)
    if diamonds:
        diamond_nodes = set()
        for d in diamonds:
            diamond_nodes.add(d.i)
            diamond_nodes.add(d.v)
            diamond_nodes.add(d.w)
            diamond_nodes.add(d.x)
        diamond_fraction = len(diamond_nodes) / max(1, len(list(dag.vertices())))
    else:
        diamond_fraction = 0.0

    return {
        "avg_out_degree": float(avg_out),
        "avg_in_degree": float(avg_in),
        "diamond_fraction": float(diamond_fraction),
        "cycle_fraction": float(cycle_fraction),
        "n_scc": float(n_scc),
    }


@dataclass
class EvolutionConfig:
    """
    Upgrade 4: evolution scoring is driven by entropy on the *condensation DAG*,
    not by an explicit cycle penalty.

    Compatibility:
      - accepts pop_size as alias for population_size
      - accepts seed as alias for random_seed
    """
    # graph size
    n_nodes: int = 8

    # evolution schedule
    population_size: int = 32
    pop_size: Optional[int] = None  # alias
    generations: int = 30
    elite_fraction: float = 0.25

    # initial graph distribution + mutation intensity
    edge_prob: float = 0.25
    mutations_per_child: int = 2

    # evaluation
    projection_mode: str = "coarsen"          # "coarsen" recommended for fidelity
    enforce_surjective_rules: bool = False   # paper-faithful ensemble toggle, NOT forced
    observer_pocket_size: int = 1
    observers_per_eval: int = 3
    steps: int = 3
    entropy_method: str = "local"            # "local" or "full" (full is expensive)
    use_condensation_for_fitness: bool = True

    # SCC suppression WITHOUT ad-hoc penalties:
    # normalize entropy by SCC density (n_scc / n_nodes), so large SCC collapses reduce fitness.
    scc_density_normalize: bool = True

    # reproducibility
    random_seed: int = 42
    seed: Optional[int] = None  # alias

    def __post_init__(self) -> None:
        if self.pop_size is not None:
            self.population_size = int(self.pop_size)
        if self.seed is not None:
            self.random_seed = int(self.seed)
        if not (0.0 < self.elite_fraction <= 1.0):
            raise ValueError("elite_fraction must be in (0,1].")
        if self.observers_per_eval <= 0:
            raise ValueError("observers_per_eval must be >= 1.")
        if self.mutations_per_child < 0:
            raise ValueError("mutations_per_child must be >= 0.")


@dataclass
class Individual:
    graph: DiGraph
    grammar_seed: int  # deterministic grammar sampling per individual


def _random_graph(n_nodes: int, edge_prob: float, rng: random.Random) -> DiGraph:
    g = DiGraph()
    for v in range(n_nodes):
        g.add_vertex(v)
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u == v:
                continue
            if rng.random() < edge_prob:
                g.add_edge(u, v)
    return g


def _mutate_graph(g_old: DiGraph, rng: random.Random, n_flips: int) -> DiGraph:
    g = DiGraph()
    for v in g_old.vertices():
        g.add_vertex(v)
    for u in g_old.vertices():
        for v in g_old.successors(u):
            g.add_edge(u, v)

    verts = list(g.vertices())
    if len(verts) < 2 or n_flips <= 0:
        return g

    for _ in range(n_flips):
        if rng.random() < 0.5:
            # remove a random existing edge
            edges: List[Tuple[Vertex, Vertex]] = []
            for u in verts:
                for v in g.successors(u):
                    edges.append((u, v))
            if edges:
                u, v = rng.choice(edges)
                try:
                    g.remove_edge(u, v)
                except AttributeError:
                    # if remove_edge isn't implemented, fallback: rebuild without that edge
                    gg = DiGraph()
                    for x in verts:
                        gg.add_vertex(x)
                    for a in verts:
                        for b in g.successors(a):
                            if not (a == u and b == v):
                                gg.add_edge(a, b)
                    g = gg
        else:
            # add a random missing edge
            u = rng.choice(verts)
            v = rng.choice(verts)
            if u != v and v not in g.successors(u):
                g.add_edge(u, v)

    return g


def initialise_population(cfg: EvolutionConfig) -> List[Individual]:
    rng = random.Random(cfg.random_seed)
    pop: List[Individual] = []
    for _ in range(cfg.population_size):
        g = _random_graph(cfg.n_nodes, cfg.edge_prob, rng)
        pop.append(Individual(graph=g, grammar_seed=rng.randint(0, 2**31 - 1)))
    return pop


def _evaluate_entropy_on_graph(
    graph_for_eval: DiGraph,
    *,
    grammar_seed: int,
    cfg: EvolutionConfig,
    eval_seed: int,
) -> float:
    """
    Deterministic evaluation of entropy growth on a given evaluation graph (usually condensation DAG).
    """
    from emsuite.core.projection import project_to_consistent
    from emsuite.physics.observers import sample_persistent_observer
    from emsuite.physics.entropy import entropy_growth_rate

    # deterministic grammar sampling
    grammar = Grammar.random_binary(
        graph_for_eval,
        seed=grammar_seed,
        enforce_surjective=cfg.enforce_surjective_rules,
    )

    # project ONCE (no double projection)
    grammar_star = project_to_consistent(grammar, mode=cfg.projection_mode)

    rng = random.Random(eval_seed)

    vals: List[float] = []
    for _ in range(cfg.observers_per_eval):
        obs = sample_persistent_observer(
            grammar_star,
            rng=rng,
            pocket_size=cfg.observer_pocket_size,
            grammar_is_projected=True,
        )
        visible = list(obs.visible)
        if not visible:
            continue
        v = float(
            entropy_growth_rate(
                grammar_star,
                visible,
                steps=cfg.steps,
                seed=rng.randint(0, 2**31 - 1),
                method=cfg.entropy_method,
            )
        )
        vals.append(max(0.0, v))

    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _fitness(ind: Individual, cfg: EvolutionConfig, generation: int) -> float:
    """
    Upgrade 4 fitness:
      - build condensation DAG if enabled
      - score entropy growth on that DAG (paper causal object)
      - normalize by SCC density to automatically disfavour collapsed SCCs
    """
    # deterministic per-individual-per-generation evaluation seed
    eval_seed = (hash((cfg.random_seed, generation, ind.grammar_seed)) & 0xFFFFFFFF)

    if cfg.use_condensation_for_fitness:
        dag, comp = ind.graph.condensation()
        n_scc = len(set(comp.values())) if comp else 0
        score = _evaluate_entropy_on_graph(dag, grammar_seed=ind.grammar_seed, cfg=cfg, eval_seed=eval_seed)
        if cfg.scc_density_normalize and cfg.n_nodes > 0:
            score *= float(n_scc) / float(cfg.n_nodes)
        return score

    # fallback: score on raw graph (not recommended for SCC arguments)
    return _evaluate_entropy_on_graph(ind.graph, grammar_seed=ind.grammar_seed, cfg=cfg, eval_seed=eval_seed)


def evolve_population(cfg: EvolutionConfig) -> Tuple[List[Individual], Dict[str, float]]:
    """
    Upgrade 4 evolution loop:
      - selection pressure is purely entropy-based (on condensation), no explicit cycle penalties
      - SCCs are disfavoured only via SCC-density normalization (a definitional density correction)
    """
    rng = random.Random(cfg.random_seed)
    pop = initialise_population(cfg)

    for gen in range(cfg.generations):
        scored: List[Tuple[float, Individual]] = []
        for ind in pop:
            scored.append((_fitness(ind, cfg, gen), ind))
        scored.sort(key=lambda x: x[0], reverse=True)

        elite_n = max(1, int(cfg.population_size * cfg.elite_fraction))
        elites = [ind for _, ind in scored[:elite_n]]

        new_pop: List[Individual] = []
        new_pop.extend(elites)

        while len(new_pop) < cfg.population_size:
            parent = rng.choice(elites)
            child_graph = _mutate_graph(parent.graph, rng, cfg.mutations_per_child)
            new_pop.append(
                Individual(
                    graph=child_graph,
                    grammar_seed=rng.randint(0, 2**31 - 1),
                )
            )
        pop = new_pop

    # report micro stats on final population (raw topology)
    agg = {
        "avg_out_degree": 0.0,
        "avg_in_degree": 0.0,
        "diamond_fraction": 0.0,
        "cycle_fraction": 0.0,
        "n_scc": 0.0,
    }
    for ind in pop:
        topo = measure_topology(ind.graph)
        for k in agg:
            agg[k] += topo[k]
    for k in agg:
        agg[k] /= float(len(pop))

    return pop, agg
