import numpy as np
import pyvrp
import pyvrp.search
from pyvrp import (
    GeneticAlgorithm,
    GeneticAlgorithmParams,
    PenaltyManager,
    PenaltyParams,
    Population,
    PopulationParams,
    Solution,
    XorShift128,
)
from pyvrp.crossover import selective_route_exchange as srex
from pyvrp.diversity import broken_pairs_distance as bpd
from pyvrp.search import (
    Exchange10,
    Exchange11,
    LocalSearch,
    NeighbourhoodParams,
    TwoOpt,
    compute_neighbours,
)
from pyvrp.stop import MaxRuntime

from utils import instance2data
from utils.filter_instance import filter_instance

from .consensus import CONSENSUS
from .simulate_instance import simulate_instance


def simulate(
    info,
    obs,
    rng,
    strategy_tlim_factor: float,
    final_dispatch: str,
    n_cycles: int,
    n_simulations: int,
    n_lookahead: int,
    consensus: str,
    consensus_params: dict = {},  # noqa: B006
    **kwargs,
):
    """
    Determine the dispatch instance by simulating the next epochs and analyzing
    those simulations.
    """
    # Return the full epoch instance for the last epoch
    current_epoch = obs["current_epoch"]
    if current_epoch == info["end_epoch"]:
        return obs["epoch_instance"]

    # Parameters
    ep_inst = obs["epoch_instance"]
    ep_size = ep_inst["is_depot"].size  # includes depot
    n_cycles = n_cycles if n_cycles > 0 else ep_size  # for DSHH and BRH

    total_sim_tlim = strategy_tlim_factor * info["epoch_tlim"]
    single_sim_tlim = total_sim_tlim / (n_cycles * n_simulations)

    to_dispatch = ep_inst["must_dispatch"].copy()
    to_postpone = np.zeros(ep_size, dtype=bool)

    for cycle_idx in range(n_cycles):
        scenarios = []

        for _ in range(n_simulations):
            sim_inst = simulate_instance(
                info,
                obs,
                rng,
                n_lookahead,
                to_dispatch,
                to_postpone,
            )

            res = _scenario_solver(sim_inst, MaxRuntime(single_sim_tlim), 42)
            sim_sol = [route.visits() for route in res.best.get_routes()]

            scenarios.append((sim_inst, sim_sol))

        # Use the consensus function to mark requests dispatched or postponed
        consensus_func = CONSENSUS[consensus]
        to_dispatch, to_postpone = consensus_func(  # type: ignore
            cycle_idx,
            scenarios,
            to_dispatch,
            to_postpone,
            **consensus_params,
        )

        # Stop the simulation run early when all requests have been marked
        if ep_size - 1 == to_dispatch.sum() + to_postpone.sum():
            break

    # Select who to dispatch
    if final_dispatch == "to_dispatch":
        selected = to_dispatch
    elif final_dispatch == "not_to_postpone":
        selected = ~to_postpone
    else:
        raise ValueError("Final dispatch action unknown.")

    return filter_instance(ep_inst, ep_inst["is_depot"] | selected)


def _scenario_solver(
    instance: dict, stop: pyvrp.stop.StoppingCriterion, seed: int
):
    gen_params = GeneticAlgorithmParams(
        repair_probability=0, intensify_probability=0, intensify_on_best=False
    )
    pen_params = PenaltyParams(
        init_time_warp_penalty=14,
        repair_booster=12,
        num_registrations_between_penalty_updates=1,
        penalty_increase=2,
        penalty_decrease=0.34,
        target_feasible=0.19,
    )
    pop_params = PopulationParams(min_pop_size=3, generation_size=8)
    nb_params = NeighbourhoodParams(
        weight_wait_time=5, weight_time_warp=18, nb_granular=16
    )

    data = instance2data(instance)
    rng = XorShift128(seed=seed)
    pen_manager = PenaltyManager(pen_params)
    pop = Population(bpd, params=pop_params)

    neighbours = compute_neighbours(data, nb_params)
    ls = LocalSearch(data, rng, neighbours)

    node_ops = [Exchange10, Exchange11, TwoOpt]
    for node_op in node_ops:
        ls.add_node_operator(node_op(data))

    init = [
        Solution.make_random(data, rng) for _ in range(pop_params.min_pop_size)
    ]
    algo = GeneticAlgorithm(
        data, pen_manager, rng, pop, ls, srex, init, gen_params
    )
    return algo.run(stop)
