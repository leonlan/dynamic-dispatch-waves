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

from .Agent import Agent
from .simulate_instance import simulate_instance


class IterativeConditionalDispatch(Agent):
    def __init__(
        self,
        seed: int,
        consensus,
        num_iterations: int,
        num_lookahead: int,
        num_scenarios: int,
        strategy_tlim_factor: float = 1,
    ):
        self.rng = np.random.default_rng(seed)
        self.consensus = consensus
        self.num_iterations = num_iterations
        self.num_lookahead = num_lookahead
        self.num_scenarios = num_scenarios
        self.strategy_tlim_factor = strategy_tlim_factor

    def act(self, observation, info) -> np.ndarray:
        ep_inst = observation["epoch_instance"]
        ep_size = ep_inst["is_depot"].size  # includes depot

        total_sim_tlim = self.strategy_tlim_factor * info["epoch_tlim"]
        single_sim_tlim = total_sim_tlim / (
            self.num_iterations * self.num_scenarios
        )

        to_dispatch = ep_inst["must_dispatch"].copy()
        to_postpone = np.zeros(ep_size, dtype=bool)

        # Dispatch everything in the last iteration
        if observation["current_epoch"] == info["end_epoch"]:
            return np.ones(ep_size, dtype=bool)

        for iter_idx in range(self.num_iterations):
            scenarios = []

            for _ in range(self.num_scenarios):
                sim_inst = simulate_instance(
                    info,
                    observation,
                    self.rng,
                    self.num_lookahead,
                    to_dispatch,
                    to_postpone,
                )

                stop = MaxRuntime(single_sim_tlim)
                res = _scenario_solver(sim_inst, stop, seed=42)
                sim_sol = [route.visits() for route in res.best.get_routes()]

                scenarios.append((sim_inst, sim_sol))

            to_dispatch, to_postpone = self.consensus(
                iter_idx, scenarios, to_dispatch, to_postpone
            )

            # Stop the run early when all requests have been marked
            if ep_size - 1 == to_dispatch.sum() + to_postpone.sum():
                break

        return to_dispatch | ep_inst["is_depot"]  # include depot


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
