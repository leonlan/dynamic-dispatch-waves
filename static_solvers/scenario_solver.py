import warnings

from pyvrp import (
    GeneticAlgorithm,
    GeneticAlgorithmParams,
    PenaltyManager,
    PenaltyParams,
    Population,
    PopulationParams,
    RandomNumberGenerator,
    Result,
    Solution,
)
from pyvrp.crossover import selective_route_exchange as srex
from pyvrp.diversity import broken_pairs_distance as bpd
from pyvrp.exceptions import EmptySolutionWarning
from pyvrp.search import (
    Exchange10,
    Exchange11,
    LocalSearch,
    NeighbourhoodParams,
    TwoOpt,
    compute_neighbours,
)
from pyvrp.stop import MaxRuntime

from .instance2data import instance2data

warnings.filterwarnings("ignore", category=EmptySolutionWarning)


def scenario_solver(instance: dict, seed: int, time_limit: float) -> Result:
    """
    Solves the given instance using a customised hybrid genetic search solver
    opimised for small time limits.

    Parameters
    ----------
    instance: dict
        The instance to solve.
    seed: int
        The seed to use for the random number generator.
    time_limit: float
        The time limit in seconds.

    Returns
    -------
    Result
        An `pyvrp.Result` instance.
    """
    gen_params = GeneticAlgorithmParams(repair_probability=0)
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
    rng = RandomNumberGenerator(seed=seed)
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
    return algo.run(MaxRuntime(time_limit))
