import warnings

from ddwp.VrpInstance import VrpInstance
from pyvrp import (
    GeneticAlgorithm,
    PenaltyManager,
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
    LocalSearch,
    NeighbourhoodParams,
    SwapRoutes,
    SwapStar,
    TwoOpt,
    compute_neighbours,
)
from pyvrp.stop import MaxRuntime

from .instance2data import instance2data

warnings.filterwarnings("ignore", category=EmptySolutionWarning)


def scenario_solver(
    instance: VrpInstance, seed: int, time_limit: float
) -> Result:
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
        A `pyvrp.Result` instance.
    """
    pop_params = PopulationParams(
        min_pop_size=5, generation_size=3, nb_elite=2, nb_close=2
    )
    nb_params = NeighbourhoodParams(nb_granular=20)

    data = instance2data(instance)
    rng = RandomNumberGenerator(seed=seed)
    pen_manager = PenaltyManager()
    pop = Population(bpd, params=pop_params)

    neighbours = compute_neighbours(data, nb_params)
    ls = LocalSearch(data, rng, neighbours)

    node_ops = [Exchange10, TwoOpt]
    for node_op in node_ops:
        ls.add_node_operator(node_op(data))

    route_ops = [SwapStar, SwapRoutes]
    for route_op in route_ops:
        ls.add_route_operator(route_op(data))

    init = [
        Solution.make_random(data, rng) for _ in range(pop_params.min_pop_size)
    ]
    algo = GeneticAlgorithm(data, pen_manager, rng, pop, ls, srex, init)

    return algo.run(MaxRuntime(time_limit))
