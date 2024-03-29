from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
from pyvrp import (
    CostEvaluator,
    GeneticAlgorithm,
    PenaltyManager,
    PenaltyParams,
    Population,
    PopulationParams,
    ProblemData,
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
    SwapRoutes,
    SwapStar,
    TwoOpt,
)
from pyvrp.stop import MaxRuntime

from ddwp.VrpInstance import VrpInstance

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
    pen_params = PenaltyParams(
        num_registrations_between_penalty_updates=10,
        penalty_increase=1.3,
        penalty_decrease=0.5,
    )
    pop_params = PopulationParams(
        min_pop_size=5, generation_size=3, nb_elite=2, nb_close=2
    )
    nb_params = NeighbourhoodParams()

    data = instance2data(instance)
    rng = RandomNumberGenerator(seed=seed)
    pen_manager = PenaltyManager(params=pen_params)
    pop = Population(bpd, params=pop_params)

    neighbours = compute_neighbours(data, nb_params)
    ls = LocalSearch(data, rng, neighbours)

    node_ops = [Exchange10, TwoOpt]
    for node_op in node_ops:
        ls.add_node_operator(node_op(data))

    route_ops = [SwapStar, SwapRoutes]
    for route_op in route_ops:
        ls.add_route_operator(route_op(data))

    if len(instance.vehicle_types) == 1:
        init = [
            Solution.make_random(data, rng)
            for _ in range(pop_params.min_pop_size)
        ]
    else:
        # In case of multiple vehicle types, we can ensure that the initial
        # solutions are feasible by running a local search with only SwapRoute.
        ls_feas = LocalSearch(data, rng, neighbours)
        ls_feas.add_route_operator(SwapRoutes(data))
        init = [
            ls_feas(Solution.make_random(data, rng), CostEvaluator(100, 100))
            for _ in range(pop_params.min_pop_size)
        ]

    algo = GeneticAlgorithm(data, pen_manager, rng, pop, ls, srex, init)

    return algo.run(MaxRuntime(time_limit))


@dataclass
class NeighbourhoodParams:
    """
    Configuration for calculating a granular neighbourhood.

    Attributes
    ----------
    weight_wait_time
        Weight given to the minimum wait time aspect of the proximity
        calculation. A large wait time indicates the clients are far apart
        in duration/time.
    weight_time_warp
        Weight given to the minimum time warp aspect of the proximity
        calculation. A large time warp indicates the clients are far apart in
        duration/time.
    nb_granular
        Number of other clients that are in each client's granular
        neighbourhood. This parameter determines the size of the overall
        neighbourhood.
    symmetric_proximity
        Whether to calculate a symmetric proximity matrix. This ensures edge
        :math:`(i, j)` is given the same weight as :math:`(j, i)`.
    symmetric_neighbours
        Whether to symmetrise the neighbourhood structure. This ensures that
        when edge :math:`(i, j)` is in, then so is :math:`(j, i)`. Note that
        this is *not* the same as ``symmetric_proximity``.

    Raises
    ------
    ValueError
        When ``nb_granular`` is non-positive.
    """

    weight_wait_time: float = 0.2
    weight_time_warp: float = 1.0
    nb_granular: int = 40
    symmetric_proximity: bool = True
    symmetric_neighbours: bool = False

    def __post_init__(self):
        if self.nb_granular <= 0:
            raise ValueError("nb_granular <= 0 not understood.")


def compute_neighbours(
    data: ProblemData, params: NeighbourhoodParams = NeighbourhoodParams()
) -> List[List[int]]:
    """
    Computes neighbours defining the neighbourhood for a problem instance.

    Parameters
    ----------
    data
        ProblemData for which to compute the neighbourhood.
    params
        NeighbourhoodParams that define how the neighbourhood is computed.

    Returns
    -------
    list
        A list of list of integers representing the neighbours for each client.
        The first element represents the depot and is an empty list.
    """
    proximity = _compute_proximity(
        data,
        params.weight_wait_time,
        params.weight_time_warp,
    )

    if params.symmetric_proximity:
        proximity = np.minimum(proximity, proximity.T)

    # TODO generalise this when we have multiple depots
    n = len(proximity)
    k = min(params.nb_granular, n - 2)  # excl. depot and self

    np.fill_diagonal(proximity, np.inf)  # cannot be in own neighbourhood
    proximity[0, :] = np.inf  # depot has no neighbours
    proximity[:, 0] = np.inf  # clients do not neighbour depot

    top_k = np.argsort(proximity, axis=1, kind="stable")[1:, :k]  # excl. depot

    if not params.symmetric_neighbours:
        return [[], *top_k.tolist()]

    # Construct a symmetric adjacency matrix and return the adjacent clients
    # as the neighbourhood structure.
    adj = np.zeros_like(proximity, dtype=bool)
    rows = np.expand_dims(np.arange(1, n), axis=1)
    adj[rows, top_k] = True
    adj = adj | adj.transpose()

    return [np.flatnonzero(row).tolist() for row in adj]


def _compute_proximity(
    data: ProblemData, weight_wait_time: float, weight_time_warp: float
) -> np.ndarray:
    """
    Computes proximity for neighborhood. Proximity is based on [1]_, with
    modification for additional VRP variants.

    Parameters
    ----------
    data
        ProblemData for which to compute proximity.
    params
        NeighbourhoodParams that define how proximity is computed.

    Returns
    -------
    np.ndarray[float]
        A numpy array of size n x n where n = data.num_clients containing
        the proximities values between all clients (depot excluded).

    References
    ----------
    .. [1] Vidal, T., Crainic, T. G., Gendreau, M., and Prins, C. (2013). A
           hybrid genetic algorithm with adaptive diversity management for a
           large class of vehicle routing problems with time-windows.
           *Computers & Operations Research*, 40(1), 475 - 489.
    """
    clients = [data.client(idx) for idx in range(data.num_clients + 1)]

    early = np.asarray([client.tw_early for client in clients])
    late = np.asarray([client.tw_late for client in clients])
    service = np.asarray([client.service_duration for client in clients])
    prize = np.asarray([client.prize for client in clients])
    release = np.asarray([client.release_time for client in clients])
    dispatch = np.asarray([client.dispatch_time for client in clients])
    duration = np.asarray(data.duration_matrix(), dtype=float)

    min_wait_time = early[None:] - duration - service[:, None] - late[:, None]

    earliest_release = np.maximum.outer(release, release) + duration[0, :]
    earliest_arrival = np.maximum(earliest_release, early[None, :])

    min_time_warp = np.maximum(
        earliest_arrival + service[None, :] + duration - late[None, :], 0
    )

    # Additional time warp due to dispatch time.
    min_time_warp += np.maximum(np.subtract.outer(release, dispatch), 0)

    return (
        np.asarray(data.distance_matrix(), dtype=float)
        + weight_wait_time * np.maximum(min_wait_time, 0)
        + weight_time_warp * np.maximum(min_time_warp, 0)
        - prize[None, :]
    )
