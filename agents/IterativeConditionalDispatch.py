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

from sampling import sample_epoch_requests
from utils import instance2data


class IterativeConditionalDispatch:
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
        # Parameters
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
                sim_inst = _simulate_instance(
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
) -> pyvrp.Result:
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


def _simulate_instance(
    info,
    obs,
    rng,
    n_lookahead: int,
    to_dispatch: np.ndarray,
    to_postpone: np.ndarray,
):
    """
    Simulates a VRPTW scenario instance with ``n_lookahead`` epochs. The
    scenario instance is created by appending the sampled requests to the
    current epoch instance.

    Parameters
    ----------
    to_dispatch
        A boolean array where True means that the corresponding request must be
        dispatched.
    to_postpone
        A boolean array where True mean that the corresponding request must be
        postponed.
    """
    current_epoch = obs["current_epoch"]
    next_epoch = current_epoch + 1
    epochs_left = info["end_epoch"] - current_epoch
    max_lookahead = min(n_lookahead, epochs_left)

    # Parameters
    static_inst = info["dynamic_context"]
    epoch_duration = info["epoch_duration"]
    ep_inst = obs["epoch_instance"]
    dispatch_time = obs["dispatch_time"]
    dist = static_inst["duration_matrix"]
    max_requests_per_epoch = info["max_requests_per_epoch"]

    # Simulation instance
    req_customer_idx = ep_inst["customer_idx"]
    req_idx = ep_inst["request_idx"]
    req_demand = ep_inst["demands"]
    req_service = ep_inst["service_times"]
    req_tw = ep_inst["time_windows"]

    # Conditional dispatching
    horizon = req_tw[0][1]
    req_release = to_postpone * epoch_duration
    req_dispatch = np.where(to_dispatch, 0, horizon)

    for epoch_idx in range(next_epoch, next_epoch + max_lookahead):
        new = sample_epoch_requests(
            rng,
            static_inst,
            epoch_idx * epoch_duration,  # next epoch start time
            (epoch_idx + 1) * epoch_duration,  # next epoch dispatch time
            max_requests_per_epoch,
        )
        n_new_reqs = new["customer_idx"].size

        # Concatenate the new feasible requests to the epoch instance
        req_customer_idx = np.concatenate(
            (req_customer_idx, new["customer_idx"])
        )

        # Simulated request indices are always negative (so we can identify them)
        sim_req_idx = -(np.arange(n_new_reqs) + 1) - len(req_idx)
        req_idx = np.concatenate((ep_inst["request_idx"], sim_req_idx))

        req_demand = np.concatenate((req_demand, new["demands"]))
        req_service = np.concatenate((req_service, new["service_times"]))

        # Normalize TW and release to dispatch time, and clip the past
        new["time_windows"] = np.maximum(
            new["time_windows"] - dispatch_time, 0
        )
        req_tw = np.concatenate((req_tw, new["time_windows"]))

        new["release_times"] = np.maximum(
            new["release_times"] - dispatch_time, 0
        )
        req_release = np.concatenate((req_release, new["release_times"]))

        # Default dispatch time is the time horizon.
        req_dispatch = np.concatenate(
            (req_dispatch, np.full(n_new_reqs, horizon))
        )

    return {
        "is_depot": static_inst["is_depot"][req_customer_idx],
        "customer_idx": req_customer_idx,
        "request_idx": req_idx,
        "coords": static_inst["coords"][req_customer_idx],
        "demands": req_demand,
        "capacity": static_inst["capacity"],
        "time_windows": req_tw,
        "service_times": req_service,
        "duration_matrix": dist[req_customer_idx][:, req_customer_idx],
        "release_times": req_release,
        "dispatch_times": req_dispatch,
    }
