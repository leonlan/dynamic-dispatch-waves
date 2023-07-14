from functools import partial
from multiprocessing import Pool

import numpy as np

from sampling import sample_epoch_requests
from static_solvers import scenario_solver

from .consensus import CONSENSUS


class IterativeConditionalDispatch:
    """
    The iterative conditional dispatch strategy repeatedly solves a set of
    sample scenarios and uses a consensus function to determine based on
    the sample scenario solutions which requests to dispatch or postpone.

    Parameters
    ----------
    seed
        The random seed.
    num_iterations
        The number of iterations to run.
    num_lookahead
        The number of future (i.e., lookahead) epochs to consider per scenario.
    num_scenarios
        The number of scenarios to sample in each iteration.
    scenario_time_limit
        The time limit for solving a single scenario instance.
    consensus
        The consensus function to use.
    consensus_params
        The parameters to pass to the consensus function.
    num_parallel_solve
        The number of scenarios to solve in parallel.
    """

    def __init__(
        self,
        seed: int,
        num_iterations: int,
        num_lookahead: int,
        num_scenarios: int,
        scenario_time_limit: float,
        consensus: str,
        consensus_params: dict,
        num_parallel_solve: int = 1,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.num_iterations = num_iterations
        self.num_lookahead = num_lookahead
        self.num_scenarios = num_scenarios
        self.scenario_time_limit = scenario_time_limit
        self.consensus_func = partial(CONSENSUS[consensus], **consensus_params)
        self.num_parallel_solve = num_parallel_solve

    def act(self, info, obs) -> np.ndarray:
        ep_inst = obs["epoch_instance"]
        ep_size = ep_inst["is_depot"].size

        # In the last epoch, all requests must be dispatched.
        if obs["current_epoch"] == info["end_epoch"]:
            return np.ones(ep_size, dtype=bool)

        to_dispatch = ep_inst["must_dispatch"].copy()
        to_postpone = np.zeros(ep_size, dtype=bool)

        # Dispatch everything in the last iteration
        if observation["current_epoch"] == info["end_epoch"]:
            return np.ones(ep_size, dtype=bool)

        for iter_idx in range(self.num_iterations):
            scenarios = []

            for _ in range(self.num_scenarios):
                instance, solution = self._sample_and_solve_scenario(
                    info, observation, to_dispatch, to_postpone
                )
                scenarios.append((instance, solution))

            to_dispatch, to_postpone = self.consensus_func(
                iter_idx, scenarios, to_dispatch, to_postpone
            )

            # Stop the run early when all requests have been marked
            if ep_size - 1 == to_dispatch.sum() + to_postpone.sum():
                break

        return to_dispatch | ep_inst["is_depot"]  # include depot

    def _sample_and_solve_scenario(
        self,
        info: dict,
        obs: dict,
        to_dispatch: np.ndarray,
        to_postpone: np.ndarray,
    ) -> tuple[dict, list[list[int]]]:
        """
        Samples and solves a single scenario instance, returning both the
        instance and resulting solution.
        """
        instance = self._sample_scenario(info, obs, to_dispatch, to_postpone)

        result = scenario_solver(instance, self.seed, self.scenario_time_limit)
        solution = [route.visits() for route in result.best.get_routes()]

        return (instance, solution)

    def _sample_scenario(
        self,
        info: dict,
        obs: dict,
        to_dispatch: np.ndarray,
        to_postpone: np.ndarray,
    ) -> dict:
        """
        Samples a VRPTW scenario instance. The scenario instance is created by
        appending the sampled requests to the current epoch instance.

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
        max_lookahead = min(self.num_lookahead, epochs_left)

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
                self.rng,
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
