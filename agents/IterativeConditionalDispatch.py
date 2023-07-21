from functools import partial
from multiprocessing import Pool

import numpy as np

from plotting.plot_dynamic_instance import save_fig
from sampling import sample_epoch_requests
from static_solvers import scenario_solver

from .consensus import CONSENSUS, utils


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

        for iter_idx in range(self.num_iterations):
            instances = [
                self._sample_scenario(info, obs, to_dispatch, to_postpone)
                for _ in range(self.num_scenarios)
            ]

            if self.num_parallel_solve == 1:
                solutions = list(map(self._solve_scenario, instances))
            else:
                with Pool(self.num_parallel_solve) as pool:
                    solutions = pool.map(self._solve_scenario, instances)

            if obs["current_epoch"] == 2:
                plot(
                    ep_inst,
                    instances,
                    solutions,
                    iter_idx,
                    to_dispatch,
                    to_postpone,
                    self.consensus_func,
                )

            to_dispatch, to_postpone = self.consensus_func(
                iter_idx,
                list(zip(instances, solutions)),
                ep_inst,
                to_dispatch,
                to_postpone,
            )

            # Stop the run early when all requests have been marked
            if ep_size - 1 == to_dispatch.sum() + to_postpone.sum():
                break

        return to_dispatch | ep_inst["is_depot"]  # include depot

    def _solve_scenario(self, instance: dict) -> list[list[int]]:
        """
        Solves a single scenario instance, returning the solution.
        """
        if instance["request_idx"].size <= 2:
            # BUG Empty or single client dispatch instance, PyVRP cannot handle
            # this (see https://github.com/PyVRP/PyVRP/issues/272).
            return [[req] for req in instance["request_idx"] if req]

        result = scenario_solver(instance, self.seed, self.scenario_time_limit)
        return [route.visits() for route in result.best.get_routes()]

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
        info
            The static problem information.
        obs
            The current epcoh observation.
        to_dispatch
            A boolean array where True means that the corresponding request must be
            dispatched.
        to_postpone
            A boolean array where True mean that the corresponding request must be
            postponed.
        """
        # Parameters
        current_epoch = obs["current_epoch"]
        next_epoch = current_epoch + 1
        epochs_left = info["end_epoch"] - current_epoch
        max_lookahead = min(self.num_lookahead, epochs_left)
        max_requests_per_epoch = info["max_requests_per_epoch"]

        static_inst = info["dynamic_context"]
        epoch_duration = info["epoch_duration"]
        dispatch_margin = info["dispatch_margin"]
        ep_inst = obs["epoch_instance"]
        departure_time = obs["departure_time"]

        # Scenario instance fields
        req_cust_idx = ep_inst["customer_idx"]
        req_idx = ep_inst["request_idx"]
        req_demand = ep_inst["demands"]
        req_service = ep_inst["service_times"]
        req_tw = ep_inst["time_windows"]
        req_release = ep_inst["release_times"]
        req_must_dispatch = ep_inst["must_dispatch"]

        # Modify the release time of postponed requests: they should start
        # at the next departure time.
        next_departure_time = departure_time + epoch_duration
        req_release[to_postpone] = next_departure_time

        # Modify the dispatch time of dispatched requests: they should start
        # at the current departure time (and at time horizon otherwise).
        horizon = req_tw[0][1]
        req_dispatch = np.where(to_dispatch, departure_time, horizon)

        for epoch_idx in range(next_epoch, next_epoch + max_lookahead):
            next_epoch_start = epoch_idx * epoch_duration
            next_epoch_depart = next_epoch_start + dispatch_margin

            new = sample_epoch_requests(
                self.rng,
                static_inst,
                next_epoch_start,
                next_epoch_depart,
                max_requests_per_epoch,
            )
            num_new_reqs = new["customer_idx"].size

            # Sampled request indices are negative so we can distinguish them
            new_req_idx = -(np.arange(num_new_reqs) + 1) - len(req_idx)

            # Concatenate the new requests to the current instance requests
            req_idx = np.concatenate((req_idx, new_req_idx))
            req_cust_idx = np.concatenate((req_cust_idx, new["customer_idx"]))
            req_demand = np.concatenate((req_demand, new["demands"]))
            req_service = np.concatenate((req_service, new["service_times"]))
            req_tw = np.concatenate((req_tw, new["time_windows"]))
            req_release = np.concatenate((req_release, new["release_times"]))

            # Default earliest dispatch time is the time horizon.
            req_dispatch = np.concatenate(
                (req_dispatch, np.full(num_new_reqs, horizon))
            )

            req_must_dispatch = np.concatenate(
                (req_must_dispatch, np.full(num_new_reqs, False))
            )

        dist = static_inst["duration_matrix"]

        return {
            "is_depot": static_inst["is_depot"][req_cust_idx],
            "customer_idx": req_cust_idx,
            "request_idx": req_idx,
            "coords": static_inst["coords"][req_cust_idx],
            "demands": req_demand,
            "capacity": static_inst["capacity"],
            "time_windows": req_tw,
            "service_times": req_service,
            "duration_matrix": dist[req_cust_idx][:, req_cust_idx],
            "release_times": req_release,
            "dispatch_times": req_dispatch,
            "must_dispatch": req_must_dispatch,
        }


def plot(
    ep_inst,
    instances,
    solutions,
    iter_idx,
    to_dispatch,
    to_postpone,
    consensus_func,
):
    # Plot epoch instance
    save_fig(
        "figs/epoch_instance.pdf",
        "Epoch instance",
        ep_inst,
    )

    scenarios = list(zip(instances, solutions))
    for idx, (instance, solution) in enumerate(scenarios):
        # Plot scenario instance
        save_fig(
            f"figs/scenario_{iter_idx}_{idx}.pdf",
            "Scenario instance",
            instance,
            postponed=to_postpone,
            dispatched=to_dispatch,
        )

        # Plot scenario solution
        save_fig(
            f"figs/scenario_solution_{iter_idx}_{idx}.pdf",
            "Scenario instance",
            instance,
            postponed=to_postpone,
            dispatched=to_dispatch,
            routes=solution,
        )

    # Plot epoch instance with dispatch frequencies as labels, only if requests
    # are not already dispatched or postponed
    counts = utils.get_dispatch_count(scenarios, to_dispatch, to_postpone)
    freqs = counts / len(scenarios)
    decided = np.flatnonzero(to_dispatch | to_postpone)
    labels = {idx: val for idx, val in enumerate(freqs) if idx not in decided}

    save_fig(
        f"figs/epoch_instance_{iter_idx}_with_labels.pdf",
        "Epoch instance with dispatch frequencies",
        ep_inst,
        labels=labels,
        postponed=to_postpone,
        dispatched=to_dispatch,
    )

    # Plot epoch instance with dispatch frequencies and the corresponding action
    new_dispatch, new_postpone = consensus_func(
        iter_idx, scenarios, ep_inst, to_dispatch, to_postpone
    )
    save_fig(
        f"figs/epoch_instance_{iter_idx}_with_labels_and_action.pdf",
        "Epoch instance with dispatch frequencies and the corresponding action",
        ep_inst,
        labels=labels,
        postponed=new_postpone,
        dispatched=new_dispatch,
    )

    # Plot epoch instance with action, but no labels
    save_fig(
        f"figs/epoch_instance_{iter_idx}_with_action.pdf",
        "Epoch instance with dispatch frequencies and the corresponding action",
        ep_inst,
        postponed=new_postpone,
        dispatched=new_dispatch,
    )
