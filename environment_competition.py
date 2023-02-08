import time
import tools
from typing import Any, Dict, Optional, List, Tuple

import numpy as np


State = Dict[str, Any]
Action = List[List[int]]
Info = Dict[str, Any]


class VRPEnvironment:
    """
    Parameters
    ----------
    seed
        Random seed.
    instance
        The static VRP instance from which requests are sampled.
    epoch_tlim
        The epoch time limit.
    max_requests_per_epoch
        The maximum number of revealed requests per epoch.
    dispatch_margin
        The preparation time needed to dispatch a set of routes. That is, when
        a set of routes are to be dispatched at epoch t, then the start time of
        the routes is `t * epoch_duration + dispatch_margin`.
    epoch_duration
        The time between two consecutive epochs.
    """

    def __init__(
        self,
        seed: int,
        instance: Dict,
        epoch_tlim: float,
        max_requests_per_epoch: int = 100,
        dispatch_margin: int = 3600,
        epoch_duration: int = 3600,
    ):
        self.rng = np.random.default_rng(seed)
        self.instance = instance
        self.epoch_tlim = epoch_tlim
        self.max_requests_per_epoch = max_requests_per_epoch
        self.dispatch_margin = dispatch_margin
        self.epoch_duration = epoch_duration

        self.is_done = True  # Requires reset to be called first

    def reset(self) -> Tuple[State, Info]:
        """
        Resets the environment.
        """
        tw = self.instance["time_windows"]

        # The start and end epochs are determined by the earliest and latest
        # opening moments of time windows, corrected by the dispatch margin.
        earliest_open = tw[1:, 0].min() - self.dispatch_margin
        latest_open = tw[1:, 0].max() - self.dispatch_margin

        self.start_epoch = int(max(earliest_open // self.epoch_duration, 0))
        self.end_epoch = int(max(latest_open // self.epoch_duration, 0))

        self.current_epoch = self.start_epoch
        self.current_time = self.current_epoch * self.epoch_duration

        self.sample_complete_dynamic_instance()

        self.is_done = False
        obs = self._next_observation()

        self.final_solutions: Dict[int, Optional[List]] = {}
        self.final_costs: Dict[int, Optional[float]] = {}

        self.start_time_epoch = time.time()

        info = {
            "dynamic_context": self.instance,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "num_epochs": self.end_epoch - self.start_epoch + 1,
            "epoch_tlim": self.epoch_tlim,
        }

        return obs, info

    def sample_complete_dynamic_instance(self):
        """
        Sample the complete dynamic instance.
        """
        # Initialize request array with dummy request for depot
        self.req_idx = np.array([0])
        self.req_customer_idx = np.array([0])
        self.req_tw = self.instance["time_windows"][0:1]
        self.req_service = self.instance["service_times"][0:1]
        self.req_demand = self.instance["demands"][0:1]
        self.req_is_dispatched = np.array([False])
        self.req_epoch = np.array([0])
        self.req_release_time = np.array([0])
        self.req_must_dispatch = np.array([False])

        for epoch_idx in range(self.current_epoch, self.end_epoch + 1):
            epoch_reqs = self.sample_epoch_requests(epoch_idx)
            n_ep_reqs = epoch_reqs["customer_idx"].size

            self.req_idx = np.concatenate(
                (self.req_idx, np.arange(n_ep_reqs) + len(self.req_idx))
            )
            self.req_customer_idx = np.concatenate(
                (self.req_customer_idx, epoch_reqs["customer_idx"])
            )
            self.req_tw = np.concatenate(
                (self.req_tw, epoch_reqs["time_windows"])
            )
            self.req_service = np.concatenate(
                (self.req_service, epoch_reqs["service_times"])
            )
            self.req_demand = np.concatenate(
                (self.req_demand, epoch_reqs["demands"])
            )
            self.req_is_dispatched = np.pad(
                self.req_is_dispatched, (0, n_ep_reqs), mode="constant"
            )
            self.req_epoch = np.concatenate(
                (self.req_epoch, np.full(n_ep_reqs, epoch_idx))
            )
            self.req_release_time = np.concatenate(
                (self.req_release_time, epoch_reqs["release_times"])
            )

    def step(
        self, solution: Action
    ) -> Tuple[Optional[State], float, bool, Info]:
        """
        Steps to the next epoch. If the submitted solution is valid, then this
        method returns the observation of the next epoch. Otherwise, the epoch
        is failed and the environment is done.
        """
        try:
            self._validate_step(solution)
        except AssertionError as error:
            self.is_done = True
            return (None, float("inf"), self.is_done, {"error": str(error)})

        cost = tools.validation.validate_dynamic_epoch_solution(
            self.ep_inst, solution
        )

        self.final_solutions[self.current_epoch] = solution
        self.final_costs[self.current_epoch] = cost

        self.current_epoch += 1
        self.current_time = self.current_epoch * self.epoch_duration
        self.is_done = self.current_epoch > self.end_epoch

        observation = self._next_observation() if not self.is_done else None
        reward = -cost

        self.start_time_epoch = time.time()
        return (observation, reward, self.is_done, {"error": None})

    def _validate_step(self, solution):
        """
        Validates if the solution was submitted on time, and whether it
        satisfies the dynamic and static constraints.
        """
        assert not self.is_done, "Environment is finished"

        # Check if solution is valid
        tools.validation.validate_dynamic_epoch_solution(
            self.ep_inst, solution
        )

        # Mark orders of submitted solution as dispatched
        for route in solution:
            assert not self.req_is_dispatched[route].any()
            self.req_is_dispatched[route] = True

        # We must not have any undispatched orders that must be dispatched
        undispatched = (self.req_must_dispatch & ~self.req_is_dispatched).any()
        assert not undispatched, "Must dispatch requests not dispatched."

    def sample_epoch_requests(self, epoch_idx, rng=None):
        """
        Samples requests from an epoch.
        """
        dist = self.instance["duration_matrix"]

        current_time = epoch_idx * self.epoch_duration
        dispatch_time = current_time + self.dispatch_margin

        n_customers = self.instance["is_depot"].size - 1  # Exclude depot
        n_samples = self.max_requests_per_epoch

        # The solution method may need to use a different rng
        if rng is None:
            rng = self.rng

        # Sample data uniformly from customers (1 to num_customers)
        cust_idx = rng.integers(n_customers, size=n_samples) + 1
        tw_idx = rng.integers(n_customers, size=n_samples) + 1
        demand_idx = rng.integers(n_customers, size=n_samples) + 1
        service_idx = rng.integers(n_customers, size=n_samples) + 1

        new_tw = self.instance["time_windows"][tw_idx]
        new_demand = self.instance["demands"][demand_idx]
        new_service = self.instance["service_times"][service_idx]

        # Filter sampled requests that cannot be served in a round trip
        earliest_arrival = np.maximum(
            dispatch_time + dist[0, cust_idx], new_tw[:, 0]
        )
        earliest_return = earliest_arrival + new_service + dist[cust_idx, 0]
        depot_closed = self.instance["time_windows"][0, 1]

        feas = (earliest_arrival <= new_tw[:, 1]) & (
            earliest_return <= depot_closed
        )
        n_new_requests = feas.sum()
        new_release = np.full(n_new_requests, dispatch_time)

        return {
            "customer_idx": cust_idx[feas],
            "time_windows": new_tw[feas],
            "demands": new_demand[feas],
            "service_times": new_service[feas],
            "release_times": new_release,
        }

    def _next_observation(self) -> State:
        """
        Returns the next observation. This consists of all requests that were
        not dispatched during the previous epoch, and newly arrived requests.
        """
        # TODO Refactor this: we should take the dynamic instance and use the
        # `filter_instance` function with mask to create a new instance.

        dist = self.instance["duration_matrix"]
        dispatch_time = self.current_time + self.dispatch_margin
        depot_closed = self.instance["time_windows"][0, 1]

        # Determine which requests are must-dispatch in the next epoch
        if self.current_epoch < self.end_epoch:
            next_dispatch_time = dispatch_time + self.epoch_duration

            earliest_arrival = np.maximum(
                next_dispatch_time + dist[0, self.req_customer_idx],
                self.req_tw[:, 0],
            )
            earliest_return_at_depot = (
                earliest_arrival
                + self.req_service
                + dist[self.req_customer_idx, 0]
            )

            self.req_must_dispatch = (earliest_arrival > self.req_tw[:, 1]) | (
                earliest_return_at_depot > depot_closed
            )
        else:
            # In the end epoch, all requests are must dispatch
            self.req_must_dispatch = self.req_idx > 0

        # Return the epoch instance. This consists of all requests that are not
        # yet dispatched nor released.
        current_reqs = self.req_idx[
            ~self.req_is_dispatched & (self.req_epoch <= self.current_epoch)
        ]
        customer_idx = self.req_customer_idx[current_reqs]

        # Normalize TW to dispatch_time, and clip the past
        time_windows = np.maximum(self.req_tw[current_reqs] - dispatch_time, 0)

        # Normalize release times to dispatch_time, and clip the past
        release_times = np.maximum(
            self.req_release_time[current_reqs] - dispatch_time, 0
        )

        self.ep_inst = {
            "is_depot": self.instance["is_depot"][customer_idx],
            "customer_idx": customer_idx,
            "request_idx": current_reqs,
            "coords": self.instance["coords"][customer_idx],
            "demands": self.req_demand[current_reqs],
            "capacity": self.instance["capacity"],
            "time_windows": time_windows,
            "service_times": self.req_service[current_reqs],
            "duration_matrix": self.instance["duration_matrix"][
                np.ix_(customer_idx, customer_idx)
            ],
            "must_dispatch": self.req_must_dispatch[current_reqs],
            "epoch": self.req_epoch[current_reqs],
            "release_time": release_times,
        }

        return {
            "current_epoch": self.current_epoch,
            "current_time": self.current_time,
            "dispatch_time": dispatch_time,
            "epoch_instance": self.ep_inst,
        }

    def get_hindsight_problem(self) -> State:
        """
        After the episode is completed, this function can be used to obtain the
        'hindsight problem', i.e., as if we had future information about all the
        requests. This includes the release times of the requests.
        """
        customer_idx = self.req_customer_idx

        # Release times indicate that a route containing this request cannot
        # dispatch before this time. This includes the margin time for dispatch
        release_times = (
            self.epoch_duration * self.req_epoch + self.dispatch_margin
        )
        release_times[self.instance["is_depot"][customer_idx]] = 0

        return {
            "is_depot": self.instance["is_depot"][customer_idx],
            "customer_idx": customer_idx,
            "request_idx": self.req_idx,
            "coords": self.instance["coords"][customer_idx],
            "demands": self.req_demand,
            "capacity": self.instance["capacity"],
            "time_windows": self.req_tw,
            "service_times": self.req_service,
            "duration_matrix": self.instance["duration_matrix"][
                np.ix_(customer_idx, customer_idx)
            ],
            "release_times": release_times,
        }
