import time
import tools
from typing import Any, Dict, Optional, List, Tuple, Union

import numpy as np


State = Dict[str, Any]
Action = List[List[int]]
Info = Dict[str, Any]


class Environment:
    """
    Parameters
    ----------
    seed
        Random seed.
    instance
        The static VRP instance from which requests are sampled.
    epoch_tlim
        The epoch time limit.
    num_epochs
        The number of epochs in which the time horizon is separated
    requests_per_epoch
        The expected number of revealed requests per epoch.
    time_window_style
        The time window style, one of ['fixed_deadline', 'variable_deadline',
        'fixed_time_window', 'variable_time_window'].
    time_window_width
        The width of the time window in number of epoch durations. Only
        applicable when a variable time window style is selected.
    """

    def __init__(
        self,
        seed: int,
        instance: Dict,
        epoch_tlim: float,
        num_epochs: int = 8,
        requests_per_epoch: Union[int, List] = 50,
        time_window_style: str = "variable_time_windows",
        time_window_width: int = 3,
    ):
        self.seed = seed
        self.instance = instance
        self.epoch_tlim = epoch_tlim
        self.requests_per_epoch = requests_per_epoch
        self.num_epochs = num_epochs
        self.time_window_style = time_window_style
        self.time_window_width = time_window_width

        self.is_done = True  # Requires reset to be called first

    def reset(self) -> Tuple[State, Info]:
        """
        Resets the environment.
        """
        self.rng = np.random.default_rng(self.seed)

        tw = self.instance["time_windows"]
        depot_closed = tw[0, 1]
        self.epoch_duration = depot_closed // (self.num_epochs + 1)

        self.start_epoch = 0
        self.end_epoch = self.num_epochs - 1
        self.current_epoch = self.start_epoch
        self.current_time = self.current_epoch * self.epoch_duration

        if isinstance(self.requests_per_epoch, int):
            self.requests_per_epoch = [
                self.requests_per_epoch
            ] * self.num_epochs

        self.sample_complete_dynamic_instance()

        self.is_done = False
        obs = self._next_observation()

        self.final_solutions: Dict[int, Optional[List]] = {}
        self.final_costs: Dict[int, Optional[float]] = {}

        info = {
            "dynamic_context": self.instance,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "num_epochs": self.num_epochs,
            "requests_per_epoch": self.requests_per_epoch,
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

        for epoch_idx in range(self.num_epochs):
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
        dispatch_time = epoch_idx * self.epoch_duration
        n_customers = self.instance["is_depot"].size - 1  # Exclude depot

        if rng is None:  # enables solution method to use different rng
            rng = self.rng

        noise = rng.uniform(0.8, 1.2)
        n_samples = int(self.requests_per_epoch[epoch_idx] * noise)

        feas = np.zeros(n_samples, dtype=bool)
        cust_idx = np.empty(n_samples, dtype=int)
        demand_idx = np.empty(n_samples, dtype=int)
        service_idx = np.empty(n_samples, dtype=int)
        old_tw = np.empty(shape=(0, 2), dtype=int)

        while not feas.all():
            # Sample data uniformly from customers (1 to num_customers)
            to_sample = np.sum(~feas)

            cust_idx = np.append(
                cust_idx[feas],
                self.rng.integers(n_customers, size=to_sample) + 1,
            )

            demand_idx = np.append(
                demand_idx[feas],
                self.rng.integers(n_customers, size=to_sample) + 1,
            )
            service_idx = np.append(
                service_idx[feas],
                self.rng.integers(n_customers, size=to_sample) + 1,
            )

            new_demand = self.instance["demands"][demand_idx]
            new_service = self.instance["service_times"][service_idx]
            new_tw = self._sample_time_windows(
                old_tw,
                self.time_window_style,
                feas,
                dispatch_time,
                epoch_idx,
                rng,
            )

            # Filter sampled requests that cannot be served in a round trip
            earliest_arrival = np.maximum(
                dispatch_time + dist[0, cust_idx], new_tw[:, 0]
            )
            earliest_return = (
                earliest_arrival + new_service + dist[cust_idx, 0]
            )
            depot_closed = self.instance["time_windows"][0, 1]

            feas = (earliest_arrival <= new_tw[:, 1]) & (
                earliest_return <= depot_closed
            )
            old_tw = new_tw[feas]

        return {
            "customer_idx": cust_idx,
            "time_windows": new_tw,
            "demands": new_demand,
            "service_times": new_service,
            "release_times": np.full(n_samples, dispatch_time),
        }

    def _sample_time_windows(
        self, old_tw, style, feas, dispatch_time, epoch_idx, rng
    ):
        n_infeas = np.sum(~feas)
        horizon = self.instance["time_windows"][0][1]
        last_dispatch_time = self.epoch_duration * self.end_epoch

        fixed_width = self.epoch_duration * self.time_window_width
        epochs_left = self.num_epochs - epoch_idx
        var_widths = self.epoch_duration * rng.integers(
            epochs_left, size=n_infeas
        )

        if style == "fixed_deadlines":
            early = dispatch_time * np.ones(n_infeas, dtype=int)
            late = np.minimum(horizon, early + fixed_width)
        elif style == "variable_deadlines":
            early = dispatch_time * np.ones(n_infeas, dtype=int)
            late = np.minimum(horizon, early + var_widths)
        elif style == "fixed_time_windows":
            early = rng.integers(dispatch_time, last_dispatch_time, n_infeas)
            late = np.minimum(horizon, early + fixed_width)
        elif style == "variable_time_windows":
            early = rng.integers(dispatch_time, last_dispatch_time, n_infeas)
            late = np.minimum(horizon, early + var_widths)
        else:
            raise ValueError("Time window style unknown.")

        new_tw = np.vstack((early, late)).T
        return np.concatenate((old_tw, new_tw))

    def _next_observation(self) -> State:
        """
        Returns the next observation. This consists of all requests that were
        not dispatched during the previous epoch, and newly arrived requests.
        """
        # TODO Refactor this: we should take the dynamic instance and use the
        # `filter_instance` function with mask to create a new instance.

        dist = self.instance["duration_matrix"]
        dispatch_time = self.current_time + self.epoch_duration
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
        release_times = self.epoch_duration * (self.req_epoch + 1)
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
