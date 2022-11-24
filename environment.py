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
    margin_dispatch
        The preparation time needed to dispatch a set of requests. That is, when
        a choice of requests are selected to be dispatched at epoch t, then the
        start time of the routes is ``t * epoch_duration + margin_dispatch``.
    epoch_duration
        The time between two consecutive epochs.
    """

    def __init__(
        self,
        seed: int,
        instance: Dict,
        epoch_tlim: int,
        max_requests_per_epoch: int = 100,
        margin_dispatch: int = 3600,
        epoch_duration: int = 3600,
    ):
        self.rng = np.random.default_rng(seed)
        self.instance = instance
        self.epoch_tlim = epoch_tlim

        self.max_requests_per_epoch = max_requests_per_epoch
        self.margin_dispatch = margin_dispatch
        self.epoch_duration = epoch_duration

        # Require reset to be called first by marking environment as done
        self.is_done = True

    def reset(self) -> Tuple[State, Info]:
        """
        Resets the environment.
        """
        tws = self.instance["time_windows"]

        # Start epoch depends on minimum earliest customer time window
        self.start_epoch = int(
            max(
                (tws[1:, 0].min() - self.margin_dispatch)
                // self.epoch_duration,
                0,
            )
        )

        # End epoch depends on maximum earliest customer time window
        self.end_epoch = int(
            max(
                (tws[1:, 0].max() - self.margin_dispatch)
                // self.epoch_duration,
                0,
            )
        )

        self.current_epoch = self.start_epoch

        # Initialize request array with dummy request for depot
        self.req_idx = np.array([0])
        self.req_customer_idx = np.array([0])
        self.req_tw = self.instance["time_windows"][0:1]
        self.req_service = self.instance["service_times"][0:1]
        self.req_demand = self.instance["demands"][0:1]
        self.req_is_dispatched = np.array([False])
        self.req_epoch = np.array([0])
        self.req_must_dispatch = np.array([False])

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

    def step(
        self, solution: Action
    ) -> Tuple[Optional[State], float, bool, Info]:
        assert not self.is_done, "Environment is finished"

        # Check time limit (2 seconds grace period)
        if self._get_elapsed_time_epoch() > self.epoch_tlim + 2:
            return self._fail_episode("Time exceeded")

        # Check if solution is valid
        try:
            driving_duration = tools.validate_dynamic_epoch_solution(
                self.ep_inst, solution
            )
        except AssertionError as e:
            return self._fail_episode(e)

        # Mark orders of current solution as dispatched
        for route in solution:
            # Route consists of 1 indexed requests
            assert not self.req_is_dispatched[route].any()
            self.req_is_dispatched[route] = True

        # We must not have any undispatched orders that must be dispatched
        assert not (self.req_must_dispatch & ~self.req_is_dispatched).any()

        self.final_solutions[self.current_epoch] = solution
        self.final_costs[self.current_epoch] = driving_duration

        self.current_epoch += 1
        self.is_done = self.current_epoch > self.end_epoch

        observation = self._next_observation() if not self.is_done else None
        reward = -driving_duration

        self.start_time_epoch = time.time()
        return (observation, reward, self.is_done, {"error": None})

    def _get_elapsed_time_epoch(self):
        assert self.start_time_epoch is not None
        return time.time() - self.start_time_epoch

    def _fail_episode(self, error):
        self.final_solutions[self.current_epoch] = None
        self.final_costs[self.current_epoch] = float("inf")
        self.is_done = True

        return (None, float("inf"), self.is_done, {"error": str(error)})

    def _next_observation(self) -> State:
        """
        Returns the next observation. This consists of all requests that were
        not dispatched during the previous epoch, and newly sampled requests.
        """
        duration_matrix = self.instance["duration_matrix"]

        current_time = self.epoch_duration * self.current_epoch
        planning_starttime = current_time + self.margin_dispatch

        # Sample uniformly
        n_customers = self.instance["is_depot"].size - 1  # Exclude depot

        # Sample data uniformly from customers (1 to num_customers)
        def sample_from_customers(k=self.max_requests_per_epoch):
            return self.rng.integers(n_customers, size=k) + 1

        cust_idx = sample_from_customers()
        tw_idx = sample_from_customers()
        demand_idx = sample_from_customers()
        service_t_idx = sample_from_customers()

        new_request_timewi = self.instance["time_windows"][tw_idx]
        # Filter data that can no longer be delivered
        # Time + margin for dispatch + drive time from depot should not exceed latest arrival
        earliest_arrival = np.maximum(
            planning_starttime + duration_matrix[0, cust_idx],
            new_request_timewi[:, 0],
        )
        # Also, return at depot in time must be feasible
        earliest_return_at_depot = (
            earliest_arrival
            + self.instance["service_times"][service_t_idx]
            + duration_matrix[cust_idx, 0]
        )
        is_feasible = (earliest_arrival <= new_request_timewi[:, 1]) & (
            earliest_return_at_depot <= self.instance["time_windows"][0, 1]
        )

        if is_feasible.any():
            num_new_requests = is_feasible.sum()
            self.req_idx = np.concatenate(
                (
                    self.req_idx,
                    np.arange(num_new_requests) + len(self.req_idx),
                )
            )
            self.req_customer_idx = np.concatenate(
                (self.req_customer_idx, cust_idx[is_feasible])
            )
            self.req_tw = np.concatenate(
                (self.req_tw, new_request_timewi[is_feasible])
            )
            self.req_service = np.concatenate(
                (
                    self.req_service,
                    self.instance["service_times"][service_t_idx[is_feasible]],
                )
            )
            self.req_demand = np.concatenate(
                (
                    self.req_demand,
                    self.instance["demands"][demand_idx[is_feasible]],
                )
            )
            self.req_is_dispatched = np.pad(
                self.req_is_dispatched,
                (0, num_new_requests),
                mode="constant",
            )
            self.req_epoch = np.concatenate(
                (
                    self.req_epoch,
                    np.full(num_new_requests, self.current_epoch),
                )
            )

        # Customers must dispatch this epoch if next epoch they will be too late
        if self.current_epoch < self.end_epoch:
            earliest_arrival = np.maximum(
                planning_starttime
                + self.epoch_duration
                + duration_matrix[0, self.req_customer_idx],
                self.req_tw[:, 0],
            )
            earliest_return_at_depot = (
                earliest_arrival
                + self.req_service
                + duration_matrix[self.req_customer_idx, 0]
            )
            self.req_must_dispatch = (earliest_arrival > self.req_tw[:, 1]) | (
                earliest_return_at_depot > self.instance["time_windows"][0, 1]
            )
        else:
            self.req_must_dispatch = self.req_idx > 0

        # Return instance based on customers not yet dispatched
        idx_undispatched = self.req_idx[~self.req_is_dispatched]
        customer_idx = self.req_customer_idx[idx_undispatched]

        # Return a VRPTW instance with undispatched requests with two
        # additional properties: customer_idx and request_idx
        time_windows = self.req_tw[idx_undispatched]

        # Renormalize time to start at planning_starttime, and clip time windows in the past (so depot will start at 0)
        time_windows = np.clip(
            time_windows - planning_starttime, a_min=0, a_max=None
        )

        self.ep_inst = {
            "is_depot": self.instance["is_depot"][customer_idx],
            "customer_idx": customer_idx,
            "request_idx": idx_undispatched,
            "coords": self.instance["coords"][customer_idx],
            "demands": self.req_demand[idx_undispatched],
            "capacity": self.instance["capacity"],
            "time_windows": time_windows,
            "service_times": self.req_service[idx_undispatched],
            "duration_matrix": self.instance["duration_matrix"][
                np.ix_(customer_idx, customer_idx)
            ],
            "must_dispatch": self.req_must_dispatch[idx_undispatched],
        }
        return {
            "current_epoch": self.current_epoch,
            "current_time": current_time,
            "planning_starttime": planning_starttime,
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
            self.epoch_duration * self.req_epoch + self.margin_dispatch
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
            # 'must_dispatch': self.request_must_dispatch,
            "release_times": release_times,
        }
