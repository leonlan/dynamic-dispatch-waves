# The MIT License (MIT)
#
# Copyright(c) 2022 ORTEC & TU/e
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
from typing import Any, Callable

import numpy as np

from utils.validation import validate_dynamic_epoch_solution

State = dict[str, Any]
Action = list[list[int]]
Info = dict[str, Any]


class Environment:
    """
    Environment for the DDWP based on [1].

    Parameters
    ----------
    seed
        Random seed.
    instance
        The static VRP instance from which requests are sampled.
    epoch_tlim
        The epoch time limit.
    instance_sampler
        The instance sampler to use.
    num_requests_per_epoch
        The maximum number of revealed requests per epoch.
    dispatch_margin
        The preparation time needed to dispatch a set of routes. That is, when
        a set of routes are to be dispatched at epoch t, then the start time of
        the routes is `t * epoch_duration + dispatch_margin`.
    epoch_duration
        The time between two consecutive epochs.

    References
    ----------
    [1] EURO meets NeurIPS 2022 vehicle routing competition.
        https://euro-neurips-vrp-2022.challenges.ortec.com/
    """

    def __init__(
        self,
        seed: int,
        instance: dict,
        epoch_tlim: float,
        instance_sampler: Callable,
        num_requests_per_epoch: int = 100,
        epoch_duration: int = 3600,
        dispatch_margin: int = 3600,
    ):
        self.seed = seed
        self.instance = instance
        self.epoch_tlim = epoch_tlim
        self.instance_sampler = instance_sampler
        self.num_requests_per_epoch = num_requests_per_epoch
        self.epoch_duration = epoch_duration
        self.dispatch_margin = dispatch_margin

        self.is_done = True  # Requires reset to be called first

    def reset(self) -> tuple[State, Info]:
        """
        Resets the environment.

        Returns
        -------
        Tuple[State, Info]
            The first epoch observation and the environment static information.
        """
        self.rng = np.random.default_rng(self.seed)
        tw = self.instance["time_windows"]

        earliest_open = tw[1:, 0].min() - self.dispatch_margin
        latest_open = tw[1:, 0].max() - self.dispatch_margin

        # The start and end epochs are determined by the earliest and latest
        # time windows of all clients, corrected by the dispatch margin.
        self.start_epoch = int(max(earliest_open // self.epoch_duration, 0))
        self.end_epoch = int(max(latest_open // self.epoch_duration, 0))

        self.current_epoch = self.start_epoch
        self.current_time = self.current_epoch * self.epoch_duration

        self.is_done = False
        self.final_solutions: dict[int, list] = {}
        self.final_costs: dict[int, float] = {}

        self._sample_complete_dynamic_instance()

        observation = self._next_observation()
        static_info = {
            "dynamic_context": self.instance,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "num_epochs": self.end_epoch - self.start_epoch + 1,
            "epoch_tlim": self.epoch_tlim,
            "epoch_duration": self.epoch_duration,
            "dispatch_margin": self.dispatch_margin,
            "num_requests_per_epoch": self.num_requests_per_epoch,
        }

        self.start_time_epoch = time.time()
        return observation, static_info

    def _sample_complete_dynamic_instance(self):
        """
        Sample the complete dynamic instance.
        """
        # Initialize request array with dummy request for depot.
        self.req_idx = np.array([0])
        self.req_customer_idx = np.array([0])
        self.req_tw = self.instance["time_windows"][0:1]
        self.req_service = self.instance["service_times"][0:1]
        self.req_demand = self.instance["demands"][0:1]
        self.req_release_time = np.array([0])
        self.req_epoch = np.array([0])  # epoch in which request is revealed
        self.req_is_dispatched = np.array([False])

        for epoch in range(self.current_epoch, self.end_epoch + 1):
            current_time = epoch * self.epoch_duration
            departure_time = current_time + self.dispatch_margin
            new_reqs = self.instance_sampler(
                self.rng,
                self.instance,
                current_time,
                departure_time,
                self.num_requests_per_epoch,
            )
            num_reqs = new_reqs["customer_idx"].size

            self.req_idx = np.concatenate(
                (self.req_idx, np.arange(num_reqs) + len(self.req_idx))
            )
            self.req_customer_idx = np.concatenate(
                (self.req_customer_idx, new_reqs["customer_idx"])
            )
            self.req_tw = np.concatenate(
                (self.req_tw, new_reqs["time_windows"])
            )
            self.req_service = np.concatenate(
                (self.req_service, new_reqs["service_times"])
            )
            self.req_demand = np.concatenate(
                (self.req_demand, new_reqs["demands"])
            )
            self.req_release_time = np.concatenate(
                (self.req_release_time, new_reqs["release_times"])
            )
            self.req_epoch = np.concatenate(
                (self.req_epoch, np.full(num_reqs, epoch))
            )
            self.req_is_dispatched = np.pad(
                self.req_is_dispatched, (0, num_reqs), mode="constant"
            )

        # Compute the latest epoch in which a request can be dispatched on time
        # as a round-trip. These requests become "must-dispatch" in that epoch.
        dist = self.instance["duration_matrix"]
        horizon = self.instance["time_windows"][0, 1]

        self.req_must_dispatch_epoch = self.req_epoch.copy()

        for epoch in range(self.current_epoch, self.end_epoch + 1):
            current_time = epoch * self.epoch_duration
            departure_time = current_time + self.dispatch_margin

            earliest_arrival = np.maximum(
                departure_time + dist[0, self.req_customer_idx],
                self.req_tw[:, 0],
            )
            earliest_return = (
                earliest_arrival
                + self.req_service
                + dist[self.req_customer_idx, 0]
            )

            feasible = (earliest_arrival <= self.req_tw[:, 1]) & (
                earliest_return <= horizon
            )
            self.req_must_dispatch_epoch[feasible] = epoch

        # Do not mark depot as must-dispatch.
        self.req_must_dispatch_epoch[0] = self.end_epoch + 1

    def step(self, solution: Action) -> tuple[State, float, bool, Info]:
        """
        Steps to the next state for the given action.

        Parameters
        ----------
        action: Action
            The action to take, i.e., the routes to dispatch.

        Returns
        -------
        State
            The next state. If the action is invalid, or when the episode is
            done, this should return an empty dictionary.
        float
            The reward for the transition. If the action is invalid, this
            should return ``float("inf")``.
        bool
            Whether the episode is done.
        Info
            Success information about the step.
        """
        try:
            assert not self.is_done, "Environment is finished."
            cost = validate_dynamic_epoch_solution(self.ep_inst, solution)
        except AssertionError as error:
            self.is_done = True
            return ({}, float("inf"), self.is_done, {"error": str(error)})

        # Mark orders of submitted solution as dispatched.
        for route in solution:
            assert not self.req_is_dispatched[route].any()
            self.req_is_dispatched[route] = True

        self.final_solutions[self.current_epoch] = solution
        self.final_costs[self.current_epoch] = cost

        self.current_epoch += 1
        self.current_time = self.current_epoch * self.epoch_duration
        self.is_done = self.current_epoch > self.end_epoch

        observation = self._next_observation() if not self.is_done else {}
        reward = -cost

        self.start_time_epoch = time.time()
        return (observation, reward, self.is_done, {"error": None})

    def _next_observation(self) -> State:
        """
        Returns the next observation. This consists of all revealed requests
        that were not dispatched in the previous epoch, and new requests.
        """
        revealed = self.req_epoch <= self.current_epoch
        not_dispatched = ~self.req_is_dispatched
        current_reqs = self.req_idx[revealed & not_dispatched]
        customer_idx = self.req_customer_idx[current_reqs]

        # Set depot time window to be at least the departure time.
        departure_time = self.current_time + self.dispatch_margin
        time_windows = self.req_tw[current_reqs]
        time_windows[0, 0] = departure_time

        # Determine the must dispatch requests.
        must_dispatch_epoch = self.req_must_dispatch_epoch[current_reqs]
        must_dispatch = must_dispatch_epoch == self.current_epoch

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
            "must_dispatch": must_dispatch,
            "epoch": self.req_epoch[current_reqs],
            "release_times": self.req_release_time[current_reqs],
        }

        return {
            "current_epoch": self.current_epoch,
            "current_time": self.current_time,
            "departure_time": departure_time,
            "epoch_instance": self.ep_inst,
        }

    def get_hindsight_problem(self) -> State:
        """
        Returns the hindsight problem, which is a static VRP instance that
        represents the dynamic instance assuming perfect information.

        Returns
        -------
        State
            A hindsight problem.
        """
        customer_idx = self.req_customer_idx

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
            "release_times": self.req_release_time,
        }
