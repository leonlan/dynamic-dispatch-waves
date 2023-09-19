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

from dataclasses import dataclass
from time import perf_counter
from typing import Optional
from warnings import warn

import numpy as np

from sampling import SamplingMethod
from utils.validation import validate_static_solution
from VrpInstance import VrpInstance

Instance = dict
Action = list[list[int]]
Info = dict


@dataclass(frozen=True)
class StaticInfo:
    """
    Static global information about the DDWP episode.

    Parameters
    ----------
    static_instance
        The static VRP instance from which requests are sampled.
    start_epoch
        The start epoch index.
    end_epoch
        The end epoch index.
    epoch_tlim
        The epoch time limit.
    epoch_duration
        The time between two consecutive epochs.
    dispatch_margin
        The preparation time needed to dispatch a set of routes. That is, when
        a set of routes are to be dispatched at epoch t, then the start time of
        the routes is `t * epoch_duration + dispatch_margin`.
    num_requests_per_epoch
        The expected number of revealed requests per epoch.
    num_vehicles_per_epoch
        The available number of primary vehicles per epoch. If None, then
        there is no limit on the number of primary vehicles.
    """

    static_instance: VrpInstance
    start_epoch: int
    end_epoch: int
    epoch_tlim: float
    epoch_duration: int
    dispatch_margin: int
    num_requests_per_epoch: list[int]
    num_vehicles_per_epoch: Optional[list[int]]


@dataclass(frozen=True)
class State:
    """
    State of an epoch.
    """

    current_epoch: int
    current_time: int
    departure_time: int
    epoch_instance: VrpInstance


class Environment:
    """
    Environment for the DDWP with equidistant epochs.

    Parameters
    ----------
    seed
        Random seed.
    instance
        The static VRP instance from which requests are sampled.
    epoch_tlim
        The epoch time limit.
    sampling_method
        The sampling method to use.
    num_requests_per_epoch
        The expected number of revealed requests per epoch.
    num_vehicles_per_epoch
        The available number of primary vehicles per epoch. If None, then
        there is no limit on the number of primary vehicles.
    start_epoch
        The start epoch.
    end_epoch
        The end epoch.
    epoch_duration
        The time between two consecutive epochs.
    dispatch_margin
        The preparation time needed to dispatch a set of routes. That is, when
        a set of routes are to be dispatched at epoch t, then the start time of
        the routes is `t * epoch_duration + dispatch_margin`.
    """

    def __init__(
        self,
        seed: int,
        instance: VrpInstance,
        epoch_tlim: float,
        sampling_method: SamplingMethod,
        start_epoch: int,
        end_epoch: int,
        num_requests_per_epoch: list[int],
        num_vehicles_per_epoch: Optional[list[int]],
        epoch_duration: int,
        dispatch_margin: int,
    ):
        self.seed = seed
        self.instance = instance
        self.epoch_tlim = epoch_tlim
        self.sampling_method = sampling_method
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.num_requests_per_epoch = num_requests_per_epoch
        self.num_vehicles_per_epoch = num_vehicles_per_epoch
        self.epoch_duration = epoch_duration
        self.dispatch_margin = dispatch_margin

        self.static_info = StaticInfo(
            static_instance=instance,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            epoch_tlim=epoch_tlim,
            epoch_duration=epoch_duration,
            dispatch_margin=dispatch_margin,
            num_requests_per_epoch=num_requests_per_epoch,
            num_vehicles_per_epoch=num_vehicles_per_epoch,
        )

        self.is_done = True  # Requires reset to be called first

    @classmethod
    def euro_neurips(
        cls,
        seed: int,
        instance: VrpInstance,
        epoch_tlim: float,
        sampling_method: SamplingMethod,
        num_requests: int = 100,
        epoch_duration: int = 3600,
        dispatch_margin: int = 3600,
        num_vehicles_per_epoch: Optional[list[int]] = None,
    ):
        """
        Creates a DDWP environment identical to the one used in [1].

        Parameters
        ----------
        seed
            Random seed.
        instance
            The static VRP instance from which requests are sampled.
        epoch_tlim
            The epoch time limit.
        sampling_method
            The sampling method to use.
        num_requests
            The expected number of revealed requests per epoch.
        epoch_duration
            The time between two consecutive epochs.
        dispatch_margin
            The preparation time needed to dispatch a set of routes. That is, when
            a set of routes are to be dispatched at epoch t, then the start time of
            the routes is `t * epoch_duration + dispatch_margin`.
        num_vehicles_per_epoch
            The available number of primary vehicles per epoch. If None, then
            there is no limit on the number of primary vehicles.

        References
        ----------
        [1] EURO meets NeurIPS 2022 vehicle routing competition.
            https://euro-neurips-vrp-2022.challenges.ortec.com/
        """
        tw = instance.time_windows
        earliest = tw[1:, 0].min() - dispatch_margin
        latest = tw[1:, 0].max() - dispatch_margin

        # The start and end epochs are determined by the earliest and latest
        # opening client time windows, corrected by the dispatch margin.
        start_epoch = int(max(earliest // epoch_duration, 0))
        end_epoch = int(max(latest // epoch_duration, 0))

        num_requests_per_epoch = [num_requests] * (end_epoch + 1)

        return cls(
            seed=seed,
            instance=instance,
            epoch_tlim=epoch_tlim,
            sampling_method=sampling_method,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            num_requests_per_epoch=num_requests_per_epoch,
            num_vehicles_per_epoch=num_vehicles_per_epoch,
            epoch_duration=epoch_duration,
            dispatch_margin=dispatch_margin,
        )

    @classmethod
    def paper(
        cls,
        seed: int,
        instance: VrpInstance,
        epoch_tlim: float,
        sampling_method: SamplingMethod,
        num_vehicles_per_epoch: Optional[list[int]] = None,
        num_requests_per_epoch: list[int] = [75] * 8,
        num_epochs: int = 8,
    ):
        """
        Creates a DDWP environment identical to the one used in [1].

        Parameters
        ----------
        seed
            Random seed.
        instance
            The static VRP instance from which requests are sampled. Note that
            the time windows are ignored in this environment.
        epoch_tlim
            The epoch time limit.
        sampling_method
            The sampling method to use.
        num_vehicles_per_epoch
            The available number of primary vehicles per epoch. If None, then
            there is no limit on the number of primary vehicles.
        num_requests_per_epoch
            The expected number of revealed requests per epoch.
        num_epochs
            The number of epochs to consider.

        References
        ----------
        [1] Lan, L., van Doorn, J., Wouda, N. A., Rijal, A., & Bhulai, S. (2023).
            An iterative conditional dispatch algorithm for the dynamic dispatch
            waves problem.
        """
        # Assume an epoch duration of one hour (in seconds) and a horizon of
        # ``num_epochs`` hours.
        epoch_duration = 3600
        horizon = num_epochs * epoch_duration
        start_epoch = 0
        end_epoch = num_epochs - 1

        # Custom depot time windows. Instance time windows are not used!
        time_windows = instance.time_windows.copy()
        time_windows[0, :] = [0, horizon]

        # Normalize the distances so that the furthest customer can be reached
        # in one hour. Service times are also scaled accordingly.
        scale = instance.duration_matrix.max() / epoch_duration
        dur_mat = np.ceil(instance.duration_matrix / scale).astype(int)
        service_times = np.ceil(instance.service_times / scale).astype(int)

        new_instance = instance.replace(
            time_windows=time_windows,
            duration_matrix=dur_mat,
            service_times=service_times,
        )

        return cls(
            seed=seed,
            instance=new_instance,
            epoch_tlim=epoch_tlim,
            sampling_method=sampling_method,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            num_requests_per_epoch=num_requests_per_epoch,
            num_vehicles_per_epoch=num_vehicles_per_epoch,
            epoch_duration=epoch_duration,
            dispatch_margin=0,
        )

    def reset(self) -> tuple[State, StaticInfo]:
        """
        Resets the environment.

        Returns
        -------
        tuple[State, StaticInfo]
            The first epoch observation and the environment static information.
        """
        self.rng = np.random.default_rng(self.seed)

        self.current_epoch = self.start_epoch
        self.current_time = self.current_epoch * self.epoch_duration
        self.num_vehicles_used: list[int] = []

        self.is_done = False
        self.final_solutions: dict[int, list] = {}
        self.final_costs: dict[int, float] = {}

        self._sample_complete_dynamic_instance()

        self.start_time_epoch = perf_counter()

        return self._next_observation(), self.static_info

    def _sample_complete_dynamic_instance(self):
        """
        Samples the complete dynamic instance.
        """
        # Initialize request array with dummy request for depot.
        self.req_idx = np.array([0])
        self.req_customer_idx = np.array([0])
        self.req_tw = self.instance.time_windows[0:1]
        self.req_service = self.instance.service_times[0:1]
        self.req_demand = self.instance.demands[0:1]
        self.req_release_time = np.array([0])
        self.req_epoch = np.array([0])  # epoch in which request is revealed
        self.req_is_dispatched = np.array([False])

        for epoch in range(self.current_epoch, self.end_epoch + 1):
            current_time = epoch * self.epoch_duration
            departure_time = current_time + self.dispatch_margin
            new_reqs = self.sampling_method(
                self.rng,
                self.instance,
                current_time,
                departure_time,
                self.epoch_duration,
                self.num_requests_per_epoch[epoch],
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
        dist = self.instance.duration_matrix
        horizon = self.instance.horizon

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

    def step(self, action: Action) -> tuple[State, float, bool]:
        """
        Steps to the next state for the given action.

        Parameters
        ----------
        action: Action
            The action to take, i.e., the routes to dispatch.

        Returns
        -------
        State
            The next state.
        float
            The epoch reward for the transition.
        bool
            Whether the episode is done.

        Raises
        ------
        RuntimeError
            If the submitted action is invalid.
        """
        elapsed = perf_counter() - self.start_time_epoch
        if elapsed > self.epoch_tlim + 3:  # grace period of 3 seconds
            msg = f"Time limit exceeded: {elapsed:.1f}s > {self.epoch_tlim}s."
            warn(msg, TimeLimitWarning)

        try:
            assert not self.is_done, "Environment is finished."

            # Convert requests to epoch instance indices.
            req2idx = {r: i for i, r in enumerate(self.ep_inst.request_idx)}
            idx_sol = [[req2idx[req] for req in route] for route in action]

            # Check that all must-dispatch requests are dispatched.
            must = np.flatnonzero(self.ep_inst.must_dispatch)
            dispatched = {req for route in idx_sol for req in route}

            msg = "Not all must-dispatch requests are dispatched."
            assert set(must).issubset(dispatched), msg

            # Check that the (static) solution is feasible.
            cost = validate_static_solution(self.ep_inst, idx_sol)

        except AssertionError as error:
            self.is_done = True
            raise RuntimeError("Invalid action.") from error

        # Mark dispatched requests as dispatched.
        for route in action:
            self.req_is_dispatched[route] = True

        # Register how many primary vehicles were used.
        self.num_vehicles_used.append(len(action))

        self.final_solutions[self.current_epoch] = action
        self.final_costs[self.current_epoch] = cost

        self.current_epoch += 1
        self.current_time = self.current_epoch * self.epoch_duration
        self.is_done = self.current_epoch > self.end_epoch

        self.start_time_epoch = perf_counter()

        return self._next_observation(), -cost, self.is_done

    def _next_observation(self) -> State:
        """
        Returns the next observation. This consists of all revealed requests
        that were not dispatched in the previous epoch, and new requests.

        Returns
        -------
        State
            The next observation. If the episode is done, returns a dummy
            observation.
        """
        if self.is_done:
            return State(-1, -1, -1, None)  # type: ignore

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

        # Determine the number of primary vehicles available.
        if self.num_vehicles_per_epoch is None:
            # Assume that the number of vehicles is equal to the number of
            # requests in the instance (minus depot).
            num_available_vehicles = max(customer_idx.size - 1, 1)
        else:
            total = sum(self.num_vehicles_per_epoch[: self.current_epoch + 1])
            num_available_vehicles = total - sum(self.num_vehicles_used)

        self.ep_inst = VrpInstance(
            is_depot=self.instance.is_depot[customer_idx],
            customer_idx=customer_idx,
            request_idx=current_reqs,
            coords=self.instance.coords[customer_idx],
            demands=self.req_demand[current_reqs],
            capacity=self.instance.capacity,
            time_windows=time_windows,
            service_times=self.req_service[current_reqs],
            duration_matrix=self.instance.duration_matrix[
                np.ix_(customer_idx, customer_idx)
            ],
            must_dispatch=must_dispatch,
            release_times=self.req_release_time[current_reqs],
            num_vehicles=num_available_vehicles,
            shift_tw_early=num_available_vehicles * [departure_time],
        )

        return State(
            self.current_epoch, self.current_time, departure_time, self.ep_inst
        )

    def get_hindsight_problem(self) -> VrpInstance:
        """
        Returns the hindsight problem, which is a static VRP instance that
        represents the dynamic instance assuming perfect information.

        Returns
        -------
        VrpInstance
            The hindsight problem instance.
        """
        customer_idx = self.req_customer_idx

        if self.num_vehicles_per_epoch is None:
            num_vehicles = customer_idx.size
            shift_tw_early = num_vehicles * [0]
        else:
            shift_tw_early = []
            for epoch, num_vehicles in enumerate(self.num_vehicles_per_epoch):
                departure = epoch * self.epoch_duration + self.dispatch_margin
                shift_tw_early.extend(num_vehicles * [departure])

            num_vehicles = sum(self.num_vehicles_per_epoch)

        return VrpInstance(
            is_depot=self.instance.is_depot[customer_idx],
            coords=self.instance.coords[customer_idx],
            customer_idx=customer_idx,
            request_idx=self.req_idx,
            demands=self.req_demand,
            capacity=self.instance.capacity,
            time_windows=self.req_tw,
            service_times=self.req_service,
            duration_matrix=self.instance.duration_matrix[
                np.ix_(customer_idx, customer_idx)
            ],
            release_times=self.req_release_time,
            num_vehicles=num_vehicles,
            shift_tw_early=shift_tw_early,
        )


class TimeLimitWarning(UserWarning):
    """
    Raised when the epoch time limit is exceeded.
    """
