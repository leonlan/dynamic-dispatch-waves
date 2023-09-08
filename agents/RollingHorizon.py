import numpy as np

from Environment import State, StaticInfo
from sampling import SamplingMethod
from static_solvers import default_solver
from VrpInstance import VrpInstance


class RollingHorizon:
    """
    The Rolling Horizon policy solves a single scenario. Based on the solution,
    it dispatches the routes that must be dispatched in the current epoch.

    Parameters
    ----------
    num_lookahead
        The number of future (i.e., lookahead) epochs to consider per scenario.
    time_limit
        The time limit for deciding on the dispatching solution.
    sampling_method
        The method to use for sampling scenarios.
    """

    def __init__(
        self,
        seed: int,
        num_lookahead: int,
        time_limit: float,
        sampling_method: SamplingMethod,
    ):
        self.seed = seed
        self.num_lookahead = num_lookahead
        self.time_limit = time_limit
        self.sampling_method = sampling_method

        self.rng = np.random.default_rng(seed)

    def act(self, info: StaticInfo, obs: State) -> list[list[int]]:
        must_dispatch = obs.epoch_instance.must_dispatch
        to_postpone = np.zeros_like(must_dispatch, dtype=bool)

        scenario = self._sample_scenario(info, obs, must_dispatch, to_postpone)
        res = default_solver(scenario, self.seed, self.time_limit)

        assert res.best.is_feasible(), "Solution is not feasible."

        solution = []
        for route in res.best.get_routes():
            num_reqs = to_postpone.size
            if any(must_dispatch[idx] for idx in route if idx < num_reqs):
                solution.append(route.visits())

        # TODO fix typing here
        return [scenario.request_idx[r].tolist() for r in solution]  # type: ignore

    def _sample_scenario(
        self,
        info: StaticInfo,
        obs: State,
        to_dispatch: np.ndarray,
        to_postpone: np.ndarray,
    ) -> VrpInstance:
        """
        Samples a VRPTW scenario instance. The scenario instance is created by
        appending the sampled requests to the current epoch instance.

        Parameters
        ----------
        info
            The static information.
        obs
            The current epoch observation.
        to_dispatch
            A boolean array where True means that the corresponding request must be
            dispatched.
        to_postpone
            A boolean array where True mean that the corresponding request must be
            postponed.
        """
        # Parameters
        current_epoch = obs.current_epoch
        next_epoch = current_epoch + 1
        epochs_left = info.end_epoch - current_epoch
        max_lookahead = min(self.num_lookahead, epochs_left)
        num_requests_per_epoch = info.num_requests_per_epoch

        static_inst = info.static_instance
        epoch_duration = info.epoch_duration
        dispatch_margin = info.dispatch_margin
        ep_inst = obs.epoch_instance
        departure_time = obs.departure_time

        # Scenario instance fields
        req_cust_idx = ep_inst.customer_idx
        req_idx = ep_inst.request_idx
        req_demand = ep_inst.demands
        req_service = ep_inst.service_times
        req_tw = ep_inst.time_windows
        req_release = ep_inst.release_times

        # Modify the release time of postponed requests: they should start
        # at the next departure time.
        next_departure_time = departure_time + epoch_duration
        req_release[to_postpone] = next_departure_time

        # Modify the dispatch time of dispatched requests: they should start
        # at the current departure time (and at time horizon otherwise).
        horizon = req_tw[0][1]
        req_dispatch = np.where(to_dispatch, departure_time, horizon)

        for epoch in range(next_epoch, next_epoch + max_lookahead):
            epoch_start = epoch * epoch_duration
            epoch_depart = epoch_start + dispatch_margin
            num_requests = num_requests_per_epoch[epoch]

            new = self.sampling_method(
                self.rng,
                static_inst,
                epoch_start,
                epoch_depart,
                epoch_duration,
                num_requests,
            )
            num_new_reqs = new["customer_idx"].size

            # Sampled request indices are negative so we can distinguish them.
            new_req_idx = -(np.arange(num_new_reqs) + 1) - len(req_idx)

            # Concatenate the new requests to the current instance requests.
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

        dist = static_inst.duration_matrix

        return VrpInstance(
            is_depot=static_inst.is_depot[req_cust_idx],
            customer_idx=req_cust_idx,
            request_idx=req_idx,
            coords=static_inst.coords[req_cust_idx],
            demands=req_demand,
            capacity=static_inst.capacity,
            time_windows=req_tw,
            service_times=req_service,
            duration_matrix=dist[req_cust_idx][:, req_cust_idx],
            release_times=req_release,
            dispatch_times=req_dispatch,
        )
