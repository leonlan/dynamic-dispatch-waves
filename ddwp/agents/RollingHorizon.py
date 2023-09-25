import numpy as np

from ddwp.Environment import State, StaticInfo
from ddwp.sampling import SamplingMethod
from ddwp.static_solvers import default_solver

from .sample_scenario import sample_scenario


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

        scenario = sample_scenario(
            info,
            obs,
            self.num_lookahead,
            self.sampling_method,
            self.rng,
            must_dispatch,
            to_postpone,
        )
        res = default_solver(scenario, self.seed, self.time_limit)

        if not res.best.is_feasible():
            raise ValueError("Scenario solution is not feasible.")

        solution = []
        for route in res.best.get_routes():
            num_reqs = to_postpone.size
            if any(must_dispatch[idx] for idx in route if idx < num_reqs):
                solution.append(route.visits())

        return [scenario.request_idx[r].tolist() for r in solution]
