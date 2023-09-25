import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np

from ddwp.Environment import State, StaticInfo
from ddwp.sampling import SamplingMethod
from ddwp.static_solvers import default_solver, scenario_solver
from ddwp.VrpInstance import VrpInstance

from .consensus import CONSENSUS, ConsensusFunction
from .sample_scenario import sample_scenario


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
    dispatch_time_limit
        The time limit for solving the dispatch instance.
    sampling_method
        The method to use for sampling scenarios.
    dispatch_choice
        Choice which requests to dispatch. Either "not_postpone" or "dispatch".
        "not_postpone" means that all requests that are not postponed are
        dispatched. "dispatch" means that all requests that are marked for
        dispatch are dispatched.
    consensus
        The name of the consensus function to use.
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
        dispatch_time_limit: float,
        sampling_method: SamplingMethod,
        dispatch_choice: str,
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
        self.dispatch_time_limit = dispatch_time_limit
        self.dispatch_choice = dispatch_choice
        self.sampling_method = sampling_method
        self.consensus_func: ConsensusFunction = partial(
            CONSENSUS[consensus], **consensus_params
        )
        self.num_parallel_solve = num_parallel_solve

    def act(self, info: StaticInfo, obs: State) -> list[list[int]]:
        """
        First determines the dispatch decisions for the current epoch, then
        solves the instance of dispatched requests.
        """
        epoch_instance = obs.epoch_instance
        to_dispatch = self._determine_dispatch(info, obs)
        dispatch_instance = epoch_instance.filter(to_dispatch)

        res = default_solver(
            dispatch_instance, self.seed, self.dispatch_time_limit
        )
        routes = [route.visits() for route in res.best.get_routes()]

        return [dispatch_instance.request_idx[r].tolist() for r in routes]

    def _determine_dispatch(self, info: StaticInfo, obs: State) -> np.ndarray:
        """
        Determines which requests to dispatch in the current epoch by solving
        a set of sample scenarios and using a consensus function to determine
        which requests to dispatch or postpone. This procedure is repeated
        for a fixed number of iterations.
        """
        ep_inst = obs.epoch_instance
        ep_size = ep_inst.dimension

        # In the last epoch, all requests must be dispatched.
        if obs.current_epoch == info.end_epoch:
            return np.ones(ep_size, dtype=bool)

        to_dispatch = ep_inst.must_dispatch.copy()
        to_postpone = np.zeros(ep_size, dtype=bool)

        for _ in range(self.num_iterations):
            scenarios = [
                sample_scenario(
                    info,
                    obs,
                    self.num_lookahead,
                    self.sampling_method,
                    self.rng,
                    to_dispatch,
                    to_postpone,
                )
                for _ in range(self.num_scenarios)
            ]

            if self.num_parallel_solve == 1:
                solutions = list(map(self._solve_scenario, scenarios))
            else:
                with Pool(self.num_parallel_solve) as pool:
                    solutions = pool.map(self._solve_scenario, scenarios)

            to_dispatch, to_postpone = self.consensus_func(
                info,
                list(zip(scenarios, solutions)),
                ep_inst,
                to_dispatch,
                to_postpone,
            )

            # Stop the run early when all requests have been marked
            if ep_size - 1 == to_dispatch.sum() + to_postpone.sum():
                break

        if self.dispatch_choice == "not_postpone":
            return ~to_postpone | ep_inst.is_depot
        else:
            return to_dispatch | ep_inst.is_depot

    def _solve_scenario(self, instance: VrpInstance) -> list[list[int]]:
        """
        Solves a single scenario instance, returning the solution.
        """
        result = scenario_solver(instance, self.seed, self.scenario_time_limit)

        if not result.best.is_feasible():
            raise RuntimeError("Infeasible scenario solution!")

        return [route.visits() for route in result.best.get_routes()]
