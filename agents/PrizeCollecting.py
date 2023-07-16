import numpy as np

from .IterativeConditionalDispatch import IterativeConditionalDispatch


class PrizeCollecting(IterativeConditionalDispatch):
    """
    A prize-collecting strategy built on top of ICD. This strategy also uses
    simulation, but "unfixes" some of the dispatch/postpone decisions of the
    concensus function. The unfixed clients are given prizes instead, and a
    prize-collecting VRP is solved to determine whether to dispatch the unfixed
    clients *now*, or postpone them to the next epoch.
    """

    def __init__(
        self,
        seed: int,
        num_iterations: int,
        num_lookahead: int,
        num_scenarios: int,
        scenario_time_limit: float,
        fix_threshold: float,
        lamda: float,
        num_parallel_solve: int = 1,
    ):
        if not (0 <= fix_threshold <= 1):
            raise ValueError("fix_threshold must be in [0, 1].")

        if lamda < 0:
            raise ValueError("lamda < 0 not understood.")

        super().__init__(
            seed,
            num_iterations,
            num_lookahead,
            num_scenarios,
            scenario_time_limit,
            "fixed_threshold",
            {
                "postpone_thresholds": [1 - fix_threshold],
                "dispatch_thresholds": [fix_threshold],
            },
            num_parallel_solve,
        )

        self.fix_threshold = fix_threshold
        self.lamda = lamda

    def act(self, static_info: dict, observation: dict) -> np.ndarray:
        pass
