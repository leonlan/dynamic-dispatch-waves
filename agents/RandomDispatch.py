import numpy as np

from static_solvers import default_solver
from utils import filter_instance


class _RandomDispatch:
    def __init__(self, seed: int, prob: float):
        if not 0 <= prob <= 1:
            raise ValueError(f"prob must be in [0, 1], got {prob}.")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.prob = prob

    def act(self, info, obs) -> np.ndarray:
        """
        Randomly dispatches not "must dispatch" requests with probability ``prob``.
        """
        instance = obs["epoch_instance"]
        to_dispatch = (
            instance["is_depot"]
            | instance["must_dispatch"]
            | (self.rng.random(instance["must_dispatch"].shape) < self.prob)
        )

        dispatch_instance = filter_instance(instance, to_dispatch)
        res = default_solver(dispatch_instance, self.seed, info["epoch_tlim"])
        routes = [route.visits() for route in res.best.get_routes()]
        ep_sol = [
            dispatch_instance["request_idx"][route].tolist()
            for route in routes
        ]
        return ep_sol


class GreedyDispatch(_RandomDispatch):
    def __init__(self, seed: int):
        super().__init__(seed, prob=1)


class LazyDispatch(_RandomDispatch):
    def __init__(self, seed: int):
        super().__init__(seed, prob=0)


class UniformDispatch(_RandomDispatch):
    def __init__(self, seed: int):
        super().__init__(seed, prob=0.5)
