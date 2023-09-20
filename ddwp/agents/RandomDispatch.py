import numpy as np

from ddwp.Environment import State, StaticInfo
from ddwp.static_solvers import default_solver


class _RandomDispatch:
    def __init__(self, seed: int, prob: float):
        if not 0 <= prob <= 1:
            raise ValueError(f"prob must be in [0, 1], got {prob}.")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.prob = prob

    def act(self, info: StaticInfo, obs: State) -> list[list[int]]:
        """
        Randomly dispatches requests (that are not must-dispatch) with
        probability ``prob``.
        """
        epoch_instance = obs.epoch_instance
        sample_shape = epoch_instance.must_dispatch.shape

        to_dispatch = (
            (self.rng.random(sample_shape) < self.prob)
            | epoch_instance.is_depot
            | epoch_instance.must_dispatch
        )
        dispatch_instance = epoch_instance.filter(to_dispatch)

        res = default_solver(dispatch_instance, self.seed, info.epoch_tlim)
        routes = [route.visits() for route in res.best.get_routes()]

        return [dispatch_instance.request_idx[r].tolist() for r in routes]


class GreedyDispatch(_RandomDispatch):
    def __init__(self, seed: int):
        super().__init__(seed, prob=1)


class LazyDispatch(_RandomDispatch):
    def __init__(self, seed: int):
        super().__init__(seed, prob=0)


class UniformDispatch(_RandomDispatch):
    def __init__(self, seed: int):
        super().__init__(seed, prob=0.5)
