import numpy as np

from .Agent import Agent


class _RandomDispatch(Agent):
    def __init__(self, seed: int, prob: float):
        if not 0 <= prob <= 1:
            raise ValueError(f"prob must be in [0, 1], got {prob}.")

        self.rng = np.random.default_rng(seed)
        self.prob = prob

    def act(self, obs, info) -> np.ndarray:
        """
        Randomly dispatches not "must dispatch" requests with probability ``prob``.
        """
        instance = obs["epoch_instance"]
        to_dispatch = (
            instance["is_depot"]
            | instance["must_dispatch"]
            | (self.rng.random(instance["must_dispatch"].shape) < self.prob)
        )
        return to_dispatch


class GreedyDispatch(_RandomDispatch):
    def __init__(self, seed: int):
        super().__init__(seed, prob=1)


class LazyDispatch(_RandomDispatch):
    def __init__(self, seed: int):
        super().__init__(seed, prob=0)


class UniformDispatch(_RandomDispatch):
    def __init__(self, seed: int):
        super().__init__(seed, prob=0.5)
