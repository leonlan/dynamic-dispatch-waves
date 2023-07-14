from typing import Protocol

import numpy as np


class Agent(Protocol):
    def __init__(self, seed, **kwargs):
        ...

    def act(self, static_info: dict, observation: dict) -> np.ndarray:
        """
        Returns a dispatch action for the given static info and observation.
        """
