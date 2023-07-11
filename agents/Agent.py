from typing import Protocol

import numpy as np


class Agent(Protocol):
    def __init__(self, seed, **kwargs):
        ...

    def act(self, observation: dict, info: dict) -> np.ndarray:
        """
        Returns a dispatch action in response to a given an observation and
        info.
        """
