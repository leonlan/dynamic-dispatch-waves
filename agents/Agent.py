from typing import Protocol

import numpy as np


class Agent(Protocol):
    def __init__(self, seed, **kwargs):
        ...

    def act(self, observation: dict, static_info: dict) -> np.ndarray:
        """
        Returns a dispatch action for a given observation and static info.
        """
