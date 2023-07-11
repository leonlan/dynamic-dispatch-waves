from typing import Protocol

import numpy as np


class Agent(Protocol):
    def act(self, observation: dict) -> np.ndarray:
        """
        Returns a dispatch action given an observation.
        """
