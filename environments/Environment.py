from abc import ABC, abstractmethod
from typing import Any, Optional

State = dict[str, Any]
Action = list[list[int]]
Info = dict[str, Any]


class Environment(ABC):
    """
    Base class describing an environment for the DDWP.
    """

    @abstractmethod
    def reset(self) -> tuple[State, Info]:
        """
        Resets the environment. This returns the initial state of the
        environment and static information about the environment.
        """
        pass

    @abstractmethod
    def step(
        self, action: Action
    ) -> tuple[Optional[State], float, bool, Info]:
        """
        Steps to the next state for the given action.

        Parameters
        ----------
        action: Action
            The action to take.

        Returns
        -------
        Optional[State]
            The next state. If the action is invalid, or when the episode is
            done, this should return ``None``.
        reward: float
            The reward for the transition. If the action is invalid, this
            should return ``float("inf")``.
        done: bool
            Whether the episode is done.
        info: Info
            Success information about the step.
        """
        pass

    @abstractmethod
    def get_hindsight_problem(self) -> State:
        """
        Returns the hindsight problem, which is a static VRP instance that
        represents the dynamic instance with perfect information.

        Returns
        -------
        State
            A hindsight problem.
        """
        pass
