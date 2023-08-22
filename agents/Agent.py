from typing import Protocol


class Agent(Protocol):
    """
    Protocol describing an agent.
    """

    def act(self, static_info: dict, observation: dict) -> list[list[int]]:
        """
        Returns a routing solution for the current epoch.

        Parameters
        ----------
        static_info
            The static information about the environment.
        observation
            The current epoch state.

        Returns
        -------
        list[list[int]]
            The dispatched routes, where the indices correspond to request
            indices.
        """
        ...
