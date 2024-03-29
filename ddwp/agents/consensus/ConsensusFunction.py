from typing import Protocol

import numpy as np

from ddwp.Environment import StaticInfo
from ddwp.VrpInstance import VrpInstance


class ConsensusFunction(Protocol):
    """
    Protocol defining a consensus function.
    """

    def __call__(
        self,
        info: StaticInfo,
        scenarios: list[tuple[VrpInstance, list[list[int]]]],
        instance: VrpInstance,
        to_dispatch: np.ndarray,
        to_postpone: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Determines which requests to dispatch and which to postpone.

        Parameters
        ----------
        info
            The static information about the problem.
        scenarios
            The set of scenarios and their solutions.
        instance
            The epoch instance.
        to_dispatch
            The requests that are marked dispatched so far.
        to_postpone
            The requests that are marked postponed so far.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Boolean arrays indicating which requests to dispatch and which to
            postpone, respectively.
        """
        ...
