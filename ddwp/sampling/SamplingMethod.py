from typing import Protocol

from numpy.random import Generator

from ddwp.VrpInstance import VrpInstance


class SamplingMethod(Protocol):
    """
    Protocol for sampling methods.
    """

    def __call__(
        self,
        rng: Generator,
        instance: VrpInstance,
        current_time: int,
        departure_time: int,
        epoch_duration: int,
        num_requests: int,
    ) -> dict:
        """
        Samples a set of requests from a base instance.

        Parameters
        ----------
        rng
            Random number generator.
        instance
            Base static VRP instance.
        current_time
            Current epoch time.
        departure_time
            Departure time of the vehicle (i.e., the request release time).
        epoch_duration
            Duration of the epoch.
        num_requests
            Number of requests to sample.

        Returns
        -------
        dict
            Data of sampled requests, including the customer indices, time windows,
            demands, service times, and release times.
        """
