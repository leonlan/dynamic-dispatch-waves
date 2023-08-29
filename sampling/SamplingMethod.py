from typing import Protocol

from numpy.random import Generator


class SamplingMethod(Protocol):
    def __call__(
        self,
        rng: Generator,
        instance: dict,
        current_time: int,
        departure_time: int,
        epoch_duration: int,
        num_requests: int,
        **kwargs,
    ) -> dict:
        ...
