from functools import partial
from typing import Callable

from .new_sampler import new_sampler
from .sample_epoch_requests import sample_epoch_requests

SAMPLING_METHODS: dict[str, Callable] = {
    "dl2": partial(
        new_sampler, time_window_type="deadlines", time_window_width=2
    ),
    "dl4": partial(
        new_sampler, time_window_type="deadlines", time_window_width=4
    ),
    "dl8": partial(
        new_sampler, time_window_type="deadlines", time_window_width=8
    ),
    "tw2": partial(
        new_sampler, time_window_type="time_windows", time_window_width=2
    ),
    "tw4": partial(
        new_sampler, time_window_type="time_windows", time_window_width=4
    ),
    "tw8": partial(
        new_sampler, time_window_type="time_windows", time_window_width=8
    ),
    "euro_neurips": sample_epoch_requests,
}
