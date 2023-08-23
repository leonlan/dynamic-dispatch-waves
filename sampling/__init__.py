from functools import partial
from typing import Callable

from .custom_time_windows import custom_time_windows
from .euro_neurips import euro_neurips

SAMPLING_METHODS: dict[str, Callable] = {
    "dl2": partial(custom_time_windows, tw_type="deadlines", tw_width=2),
    "dl4": partial(custom_time_windows, tw_type="deadlines", tw_width=4),
    "dl8": partial(custom_time_windows, tw_type="deadlines", tw_width=8),
    "tw2": partial(custom_time_windows, tw_type="time_windows", tw_width=2),
    "tw4": partial(custom_time_windows, tw_type="time_windows", tw_width=4),
    "tw8": partial(custom_time_windows, tw_type="time_windows", tw_width=8),
    "euro_neurips": euro_neurips,
}
