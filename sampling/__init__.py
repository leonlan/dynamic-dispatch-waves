from functools import partial

from .custom_time_windows import custom_time_windows
from .euro_neurips import euro_neurips
from .SamplingMethod import SamplingMethod

SAMPLING_METHODS: dict[str, SamplingMethod] = {
    "DL2": partial(custom_time_windows, tw_type="deadlines", tw_width=2),
    "DL4": partial(custom_time_windows, tw_type="deadlines", tw_width=4),
    "DL8": partial(custom_time_windows, tw_type="deadlines", tw_width=8),
    "TW2": partial(custom_time_windows, tw_type="time_windows", tw_width=2),
    "TW4": partial(custom_time_windows, tw_type="time_windows", tw_width=4),
    "TW8": partial(custom_time_windows, tw_type="time_windows", tw_width=8),
    "euro_neurips": euro_neurips,
}
