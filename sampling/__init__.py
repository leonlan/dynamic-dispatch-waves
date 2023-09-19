from functools import partial

from .custom_time_windows import custom_time_windows
from .euro_neurips import euro_neurips
from .SamplingMethod import SamplingMethod

SAMPLING_METHODS: dict[str, SamplingMethod] = {
    f"{name}{tw_width}": partial(custom_time_windows, tw_type, tw_width)
    for name, tw_type in [("DL", "deadlines"), ("TW", "time_windows")]
    for tw_width in range(1, 9)
}
SAMPLING_METHODS["euro_neurips"] = euro_neurips
