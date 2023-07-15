from typing import Callable

from .fixed_threshold import fixed_threshold
from .hamming_distance import hamming_distance

# This dict stores consensus functions that can be used to determine which
# requests to dispatch or to postpone during the scenario iterations. A
# consensus function takes as input:
#
# * iteration_idx: the current sampling iteration index
# * scenarios: the set of of scenario instances and solutions
# * to_dispatch: the requests that are marked dispatched so far
# * to_postpone: the requests that are marked postponed so far
# * kwargs: any additional keyword arguments taken from the configuration
#           object's consensus parameters.
#
# Using these arguments, the consensus function should return which requests
# to dispatch and which to postpone.
CONSENSUS: dict[str, Callable] = {
    "fixed_threshold": fixed_threshold,
    "hamming_distance": hamming_distance,
}
