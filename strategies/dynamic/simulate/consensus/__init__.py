from .adaptive_threshold import adaptive_threshold
from .fixed_threshold import fixed_threshold

# This dict stores consensus functions that can be used to determine which
# requests to dispatch or to postpone during the simulation cycles. A consensus
# function takes as input:
#
# * cycle_idx: the current simulation cycle index
# * solution_pool: the pool of simulation solutions
# * to_dispatch: the requests that are marked dispatched so far
# * to_postpone: the requests that are marked postponed so far
# * kwargs: any additional keyword arguments taken from the configuration
#           object's consensus parameters.
#
# Using these arguments, the consensus function should return which requests
# to dispatch and which to postpone.
CONSENSUS = {
    "adaptive_threshold": adaptive_threshold,
    "fixed_threshold": fixed_threshold,
}