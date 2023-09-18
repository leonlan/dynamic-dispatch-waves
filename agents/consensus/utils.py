import numpy as np

Scenario = tuple[dict, list[list[int]]]


def select_postpone_on_threshold(
    scenarios: list[Scenario],
    old_dispatch: np.ndarray,
    old_postpone: np.ndarray,
    postpone_threshold: float,
):
    """
    Returns a boolean array indicating which requests should be postponed:
    a request is postponed if it is postponed in at least `postpone_threshold`
    fraction of the scenarios.
    """
    dispatch_count = get_dispatch_count(scenarios, old_dispatch, old_postpone)
    postpone_count = len(scenarios) - dispatch_count
    postpone_count[0] = 0  # do not postpone depot

    return postpone_count >= postpone_threshold * len(scenarios)


def select_dispatch_on_threshold(
    scenarios: list[Scenario],
    old_dispatch: np.ndarray,
    old_postpone: np.ndarray,
    dispatch_threshold: float,
) -> np.ndarray:
    """
    Returns a boolean array indicating which requests should be dispatched:
    a request is dispatched if it is dispatched in at least `dispatch_threshold`
    fraction of the scenarios.
    """
    dispatch_count = get_dispatch_count(scenarios, old_dispatch, old_postpone)
    return dispatch_count >= dispatch_threshold * len(scenarios)


def get_dispatch_count(
    scenarios: list[Scenario], old_dispatch, old_postpone
) -> np.ndarray:
    """
    Returns a vector containing the number of scenarios in which each request
    was dispatched.
    """
    mat = get_dispatch_matrix(scenarios, old_dispatch, old_postpone)
    return np.sum(mat, axis=0)


def get_dispatch_matrix(
    scenarios: list[Scenario], to_dispatch: np.ndarray, to_postpone: np.ndarray
) -> np.ndarray:
    """
    Returns the dispatch matrix. Each row in this matrix corresponds to the
    dispatch action for a given scenario, where 1 means that the request was
    dispatched in this scenario, and 0 means it was postponed.
    """
    num_reqs = to_dispatch.size  # including depot
    dispatch_matrix = np.zeros((len(scenarios), num_reqs), dtype=int)

    for scenario_idx, (instance, solution) in enumerate(scenarios):
        for route in solution:
            if is_dispatched_route(instance, route, to_dispatch, to_postpone):
                dispatch_matrix[scenario_idx, route] += 1

    return dispatch_matrix


def is_dispatched_route(instance, route, to_dispatch, to_postpone):
    """
    Determines whether or not the passed route was dispatched in the scenario
    instance. A route is considered dispatched if:
    * at least one the requests on the route is already marked dispatched, or
    * the route does not contain postponed or sampled requests AND the route
      cannot be postponed to the next epoch without violating feasibility.
      In other words, the "route latest start" is less than the next epoch
      start time.
    """
    num_reqs = to_dispatch.size  # Sampled request indices are >= `num_reqs`.

    if any(to_dispatch[idx] for idx in route if idx < num_reqs):
        # At least one request on the route was already dispatched.
        return True

    # Sampled request indices are larger than the number of requests
    has_sampled_reqs = any(idx >= num_reqs for idx in route)
    has_postponed_reqs = any(
        to_postpone[idx] for idx in route if idx < num_reqs
    )

    # Postpone routes that contain sampled or postponed requests.
    if has_sampled_reqs or has_postponed_reqs:
        return False

    # Dispatch routes if they cannot be postponed to the next epoch.
    # This is the case when the route has no must-dispatch, sampled
    # or postponed requests.
    return not can_postpone_route(instance, route)


def can_postpone_route(instance, route):
    """
    Checks if the route can be postponed to the next epoch, i.e., the route
    remains feasible if it starts one epoch duration later.

    # TODO Replace this with route.earliest_start() and route.slack() in PyVRP.
    See https://github.com/PyVRP/PyVRP/pull/241.
    """
    tour = [0] + route + [0]
    tws = instance.time_windows
    dist = instance.duration_matrix
    service = instance.service_times

    # HACK The next epoch time is inferred from the smallest non-zero release
    # times. If all release times are zero, then this scenario contains no
    # sampled requests, and we assume that the route must be dispatched.
    release_times = instance.release_times
    non_zero_release = np.flatnonzero(release_times)

    if non_zero_release.size == 0:
        return False

    next_epoch_time = np.min(release_times[non_zero_release])
    current_time = next_epoch_time

    for idx in range(len(tour) - 1):
        pred, succ = tour[idx], tour[idx + 1]

        earliest_arrival, latest_arrival = tws[succ]
        arrival_time = current_time + dist[pred, succ]
        current_time = max(arrival_time, earliest_arrival)

        if current_time <= latest_arrival:
            return False

        current_time += service[succ]

    return True


def verify_action(old_dispatch, old_postpone, new_dispatch, new_postpone):
    """
    Checks that (1) the old actions are a subset of the new actions, (2) the
    depot is never part of any action, and (3) a request is not dispatch and
    postpone at the same time.
    """
    # Preserve old actions
    assert np.all(old_dispatch <= new_dispatch)
    assert np.all(old_postpone <= new_postpone)

    # Depot is never part of any action
    assert not new_dispatch[0]
    assert not new_postpone[0]

    # A request cannot be dispatched and postponed at the same time
    assert not np.any(new_dispatch & new_postpone)
