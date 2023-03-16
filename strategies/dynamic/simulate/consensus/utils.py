import numpy as np


def get_dispatch_count(scenarios, to_dispatch, to_postpone):
    """
    Computes the dispatch counts for the given solved scenarios.
    """
    dispatch_matrix = get_dispatch_matrix(scenarios, to_dispatch, to_postpone)
    return dispatch_matrix.sum(axis=0)


def get_dispatch_matrix(scenarios, to_dispatch, to_postpone):
    """
    Returns a matrix, where each row corresponds to the scenario action. The
    scenario action is a binary vector, where 1 means that the request was
    dispatched in this scenario, and 0 means it is postponed.
    """
    n_reqs = to_dispatch.size  # including depot
    dispatch_matrix = np.zeros((len(scenarios), n_reqs), dtype=int)

    for scenario_idx, (inst, sol) in enumerate(scenarios):
        for route in sol:
            if is_dispatched(inst, route, to_dispatch, to_postpone):
                dispatch_matrix[scenario_idx, route] += 1

    return dispatch_matrix


def is_dispatched(instance, route, to_dispatch, to_postpone):
    """
    Determines whether the passed-in route was a dispatched route in the
    simulations or not. A route is considered dispatched w.r.t. the current
    instance if:
    - at least one the requests is marked dispatched, or
    - the route does not contain postponed or simulated requests and the route
      cannot be postponed to the next epoch without violating feasibility.
    """
    n_reqs = to_dispatch.size

    has_to_dispatch = any(to_dispatch[idx] for idx in route if idx < n_reqs)

    if has_to_dispatch:
        return True

    has_postponed_reqs = any(to_postpone[idx] for idx in route if idx < n_reqs)
    has_simulated_reqs = any(idx >= n_reqs for idx in route)

    if has_postponed_reqs or has_simulated_reqs:
        return False

    return not can_postpone_route(instance, route)


def can_postpone_route(instance, route):
    """
    Checks if the route can be postponed to the next epoch, i.e., the route
    remains feasible if it starts one epoch duration later.
    """
    tour = [0] + route + [0]
    tws = instance["time_windows"]
    dist = instance["duration_matrix"]
    service = instance["service_times"]

    # HACK The next epoch time is inferred from the smallest non-zero release
    # times. We can also infer this from the environment.
    release_times = instance["release_times"]
    current_time = np.min(release_times[np.nonzero(release_times)])

    for idx in range(len(tour) - 1):
        pred, succ = tour[idx], tour[idx + 1]

        earliest_arrival, latest_arrival = tws[succ]
        arrival_time = current_time + dist[pred, succ]
        current_time = max(arrival_time, earliest_arrival)

        if current_time <= latest_arrival:
            return False

        current_time += service[succ]

    return True
