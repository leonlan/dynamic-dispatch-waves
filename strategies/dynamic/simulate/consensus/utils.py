def is_dispatched(instance, route, to_dispatch, to_postpone):
    """
    Returns true if the route contains requests that are marked `to_dispatch`,
    or if the route, consisting of epoch requests, cannot be postponed to the
    next epoch.
    """
    n_reqs = to_dispatch.size

    # TODO document this
    is_to_postpone = any(to_postpone[idx] for idx in route if idx < n_reqs)
    if is_to_postpone:
        return False

    is_to_dispatch = any(to_dispatch[idx] for idx in route if idx < n_reqs)
    only_ep_reqs = all(idx < n_reqs for idx in route)
    cannot_postpone = not can_postpone_route(instance, route)

    return is_to_dispatch or (only_ep_reqs and cannot_postpone)


def can_postpone_route(instance, route):
    """
    Checks if the route can be postponed to the next epoch, i.e., the route
    remains feasible if it starts one epoch duration later.
    """
    tour = [0] + route + [0]
    tws = instance["time_windows"]
    dist = instance["duration_matrix"]
    service = instance["service_times"]

    current_time = (
        3600  # TODO This should be the epoch duration from the environment
    )

    for idx in range(len(tour) - 1):
        pred, succ = tour[idx], tour[idx + 1]

        earliest_arrival, latest_arrival = tws[succ]
        arrival_time = current_time + dist[pred, succ]
        current_time = max(arrival_time, earliest_arrival)

        if current_time <= latest_arrival:
            return False

        current_time += service[succ]

    return True
