def is_dispatched(route, to_dispatch):
    """
    Returns true if the route contains requests that are marked `to_dispatch`.
    """
    return any(to_dispatch[idx] for idx in route if idx < to_dispatch.size)


def is_postponed(route, to_postpone):
    """
    Returns true if the route contains requests that are marked `to_postpone`
    or sampled requests.
    """
    ep_size = to_postpone.size
    return any(to_postpone[idx] for idx in route if idx < ep_size) or any(
        idx >= ep_size for idx in route
    )
