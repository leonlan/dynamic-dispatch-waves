import numpy as np


def hamming_distance(solutions, to_dispatch, to_postpone, **kwargs):
    """
    Selects the solution with the smallest average Hamming distance w.r.t.
    the other solutions. The requests of this solution are marked dispatch.
    """
    n_simulations = len(solutions)
    ep_size = to_dispatch.size
    dispatch_actions = np.zeros((n_simulations, ep_size), dtype=int)

    for sim_idx, sol in enumerate(solutions):
        for route in sol:
            if any(to_dispatch[idx] for idx in route if idx < ep_size):
                dispatch_actions[sim_idx, route] += 1

    # Mean absolute error a.k.a. average Hamming distance
    mae = (abs(dispatch_actions - dispatch_actions.mean(axis=0))).mean(axis=1)
    to_dispatch = dispatch_actions[mae.argsort()[0]].astype(bool)

    return to_dispatch, to_postpone
