from functools import partial, reduce

import numpy as np


def _as_tuple(x):
    """Convert non-tuple input to being a single element of a tuple"""
    return (x if isinstance(x, tuple) else x,)


def _merge_nodes(inst, i, j):
    """Merge two nodes of a node-view instance in place"""
    n = inst.pop(i)
    m = inst.pop(j)

    k = _as_tuple(i) + _as_tuple(j)

    # Use the concatenation scheme of Vidal 2013 to merge two nodes
    min_time_between = n["service_times"] + n["duration_to"][j]
    min_waiting_time = max(
        m["time_windows"][0] - min_time_between - n["time_windows"][1], 0
    )
    min_time_warping = max(
        n["time_windows"][0] + min_time_between - m["time_windows"][1], 0
    )

    # Merging nodes isn't allowed if the concatenation is certain to violate time windows
    assert min_time_warping == 0

    tw_start = (
        max(m["time_windows"][0] - min_time_between, n["time_windows"][0])
        - min_waiting_time
    )
    tw_close = (
        min(m["time_windows"][1] - min_time_between, n["time_windows"][1])
        + min_time_warping
    )

    inst[k] = {
        "is_depot": n["is_depot"] | m["is_depot"],
        "customer_idx": _as_tuple(n["customer_idx"])
        + _as_tuple(m["customer_idx"]),
        "request_idx": _as_tuple(n["request_idx"])
        + _as_tuple(m["request_idx"]),
        "coords": n["coords"],
        "demands": n["demands"] + m["demands"],
        "time_windows": np.array([tw_start, tw_close]),
        "service_times": n["service_times"]
        + m["service_times"]
        + n["duration_to"][j]
        + min_waiting_time,
        "must_dispatch": n["must_dispatch"] | m["must_dispatch"],
        "capacity": n["capacity"],
        "duration_to": m["duration_to"],
    }

    # Update the travel durations from all (other) nodes towards the old & new current nodes
    for key in inst:
        inst[key]["duration_to"][k] = inst[key]["duration_to"].pop(i)
        inst[key]["duration_to"].pop(j)

    return k


def __get_dtype(key):
    objects = ["customer_idx", "request_idx", "coords"]
    return object if key in objects else None


def __get_attributes(inst, keys):
    return {
        k: np.array([inst[i][k] for i in inst], dtype=__get_dtype(k))
        for k in keys
        if k not in ["capacity", "duration_to"]
    }


def __get_duration_matrix(inst):
    return np.array([[inst[n]["duration_to"][m] for m in inst] for n in inst])


def _to_attr_view(inst):
    """
    Converts an instance of nested node-attr-value pairs to an instance as dictionary of attr-array pairs
    """
    first = inst[next(iter(inst))]

    return {
        **__get_attributes(inst, first.keys()),
        "capacity": first["capacity"],
        "duration_matrix": __get_duration_matrix(inst),
    }


def __get_node_attributes(inst, i):
    return {
        k: inst[k][i] for k in inst if k not in ["capacity", "duration_matrix"]
    }


def __get_node_durations(inst, i, n):
    return dict(zip(range(n), inst["duration_matrix"][i]))


def _to_node_view(inst):
    """
    Converts an instance as dictionary of attr-array pairs to an instance of nested node-attr-value pairs
    """
    n = inst["request_idx"].size
    return {
        i: {
            **__get_node_attributes(inst, i),
            "capacity": inst["capacity"],
            "duration_to": __get_node_durations(inst, i, n),
        }
        for i in range(n)
    }


def extract_subsequences(sequence, lmin, lmax):
    """
    Extracts all subsequences of lengths [lmin, lmax] from the passed-in sequence.
    """
    n = len(sequence)

    for l in range(lmin, min(lmax, n) + 1):
        for subsequence in zip(*(sequence[i:] for i in range(l))):
            yield subsequence


def fix_subsequences(inst, subsequences):
    """Create a new instance in which all subsequences are concatenated to a single node"""
    inst = _to_node_view(inst)

    for subsequence in subsequences:
        reduce(partial(_merge_nodes, inst), subsequence)

    return _to_attr_view(inst)
