import numpy as np
import vrplib


def _readlines(filename):
    try:
        with open(filename, "r") as f:
            return f.readlines()
    except:  # noqa: E722
        with open(filename, "rb") as f:
            return [
                line.decode("utf-8", errors="ignore").strip()
                for line in f.readlines()
            ]


def read_vrplib(filename, instance_format="vrplib"):
    """
    Read a VRPLIB instance from file and return an `instance` dict, containing
    - 'is_depot': boolean np.array. True for depot; False otherwise.
    - 'coords': np.array of locations (incl. depot)
    - 'demands': np.array of location demands (incl. depot with demand zero)
    - 'capacity': int of vehicle capacity
    - 'time_windows': np.array of [l, u] time windows per client (incl. depot)
    - 'service_times': np.array of service times at each client (incl. depot)
    - 'duration_matrix': distance matrix between clients (incl. depot)
    """
    instance = vrplib.read_instance(filename, instance_format=instance_format)
    n_locations = instance.get("dimension", len(instance["node_coord"]))
    horizon = instance["time_window"][0][1]  # depot latest tw

    return {
        "is_depot": np.array([1] + [0] * (n_locations - 1), dtype=bool),
        "coords": instance["node_coord"],
        "demands": instance["demand"],
        "capacity": instance["capacity"],
        "time_windows": instance["time_window"],
        "service_times": instance["service_time"],
        "duration_matrix": instance["edge_weight"].astype(int),
        "release_times": instance["release_time"]
        if "release_time" in instance
        else np.zeros(n_locations, dtype=int),
        "dispatch_times": instance["dispatch_time"]
        if "dispatch_time" in instance
        else np.ones(n_locations, dtype=int) * horizon,
    }


def tabulate(headers, rows) -> str:
    # These lengths are used to space each column properly.
    lengths = [len(header) for header in headers]

    for row in rows:
        for idx, cell in enumerate(row):
            lengths[idx] = max(lengths[idx], len(str(cell)))

    header = [
        "  ".join(f"{h:<{l}s}" for l, h in zip(lengths, headers)),
        "  ".join("-" * l for l in lengths),
    ]

    content = [
        "  ".join(f"{str(c):>{l}s}" for l, c in zip(lengths, row))
        for row in rows
    ]

    return "\n".join(header + content)
