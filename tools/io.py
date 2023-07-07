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


def inst_to_vars(inst):
    # Instance is a dict that has the following entries:
    # - 'is_depot': boolean np.array. True for depot; False otherwise.
    # - 'coords': np.array of locations (incl. depot)
    # - 'demands': np.array of location demands (incl. depot with demand zero)
    # - 'capacity': int of vehicle capacity
    # - 'time_windows': np.array of [l, u] time windows per client (incl. depot)
    # - 'service_times': np.array of service times at each client (incl. depot)
    # - 'duration_matrix': distance matrix between clients (incl. depot)
    # - optional 'release_times': earliest possible time to leave depot
    # - optional 'latest_dispatch': latest possible time to leave depot

    # Notice that the dictionary key names are not entirely free-form: these
    # should match the argument names defined in the C++/Python bindings.
    if "release_times" in inst:
        release_times = inst["release_times"]
    else:
        release_times = np.zeros_like(inst["service_times"])

    if "latest_dispatch" in inst:
        latest_dispatch = inst["latest_dispatch"]
    else:
        # Default latest dispatch is equal to the latest depot time window
        horizon = inst["time_windows"][0][1]
        latest_dispatch = np.ones_like(inst["service_times"]) * horizon

    assert len(release_times) == len(latest_dispatch)

    return {
        "coords": inst["coords"],
        "demands": inst["demands"],
        "vehicle_cap": inst["capacity"],
        "time_windows": inst["time_windows"],
        "service_durations": inst["service_times"],
        "duration_matrix": inst["duration_matrix"],
        "release_times": release_times,
        "latest_dispatch": latest_dispatch,
    }


def read_vrptw_solution(filename, return_extra=False):
    """Reads a VRPTW solution in VRPLib format (one route per row)"""
    solution = []
    extra = {}

    for line in _readlines(filename):
        if line.startswith("Route"):
            solution.append(
                np.array(
                    [
                        int(node)
                        for node in line.split(":")[-1].strip().split(" ")
                    ]
                )
            )
        else:
            if len(line.strip().split(" ")) == 2:
                key, val = line.strip().split(" ")
                extra[key] = val

    if return_extra:
        return solution, extra
    return solution


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
        "latest_dispatch": instance["latest_dispatch"]
        if "latest_dispatch" in instance
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


def write_vrplib(
    filename, instance, name="problem", euclidean=False, is_vrptw=True
):
    # LKH/VRP does not take floats (HGS seems to do)

    coords = instance["coords"]
    demands = instance["demands"]
    is_depot = instance["is_depot"]
    duration_matrix = instance["duration_matrix"]
    capacity = instance["capacity"]
    assert (np.diag(duration_matrix) == 0).all()
    assert (demands[~is_depot] > 0).all()

    with open(filename, "w") as f:
        f.write(
            "\n".join(
                [
                    "{} : {}".format(k, v)
                    for k, v in [
                        ("NAME", name),
                        (
                            "COMMENT",
                            "ORTEC",
                        ),  # For HGS we need an extra row...
                        ("TYPE", "CVRP"),
                        ("DIMENSION", len(coords)),
                        (
                            "EDGE_WEIGHT_TYPE",
                            "EUC_2D" if euclidean else "EXPLICIT",
                        ),
                    ]
                    + (
                        []
                        if euclidean
                        else [("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")]
                    )
                    + [("CAPACITY", capacity)]
                ]
            )
        )
        f.write("\n")

        if not euclidean:
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in duration_matrix:
                f.write("\t".join(map(str, row)))
                f.write("\n")

        f.write("NODE_COORD_SECTION\n")
        f.write(
            "\n".join(
                [
                    "{}\t{}\t{}".format(i + 1, x, y)
                    for i, (x, y) in enumerate(coords)
                ]
            )
        )
        f.write("\n")

        f.write("DEMAND_SECTION\n")
        f.write(
            "\n".join(
                ["{}\t{}".format(i + 1, d) for i, d in enumerate(demands)]
            )
        )
        f.write("\n")

        f.write("DEPOT_SECTION\n")
        for i in np.flatnonzero(is_depot):
            f.write(f"{i+1}\n")
        f.write("-1\n")

        if is_vrptw:
            service_t = instance["service_times"]
            timewi = instance["time_windows"]

            # Following LKH convention
            f.write("SERVICE_TIME_SECTION\n")
            f.write(
                "\n".join(
                    [
                        "{}\t{}".format(i + 1, s)
                        for i, s in enumerate(service_t)
                    ]
                )
            )
            f.write("\n")

            f.write("TIME_WINDOW_SECTION\n")
            f.write(
                "\n".join(
                    [
                        "{}\t{}\t{}".format(i + 1, l, u)
                        for i, (l, u) in enumerate(timewi)
                    ]
                )
            )
            f.write("\n")

            if "release_times" in instance:
                release_times = instance["release_times"]

                f.write("RELEASE_TIME_SECTION\n")
                f.write(
                    "\n".join(
                        [
                            "{}\t{}".format(i + 1, s)
                            for i, s in enumerate(release_times)
                        ]
                    )
                )
                f.write("\n")

            if "latest_dispatch" in instance:
                f.write("LATEST_DISPATCH_SECTION\n")
                f.write(
                    "\n".join(
                        [
                            "{}\t{}".format(i + 1, s)
                            for i, s in enumerate(instance["latest_dispatch"])
                        ]
                    )
                )
                f.write("\n")

        f.write("EOF\n")
