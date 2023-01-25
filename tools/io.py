import numpy as np


def _readlines(filename):
    try:
        with open(filename, "r") as f:
            return f.readlines()
    except:
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

    return dict(
        coords=inst["coords"],
        demands=inst["demands"],
        vehicle_cap=inst["capacity"],
        time_windows=inst["time_windows"],
        service_durations=inst["service_times"],
        duration_matrix=inst["duration_matrix"],
        release_times=release_times,
        latest_dispatch=latest_dispatch,
    )


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


def read_vrplib(filename, rounded=True):
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
    loc = []
    demand = []
    mode = ""
    capacity = None
    edge_weight_type = None
    edge_weight_format = None
    duration_matrix = []
    service_t = []
    timewi = []
    release_times = []
    latest_dispatch = []
    with open(filename, "r") as f:

        for line in f:
            line = line.strip(" \t\n")
            if line == "":
                continue
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(" : ")[1])
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(" : ")[1]
            elif line.startswith("EDGE_WEIGHT_FORMAT"):
                edge_weight_format = line.split(" : ")[1]
            elif line == "NODE_COORD_SECTION":
                mode = "coord"
            elif line == "DEMAND_SECTION":
                mode = "demand"
            elif line == "DEPOT_SECTION":
                mode = "depot"
            elif line == "EDGE_WEIGHT_SECTION":
                mode = "edge_weights"
                assert edge_weight_type == "EXPLICIT"
                assert edge_weight_format == "FULL_MATRIX"
            elif line == "TIME_WINDOW_SECTION":
                mode = "time_windows"
            elif line == "SERVICE_TIME_SECTION":
                mode = "service_t"
            elif line == "RELEASE_TIME_SECTION":
                mode = "release_time"
            elif line == "LATEST_DISPATCH_SECTION":
                mode = "latest_dispatch"
            elif line == "EOF":
                break
            elif mode == "coord":
                (
                    node,
                    x,
                    y,
                ) = (
                    line.split()
                )  # Split by whitespace or \t, skip duplicate whitespace
                node = int(node)
                x, y = (int(x), int(y)) if rounded else (float(x), float(y))

                if node == 1:
                    depot = (x, y)
                else:
                    assert (
                        node == len(loc) + 2
                    )  # 1 is depot, 2 is 0th location
                    loc.append((x, y))
            elif mode == "demand":
                node, d = [int(v) for v in line.split()]
                if node == 1:
                    assert d == 0
                demand.append(d)
            elif mode == "edge_weights":
                duration_matrix.append(
                    list(map(int if rounded else float, line.split()))
                )
            elif mode == "service_t":
                node, t = line.split()
                node = int(node)
                t = int(t) if rounded else float(t)
                if node == 1:
                    assert t == 0
                assert node == len(service_t) + 1
                service_t.append(t)
            elif mode == "time_windows":
                node, l, u = line.split()
                node = int(node)
                l, u = (int(l), int(u)) if rounded else (float(l), float(u))
                assert node == len(timewi) + 1
                timewi.append([l, u])
            elif mode == "release_time":
                node, release_time = line.split()
                release_time = int(release_time)
                release_times.append(release_time)
            elif mode == "latest_dispatch":
                node, dispatch_time = line.split()
                dispatch_time = int(dispatch_time)
                latest_dispatch.append(dispatch_time)

    horizon = timewi[0][1]  # time horizon, i.e., depot latest tw
    return {
        "is_depot": np.array([1] + [0] * len(loc), dtype=bool),
        "coords": np.array([depot] + loc),
        "demands": np.array(demand),
        "capacity": capacity,
        "time_windows": np.array(timewi),
        "service_times": np.array(service_t),
        "duration_matrix": np.array(duration_matrix)
        if len(duration_matrix) > 0
        else None,
        "release_times": np.array(release_times)
        if release_times
        else np.zeros(len(loc) + 1, dtype=int),
        "latest_dispatch": np.array(latest_dispatch)
        if latest_dispatch
        else np.ones(len(loc) + 1, dtype=int) * horizon,
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

