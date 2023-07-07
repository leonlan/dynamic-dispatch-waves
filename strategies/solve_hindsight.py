from numbers import Number

import numpy as np
from pyvrp import Client, CostEvaluator, Model, ProblemData, VehicleType
from pyvrp.stop import MaxRuntime

_INT_MAX = np.iinfo(np.int32).max


def solve_hindsight(env, solver_seed: int):
    """
    Solve the dynamic VRPTW problem using the oracle strategy, i.e., the
    problem is solved as static VRPTW with release dates using the information
    that is known in hindsight. The found solution is then submitted to the
    environment. The given seed is passed to the static solver.
    """
    observation, info = env.reset()
    hindsight_inst = env.get_hindsight_problem()

    model = Model.from_data(instance2data(hindsight_inst))
    res = model.solve(MaxRuntime(info["epoch_tlim"]), seed=solver_seed)

    best = res.best
    routes = [route.visits() for route in best.get_routes() if route]
    observation, _ = env.reset()

    # Submit the solution from the hindsight problem
    while not env.is_done:
        ep_inst = observation["epoch_instance"]
        requests = set(ep_inst["request_idx"])

        # This is a proxy to extract the routes from the hindsight
        # solution that are dispatched in the current epoch.
        ep_sol = [
            route
            for route in routes
            if len(requests.intersection(route)) == len(route)
        ]

        observation, _, _, info = env.step(ep_sol)
        assert info["error"] is None, f"{info['error']}"

    cost_eval = CostEvaluator(0, 0)
    assert sum(env.final_costs.values()) == cost_eval.cost(best)

    return env.final_costs, env.final_solutions


def instance2data(instance) -> ProblemData:
    """
    Converts an instance to a pyvrp model.
    """
    # A priori checks
    if "dimension" in instance:
        dimension: int = instance["dimension"]
    else:
        if "demands" not in instance:
            raise ValueError("File should either contain dimension or demands")
        dimension = len(instance["demands"])

    depots: np.ndarray = instance.get("depot", np.array([0]))
    num_vehicles: int = instance.get("vehicles", dimension - 1)
    capacity: int = instance.get("capacity", _INT_MAX)

    distances: np.ndarray = instance["duration_matrix"]

    demands: np.ndarray = instance["demands"]
    coords: np.ndarray = instance["coords"]
    durations = distances
    time_windows: np.ndarray = instance["time_windows"]

    if "service_times" in instance:
        if isinstance(instance["service_times"], Number):
            # Some instances describe a uniform service time as a single value
            # that applies to all clients.
            service_times = np.full(dimension, instance["service_times"], int)
            service_times[0] = 0
        else:
            service_times = instance["service_times"]
    else:
        service_times = np.zeros(dimension, dtype=int)

    if "release_times" in instance:
        release_times: np.ndarray = instance["release_times"]
    else:
        release_times = np.zeros(dimension, dtype=int)

    if "dispatch_times" in instance:
        dispatch_times: np.ndarray = instance["dispatch_times"]
    else:
        horizon = time_windows.max()
        dispatch_times = horizon * np.ones(dimension, dtype=int)

    prizes = instance.get("prizes", np.zeros(dimension, dtype=int))

    # Checks
    if len(depots) != 1 or depots[0] != 0:
        raise ValueError(
            "Source file should contain single depot with index 1 "
            + "(depot index should be 0 after subtracting offset 1)"
        )

    if demands[0] != 0:
        raise ValueError("Demand of depot must be 0")

    if time_windows[0, 0] != 0:
        raise ValueError("Depot start of time window must be 0")

    if service_times[0] != 0:
        raise ValueError("Depot service duration must be 0")

    if release_times[0] != 0:
        raise ValueError("Depot release time must be 0")

    if dispatch_times[0] != time_windows[0, 1]:
        raise ValueError("Depot end of time window must be dispatch time")

    if (time_windows[:, 0] > time_windows[:, 1]).any():
        raise ValueError("Time window cannot start after end")

    clients = [
        Client(
            coords[idx][0],  # x
            coords[idx][1],  # y
            demands[idx],
            service_times[idx],
            time_windows[idx][0],  # TW early
            time_windows[idx][1],  # TW late
            release_times[idx],
            prizes[idx],
            np.isclose(prizes[idx], 0),  # required only when prize is zero
        )
        for idx in range(dimension)
    ]
    vehicle_types = [VehicleType(capacity, num_vehicles)]

    return ProblemData(
        clients,
        vehicle_types,
        distances,
        durations,
    )
