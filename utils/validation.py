import numpy as np

# TODO Rewrite this


def _validate_all_customers_visited(solution, num_customers):
    flat_solution = np.array([stop for route in solution for stop in route])
    assert len(flat_solution) == num_customers, "Not all customers are visited"
    visited = np.zeros(num_customers + 1)  # Add padding for depot
    visited[flat_solution] = True
    assert visited[1:].all(), "Not all customers are visited"


def _validate_route_capacity(route, demands, capacity):
    assert (
        sum(demands[route]) <= capacity
    ), f"Capacity validated for route, {sum(demands[route])} > {capacity}"


def _validate_route_dispatch_windows(route, release_times, dispatch_times):
    if route:
        assert max(release_times[route]) <= min(dispatch_times[route])


def _validate_route_time_windows(
    route, dist, timew, service_t, release_t=None
):
    depot = 0  # For readability, define variable
    earliest_start_depot, latest_arrival_depot = timew[depot]
    if release_t is not None:
        earliest_start_depot = max(
            earliest_start_depot, release_t[route].max()
        )
    current_time = earliest_start_depot + service_t[depot]

    prev_stop = depot
    for stop in route:
        earliest_arrival, latest_arrival = timew[stop]
        arrival_time = current_time + dist[prev_stop, stop]
        # Wait if we arrive before earliest_arrival
        current_time = max(arrival_time, earliest_arrival)
        assert (
            current_time <= latest_arrival
        ), f"Time window violated for stop {stop}: {current_time} not in ({earliest_arrival}, {latest_arrival})"
        current_time += service_t[stop]
        prev_stop = stop
    current_time += dist[prev_stop, depot]
    assert (
        current_time <= latest_arrival_depot
    ), f"Time window violated for depot: {current_time} not in ({earliest_start_depot}, {latest_arrival_depot})"


def compute_route_driving_time(route, duration_matrix):
    """
    Computes the total route driving time, excluding waiting and service time.
    """
    return (
        duration_matrix[0, route[0]]
        + duration_matrix[route[:-1], route[1:]].sum()
        + duration_matrix[route[-1], 0]
    )


def compute_solution_driving_time(instance, solution):
    return sum(
        [
            compute_route_driving_time(route, instance["duration_matrix"])
            for route in solution
        ]
    )


def validate_static_solution(
    instance, solution, allow_skipped_customers=False
):
    if not allow_skipped_customers:
        _validate_all_customers_visited(solution, len(instance["coords"]) - 1)

    for route in solution:
        _validate_route_capacity(
            route, instance["demands"], instance["capacity"]
        )
        _validate_route_time_windows(
            route,
            instance["duration_matrix"],
            instance["time_windows"],
            instance["service_times"],
            instance.get("release_times", None),
        )

        if "dispatch_times" in instance:
            _validate_route_dispatch_windows(
                route, instance["release_times"], instance["dispatch_times"]
            )

    return compute_solution_driving_time(instance, solution)
