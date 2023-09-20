import pickle
from copy import copy, deepcopy
from typing import Tuple

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal, assert_raises
from pytest import mark

from pyvrp import (
    Client,
    ProblemData,
    RandomNumberGenerator,
    Route,
    Solution,
    VehicleType,
)
from pyvrp.tests.helpers import read


@mark.parametrize(
    "routes",
    [
        [[3, 4], [1, 2], []],
        [[3, 4], [], [1, 2]],
    ],
)
def test_route_constructor_raises_for_empty_routes(ok_small, routes):
    """
    Tests that constructing a ``Solution`` with empty routes fails.
    """
    with assert_raises(RuntimeError):
        Solution(ok_small, routes)


def test_route_constructor_with_different_vehicle_types(ok_small):
    """
    Tests that Solution's route constructor respects vehicle types.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(10, 1), VehicleType(20, 2)]
    )

    sol = Solution(data, [Route(data, [3, 4], 0), Route(data, [1, 2], 1)])

    # We expect Solution to return routes with the correct vehicle types.
    routes = sol.get_routes()
    assert_equal(len(routes), 2)

    assert_equal(routes[0].visits(), [3, 4])
    assert_equal(routes[0].vehicle_type(), 0)
    assert_equal(routes[0], Route(data, [3, 4], 0))

    assert_equal(routes[1].visits(), [1, 2])
    assert_equal(routes[1].vehicle_type(), 1)
    assert_equal(routes[1], Route(data, [1, 2], 1))


def test_route_eq(ok_small):
    """
    Tests ``Route``'s equality operator.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(10, 1), VehicleType(20, 2)]
    )

    route1 = Route(data, [1, 2], 0)
    assert_(route1 == route1)  # should equal self

    route2 = Route(data, [1, 2], 0)
    assert_equal(route1, route2)  # same route/vehicle type; different object

    route3 = Route(data, [1, 2], 1)
    assert_(route2 != route3)  # different vehicle type

    route4 = Route(data, [2, 1], 0)
    assert_(route2 != route4)  # same vehicle type, different visits

    assert_(route1 != "test")
    assert_(route1 != 0)
    assert_(route1 != -1.0)


def test_random_constructor_cycles_over_routes(ok_small):
    """
    Tests that a randomly constructed solution fills all available vehicles
    in turn.
    """
    # The OkSmall instance has four clients and three vehicles. Since 1 client
    # per vehicle would not work (insufficient vehicles), each route is given
    # two clients (and the last route should be empty).
    rng = RandomNumberGenerator(seed=42)

    sol = Solution.make_random(ok_small, rng)
    routes = sol.get_routes()

    assert_equal(sol.num_routes(), 2)
    assert_equal(len(routes), 2)

    for idx, size in enumerate([2, 2]):
        assert_equal(len(routes[idx]), size)


def test_route_constructor_raises_too_many_vehicles(ok_small):
    """
    Tests that constructing a solution with more routes than available in the
    data instance raises.
    """
    assert_equal(ok_small.num_vehicles, 3)

    # Only two routes should not raise.
    sol = Solution(ok_small, [[1, 2], [4, 3]])
    assert_equal(len(sol.get_routes()), 2)

    # Three routes should not raise.
    Solution(ok_small, [[1, 2], [4], [3]])

    # More than three routes should raise, since we only have three vehicles.
    with assert_raises(RuntimeError):
        Solution(ok_small, [[1], [2], [3], [4]])

    # Now test the case with multiple vehicle types.
    data = ok_small.replace(
        vehicle_types=[VehicleType(10, 2), VehicleType(20, 1)]
    )

    # Only two routes (of type 0) should not raise.
    sol = Solution(data, [[1, 2], [4, 3]])
    assert_equal(len(sol.get_routes()), 2)

    # One route of both vehicle types should not raise.
    sol = Solution(data, [Route(data, [1, 2], 0), Route(data, [4, 3], 1)])
    assert_equal(len(sol.get_routes()), 2)

    # Two routes of type 1 and one of type 2 should not raise as we have those.
    sol = Solution(
        data,
        [Route(data, [1], 0), Route(data, [2], 0), Route(data, [4, 3], 1)],
    )
    assert_equal(len(sol.get_routes()), 3)

    # Two routes of vehicle type 1 should raise since we have only one.
    with assert_raises(RuntimeError):
        sol = Solution(data, [Route(data, [1, 2], 1), Route(data, [4, 3], 1)])

    # Three routes should raise since they are considered to be type 0.
    with assert_raises(RuntimeError):
        Solution(data, [[1, 2], [4], [3]])


def test_route_constructor_raises_for_multiple_visits(ok_small):
    """
    Tests that visiting the same client more than once raises.
    """
    with assert_raises(RuntimeError):
        Solution(ok_small, [[1, 2], [1, 3, 4]])  # client 1 is visited twice

    with assert_raises(RuntimeError):
        Solution(
            ok_small, [[1, 2], [1, 3, 4], [1]]
        )  # client 1 is visited thrice


def test_route_constructor_allows_incomplete_solutions(ok_small_prizes):
    """
    Tests that not visiting a client at all is allowed, but turns the solution
    infeasible (unless the client is not required). Allowing this is useful
    for both prize collecting, and in LNS settings where an incomplete solution
    is subsequently repaired.
    """
    # Client 1 is required but not visited.
    sol = Solution(ok_small_prizes, [[2], [3, 4]])
    assert_(not sol.is_complete())
    assert_(not sol.is_feasible())

    # All required clients are visited, but the solution is not feasible.
    sol = Solution(ok_small_prizes, [[1], [2, 3, 4]])
    assert_(not sol.is_feasible())
    assert_(sol.is_complete())

    # All required clients are visited and the solution is feasible.
    sol = Solution(ok_small_prizes, [[1]])
    assert_(sol.is_feasible())
    assert_(sol.is_complete())


def test_get_neighbours(ok_small):
    """
    Tests that accessing the neighbour structur of (pred, succ) pairs for each
    client in the solution works correctly.
    """
    assert_equal(ok_small.num_clients, 4)

    sol = Solution(ok_small, [[3], [1, 2]])  # client 4 is not in the solution
    assert_(not sol.is_complete())

    neighbours = sol.get_neighbours()
    expected = [
        None,  # 0: is depot
        (0, 2),  # 1: between depot (0) to 2
        (1, 0),  # 2: between 1 and depot (0)
        (0, 0),  # 3: between depot (0) and depot (0)
        None,  # 4: unassigned
    ]

    for client in range(ok_small.num_clients + 1):  # incl. depot
        assert_equal(neighbours[client], expected[client])


def test_feasibility(ok_small):
    """
    Tests that solutions are infeasible when they have load or time window
    violations.
    """
    # This solution is infeasible due to both load and time window violations.
    sol = Solution(ok_small, [[1, 2, 3, 4]])
    assert_(not sol.is_feasible())

    # First route has total load 18, but vehicle capacity is only 10.
    assert_(sol.has_excess_load())

    # Client 4 has TW [8400, 15300], but client 2 cannot be visited before
    # 15600, so there must be time warp on the single-route solution.
    assert_(sol.has_time_warp())

    # Let's try another solution that's actually feasible.
    sol = Solution(ok_small, [[1, 2], [3], [4]])
    assert_(sol.is_feasible())
    assert_(not sol.has_excess_load())
    assert_(not sol.has_time_warp())


def test_feasibility_release_times():
    """
    Tests solutions can be infeasible due to release time violations, which
    adds time warp.
    """
    data = read("data/OkSmallReleaseTimes.txt")

    # Client 1 is released at 20'000, but client 2 time window ends at 19'500,
    # so this solution must be infeasible due to time-warping. We arrive at
    # client 1 at time 20'000 + 1'544 = 21'544 before the TW closes (22'500).
    # We arrive at client 2 at 21'544 + 360 + 1'992 = 23'896, so we incur a
    # time warp of 23'896 - 19'500 = 4'396.
    sol = Solution(data, [[1, 2], [3], [4]])
    assert_(not sol.is_feasible())
    assert_allclose(sol.time_warp(), 4396)

    # Visiting clients 2 and 3 together is feasible: both clients are released
    # at time 5'000. We arrive at client 2 at 5'000 + 1'944 and wait till the
    # TW opens (12'000). We arrive at client 3 at 12'000 + 360 + 621 = 12'981,
    # which is before the TW closes (15'300).
    sol = Solution(data, [[1], [2, 3], [4]])
    assert_(sol.is_feasible())


def test_feasibility_dispatch_times():
    data = read("data/OkSmallDispatchTimes.txt")

    # This example is similar to the example in release times, but now we have
    # additional time warp due to the dispatch times being earlier than the
    # release times. Client 1 is released at 20'000 and must be dispatched by
    # 20'000, but client 2 is released at 5'000 and must be dispatched by
    # 10'000. So the route earliest release is 20'000 and the route latest
    # dispatch is 10'000, so we incur an additional time warp of 20'000 -
    # 10'000 = 10'000 to start the route at release time.
    sol = Solution(data, [[1, 2], [3], [4]])
    assert_(not sol.is_feasible())
    assert_equal(sol.time_warp(), 14396)

    # Visiting clients 2 and 3 together is feasible: both clients are released
    # at time 5'000, and must be dispatched by min(10'000, 45'000) = 10'000.
    # We arrive at client 2 at 5'000 + 1'944 and wait till the TW opens
    # (12'000). We arrive at client 3 at 12'000 + 360 + 621 = 12'981, which is
    # before the TW closes (15'300).
    sol = Solution(data, [[1], [2, 3], [4]])
    assert_(sol.is_feasible())


def test_distance_calculation(ok_small):
    """
    Tests that route distance calculations are correct, and that the overall
    Solution's distance is the sum of the route distances.
    """
    sol = Solution(ok_small, [[1, 2], [3], [4]])
    routes = sol.get_routes()

    # Solution is feasible, so all its routes should also be feasible.
    assert_(sol.is_feasible())
    assert_(all(route.is_feasible() for route in routes))

    # Solution distance should be equal to all routes' distances. These we
    # check separately.
    assert_allclose(sol.distance(), sum(route.distance() for route in routes))

    expected = ok_small.dist(0, 1) + ok_small.dist(1, 2) + ok_small.dist(2, 0)
    assert_allclose(routes[0].distance(), expected)

    expected = ok_small.dist(0, 3) + ok_small.dist(3, 0)
    assert_allclose(routes[1].distance(), expected)

    expected = ok_small.dist(0, 4) + ok_small.dist(4, 0)
    assert_allclose(routes[2].distance(), expected)


def test_excess_load_calculation(ok_small):
    """
    Tests the Solution's excess load calculation on a single-route case.
    """
    sol = Solution(ok_small, [[4, 3, 1, 2]])
    assert_(sol.has_excess_load())
    assert_(not sol.has_time_warp())

    # All clients are visited on the same route/by the same vehicle. The total
    # demand is 18, but the vehicle capacity is only 10.
    assert_allclose(sol.excess_load(), 18 - ok_small.vehicle_type(0).capacity)


def test_excess_load_calculation_with_multiple_vehicle_capacities(ok_small):
    """
    Tests that vehicles of different capacities result in different (excess)
    load calaculations.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(10, 2), VehicleType(20, 1)]
    )

    # This instance has capacities 10 and 20 for vehicle type 0 and 1. The
    # total demand is 18 so if all demand is put in vehicle type 0 the
    # excess_load is 18 - 10 = 8.
    sol = Solution(data, [Route(data, [1, 2, 3, 4], 0)])
    assert_(sol.has_excess_load())
    assert_allclose(sol.excess_load(), 8)

    # With vehicle type 1, the capacity 20 is larger than 18.
    sol = Solution(data, [Route(data, [1, 2, 3, 4], 1)])
    assert_(not sol.has_excess_load())
    assert_allclose(sol.excess_load(), 0)


def test_route_access_methods(ok_small):
    """
    Tests that accessing route statistics returns the correct numbers.
    """
    sol = Solution(ok_small, [[1, 3], [2, 4]])
    routes = sol.get_routes()

    # Test route access: getting the route plan should return a simple list, as
    # given to the solution above.
    assert_equal(routes[0].visits(), [1, 3])
    assert_equal(routes[1].visits(), [2, 4])

    # There's no excess load, so all excess load should be zero.
    assert_(not sol.has_excess_load())
    assert_allclose(routes[0].excess_load(), 0)
    assert_allclose(routes[1].excess_load(), 0)

    # Total route demand.
    demands = [
        ok_small.client(idx).demand for idx in range(ok_small.num_clients + 1)
    ]

    assert_allclose(routes[0].demand(), demands[1] + demands[3])
    assert_allclose(routes[1].demand(), demands[2] + demands[4])

    # The first route is not feasible due to time warp, but the second one is.
    # See also the tests below.
    assert_(not routes[0].is_feasible())
    assert_(routes[1].is_feasible())

    # Total service duration.
    services = [
        ok_small.client(idx).service_duration
        for idx in range(ok_small.num_clients + 1)
    ]
    assert_allclose(routes[0].service_duration(), services[1] + services[3])
    assert_allclose(routes[1].service_duration(), services[2] + services[4])


def test_route_time_warp_calculations(ok_small):
    """
    Tests route time warp calculations.
    """
    sol = Solution(ok_small, [[1, 3], [2, 4]])
    routes = sol.get_routes()

    # There is time warp on the first route: duration(0, 1) = 1'544, so we
    # arrive at 1 before its opening window of 15'600. Service (360) thus
    # starts at 15'600, and completes at 15'600 + 360. Then we drive for
    # duration(1, 3) = 1'427, where we arrive after 15'300 (its closing time
    # window). This is where we incur time warp: we need to 'warp' to 15'300.
    assert_(sol.has_time_warp())
    assert_(routes[0].has_time_warp())
    assert_allclose(routes[0].time_warp(), 15_600 + 360 + 1_427 - 15_300)

    # The second route has no time warp, so the overall solution time warp is
    # all incurred on the first route.
    assert_(not routes[1].has_time_warp())
    assert_allclose(routes[1].time_warp(), 0)
    assert_allclose(sol.time_warp(), routes[0].time_warp())


def test_route_wait_time_calculations():
    """
    Tests route wait time and slack calculations.
    """
    data = read("data/OkSmallWaitTime.txt")
    sol = Solution(data, [[1, 3], [2, 4]])
    routes = sol.get_routes()

    # In the second route, the time window of client 2 closes at 15'000. After
    # service and travel, we then arrive at client 4 before its time window is
    # open, so we have to wait. In particular, we have to wait:
    #   twEarly(4) - duration(2, 4) - serv(2) - twLate(2)
    #     = 18'000 - 1'090 - 360 - 15'000
    #     = 1'550.
    assert_allclose(routes[1].wait_duration(), 1_550)

    # Since there is waiting time, there is no slack in the schedule. We should
    # thus start as late as possible, at:
    #   twLate(2) - duration(0, 2)
    #     = 15'000 - 1'944
    #     = 13'056.
    assert_allclose(routes[1].slack(), 0)
    assert_allclose(routes[1].start_time(), 13_056)

    # So far we have tested a route that had wait duration, but not time warp.
    # We now test a solution with a route that has both.
    sol = Solution(data, [[1, 2, 4], [3]])
    route, *_ = sol.get_routes()

    # This route has the same wait time as explained above. The time warp is
    # incurred earlier in the route, between 1 -> 2:
    #   twEarly(1) + serv(1) + duration(1, 2) - twLate(2)
    #     = 15'600 + 360 + 1'992 - 15'000
    #     = 2'952.
    assert_allclose(route.time_warp(), 2_952)
    assert_allclose(route.wait_duration(), 1_550)
    assert_allclose(route.slack(), 0)

    # Finally, the overall route duration should be equal to the sum of the
    # travel, service, and waiting durations.
    assert_allclose(
        route.duration(),
        route.travel_duration()
        + route.service_duration()
        + route.wait_duration(),
    )


def test_route_start_and_end_time_calculations(ok_small):
    """
    Tests route start time, slack, and end time calculations for cases with
    and without time warp or wait duration.
    """
    sol = Solution(ok_small, [[1, 3], [2, 4]])
    routes = sol.get_routes()

    # The first route has timewarp, so there is no slack in the schedule. We
    # should thus depart as soon as possible to arrive at the first client the
    # moment its time window opens.
    start_time = ok_small.client(1).tw_early - ok_small.duration(0, 1)
    end_time = start_time + routes[0].duration() - routes[0].time_warp()

    assert_(routes[0].has_time_warp())
    assert_allclose(routes[0].slack(), 0)
    assert_allclose(routes[0].start_time(), start_time)
    assert_allclose(routes[0].end_time(), end_time)

    # The second route has no time warp. The latest it can start is calculated
    # backwards from the closing of client 4's time window:
    #   twLate(4) - duration(2, 4) - serv(2) - duration(0, 2)
    #     = 19'500 - 1'090 - 360 - 1'944
    #     = 16'106.
    #
    # Because client 4 has a large time window, the earliest this route can
    # start is determined completely by client 2: we should not arrive before
    # its time window, because that would incur needless waiting. We should
    # thus not depart before:
    #   twEarly(2) - duration(0, 2)
    #     = 12'000 - 1'944
    #     = 10'056.
    assert_(not routes[1].has_time_warp())
    assert_allclose(routes[1].wait_duration(), 0)
    assert_allclose(routes[1].start_time(), 10_056)
    assert_allclose(routes[1].slack(), 16_106 - 10_056)

    # The overall route duration is given by:
    #   duration(0, 2) + serv(2) + duration(2, 4) + serv(4) + duration(4, 0)
    #     = 1'944 + 360 + 1'090 + 360 + 1'475
    #     = 5'229.
    assert_allclose(routes[1].duration(), 1_944 + 360 + 1_090 + 360 + 1_475)
    assert_allclose(routes[1].end_time(), 10_056 + 5_229)


def test_route_release_time():
    """
    Tests that routes return the correct release times, and, when feasible,
    start after the release time.
    """
    data = read("data/OkSmallReleaseTimes.txt")
    sol = Solution(data, [[1, 3], [2, 4]])
    routes = sol.get_routes()

    # The client release times are 20'000, 5'000, 5'000 and 1'000. So the first
    # route has a release time of max(20'000, 5'000) = 20'000, and the second
    # has a release time of max(5'000, 1'000) = 5'000.
    assert_allclose(routes[0].release_time(), 20_000)
    assert_allclose(routes[1].release_time(), 5_000)

    # Second route is feasible, so should have start time not smaller than
    # release time.
    assert_(not routes[1].has_time_warp())
    assert_(routes[1].start_time() > routes[1].release_time())


def test_route_dispatch_time():
    data = read("data/OkSmallDispatchTimes.txt")
    sol = Solution(data, [[1, 3], [2, 4]])
    routes = sol.get_routes()

    # The client dispatch times are 20'000, 10'000, 45'000 and 10'000.
    # So the first route has a dispatch time of min(20'000, 45'000) = 20'000,
    # and the second route has a dispatch time of min(10'000, 10'000) = 10'000.
    assert_allclose(routes[0].dispatch_time(), 20000)
    assert_allclose(routes[1].dispatch_time(), 10000)


@mark.parametrize(
    "dist_mat",
    [
        np.full((3, 3), fill_value=100, dtype=int),
        np.full((3, 3), fill_value=1, dtype=int),
        np.full((3, 3), fill_value=1000, dtype=int),
    ],
)
def test_time_warp_for_a_very_constrained_problem(dist_mat):
    """
    This tests an artificial instance where the second client cannot be reached
    directly from the depot in a feasible solution, but only after the first
    client.
    """
    dur_mat = [
        [0, 1, 10],  # cannot get to 2 from depot within 2's time window
        [1, 0, 1],
        [1, 1, 0],
    ]

    data = ProblemData(
        clients=[
            Client(x=0, y=0, tw_late=10),
            Client(x=1, y=0, tw_late=5),
            Client(x=2, y=0, tw_late=5),
        ],
        vehicle_types=[VehicleType(0, 2)],
        distance_matrix=dist_mat,
        duration_matrix=dur_mat,
    )

    # This solution directly visits the second client from the depot, which is
    # not time window feasible.
    infeasible = Solution(data, [[1], [2]])
    assert_(infeasible.has_time_warp())
    assert_(not infeasible.has_excess_load())
    assert_(not infeasible.is_feasible())

    # But visiting the second client after the first is feasible.
    feasible = Solution(data, [[1, 2]])
    assert_(not feasible.has_time_warp())
    assert_(not feasible.has_excess_load())
    assert_(feasible.is_feasible())

    assert_allclose(
        feasible.distance(),
        dist_mat[0, 1] + dist_mat[1, 2] + dist_mat[2, 0],
    )


def test_time_warp_return_to_depot():
    """
    This tests wether the calculated total duration and time warp includes the
    travel back to the depot.
    """
    data = ProblemData(
        clients=[Client(x=0, y=0, tw_late=1), Client(x=1, y=0)],
        vehicle_types=[VehicleType(0, 1)],
        distance_matrix=np.asarray([[0, 0], [0, 0]]),
        duration_matrix=np.asarray([[0, 1], [1, 0]]),
    )

    sol = Solution(data, [[1]])
    route, *_ = sol.get_routes()

    # Travel from depot to client and back gives duration 1 + 1 = 2. This is 1
    # more than the depot time window 1, giving a time warp of 1.
    assert_allclose(route.duration(), 2)
    assert_allclose(data.client(0).tw_late, 1)
    assert_allclose(sol.time_warp(), 1)


def tests_that_not_specifying_the_vehicle_type_assumes_a_default(ok_small):
    """
    Not specifying the vehicle type when providing a list of visits uses the
    first vehicle type to complete the routes. That could result in a solution
    using too many vehicles of the first type.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(10, 2), VehicleType(20, 1)]
    )

    sol = Solution(data, [[1, 2, 3, 4]])
    assert_equal(sol.num_routes(), 1)

    sol = Solution(data, [Route(data, [1, 2, 3, 4], 0)])
    assert_equal(sol.num_routes(), 1)

    sol = Solution(data, [[1, 2], [3, 4]])
    assert_equal(sol.num_routes(), 2)

    with assert_raises(RuntimeError):
        # This raises since we don't specify route types, which means we create
        # 3 routes of type 0 whereas we only have 2 available.
        sol = Solution(data, [[1], [2], [3, 4]])

    # It works if we specify the correct vehicle types.
    sol = Solution(
        data,
        [Route(data, [1], 0), Route(data, [2], 0), Route(data, [3, 4], 1)],
    )
    assert_equal(sol.num_routes(), 3)

    # But not if we violate the number of available vehicles of that type.
    with assert_raises(RuntimeError):
        sol = Solution(
            data,
            [Route(data, [1], 0), Route(data, [2], 1), Route(data, [3, 4], 1)],
        )


def test_copy(ok_small):
    """
    Tests that copied solutions are equal to the original solution, but not
    the exact same object.
    """
    sol = Solution(ok_small, [[1, 2, 3, 4]])
    copy_sol = copy(sol)
    deepcopy_sol = deepcopy(sol)

    # Copied solutions are equal to the original solution.
    assert_(sol == copy_sol)
    assert_(sol == deepcopy_sol)

    # But they are not the same object.
    assert_(sol is not copy_sol)
    assert_(sol is not deepcopy_sol)


def test_eq(ok_small):
    """
    Tests the solution's equality operator.
    """
    sol1 = Solution(ok_small, [[1, 2, 3, 4]])
    sol2 = Solution(ok_small, [[1, 2], [3], [4]])
    sol3 = Solution(ok_small, [[1, 2, 3, 4]])

    assert_(sol1 == sol1)  # Solutions should be equal to themselves
    assert_(sol2 == sol2)
    assert_(sol1 != sol2)  # different routes, so should not be equal
    assert_(sol1 == sol3)  # same routes, different solution

    sol4 = Solution(ok_small, [[1, 2, 3], [4]])
    sol5 = Solution(ok_small, [[4], [1, 2, 3]])

    assert_(sol4 == sol5)  # routes are the same, but in different order

    # And a few tests against things that are not solutions, just to be sure
    # there's also a type check in there somewhere.
    assert_(sol4 != 1)
    assert_(sol4 != "abc")
    assert_(sol5 != 5)
    assert_(sol5 != "cd")


def test_eq_with_multiple_vehicle_types(ok_small):
    """
    Tests that two solutions are not considered equal if they have the same
    routes (orders of clients) but served by different vehicle types.
    """
    # Make sure capacities are different but large enough (>18) to have no
    # violations so have the same attributes, such that we actually test if the
    # assignments are used for the equality comparison.
    data = ok_small.replace(
        vehicle_types=[VehicleType(20, 2), VehicleType(30, 1)]
    )

    # These two should be the same
    sol1 = Solution(data, [[1, 2, 3, 4]])
    sol2 = Solution(data, [Route(data, [1, 2, 3, 4], 0)])
    # Create solution with different vehicle type
    sol3 = Solution(data, [Route(data, [1, 2, 3, 4], 1)])

    # First two solution have one route with the same vehicle type
    assert_(sol1 == sol2)
    # Solution 3 is different since the route has a different vehicle type
    assert_(sol1 != sol3)

    # Order should not matter so these should be the same
    sol1 = Solution(data, [Route(data, [1, 2], 0), Route(data, [3, 4], 1)])
    sol2 = Solution(data, [Route(data, [3, 4], 1), Route(data, [1, 2], 0)])
    assert_(sol1 == sol2)

    # But changing the vehicle types should be different
    sol3 = Solution(data, [Route(data, [1, 2], 1), Route(data, [3, 4], 0)])
    assert_(sol1 != sol3)


def test_eq_unassigned():
    """
    Tests the equality operator for solutions with unassigned clients.
    """
    clients = [
        Client(x=0, y=0),
        Client(x=0, y=1, required=False),
        Client(x=1, y=0, required=False),
    ]
    vehicle_types = [VehicleType(1, 2)]
    dist = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

    data = ProblemData(clients, vehicle_types, dist, dist)

    sol1 = Solution(data, [[1]])
    sol2 = Solution(data, [[1]])
    sol3 = Solution(data, [[2]])

    assert_(sol1 == sol2)
    assert_(sol1 != sol3)


def test_duplicate_vehicle_types(ok_small):
    """
    Tests that it is allowed to have duplicate vehicle types. These will be
    considered completely different during optimisation.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(10, 1), VehicleType(10, 1)]
    )

    sol1 = Solution(data, [Route(data, [1, 2, 3, 4], 0)])
    sol2 = Solution(data, [Route(data, [1, 2, 3, 4], 1)])

    assert_(sol1 != sol2)


@mark.parametrize(
    "vehicle_types",
    [[VehicleType(10, 3)], [VehicleType(10, 2), VehicleType(20, 1)]],
)
def test_str_contains_routes(ok_small, vehicle_types):
    """
    Tests that the Solution's string representation contains each route.
    """
    data = ok_small.replace(vehicle_types=vehicle_types)
    rng = RandomNumberGenerator(seed=2)

    for _ in range(5):  # let's do this a few times to really make sure
        sol = Solution.make_random(data, rng)
        str_representation = str(sol).splitlines()
        routes = sol.get_routes()

        # There should be no more than len(routes) lines (each detailing a
        # single route).
        assert_equal(len(str_representation), len(routes))

        # Each line should contain a route, where each route should contain
        # every client that is in the route as returned by get_routes().
        for route, str_route in zip(routes, str_representation):
            for client in route:
                assert_(str(client) in str_route)


def test_hash(ok_small):
    """
    Tests that solutions that compare the same have the same hash.
    """
    rng = RandomNumberGenerator(seed=2)

    sol1 = Solution.make_random(ok_small, rng)
    sol2 = Solution.make_random(ok_small, rng)

    # Two random solutions. They're not the same, so the hashes should not be
    # the same either (unless there's a collision, which is not the case here).
    assert_(sol1 != sol2)
    assert_(hash(sol1) != hash(sol2))

    sol3 = deepcopy(sol2)  # is a direct copy

    # These two are the same solution, so their hashes should be the same too.
    assert_equal(sol2, sol3)
    assert_equal(hash(sol2), hash(sol3))


def test_route_centroid(ok_small):
    """
    Tests that each route's center point is the center point of all clients
    visited by that route.
    """
    x = np.array([ok_small.client(client).x for client in range(5)])
    y = np.array([ok_small.client(client).y for client in range(5)])

    routes = [
        Route(ok_small, [1, 2], 0),
        Route(ok_small, [3], 0),
        Route(ok_small, [4], 0),
    ]

    for route in routes:
        x_center, y_center = route.centroid()
        assert_allclose(x_center, x[route].mean())
        assert_allclose(y_center, y[route].mean())


def test_solution_can_be_pickled(ok_small):
    """
    Tests that a solution can be serialised and unserialised.
    """
    rng = RandomNumberGenerator(seed=2)

    before_pickle = Solution.make_random(ok_small, rng)
    bytes = pickle.dumps(before_pickle)
    after_pickle = pickle.loads(bytes)

    assert_equal(after_pickle, before_pickle)


def test_route_can_be_pickled(rc208):
    """
    Tests that individual routes can be serialised and unserialised.
    """
    rng = RandomNumberGenerator(seed=2)
    sol = Solution.make_random(rc208, rng)

    for before_pickle in sol.get_routes():
        bytes = pickle.dumps(before_pickle)
        after_pickle = pickle.loads(bytes)

        assert_equal(after_pickle, before_pickle)


@mark.parametrize(
    ("assignment", "expected"), [((0, 0), 0), ((0, 1), 10), ((1, 1), 20)]
)
def test_fixed_vehicle_cost(
    ok_small, assignment: Tuple[int, int], expected: int
):
    """
    Tests that the solution tracks the total fixed vehicle costs of the
    vehicles used for its routes.
    """
    # First vehicle type is free, second costs 10 per vehicle. The solution
    # should be able to track this.
    data = ok_small.replace(
        vehicle_types=[
            VehicleType(10, 2, fixed_cost=0),
            VehicleType(10, 2, fixed_cost=10),
        ]
    )

    routes = [
        Route(data, [1, 2], assignment[0]),
        Route(data, [3, 4], assignment[1]),
    ]

    sol = Solution(data, routes)
    assert_allclose(sol.fixed_vehicle_cost(), expected)


@mark.parametrize(
    ("tw_early", "tw_late", "expected"),
    [
        (0, 0, 20_277),  # cannot be back at the depot before 20'277
        (0, 20_000, 277),  # larger shift window decreases time warp
        (0, 20_277, 0),  # and in this case there is no more time warp
        (15_000, 20_000, 1_221),  # minimum route duration is 6'221
        (10_000, 20_000, 277),  # before earliest possible return
    ],
)
def test_route_shift_duration(
    ok_small, tw_early: int, tw_late: int, expected: int
):
    """
    Tests that Route computes time warp due to shift durations correctly on a
    simple, two-client route.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(10, 2, tw_early=tw_early, tw_late=tw_late)]
    )

    # Overall route duration is, at the bare minimum, dist(0, 1) + dist(1, 2)
    # + dist(2, 0) + serv(1) + serv(2). That's 1'544 + 1'992 + 1'965 + 360
    # + 360 = 6'221. We cannot service client 1 before 15'600, and it takes
    # 1'544 to get there from the depot, so we leave at 14'056. Thus, the
    # earliest complete time is 14'056 + 6'221 = 20'277.
    route = Route(data, [1, 2], vehicle_type=0)
    assert_allclose(route.time_warp(), expected)
