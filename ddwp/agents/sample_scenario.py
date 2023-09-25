from copy import copy

import numpy as np
from pyvrp import VehicleType

from ddwp.Environment import State, StaticInfo
from ddwp.sampling import SamplingMethod
from ddwp.VrpInstance import VrpInstance


def sample_scenario(
    info: StaticInfo,
    obs: State,
    num_lookahead: int,
    sampling_method: SamplingMethod,
    rng: np.random.Generator,
    to_dispatch: np.ndarray,
    to_postpone: np.ndarray,
) -> VrpInstance:
    """
    Samples a VRPTW scenario instance. The scenario instance is created by
    appending the sampled requests to the current epoch instance.

    Parameters
    ----------
    info
        The static information.
    obs
        The current epoch observation.
    to_dispatch
        A boolean array where True means that the corresponding request must be
        dispatched.
    to_postpone
        A boolean array where True mean that the corresponding request must be
        postponed.
    """
    # Parameters
    current_epoch = obs.current_epoch
    next_epoch = current_epoch + 1
    epochs_left = info.end_epoch - current_epoch
    max_lookahead = min(num_lookahead, epochs_left)
    num_requests_per_epoch = info.num_requests_per_epoch

    static_inst = info.static_instance
    epoch_duration = info.epoch_duration
    dispatch_margin = info.dispatch_margin
    ep_inst = obs.epoch_instance
    departure_time = obs.departure_time

    # Scenario instance fields
    req_cust_idx = ep_inst.customer_idx
    req_idx = ep_inst.request_idx
    req_demand = ep_inst.demands
    req_service = ep_inst.service_times
    req_tw = ep_inst.time_windows
    req_release = ep_inst.release_times

    # Modify the release time of postponed requests: they should start
    # at the next departure time.
    next_departure_time = departure_time + epoch_duration
    req_release[to_postpone] = next_departure_time

    # Modify the dispatch time of dispatched requests: they should start
    # at the current departure time (and at time horizon otherwise).
    horizon = req_tw[0][1]
    req_dispatch = np.where(to_dispatch, departure_time, horizon)

    vehicle_types = copy(ep_inst.vehicle_types)

    for epoch in range(next_epoch, next_epoch + max_lookahead):
        epoch_start = epoch * epoch_duration
        epoch_depart = epoch_start + dispatch_margin
        num_requests = num_requests_per_epoch[epoch]

        new = sampling_method(
            rng,
            static_inst,
            epoch_start,
            epoch_depart,
            epoch_duration,
            num_requests,
        )
        num_new_reqs = new["customer_idx"].size

        # Sampled request indices are negative so we can distinguish them.
        new_req_idx = -(np.arange(num_new_reqs) + 1) - len(req_idx)

        # Concatenate the new requests to the current instance requests.
        req_idx = np.concatenate((req_idx, new_req_idx))
        req_cust_idx = np.concatenate((req_cust_idx, new["customer_idx"]))
        req_demand = np.concatenate((req_demand, new["demands"]))
        req_service = np.concatenate((req_service, new["service_times"]))
        req_tw = np.concatenate((req_tw, new["time_windows"]))
        req_release = np.concatenate((req_release, new["release_times"]))

        # Default earliest dispatch time is the time horizon.
        req_dispatch = np.concatenate(
            (req_dispatch, np.full(num_new_reqs, horizon))
        )

        if info.num_vehicles_per_epoch is None:
            # No restriction means that we don't have to distinguish between,
            # vehicle types, so we use one vehicle per request.
            vehicle_types = [
                VehicleType(
                    ep_inst.capacity,
                    req_cust_idx.size - 1,
                    tw_early=departure_time,
                    tw_late=horizon,
                )
            ]
        else:
            num_primary = info.num_vehicles_per_epoch[epoch]
            if num_primary > 0:
                vehicle_types.append(
                    VehicleType(
                        ep_inst.capacity,
                        num_primary,
                        tw_early=epoch_depart,
                        tw_late=horizon,
                    )
                )

            if (num_secondary := num_new_reqs - num_primary) > 0:
                vehicle_types.append(
                    VehicleType(
                        ep_inst.capacity,
                        num_secondary,
                        tw_early=epoch_depart,
                        tw_late=horizon,
                        fixed_cost=info.secondary_fleet_fixed_cost,
                    )
                )

    dist = static_inst.duration_matrix

    return VrpInstance(
        is_depot=static_inst.is_depot[req_cust_idx],
        customer_idx=req_cust_idx,
        request_idx=req_idx,
        coords=static_inst.coords[req_cust_idx],
        demands=req_demand,
        capacity=static_inst.capacity,
        time_windows=req_tw,
        service_times=req_service,
        duration_matrix=dist[req_cust_idx][:, req_cust_idx],
        release_times=req_release,
        dispatch_times=req_dispatch,
        vehicle_types=vehicle_types,
    )
