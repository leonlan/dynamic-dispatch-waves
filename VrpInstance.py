from copy import copy
from typing import Optional

import numpy as np
import numpy.typing as npt


class VrpInstance:
    """
    Class representing a static VRP instance.

    Parameters
    ----------
    is_depot
        Boolean array indicating whether a request is a depot.
    coords
        Request location coordinates.
    demands
        Request demands.
    capacity
        Vehicle capacity.
    time_windows
        Request time windows.
    service_times
        Request service times.
    duration_matrix
        Request duration matrix.
    release_times
        Request release times.
    customer_idx
        Request customer indices. Used to identify which requests belong to
        which customer.
    request_idx
        Request indices.
    dispatch_times
        Request dispatch times.
    must_dispatch
        Boolean array indicating whether a request must be dispatched.
    prizes
        Request prizes.
    """

    def __init__(
        self,
        is_depot: npt.NDArray[np.bool_],
        coords: npt.NDArray[np.int_],
        demands: npt.NDArray[np.int_],
        capacity: int,
        time_windows: npt.NDArray[np.int_],
        service_times: npt.NDArray[np.int_],
        duration_matrix: npt.NDArray[np.int_],
        release_times: npt.NDArray[np.int_],
        customer_idx: Optional[npt.NDArray[np.int_]] = None,
        request_idx: Optional[npt.NDArray[np.int_]] = None,
        dispatch_times: Optional[npt.NDArray[np.int_]] = None,
        must_dispatch: Optional[npt.NDArray[np.bool_]] = None,
        prizes: Optional[npt.NDArray[np.int_]] = None,
    ):
        self._is_depot = is_depot
        self._coords = coords
        self._demands = demands
        self._capacity = capacity
        self._time_windows = time_windows
        self._service_times = service_times
        self._duration_matrix = duration_matrix
        self._release_times = release_times
        self._customer_idx = _set_if_none(
            customer_idx, np.arange(self.dimension)
        )
        self._request_idx = _set_if_none(
            request_idx, np.arange(self.dimension)
        )
        self._dispatch_times = _set_if_none(
            dispatch_times, np.ones(self.dimension, dtype=int) * self.horizon
        )
        self._must_dispatch = _set_if_none(
            must_dispatch, np.zeros(self.dimension, dtype=bool)
        )
        self._prizes = _set_if_none(
            prizes, np.zeros(self.dimension, dtype=int)
        )

    @property
    def is_depot(self) -> npt.NDArray[np.bool_]:
        return self._is_depot

    @property
    def coords(self) -> npt.NDArray[np.int_]:
        return self._coords

    @property
    def demands(self) -> npt.NDArray[np.int_]:
        return self._demands

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def time_windows(self) -> npt.NDArray[np.int_]:
        return self._time_windows

    @property
    def service_times(self) -> npt.NDArray[np.int_]:
        return self._service_times

    @property
    def duration_matrix(self) -> npt.NDArray[np.int_]:
        return self._duration_matrix

    @property
    def release_times(self) -> npt.NDArray[np.int_]:
        return self._release_times

    @property
    def customer_idx(self) -> npt.NDArray[np.int_]:
        return self._customer_idx

    @property
    def request_idx(self) -> npt.NDArray[np.int_]:
        return self._request_idx

    @property
    def dispatch_times(self) -> npt.NDArray[np.int_]:
        return self._dispatch_times

    @property
    def must_dispatch(self) -> npt.NDArray[np.bool_]:
        return self._must_dispatch

    @property
    def prizes(self) -> npt.NDArray[np.int_]:
        return self._prizes

    @property
    def horizon(self) -> int:
        return self._time_windows[0, 1]

    @property
    def dimension(self) -> int:
        return self._coords.shape[0]

    def replace(
        self,
        is_depot: Optional[npt.NDArray[np.bool_]] = None,
        coords: Optional[npt.NDArray[np.int_]] = None,
        demands: Optional[npt.NDArray[np.int_]] = None,
        capacity: Optional[int] = None,
        time_windows: Optional[npt.NDArray[np.int_]] = None,
        service_times: Optional[npt.NDArray[np.int_]] = None,
        duration_matrix: Optional[npt.NDArray[np.int_]] = None,
        release_times: Optional[npt.NDArray[np.int_]] = None,
        customer_idx: Optional[npt.NDArray[np.int_]] = None,
        request_idx: Optional[npt.NDArray[np.int_]] = None,
        dispatch_times: Optional[npt.NDArray[np.int_]] = None,
        must_dispatch: Optional[npt.NDArray[np.bool_]] = None,
        prizes: Optional[npt.NDArray[np.int_]] = None,
    ) -> "VrpInstance":
        return VrpInstance(
            is_depot=_copy_if_none(is_depot, self.is_depot),
            coords=_copy_if_none(coords, self.coords),
            demands=_copy_if_none(demands, self.demands),
            capacity=_copy_if_none(capacity, self.capacity),
            time_windows=_copy_if_none(time_windows, self.time_windows),
            service_times=_copy_if_none(service_times, self.service_times),
            duration_matrix=_copy_if_none(
                duration_matrix, self.duration_matrix
            ),
            release_times=_copy_if_none(release_times, self.release_times),
            customer_idx=_copy_if_none(customer_idx, self.customer_idx),
            request_idx=_copy_if_none(request_idx, self.request_idx),
            dispatch_times=_copy_if_none(dispatch_times, self.dispatch_times),
            must_dispatch=_copy_if_none(must_dispatch, self.must_dispatch),
            prizes=_copy_if_none(prizes, self.prizes),
        )

    def filter(self, mask: npt.NDArray[np.bool_]) -> "VrpInstance":
        """
        Filters requests from the current instance using a given mask and
        returns a new VrpInstance.

        Parameters
        ----------
        mask : npt.NDArray[np.bool_]
            A boolean mask of length ``dimension`` indicating which requests
            should be kept.

        Returns
        -------
        VrpInstance
            A new VrpInstance with the filtered requests.
        """

        return self.replace(
            is_depot=self.is_depot[mask],
            coords=self.coords[mask, :],
            demands=self.demands[mask],
            time_windows=self.time_windows[mask, :],
            service_times=self.service_times[mask],
            duration_matrix=self.duration_matrix[mask][:, mask],
            release_times=self.release_times[mask],
            customer_idx=self.customer_idx[mask],
            request_idx=self.request_idx[mask],
            dispatch_times=self.dispatch_times[mask],
            must_dispatch=self.must_dispatch[mask],
            prizes=self.prizes[mask],
        )


def _set_if_none(value, default):
    return default if value is None else value


def _copy_if_none(value, to_be_copied):
    return copy(to_be_copied) if value is None else value
