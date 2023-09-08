from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass
class VrpInstance:
    """
    A static VRP instance.
    """

    is_depot: npt.NDArray[np.bool_]
    coords: npt.NDArray[np.int_]
    demands: npt.NDArray[np.int_]
    capacity: int
    time_windows: npt.NDArray[np.int_]
    service_times: npt.NDArray[np.int_]
    duration_matrix: npt.NDArray[np.int_]
    release_times: npt.NDArray[np.int_]
    customer_idx: Optional[npt.NDArray[np.int_]] = None
    request_idx: Optional[npt.NDArray[np.int_]] = None
    dispatch_times: Optional[npt.NDArray[np.int_]] = None
    must_dispatch: Optional[npt.NDArray[np.bool_]] = None
    epoch: Optional[npt.NDArray[np.int_]] = None
    prizes: Optional[npt.NDArray[np.int_]] = None


@dataclass
class EpochInstance(VrpInstance):
    customer_idx: npt.NDArray[np.int_] = np.array([])
    request_idx: npt.NDArray[np.int_] = np.array([])
    must_dispatch: npt.NDArray[np.bool_] = np.array([])
    epoch: npt.NDArray[np.int_] = np.array([])
