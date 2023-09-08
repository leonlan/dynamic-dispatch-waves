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
    capacity: float
    time_windows: npt.NDArray[np.int_]
    service_times: npt.NDArray[np.int_]
    duration_matrix: npt.NDArray[np.int_]
    release_times: npt.NDArray[np.int_]
    customer_idx: Optional[npt.NDArray[np.int_]] = None
    request_idx: Optional[npt.NDArray[np.int_]] = None
    dispatch_times: Optional[npt.NDArray[np.int_]] = None
