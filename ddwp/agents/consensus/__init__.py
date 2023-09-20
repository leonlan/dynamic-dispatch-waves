from typing import Callable

from .ConsensusFunction import ConsensusFunction
from .fixed_threshold import fixed_threshold
from .hamming_distance import hamming_distance
from .prize_collecting import prize_collecting

CONSENSUS: dict[str, Callable] = {
    "fixed_threshold": fixed_threshold,
    "hamming_distance": hamming_distance,
    "prize_collecting": prize_collecting,
}
