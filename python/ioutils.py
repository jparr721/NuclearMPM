import os
from dataclasses import dataclass

import numpy as np


@dataclass
class SimResult(object):
    x: np.ndarray
    v: np.ndarray
    F: np.ndarray
    C: np.ndarray
    Jp: float
    lame: np.ndarray


def process_tmp():
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")

    keys = []
    for f in os.listdir(tmp):
        n, _ = f.split("_")
        keys.append(int(n))

    keys = list(sorted(set(keys)))
