import os
import pickle
import re
from collections import defaultdict
from typing import Dict

import numpy as np
from loguru import logger
from tqdm import tqdm


def to_3d(idx: int, max_: int):
    x = idx // max_
    idx -= x * max_
    y = idx % max_
    return x, y


class SimResult(object):
    def __init__(self):
        self.x: np.ndarray
        self.v: np.ndarray
        self.F: np.ndarray
        self.C: np.ndarray
        self.Jp: float
        self.lame: np.ndarray
        self.velocity: np.ndarray
        self.mass: np.ndarray
        self.timestep: float


def process_tmp(tmp: str):
    logger.info(f"Loading from {tmp}")

    def process_valuekey(fullpath: str, valuekey: str):
        if valuekey == "timestep":
            timestep = np.loadtxt(fullpath)
            return timestep
        if valuekey == "x":
            x = np.loadtxt(fullpath)
            x = x.reshape(len(x) // 2, 2)
            return x
        if valuekey == "v":
            v = np.loadtxt(fullpath)
            v = v.reshape(len(v) // 2, 2)
            return v
        if valuekey == "F":
            F = np.loadtxt(fullpath)
            F = F.reshape(F.shape[0] // 2, 2, 2)
            return F
        if valuekey == "C":
            C = np.loadtxt(fullpath)
            C = C.reshape(C.shape[0] // 2, 2, 2)
            return C
        if valuekey == "Jp":
            Jp = np.loadtxt(fullpath)
            return Jp
        if valuekey == "lame":
            lame = np.loadtxt(fullpath)
            return lame
        if valuekey == "mass":
            res = 65
            mass = np.loadtxt(fullpath)
            mass = mass.reshape(res, res)
            return mass

        if valuekey == "velocity":
            res = 65
            velocity = np.loadtxt(fullpath)
            velocity = velocity.reshape(res, res, 2)
            return velocity
        else:
            logger.error(f"ValueKey {valuekey} is invalid")

    # Round up all the dict keys so we can process filenames more easily.
    filepaths = []
    for f in os.listdir(tmp):
        filepaths.append(f)

    filepaths.sort(
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split("(\\d+)", s)],
    )
    filepaths = [tmp + "/" + keyname for keyname in filepaths]

    results: Dict[str, SimResult] = defaultdict(SimResult)

    for fullpath in tqdm(filepaths):
        # *_...txt
        fname = os.path.basename(fullpath)

        # 0 ...
        n, end = fname.split("_")

        # Our dict key for the results
        dictkey = f"{n}_"

        # value, extension
        valuekey, _ = end.split(".")

        results[dictkey].__dict__[valuekey] = process_valuekey(fullpath, valuekey)

    logger.success("Files loaded, saving pickle")
    with open("results.pickle", "wb+") as output:
        pickle.dump(results, output)
    logger.success("Done")
