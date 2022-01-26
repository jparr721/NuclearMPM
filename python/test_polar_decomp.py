import numpy as np
from nuclear_mpm import nclr_polar
from scipy.linalg import polar


def test_polar_decomp_1000():
    for _ in range(1000):
        a = np.random.rand(3, 3)
        r, s = polar(a)
        rr, ss = nclr_polar(a)
        assert np.isclose(r, rr).all()
        assert np.isclose(s, ss).all()
