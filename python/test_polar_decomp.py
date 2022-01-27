import numpy as np
from nuclear_mpm import nclr_polar


def polar(m: np.ndarray):
    """Perform polar decomposition (A=UP) for 3x3 matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Polar_decomposition.

    Args:
        m (np.ndarray): input 3x3 matrix `m`.

    Returns:
        Decomposed 3x3 matrices `U` and `P`.
    """
    w, s, vh = np.linalg.svd(m, full_matrices=False)

    u = w.dot(vh)
    # a = up
    p = (vh.T.conj() * s).dot(vh)
    return u, p


def test_polar_decomp_1000():
    for _ in range(1000):
        a = np.random.rand(3, 3)
        r, s = polar(a)

        rr, ss = nclr_polar(a)
        assert np.isclose(r, rr).all()
        assert np.isclose(s, ss).all()
