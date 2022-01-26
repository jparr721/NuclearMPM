import numpy as np
from nuclear_mpm import nclr_polar, nclr_stress


def check(
    F: np.ndarray,
    inv_dx: float,
    mu: float,
    lambda_: float,
    dt: float,
    volume: float,
    mass: float,
    C: np.ndarray,
):
    J = np.linalg.det(F)

    r, _ = nclr_polar(F)

    D_inv = 4 * inv_dx * inv_dx

    PF = (2 * mu * (F - r) @ F.T) + lambda_ * (J - 1) * J
    stress = -(dt * volume) * (D_inv * PF)
    return stress + mass * C


def test_stress_1000():
    for _ in range(1000):
        f = np.random.rand(3, 3)
        c = np.random.rand(3, 3)
        inv_dx = 1 / 64
        mu = 20
        lambda_ = 30
        dt = 1e-4
        volume = 1
        mass = 1

        assert np.isclose(
            check(f, inv_dx, mu, lambda_, dt, volume, mass, c),
            nclr_stress(f, inv_dx, mu, lambda_, dt, volume, mass, c),
        ).all()
