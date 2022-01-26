import numpy as np
from nuclear_mpm import nclr_grid_op


def grid_op(
    grid_resolution: int,
    dx: float,
    dt: float,
    gravity: float,
    grid_velocity: np.ndarray,
    grid_mass: np.ndarray,
):
    v_allowed = dx * 0.9 / dt
    boundary = 3
    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1):
            for k in range(grid_resolution + 1):
                if grid_mass[i, j, k][0] > 0:
                    grid_velocity[i, j, k] /= grid_mass[i, j, k][0]
                    grid_velocity[i, j, k][1] += dt * gravity

                    grid_velocity[i, j, k] = np.clip(
                        grid_velocity[i, j, k], -v_allowed, v_allowed
                    )

                I = [i, j, k]
                for d in range(3):
                    if I[d] < boundary and grid_velocity[i, j, k][d] < 0:
                        grid_velocity[i, j, k][d] = 0
                    if (
                        I[d] >= grid_resolution - boundary
                        and grid_velocity[i, j, k][d] > 0
                    ):
                        grid_velocity[i, j, k][d] = 0


def test_grid_op():
    res = 10
    gv = np.zeros((res + 1, res + 1, res + 1, 3))
    gm = np.ones((res + 1, res + 1, res + 1, 1))

    grid_op(res, 1 / res, 1e-4, -200, gv, gm)

    ggv = np.zeros((res + 1, res + 1, res + 1, 3), dtype=np.float64)
    ggm = np.ones((res + 1, res + 1, res + 1, 1), dtype=np.float64)
    nclr_grid_op(res, 3.0, 1 / res, 1e-4, -200.0, ggv, ggm)

    assert np.isclose(gv[gv.nonzero()], ggv[ggv.nonzero()]).all()
    assert np.isclose(gm[gm.nonzero()], ggm[ggm.nonzero()]).all()
