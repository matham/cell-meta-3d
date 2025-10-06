import numpy as np
import pytest

from cell_meta_3d.measure import CellSizeCalc, gaussian_func


def test_empty_cube():
    cube = np.zeros((20, 50, 50))
    calc = CellSizeCalc(cube_size_um=(100, 50, 50), voxel_size=(5, 1, 1))
    (
        center,
        r_lat,
        lat_line,
        r_lat_params,
        r_axial,
        ax_line,
        r_axial_params,
    ) = calc(cube[None, ...])
    z, y, x = center[0, :]

    line = np.arange(lat_line.shape[1])
    params = r_lat_params[0]
    assert np.all(
        np.abs(
            gaussian_func(
                line,
                params["a"],
                params["offset"],
                params["sigma"],
                params["c"],
            )
        )
        < 0.2
    )
    line = np.arange(ax_line.shape[1])
    params = r_axial_params[0]
    assert np.all(
        np.abs(
            gaussian_func(
                line,
                params["a"],
                params["offset"],
                params["sigma"],
                params["c"],
            )
        )
        < 0.2
    )

    assert np.allclose(lat_line, 0)
    assert np.allclose(ax_line, 0)
    assert z == 10
    assert y == 25
    assert x == 25


@pytest.mark.parametrize(
    "x_start,y_max,x_r", [(-1, 1, 3.33), (0, 1, 3.33), (1, 0.883, 2.48)]
)
def test_shifted_gauss_center(x_start, y_max, x_r):
    calc = CellSizeCalc()

    x = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    x = x[x_start + 1 :]
    y = gaussian_func(x, 1, 0, 2, 0)
    r, [a, offset, sigma, c], _ = calc.get_radius_from_gaussian(
        y,
        decay_fraction=0.25,
        max_n=10,
        left_max_offset=-1,
    )

    assert np.isclose(y_max, np.max(y), atol=0.1, rtol=0.05)
    assert np.isclose(r, x_r, atol=0.1, rtol=0.05)
    assert np.isclose(x_start, -offset, atol=0.1, rtol=0.05)
