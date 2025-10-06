import numpy as np

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
