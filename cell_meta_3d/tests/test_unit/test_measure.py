import numpy as np

from cell_meta_3d.measure import CellSizeCalc, gaussian_func


def test_empty_cube():
    cube = np.zeros((20, 50, 50))
    calc = CellSizeCalc(cube_size=(100, 50, 50), voxel_size=(5, 1, 1))
    center, r_lat_data, r_axial_data, lat_line, ax_line = calc(cube[None, ...])
    z, y, x = center[0, :]

    line = np.arange(lat_line.shape[1])
    assert np.all(np.abs(gaussian_func(line, *r_lat_data[0, 1:])) < 0.2)
    line = np.arange(ax_line.shape[1])
    assert np.all(np.abs(gaussian_func(line, *r_axial_data[0, 1:])) < 0.2)

    assert np.allclose(lat_line, 0)
    assert np.allclose(ax_line, 0)
    assert z == 10
    assert y == 25
    assert x == 25
