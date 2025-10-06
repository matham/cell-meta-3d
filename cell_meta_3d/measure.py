import math
from collections.abc import Callable, Sequence
from multiprocessing import shared_memory
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit


def _expand_num_triplet(
    value: float | tuple[float, float, float],
) -> tuple[float, float, float]:
    if not isinstance(value, Sequence):
        return value, value, value
    return tuple(value)


def _arr_index(n: int, indices: Sequence[int], values: Sequence) -> tuple:
    index = [
        slice(None),
    ] * n
    for i, value in zip(indices, values, strict=True):
        index[i] = value
    return tuple(index)


def _norm_by_size(
    value: tuple[float, float, float], size: tuple[float, float, float]
) -> tuple[int, int, int]:
    normed = (
        int(round(value[0] / size[0])),
        int(round(value[1] / size[1])),
        int(round(value[2] / size[2])),
    )
    return normed


def gaussian_func(x, a, offset, sigma, c):
    return a * np.exp(-np.square(x - offset) / (2 * sigma**2)) + c


_gauss_names = ["a", "offset", "sigma", "c"]
_gauss_names += [f"{name}_std" for name in _gauss_names]


def default_center(ax: int, size: int) -> int:
    return int(round(size / 2))


class CellSizeCalc:

    axial_dim: int
    voxel_size: tuple[float, float, float]
    cube_voxels: tuple[int, int, int]

    initial_center_search_radius_voxels: tuple[int, int, int]
    """The radius around the center, in each dim, along which to search for
    a higher central intensity. If we find a better intensity (using the volume
    initial_center_search_volume_voxels around that point), we use that as the
    new cell center.

    The radius is in addition to the center. E.g. if the radius is `2`, then
    we'll consider for a total of 5 center points: the center, and 2 voxels on
    each side of the original center.

    Unit is in voxels.
    """
    initial_center_search_volume_voxels: tuple[int, int, int]
    """The volume size to use when searching for the best center. We compute
    the overall intensity of a volume of this size, around a potential cell
    center. A value of zero, means just use the center voxel. Otherwise, it's
    centered on the center voxel. E.g. a value of 1 would be the center voxel
    plus a voxel on one side.

    Unit is in voxels.
    """

    lateral_intensity_algorithm: Literal["center_line", "area", "area_margin"]
    lateral_max_radius_voxels: int
    """From the (adjusted) center of the cell, we get the axial plane located
    at that center. Then, from the lateral center, we extract the 4 lines
    within the plane that start from the center and extend radially outward.
    These 4 lines are in each direction for the 2 lateral axes.

    `lateral_max_radius_voxels` indicates how many voxels (not including the
    center) to use to extend that line in any direction. The average of the 4
    lines is then used to estimate the intensity drop-off from the center.
    """
    lateral_decay_algorithm: Literal["gaussian", "manual"]
    lateral_decay_len_voxels: int
    lateral_decay_fraction: float

    axial_intensity_algorithm: Literal[
        "center_line", "volume", "volume_margin"
    ]
    axial_max_radius_voxels: int
    axial_decay_algorithm: Literal["gaussian", "manual"]
    axial_decay_len_voxels: int
    axial_decay_fraction: float

    decay_gaussian_bounds: Sequence[float]

    # the center of the cube
    cube_center_voxels: np.ndarray

    # 4d tuple of slices, indicating the sub-volume of the cube we need, in
    # order to search for a better cell center. For each dim (first is
    # slice(None) for batch dim) it's the slice to extract the volume of that
    # dim
    _center_search_data_indices: tuple[slice, slice, slice, slice] | None
    # the index in the cube of the first (start) voxel we consider as new
    # center for each of the 3d. E.g. if the original center is 10, and we
    # search a radius of 2 on each side, this would be 8 for that dim. It's
    # cube_center_voxels if there's no shifts. It has an extra first dim
    _center_search_start: np.ndarray

    # the np masks buffer that helps us calc circles of different sizes, at
    # different offsets from the original cube center
    _circle_masks: np.ndarray | None = None
    # The underlying _circle_masks buffer is a shared memory so it can be
    # shared across multiple processes without each needed a full copy
    _circle_masks_buffer: shared_memory.SharedMemory | None = None
    # only in the main process that created the buffer is this true
    _circle_masks_buffer_created: bool = False
    _sphere_masks: np.ndarray

    _gauss_dtype = np.dtype([(name, np.float64) for name in _gauss_names])

    def __init__(
        self,
        axial_dim: int = 0,
        voxel_size: tuple[float, float, float] = (5, 1, 1),
        cube_size_um: float | tuple[float, float, float] = (100, 50, 50),
        cuboid_center_func: Callable[[int, int], int] = default_center,
        initial_center_search_radius_um: float | tuple[float, float, float] = (
            10,
            3,
            3,
        ),
        initial_center_search_volume_um: float | tuple[float, float, float] = (
            15,
            3,
            3,
        ),
        lateral_intensity_algorithm: Literal[
            "center_line", "area", "area_margin"
        ] = "area_margin",
        lateral_max_radius_um: float = 20,
        lateral_decay_length_um: float = 12,
        lateral_decay_fraction: float = 1 / math.e,
        lateral_decay_algorithm: Literal["gaussian", "manual"] = "gaussian",
        axial_intensity_algorithm: Literal[
            "center_line", "volume", "volume_margin"
        ] = "center_line",
        axial_max_radius_um: float = 35,
        axial_decay_length_um: float = 35,
        axial_decay_fraction: float = 1 / math.e,
        axial_decay_algorithm: Literal["gaussian", "manual"] = "gaussian",
        decay_gaussian_bounds: Sequence[float] = (
            0.1,
            1.0,
            -0.25,
            3,
            0.1,
            10.0,
            -1,
            1,
        ),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axial_dim = axial_dim
        self.voxel_size = voxel_size
        self.lateral_intensity_algorithm = lateral_intensity_algorithm
        self.lateral_decay_algorithm = lateral_decay_algorithm
        self.lateral_decay_fraction = lateral_decay_fraction
        self.axial_intensity_algorithm = axial_intensity_algorithm
        self.axial_decay_algorithm = axial_decay_algorithm
        self.axial_decay_fraction = axial_decay_fraction

        cube_size_um = _expand_num_triplet(cube_size_um)
        self.cube_voxels = _norm_by_size(cube_size_um, voxel_size)

        initial_center_search_radius_um = _expand_num_triplet(
            initial_center_search_radius_um
        )
        self.initial_center_search_radius_voxels = _norm_by_size(
            initial_center_search_radius_um, voxel_size
        )

        initial_center_search_volume_um = _expand_num_triplet(
            initial_center_search_volume_um
        )
        self.initial_center_search_volume_voxels = _norm_by_size(
            initial_center_search_volume_um, voxel_size
        )

        lat_voxels = [r for i, r in enumerate(voxel_size) if i != axial_dim]
        lat_vox = sum(lat_voxels) / 2
        axial_vox = voxel_size[axial_dim]

        self.lateral_max_radius_voxels = int(
            round(lateral_max_radius_um / lat_vox)
        )
        self.lateral_decay_len_voxels = int(
            round(lateral_decay_length_um / lat_vox)
        )

        self.axial_max_radius_voxels = int(
            round(axial_max_radius_um / axial_vox)
        )
        self.axial_decay_len_voxels = int(
            round(axial_decay_length_um / axial_vox)
        )

        self.cube_center_voxels = np.array(
            [cuboid_center_func(i, v) for i, v in enumerate(self.cube_voxels)],
            dtype=np.int_,
        )

        self.decay_gaussian_bounds = decay_gaussian_bounds

        self._verify_lateral_parameters()
        self._verify_axial_parameters()

        self._calc_find_pos_center_window()
        if self.lateral_intensity_algorithm.startswith("area"):
            self._make_circle_masks()

        if self.axial_intensity_algorithm.startswith("volume"):
            self._make_sphere_masks()

    def __del__(self):
        # we have to close the ref to the shared memory
        if self._circle_masks_buffer is not None:
            # every instance must close the ref
            self._circle_masks_buffer.close()
            if self._circle_masks_buffer_created:
                # in the main process that created it, we must also fully
                # delete it. Presumably, when this instance is deleted, all
                # the sub-processes are already closed, otherwise if they try
                # to access the memory it may crash
                self._circle_masks_buffer.unlink()
            self._circle_masks_buffer = None

    def __getstate__(self):
        state = self.__dict__.copy()

        # when copying the instance to create a new one, use underlying shared
        # mem for masks
        state["_circle_masks_buffer_created"] = False
        if state["_circle_masks_buffer"] is not None:
            # we need the name of the buffer and its shape for __setstate__
            state["_circle_masks_buffer"] = state["_circle_masks_buffer"].name
        if state["_circle_masks"] is not None:
            state["_circle_masks"] = state["_circle_masks"].shape
        return state

    def __setstate__(self, state):
        if state["_circle_masks_buffer"] is not None:
            # see __getstate__. We share the underlying memory
            shm = shared_memory.SharedMemory(
                name=state["_circle_masks_buffer"],
                create=False,
                track=False,
            )
            state["_circle_masks_buffer"] = shm

        if (
            state["_circle_masks"] is not None
            and state["_circle_masks_buffer"] is not None
        ):
            masks = np.ndarray(
                state["_circle_masks"],
                dtype=np.bool,
                buffer=state["_circle_masks_buffer"].buf,
            )
            state["_circle_masks"] = masks
        else:
            state["_circle_masks"] = None

        self.__dict__.update(state)

    @property
    def lateral_dims(self) -> list[int]:
        """Returns the dim indices of the data, that are lateral dimensions."""
        return [i for i in range(3) if i != self.axial_dim]

    def _parameters_names(
        self, algorithm: Literal["gaussian", "manual"]
    ) -> list[str]:
        match algorithm:
            case "gaussian":
                return _gauss_names
            case "manual":
                return []
            case _:
                raise ValueError

    @property
    def lateral_parameters_names(self):
        return self._parameters_names(self.lateral_decay_algorithm)

    @property
    def axial_parameters_names(self):
        return self._parameters_names(self.lateral_decay_algorithm)

    @property
    def lateral_line_length(self):
        return self.lateral_max_radius_voxels + 1

    @property
    def axial_line_length(self):
        return self.axial_max_radius_voxels + 1

    def __call__(self, data: np.ndarray) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
    ]:
        """
        Ideally, we would shift the center by the amount estimate during
        radius estimation (e.g. with Gaussian), but currently we don't know
        the direction of the shift.

        :param data: Shape is 4D: batch, and the 3 data dims in the same order
            as the input parameters (e.g. `voxel_size`).
        :return:
        """
        if len(data.shape) != 4:
            # batch dim is first
            raise ValueError
        if data.shape[1:] != self.cube_voxels:
            raise ValueError
        # get the 3d indices of the (better) cube centers
        center = self.find_pos_center_max(data)

        # get the intensity decay line from the center in the lateral direction
        match self.lateral_intensity_algorithm:
            case "center_line":
                lat_line = self.get_center_2d_falloff_line(data, center)
            case "area":
                lat_line = self.get_area_falloff_line(data, center, False)
            case "area_margin":
                lat_line = self.get_area_falloff_line(data, center, True)
            case _:
                raise ValueError

        # calculate the lateral radius based on the decay
        r_lat, r_lat_params = self._get_decay_radius(
            lat_line,
            self.lateral_decay_algorithm,
            self.lateral_decay_fraction,
            self.lateral_decay_len_voxels + 1,
        )

        # get the intensity decay line from the center in the axial direction
        match self.axial_intensity_algorithm:
            case "center_line":
                ax_line = self.get_center_1d_falloff_line(data, center)
            case "volume":
                ax_line = self.get_volume_falloff_line(data, center, False)
            case "volume_margin":
                ax_line = self.get_volume_falloff_line(data, center, True)
            case _:
                raise ValueError

        # calculate the axial radius based on the decay
        r_axial, r_axial_params = self._get_decay_radius(
            ax_line,
            self.axial_decay_algorithm,
            self.axial_decay_fraction,
            self.axial_decay_len_voxels + 1,
        )

        return (
            center,
            r_lat,
            lat_line,
            r_lat_params,
            r_axial,
            ax_line,
            r_axial_params,
        )

    def _get_decay_radius(
        self,
        line: np.ndarray,
        algorithm: Literal["gaussian", "manual"],
        fraction: float,
        len_voxels: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        params_arr = None
        r_arr = np.empty(len(line), dtype=np.float64)

        match algorithm:
            case "gaussian":
                params_arr = np.empty(len(line), dtype=self._gauss_dtype)

                for i in range(len(line)):
                    r, params, err = self.get_radius_from_gaussian(
                        line[i, :],
                        fraction,
                        len_voxels,
                        *self.decay_gaussian_bounds,
                    )
                    r_arr[i] = r
                    params_arr[_gauss_names][i] = *params, *err
            case "manual":
                r = self.get_radius_from_decay(
                    line,
                    fraction,
                    len_voxels,
                )
                r_arr[:] = r
            case _:
                raise ValueError

        return r_arr, params_arr

    def _calc_find_pos_center_window(self) -> None:
        """Pre-calculates the sub-regions of the cuboid we need in order to
        search for a new and better center.
        """
        center_data_indices = [slice(None)]
        center_search_offset = []

        # only pre-calc if we adjust the cube center in at least one direction
        if not any(self.initial_center_search_radius_voxels):
            self._center_search_data_indices = None
            # no shift
            self._center_search_start = self.cube_center_voxels[None, :]
            return

        for c, dim, sides, win in zip(
            self.cube_center_voxels,
            self.cube_voxels,
            self.initial_center_search_radius_voxels,
            self.initial_center_search_volume_voxels,
            strict=True,
        ):
            # we search sides voxels on each side, not including the original c
            if not sides:
                start = end = c
            else:
                start = c - sides
                end = c + sides + 1

            # win of zero means just use center voxel
            if win < 0:
                raise ValueError(
                    f"Requested a center search volume width of {win} voxels. "
                    f"We need at least a window of 0 voxels"
                )
            # split window into (unequal) halves (if not even)
            half_win = win // 2
            rest_win = win - half_win

            # we have to add to start that many voxels, not including start
            start -= half_win
            # the center is inclusive so adding rest_win, we have window size
            # of win + 1. We need +1 because win of zero means just the center
            end += rest_win

            center_data_indices.append(slice(start, end))
            if start < 0 or end > dim:
                raise ValueError(
                    f"The cube size of {dim} voxels is too small for a center "
                    f"adjustment search of {sides} voxels on each side, with a"
                    f" volume window size of {win} voxels. For a cube center "
                    f"of {c} voxels, we would have needed a sub-region of "
                    f"[{start}, {end}) with size {end - start} voxels, which "
                    f"extends beyond the full cube"
                )

            center_search_offset.append(c - sides)

        self._center_search_data_indices = tuple(center_data_indices)
        self._center_search_start = np.array(
            [center_search_offset], dtype=np.int_
        )

    def find_pos_center_max(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Searches the cube for a better center of the cell, by looking around
        the center for a brighter area and picking its center voxel.

        :param data: A 4D array of batch X 3 dimensions with the cube
            intensity values.
        :return: A 2D array of Nx3 containing the index to the new
            center of the cell in the cube. The original center is
            `cube_center_voxels`. If we don't find a better center, it stays
            the same.
        """
        n = len(data)

        if self._center_search_data_indices is None:
            # no search - the center is the original center. Just add batch dim
            max_idx = self.cube_center_voxels[None, ...]
            return np.repeat(max_idx, n, axis=0)

        # extract the padded 3d sub-region to search
        data = data[self._center_search_data_indices]
        # create a bunch of 3d sliding windows across the 3 data dims. For each
        # voxel in the data (not including the padding, only in the search
        # radius) we generate a volume of the asked size. So we end up with 7D.
        # The 1st dim is batch, 2nd - 4th the search area (e.g. if the search r
        # for a dim is 1, the size of that dim is 3), 5th - 7th is the win_size
        # We add one to window, because zero means just the center voxel etc.
        win_size = [v + 1 for v in self.initial_center_search_volume_voxels]
        windows = sliding_window_view(data, win_size, axis=(1, 2, 3))
        # for each voxel in the search are, add up the total intensity of the
        # volume around that voxel. Back to 4D of the search area size
        intensity = np.sum(windows, axis=(4, 5, 6), dtype=np.float64)

        # flatten the search area so we can search more easily
        flat_intensity = intensity.reshape((n, -1))
        # check if all intensities are the same for a given cube
        all_same = np.all(
            flat_intensity[:, 0][:, None] == flat_intensity, axis=1
        )

        flat_max = flat_intensity.argmax(axis=1)
        # convert back to 2D indices, with 2nd dim the max index of each cube
        max_idx = np.column_stack(
            np.unravel_index(flat_max, intensity.shape[1:])
        )
        assert len(max_idx) == n

        # the indices of the max is relative to the start of the search area
        max_idx += self._center_search_start
        # set it back to center if they were all the same
        max_idx = np.add(
            max_idx,
            [self.initial_center_search_radius_voxels],
            out=max_idx,
            where=all_same[:, None],
        )
        return max_idx

    def _verify_lateral_parameters(self):
        center_offsets = [
            self.initial_center_search_radius_voxels[i]
            for i in self.lateral_dims
        ]
        center = self.cube_center_voxels
        sizes = [self.cube_voxels[i] for i in self.lateral_dims]
        r = self.lateral_max_radius_voxels
        decay = self.lateral_decay_len_voxels

        if decay > r:
            raise ValueError(
                f"Requested fit of line with size {decay} voxels. This is "
                f"larger than the size of the requested lateral line of {r}"
                f" voxels"
            )

        for c, c_offset, size in zip(
            center, center_offsets, sizes, strict=False
        ):
            # number of elements inclusive of the center
            right = size - (c + c_offset)
            # same - number of elements inclusive of the center
            left = c - c_offset + 1

            # don't need to check for decay because decay <= r
            # number of elements is center plus r
            if right + r >= size:
                raise ValueError(
                    f"Requested lateral line with size {r} voxels and "
                    f"potential center offset of {c_offset} voxels. This is "
                    f"larger than the size of the cube {size} voxels"
                )
            if left - r < 0:
                raise ValueError(
                    f"Requested lateral line with size {r} voxels and "
                    f"potential center offset of negative {c_offset} voxels. "
                    f"This is larger than the size of the cube {size} voxels"
                )

    def _verify_axial_parameters(self):
        c_offset = self.initial_center_search_radius_voxels[self.axial_dim]
        c = self.cube_center_voxels[self.axial_dim]
        size = self.cube_voxels[self.axial_dim]
        r = self.axial_max_radius_voxels
        decay = self.axial_decay_len_voxels

        if decay > r:
            raise ValueError(
                f"Requested fit of line with size {decay} voxels. This is "
                f"larger than the size of the requested axial line of {r}"
                f" voxels"
            )

        # number of elements inclusive of the center
        right = size - (c + c_offset)
        # same - number of elements inclusive of the center
        left = c - c_offset + 1

        # don't need to check for decay because decay <= r
        # number of elements is center plus r
        if right + r >= size:
            raise ValueError(
                f"Requested axial line with size {r} voxels and "
                f"potential center offset of {c_offset} voxels. This is "
                f"larger than the size of the cube {size} voxels"
            )
        if left - r < 0:
            raise ValueError(
                f"Requested axial line with size {r} voxels and "
                f"potential center offset of negative {c_offset} voxels. "
                f"This is larger than the size of the cube {size} voxels"
            )

    def get_center_2d_falloff_line(
        self,
        data: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        """
        Uses the 4 lateral lines emanating from the center, in the center axial
        plane, to calculate the average intensity line starting from the
        center.

        :param data: A 4D array of batch X 3 dimensions with the cube
            intensity values.
        :param center: A 2D array of Nx3 containing the index to the
            center of the cell in the cube.
        :return: A 2D array of NxK. Where K is `lateral_max_radius_voxels` + 1.
            And for each cube contains the average lateral intensity starting
            from the center going outward.

            The line for each batch item is normalized to be in the [0, 1]
            range.
        """
        axial_axis = self.axial_dim
        # zero means just the center etc.
        n_points = self.lateral_max_radius_voxels + 1
        lat_axes = self.lateral_dims
        n = len(data)
        n_range = np.arange(n)

        # center is Nx3. Convert to N by getting the axial center value for
        # each batch item
        axial_center = center.take(axial_axis, axis=1)
        # these are the center values for the 1st and 2nd lat axes
        lat_c1 = center[:, lat_axes[0]]
        lat_c2 = center[:, lat_axes[1]]

        # data is 4D with first axis batch. Use centers to index the axial dim
        # to get the center plane for each batch item. We end up with 3-d array
        planes = data[
            _arr_index(4, [0, axial_axis + 1], [n_range, axial_center])
        ]
        # from the Nx3 planes, for the first lat axis, index the 2D array of
        # that axis at its center. This will return a line along the 2nd lat
        # axis at the center of the 1st lat axis. Same for the 2nd axis
        ax2_line = planes[_arr_index(3, [0, 1], [n_range, lat_c1])]
        ax1_line = planes[_arr_index(3, [0, 2], [n_range, lat_c2])]

        # For the line along the 2nd lat axis, get the subsection of the line
        # that starts at the center of that (2nd) axis and goes out by desired
        # radius from there
        line1 = ax2_line[
            n_range[:, None], lat_c2[:, None] + np.arange(n_points)[None, :]
        ]
        # similarly we do for the opposite direction of the line
        line2 = ax2_line[
            n_range[:, None],
            lat_c2[:, None] + np.arange(0, -n_points, -1)[None, :],
        ]
        # and the same for the first lat axis
        line3 = ax1_line[
            n_range[:, None], lat_c1[:, None] + np.arange(n_points)[None, :]
        ]
        line4 = ax1_line[
            n_range[:, None],
            lat_c1[:, None] + np.arange(0, -n_points, -1)[None, :],
        ]

        # divide now to be sure no overflow, if original values were large
        line = line1 / 4 + line2 / 4 + line3 / 4 + line4 / 4
        line -= line.min(axis=1, keepdims=True)

        out_line = np.zeros_like(line)
        max_val = line.max(axis=1, keepdims=True)
        np.divide(line, max_val, out=out_line, where=max_val > 0)

        return out_line

    def get_center_1d_falloff_line(
        self,
        data: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        """
        Uses the 2 lateral lines emanating from the center along the axial
        direction to calculate the average intensity line starting from the
        center.

        :param data: A 4D array of batch X 3 dimensions with the cube
            intensity values.
        :param center: A 2D array of Nx3 containing the index to the
            center of the cell in the cube.
        :return: A 2D array of NxK. Where K is `axial_max_radius_voxels` + 1.
            And for each cube contains the average axial intensity starting
            from the center going outward.

            The line for each batch item is normalized to be in the [0, 1]
            range.
        """
        axial_axis = self.axial_dim
        # zero means just the center etc.
        n_points = self.axial_max_radius_voxels + 1
        lat_axes = self.lateral_dims
        n = len(data)
        n_range = np.arange(n)

        # center is Nx3. Convert to N by getting the axial center value for
        # each batch item
        axial_center = center.take(axial_axis, axis=1)
        # these are the center values for the 1st and 2nd lat axes
        lat_c1 = center[:, lat_axes[0]]
        lat_c2 = center[:, lat_axes[1]]

        # data is 4-d with first axis batch. Use centers to index the 1st and
        # 2nd lateral dims to get the center axial line for each batch item
        planes = data[_arr_index(4, [0, lat_axes[0] + 1], [n_range, lat_c1])]
        # lat axes are ordered so 2nd axis is going to be shifted down by one
        # so no need to add 1 for batch dim
        lines = planes[_arr_index(3, [0, lat_axes[1]], [n_range, lat_c2])]

        # from the lateral center line, locate the axial center and get the
        # line from there on each direction
        line1 = lines[
            n_range[:, None],
            axial_center[:, None] + np.arange(n_points)[None, :],
        ]
        line2 = lines[
            n_range[:, None],
            axial_center[:, None] + np.arange(0, -n_points, -1)[None, :],
        ]

        # divide now to be sure no overflow, if original values were large
        line = line1 / 2 + line2 / 2
        line -= line.min(axis=1, keepdims=True)

        out_line = np.zeros_like(line)
        max_val = line.max(axis=1, keepdims=True)
        np.divide(line, max_val, out=out_line, where=max_val > 0)

        return out_line

    def _make_circle_masks(
        self,
    ) -> None:
        """
        Generates the masks we use in get_area_falloff_line to quickly
        calculate the intensity over a circle with possible offset from
        center.
        """
        # _verify_lateral_parameters ensures we have enough cube size
        max_r = self.lateral_max_radius_voxels
        r_off1, r_off2 = [
            self.initial_center_search_radius_voxels[i]
            for i in self.lateral_dims
        ]

        # dims are batch, c1, c2, mask_r, dim1, dim2
        masks_shape = (
            1,
            r_off1 * 2 + 1,
            r_off2 * 2 + 1,
            max_r + 1,
            max_r * 2 + 1 + r_off1 * 2,
            max_r * 2 + 1 + r_off2 * 2,
        )

        # create a shared memory numpy array, that is shared with sub-processes
        single_bytes = np.zeros(1, dtype=np.bool).nbytes
        total_bytes = math.prod(masks_shape) * single_bytes
        shm = shared_memory.SharedMemory(
            create=True, size=total_bytes, track=False
        )

        masks = np.ndarray(masks_shape, dtype=np.bool, buffer=shm.buf)
        masks[...] = 0

        # grid the plane with the largest coordinates we can have. Both max_r
        # and offsets are in addition to the center
        dist1 = np.arange(-max_r - r_off1, max_r + r_off1 + 1)[:, None]
        dist2 = np.arange(-max_r - r_off2, max_r + r_off2 + 1)[None, :]

        for off1 in range(-r_off1, r_off1 + 1):
            for off2 in range(-r_off2, r_off2 + 1):
                for r in range(max_r + 1):
                    dist = np.sqrt(
                        np.square(dist1 - off1) + np.square(dist2 - off2)
                    )
                    masks[0, off1 + r_off1, off2 + r_off2, r, :, :] = dist <= r

        self._circle_masks_buffer = shm
        self._circle_masks_buffer_created = True
        self._circle_masks = masks

    def _make_sphere_masks(
        self,
    ) -> None:
        axial_axis = self.axial_dim

        max_r_lat = self.lateral_max_radius_voxels
        max_r_ax = self.axial_max_radius_voxels

        r_off = self.initial_center_search_radius_voxels
        r_off1_lat, r_off2_lat = [
            v for i, v in enumerate(r_off) if i != axial_axis
        ]
        r_off_lat = max(r_off1_lat, r_off2_lat)
        r_off_ax = r_off[axial_axis]

        dim_ax = self.cube_voxels[axial_axis]
        c_ax = dim_ax // 2

        if (
            c_ax - max_r_ax - r_off_ax < 0
            or c_ax + max_r_ax + r_off_ax >= dim_ax
        ):
            raise ValueError

        # dims are batch, c1_lat, c2_lat, c_ax, r1_lat, r2_lat, mask_r, dim_ax
        masks = np.zeros(
            (
                1,
                r_off1_lat * 2 + 1,
                r_off2_lat * 2 + 1,
                r_off_ax * 2 + 1,
                max_r_lat + 1,
                max_r_ax + 1,
                max_r_ax * 2 + 1 + r_off_ax * 2,  # todo: order of axes
                max_r_lat * 2 + 1 + r_off_lat * 2,
                max_r_lat * 2 + 1 + r_off_lat * 2,
            )
        )

        dist1 = np.arange(-max_r_lat - r_off_lat, max_r_lat + r_off_lat + 1)[
            :, None
        ]
        dist2 = np.arange(-max_r_lat - r_off_lat, max_r_lat + r_off_lat + 1)[
            None, :
        ]
        plane_dims = self.lateral_dims
        dist3 = np.expand_dims(
            np.abs(np.arange(-max_r_ax - r_off_ax, max_r_ax + r_off_ax + 1)),
            plane_dims,
        )

        for off1_lat in range(-r_off1_lat, r_off1_lat + 1):
            dist1_ = dist1 - off1_lat
            for off2_lat in range(-r_off2_lat, r_off2_lat + 1):
                dist2_ = dist2 - off2_lat
                for off_ax in range(-r_off_ax, r_off_ax + 1):
                    dist3_ = dist3 - off_ax
                    for r_lat in range(max_r_lat + 1):
                        dist_12 = (
                            1
                            - (np.square(dist1_) + np.square(dist2_))
                            / r_lat**2
                        )
                        valid_12 = dist_12 >= 0
                        for r_ax in range(max_r_ax + 1):
                            dist3_max = np.ones_like(dist_12) * -1
                            dist3_max[valid_12] = np.sqrt(
                                dist_12[valid_12] * r_ax**2
                            )
                            masks[
                                0,
                                off1_lat + r_off1_lat,
                                off2_lat + r_off2_lat,
                                off_ax + r_off_ax,
                                r_lat,
                                r_ax,
                                :,
                                :,
                                :,
                            ] = dist3_ <= np.expand_dims(dist3_max, axial_axis)

        self._sphere_masks = masks

    def get_area_falloff_line(
        self,
        data: np.ndarray,
        center: np.ndarray,
        margin: bool,
    ) -> np.ndarray:
        """


        :param data: A 4D array of batch X 3 dimensions with the cube
            intensity values.
        :param center: A 2D array of Nx3 containing the index to the
            center of the cell in the cube.
        :param margin:
        :return: A 2D array of NxK. Where K is `lateral_max_radius_voxels` + 1.
            And for each cube contains the average lateral intensity of a
            circle with given radius starting from the center going outward.

            The line for each batch item is normalized to be in the [0, 1]
            range.
        """
        max_r = self.lateral_max_radius_voxels
        r_off1, r_off2 = [
            self.initial_center_search_radius_voxels[i]
            for i in self.lateral_dims
        ]
        c1, c2 = [self.cube_center_voxels[i] for i in self.lateral_dims]

        axial_axis = self.axial_dim
        lat_axes = self.lateral_dims
        # get the offset relative to most negative smallest center offset
        rel_center = center - self._center_search_start
        # these are the center values for the 1st and 2nd lat axes
        c1_rel = rel_center[:, lat_axes[0]]
        c2_rel = rel_center[:, lat_axes[1]]
        ax_c = center[:, axial_axis]

        n = len(data)
        n_zeros = np.zeros(n, dtype=np.int_)

        # get axial center plane for each batch item to end up with 3D data
        data_idx = _arr_index(4, [0, axial_axis + 1], [np.arange(n), ax_c])
        data = data[data_idx]
        # extract mask size area in the center of planes
        data = data[
            :,
            c1 - max_r - r_off1 : c1 + max_r + r_off1 + 1,
            c2 - max_r - r_off2 : c2 + max_r + r_off2 + 1,
        ]
        # select masks matching the correct center offset
        masks_idx = _arr_index(6, [0, 1, 2], [n_zeros, c1_rel, c2_rel])
        # masks will ow be 4D: batch, mask_r, dim1, dim2
        masks = self._circle_masks[masks_idx]

        # flatten dim1 and dim2 so we only have 3D: batch, mask_r, dim
        masks_flat = np.reshape(masks, (n, max_r + 1, -1))
        data_flat = np.reshape(data, (n, -1))[:, None, :]

        # calculate the masked intensity over the mask as well as the mask size
        # this gives us a 2D array of NxR
        intensity_sum = np.sum(
            masks_flat * data_flat, axis=2, dtype=np.float64
        )
        mask_size = np.sum(masks_flat, axis=2, dtype=np.float64)

        # average intensity of the mask
        intensity = intensity_sum / mask_size
        # if we want to calculate the intensity of the pixels in the current
        # circle that was not in the last circle, do it
        if margin:
            # only set for R larger than the first (0). For the first it's just
            # the original value
            intensity[:, 1:] = (
                intensity_sum[:, 1:] - intensity_sum[:, :-1]
            ) / (mask_size[:, 1:] - mask_size[:, :-1])

        intensity -= intensity.min(axis=1, keepdims=True)

        out_line = np.zeros_like(intensity)
        max_val = intensity.max(axis=1, keepdims=True)
        np.divide(intensity, max_val, out=out_line, where=max_val > 0)

        return out_line

    def get_radius_from_gaussian(
        self,
        data: np.ndarray,
        decay_fraction: float,
        max_n: int,
        min_scale: float = 0.1,
        max_scale: float = 1.0,
        left_max_offset: float = -0.25,
        right_max_offset: float = 3,
        min_sigma: float = 0.1,
        max_sigma: float = 10.0,
        min_y_offset: float = -1,
        max_y_offset: float = 1,
    ) -> tuple[float, list[float], list[float]]:
        data = data[:max_n]
        n = len(data)
        bounds = (
            [min_scale, left_max_offset, min_sigma, min_y_offset],
            [max_scale, right_max_offset, max_sigma, max_y_offset],
        )
        bad_result = -1, [0, 0, 0, 0], [0, 0, 0, 0]

        try:
            (a, offset, sigma, c), pcov = curve_fit(
                gaussian_func,
                np.arange(n),
                data,
                p0=[1, 0, 0.5 * (max_n - 1), 0],
                bounds=bounds,
            )
            perr = np.sqrt(np.diag(pcov))
        except (RuntimeError, ValueError):
            return bad_result

        assert a > 0
        if offset >= 0:
            # if max of the data is shifted to the right (i.e. data[i] >
            # data[0] for some i != 0), we treat it as if the max is at zero.
            # So we drop offset and solve for r relative to max, which is now
            # at i == 0. The radius is then the i relative to i == 0.

            # First, if desired drop-off cannot be reached, return bad value
            desired_val = decay_fraction * (a + c)
            if c > desired_val:
                return bad_result

            temp = (a * decay_fraction + (decay_fraction - 1) * c) / a
            temp = -2 * sigma**2 * math.log(temp)
            assert temp >= 0
            r = math.sqrt(temp)
        else:
            # the curve is shifted to left, so gaussian max is at i < 0 (i.e.
            # real data starts after max). So we find the i, where the curve is
            # at the desired fraction of curve at i == 0. That i is the radius.

            # First, if desired drop-off cannot be reached, return bad value
            y_0_unity = math.exp(-(offset**2) / (2 * sigma**2))
            y_0 = a * y_0_unity + c
            desired_val = decay_fraction * y_0
            if c > desired_val:
                return bad_result

            temp = decay_fraction * y_0_unity + (decay_fraction - 1) * c / a
            temp = -2 * sigma**2 * math.log(temp)
            assert temp >= 0
            r = math.sqrt(temp) + offset

        # r is relative to "offset" from center
        return r, [a, offset, sigma, c], perr.tolist()

    def get_radius_from_decay(
        self,
        data: np.ndarray,
        decay_fraction: float,
        max_n: int,
    ) -> np.ndarray:
        data = data[:, :max_n]

        less_mask = data <= decay_fraction
        has_max = np.any(less_mask, axis=1)

        r = np.where(has_max, np.argmax(less_mask, axis=1), -1)
        return r
