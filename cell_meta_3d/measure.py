import math
import sys
from collections.abc import Callable, Sequence
from functools import cache, cached_property
from multiprocessing import shared_memory
from typing import Literal

import numpy as np
import scipy.ndimage
import skimage
from cellfinder.core.detect.filters.volume.laplacian_filter import (
    get_27_stencil,
)
from scipy.optimize import curve_fit

_shared_mem_kwargs = {}
if (sys.version_info.major, sys.version_info.minor) >= (3, 13):
    _shared_mem_kwargs = {"track": False}


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
    """The radius around the center, in each dim, along which to iteratively
    search for a higher central intensity. If we find a better intensity at
    each step in the volume of size 1 around the current center, we use that
    as the new cell center for the iteration. Until it stabilizes.

    The radius is in addition to the center. E.g. if the radius is `2`, then
    we'll consider for a total of 5 center points: the center, and 2 voxels on
    each side of the original center.

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

    decay_fraction: float

    # the center of the cube
    cube_center_voxels: np.ndarray

    # the index in the cube of the first (start) voxel we consider as new
    # center for each of the 3d. E.g. if the original center is 10, and we
    # search a radius of 2 on each side, this would be 8 for that dim. It's
    # cube_center_voxels if there's no shifts. It has an extra first dim
    _center_search_start: np.ndarray
    # same as above, but last index we include in search (12 in above example)
    _center_search_end: np.ndarray

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
        self.decay_fraction = (
            2 * lateral_decay_fraction + axial_decay_fraction
        ) / 3

        cube_size_um = _expand_num_triplet(cube_size_um)
        self.cube_voxels = _norm_by_size(cube_size_um, voxel_size)

        initial_center_search_radius_um = _expand_num_triplet(
            initial_center_search_radius_um
        )
        self.initial_center_search_radius_voxels = _norm_by_size(
            initial_center_search_radius_um, voxel_size
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

        if any(self.initial_center_search_radius_voxels):
            self._center_search_start = (
                self.cube_center_voxels
                - self.initial_center_search_radius_voxels
            )[None, ...]
            self._center_search_end = (
                self.cube_center_voxels
                + self.initial_center_search_radius_voxels
            )[None, ...]
        else:
            self._center_search_start = self.cube_center_voxels[None, ...]
            self._center_search_end = self.cube_center_voxels[None, ...]

        self.decay_gaussian_bounds = decay_gaussian_bounds

        self._verify_lateral_parameters()
        self._verify_axial_parameters()

        self._check_do_center_search()
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
                **_shared_mem_kwargs,
            )
            state["_circle_masks_buffer"] = shm

        if (
            state["_circle_masks"] is not None
            and state["_circle_masks_buffer"] is not None
        ):
            masks = np.ndarray(
                state["_circle_masks"],
                dtype=bool,
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

    @cached_property
    def laplacian_kernel(self):
        return get_27_stencil(self.voxel_size)

    def __call__(self, data: np.ndarray) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray,
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

        center_values = self.get_center_values(data, center)
        data_min = np.min(data, axis=(1, 2, 3))
        segmentation_intensity_threshold = self.get_segmentation_threshold(
            data, data_min, center, center_values, self.decay_fraction
        )
        segmentation_mask = self.get_segmentation_mask(
            data,
            data_min,
            center,
            segmentation_intensity_threshold,
            self.laplacian_kernel,
        )

        return (
            center,
            r_lat,
            lat_line,
            r_lat_params,
            r_axial,
            ax_line,
            r_axial_params,
            segmentation_mask,
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

    @cache
    def _check_do_center_search(self) -> bool:
        """
        Checks that cubes is big enough for the requested search size with
        centered search window of 3.
        """
        # only pre-calc if we can adjust the cube center by at least one in
        # each direction
        if not all(r for r in self.initial_center_search_radius_voxels):
            return False

        for c, dim, sides in zip(
            self.cube_center_voxels,
            self.cube_voxels,
            self.initial_center_search_radius_voxels,
            strict=True,
        ):
            # we search sides voxels on each side, not including the original c
            start = c - sides
            end = c + sides + 1

            # we have to add to start/end that many voxels for a window of size
            # 3 around center
            start -= 1
            end += 1

            if start < 0 or end > dim:
                raise ValueError(
                    f"The cube size of {dim} voxels is too small for a center "
                    f"adjustment search of {sides} voxels on each side, with a"
                    f" centered window size of {3} voxels. For a cube center "
                    f"of {c} voxels, we would have needed a sub-region of "
                    f"[{start}, {end}) with size {end - start} voxels, which "
                    f"extends beyond the full cube"
                )

        return True

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
        centers = np.repeat(self.cube_center_voxels[None, ...], n, axis=0)

        # only pre-calc if we adjust the cube center in at least one direction
        if not self._check_do_center_search():
            # no search - the center is the original center
            return centers

        starts = self._center_search_start
        ends = self._center_search_end

        # mask indicating which of the original batch items center is still
        # being updated
        remaining_mask = np.ones(data.shape[0], dtype=bool)
        # batch data containing only batch items whose center is still being
        # updated
        remaining_data = data
        rem_n = n

        # indices in each dim to the center and the item before and after it so
        # as to get a window on 3 around center
        center_vol_indices_rem = [
            np.concatenate(
                [centers[:, :1] - 1, centers[:, :1], centers[:, :1] + 1],
                axis=1,
            )[:, :, None, None],
            np.concatenate(
                [centers[:, 1:2] - 1, centers[:, 1:2], centers[:, 1:2] + 1],
                axis=1,
            )[:, None, :, None],
            np.concatenate(
                [centers[:, 2:3] - 1, centers[:, 2:3], centers[:, 2:3] + 1],
                axis=1,
            )[:, None, None, :],
        ]

        win_center_index = np.ravel_multi_index((1, 1, 1), (3, 3, 3))
        while True:
            batch_ind = np.arange(rem_n)

            # get the 3x3x3 volume around current center of each item
            center_vol = remaining_data[batch_ind, *center_vol_indices_rem]
            cval = center_vol[:, 1, 1, 1]
            data_win = center_vol.reshape((rem_n, -1))

            i_max = np.argmax(data_win, axis=1)
            # for which batch items did the location of the max value not shift
            # from last center? Drop those batch items
            same = np.logical_or(
                i_max == win_center_index, cval == data_win[batch_ind, i_max]
            )
            not_same = np.logical_not(same)

            remaining_data = remaining_data[not_same, ...]
            rem_n = remaining_data.shape[0]

            # if we dropped all remaining data items, there's nothing to do
            if not rem_n:
                break

            remaining_mask[remaining_mask] = not_same

            i_max = i_max[not_same]
            offsets = np.unravel_index(i_max, (3, 3, 3))

            for i, offset in enumerate(offsets):
                center_vol_indices_rem[i] = center_vol_indices_rem[i][
                    not_same, ...
                ]
                # shift the center zero or plus or minus one for the volume
                # indices
                center_vol_indices_rem[i] += offset[:, None] - 1
                # update the center to new value
                centers[remaining_mask, i] = np.take(
                    center_vol_indices_rem[i], 1, axis=i + 1
                )[:, 0, 0]

            # check if we hit the edge in any dim for remaining data
            rem_centers = centers[remaining_mask, :]
            hit_edge = np.any(
                np.logical_or(rem_centers == starts, rem_centers == ends),
                axis=1,
            )
            no_edge = np.logical_not(hit_edge)

            # keep only those who didn't hit edge
            remaining_data = remaining_data[no_edge, ...]
            rem_n = remaining_data.shape[0]
            if not rem_n:
                break

            remaining_mask[remaining_mask] = no_edge
            for i in range(3):
                center_vol_indices_rem[i] = center_vol_indices_rem[i][
                    no_edge, ...
                ]

        return centers

    def _verify_lateral_parameters(self):
        center_offsets = [
            self.initial_center_search_radius_voxels[i]
            for i in self.lateral_dims
        ]
        center = [self.cube_center_voxels[i] for i in self.lateral_dims]
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
            center, center_offsets, sizes, strict=True
        ):
            # last element exclusive of the center
            right = c + c_offset + r
            # same - first element exclusive of the center
            left = c - c_offset - r

            # don't need to check for decay because decay <= r
            # number of elements is center plus r
            if right >= size:
                raise ValueError(
                    f"Requested lateral line with size {r} voxels and "
                    f"potential center offset of {c_offset} voxels from "
                    f"the center at {c}. This is "
                    f"larger than the size of the cube {size} voxels"
                )
            if left < 0:
                raise ValueError(
                    f"Requested lateral line with size {r} voxels and "
                    f"potential center offset of negative {c_offset} voxels "
                    f"from the center at {c}. "
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
                f"potential center offset of {c_offset} voxels from "
                f"the center at {c}. This is "
                f"larger than the size of the cube {size} voxels"
            )
        if left - r < 0:
            raise ValueError(
                f"Requested axial line with size {r} voxels and "
                f"potential center offset of negative {c_offset} voxels "
                f"from the center at {c}. "
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
        single_bytes = np.zeros(1, dtype=bool).nbytes
        total_bytes = math.prod(masks_shape) * single_bytes
        shm = shared_memory.SharedMemory(
            create=True,
            size=total_bytes,
            **_shared_mem_kwargs,
        )

        masks = np.ndarray(masks_shape, dtype=bool, buffer=shm.buf)
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

    def get_center_values(
        self, data: np.ndarray, center: np.ndarray
    ) -> np.ndarray:
        return data[
            np.arange(data.shape[0]), center[:, 0], center[:, 1], center[:, 2]
        ]

    def get_segmentation_threshold(
        self,
        data: np.ndarray,
        global_min_data: np.ndarray,
        center: np.ndarray,
        center_values: np.ndarray,
        decay_fraction: float,
    ) -> np.ndarray:
        """
        Gets threshold by finding local min closest to cell center and using
        that as local min value.
        """
        n = data.shape[0]
        local_min_data = np.empty(n)

        for i in range(n):
            item_data = data[i, ...]
            cval = center_values[i]

            local_min_coords = skimage.feature.peak_local_max(
                -(item_data - global_min_data[i]),
                min_distance=1,
                exclude_border=False,
                num_peaks_per_label=1,
            )

            min_global = global_min_data[i]
            min_local = min_global
            if len(local_min_coords):
                if len(local_min_coords) == 1:
                    min_local = item_data[*local_min_coords[0]]
                else:
                    dist = np.sum(
                        np.square(local_min_coords - center[i][None, :]),
                        axis=1,
                    )
                    min_i = np.argmin(dist)
                    min_local = item_data[*local_min_coords[min_i]]

                if min_local >= cval:
                    min_local = min_global

            local_min_data[i] = min_local

            if np.isclose(min_global, cval):
                raise ValueError(
                    "Center intensity is too similar to the global min"
                )

        threshold = (
            center_values - local_min_data
        ) * decay_fraction + local_min_data

        return threshold

    def get_segmentation_mask(
        self,
        data: np.ndarray,
        global_min_data: np.ndarray,
        center: np.ndarray,
        threshold: np.ndarray,
        kernel: np.ndarray,
    ):
        n = data.shape[0]
        above_threshold_data = data > threshold[:, None, None, None]
        laplacian_data = scipy.ndimage.convolve(
            data, kernel, axes=(1, 2, 3), mode="nearest"
        )
        segmentation_mask = np.empty(data.shape, dtype=bool)

        for i in range(n):
            item_data = data[i, ...]
            c1, c2, c3 = center[i]
            above_threshold = above_threshold_data[i, ...]

            local_max_mask = np.zeros(item_data.shape, dtype=bool)

            local_max_coords = skimage.feature.peak_local_max(
                item_data,
                min_distance=1,
                exclude_border=False,
                num_peaks_per_label=1,
            )
            local_max_mask[tuple(local_max_coords.T)] = True
            local_max_mask[c1, c2, c3] = True

            local_max_coords = skimage.feature.peak_local_max(
                laplacian_data[i, ...],
                min_distance=1,
                exclude_border=False,
                num_peaks_per_label=1,
            )
            local_max_mask[tuple(local_max_coords.T)] = True

            peak_markers = skimage.measure.label(
                local_max_mask, connectivity=2
            )
            labels = skimage.segmentation.watershed(
                -(item_data - global_min_data[i]),
                markers=peak_markers,
                mask=above_threshold,
                connectivity=3,
                compactness=50,
            )

            inside = labels == labels[c1, c2, c3]
            segmentation_mask[i, ...] = np.logical_and(above_threshold, inside)

        return segmentation_mask
