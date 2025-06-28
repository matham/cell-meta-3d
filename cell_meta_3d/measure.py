import math
from collections.abc import Sequence
from numbers import Number
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit


def _expand_num_triplet(
    value: Number | tuple[Number, Number, Number],
) -> tuple[Number, Number, Number]:
    if isinstance(value, Number):
        return value, value, value
    return value


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
    return tuple(int(round(v / s)) for v, s in zip(value, size, strict=True))


def gaussian_func(x, a, offset, sigma, c):
    return a * np.exp(-np.square(x - offset) / (2 * sigma**2)) + c


class CellSizeCalc:

    axial_dim: int
    voxel_size: tuple[float, float, float]
    cube_voxels: tuple[int, int, int]

    initial_center_search_radius_voxels: tuple[int, int, int]
    initial_center_search_volume_voxels: tuple[int, int, int]

    lateral_intensity_algorithm: Literal["center_line", "area", "area_margin"]
    lateral_max_radius_voxels: int
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

    _center_search_data_indices: tuple[slice, slice, slice, slice]
    _center_search_window: tuple[int, int, int, int]
    _center_search_offset: np.ndarray

    _circle_masks: np.ndarray
    _sphere_masks: np.ndarray

    def __init__(
        self,
        axial_dim: int = 0,
        voxel_size: tuple[float, float, float] = (5, 1, 1),
        cube_size: float | tuple[float, float, float] = (100, 50, 50),
        initial_center_search_radius: float | tuple[float, float, float] = (
            10,
            3,
            3,
        ),
        initial_center_search_volume: float | tuple[float, float, float] = (
            15,
            3,
            3,
        ),
        lateral_intensity_algorithm: Literal[
            "center_line", "area", "area_margin"
        ] = "area_margin",
        lateral_max_radius: float = 20,
        lateral_decay_length: float = 12,
        lateral_decay_fraction: float = 1 / math.e,
        lateral_decay_algorithm: Literal["gaussian", "manual"] = "gaussian",
        axial_intensity_algorithm: Literal[
            "center_line", "volume", "volume_margin"
        ] = "center_line",
        axial_max_radius: float = 40,
        axial_decay_length: float = 35,
        axial_decay_fraction: float = 1 / math.e,
        axial_decay_algorithm: Literal["gaussian", "manual"] = "gaussian",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axial_dim = axial_dim
        self.lateral_intensity_algorithm = lateral_intensity_algorithm
        self.lateral_decay_algorithm = lateral_decay_algorithm
        self.lateral_decay_fraction = lateral_decay_fraction
        self.axial_intensity_algorithm = axial_intensity_algorithm
        self.axial_decay_algorithm = axial_decay_algorithm
        self.axial_decay_fraction = axial_decay_fraction

        cube_size = _expand_num_triplet(cube_size)
        self.cube_voxels = _norm_by_size(cube_size, voxel_size)

        initial_center_search_radius = _expand_num_triplet(
            initial_center_search_radius
        )
        self.initial_center_search_radius_voxels = _norm_by_size(
            initial_center_search_radius, voxel_size
        )

        initial_center_search_volume = _expand_num_triplet(
            initial_center_search_volume
        )
        self.initial_center_search_volume_voxels = _norm_by_size(
            initial_center_search_volume, voxel_size
        )

        lat_voxels = [r for i, r in enumerate(voxel_size) if i != axial_dim]
        lat_vox = sum(lat_voxels) / 2
        axial_vox = voxel_size[axial_dim]

        self.lateral_max_radius_voxels = int(
            round(lateral_max_radius / lat_vox)
        )
        self.lateral_decay_len_voxels = int(
            round(lateral_decay_length / lat_vox)
        )

        self.axial_max_radius_voxels = int(round(axial_max_radius / axial_vox))
        self.axial_decay_len_voxels = int(
            round(axial_decay_length / axial_vox)
        )

        self._calc_find_pos_center_window()
        self._make_circle_masks()
        self._make_sphere_masks()

    @property
    def lateral_dims(self) -> list[int]:
        return [i for i in range(3) if i != self.axial_dim]

    def __call__(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(data.shape) != 4:
            raise ValueError
        center = self.find_pos_center_max(data)

        match self.lateral_intensity_algorithm:
            case "center_line":
                lat_line = self.get_center_2d_falloff_line(data, center)
            case "area":
                lat_line = self.get_area_falloff_line(data, center, False)
            case "area_margin":
                lat_line = self.get_area_falloff_line(data, center, True)
            case _:
                raise ValueError

        r_lat_data = self._get_decay_radius(
            lat_line,
            self.lateral_decay_algorithm,
            self.lateral_decay_fraction,
            self.lateral_decay_len_voxels,
        )

        match self.axial_intensity_algorithm:
            case "center_line":
                ax_line = self.get_center_1d_falloff_line(data, center)
            case "volume":
                ax_line = self.get_volume_falloff_line(data, center, False)
            case "volume_margin":
                ax_line = self.get_volume_falloff_line(data, center, True)
            case _:
                raise ValueError

        r_axial_data = self._get_decay_radius(
            ax_line,
            self.axial_decay_algorithm,
            self.axial_decay_fraction,
            self.axial_decay_len_voxels,
        )

        return center, r_lat_data, r_axial_data, lat_line, ax_line

    def _get_decay_radius(
        self,
        line: np.ndarray,
        algorithm: Literal["gaussian", "manual"],
        fraction: float,
        len_voxels: int,
    ) -> np.ndarray:
        output = np.zeros((len(line), 5))
        match algorithm:
            case "gaussian":
                for i, item in enumerate(line):
                    output[i, :] = self.get_radius_from_gaussian(
                        item,
                        fraction,
                        len_voxels,
                    )
            case "manual":
                r = self.get_radius_from_decay(
                    line,
                    fraction,
                    len_voxels,
                )
                output[:, 0] = r
            case _:
                raise ValueError

        return output

    def _calc_find_pos_center_window(self) -> None:
        center_data_indices = [slice(None)]
        center_search_window = [1]
        center_search_offset = []

        for dim, sides, win in zip(
            self.cube_voxels,
            self.initial_center_search_radius_voxels,
            self.initial_center_search_volume_voxels,
            strict=True,
        ):
            c = dim // 2
            left_win = win // 2
            right_win = win - left_win

            start = c - sides - left_win
            end = c + sides + 1 + right_win
            center_data_indices.append(slice(start, end))
            if start < 0 or end >= dim:
                raise ValueError

            center_search_window.append(2 * sides + 1)
            center_search_offset.append(c - sides)

        self._center_search_data_indices = tuple(center_data_indices)
        self._center_search_window = tuple(center_search_window)
        self._center_search_offset = np.array(
            [center_search_offset], dtype=np.int_
        )

    def find_pos_center_max(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        n = len(data)

        data = data[self._center_search_data_indices]
        windows = sliding_window_view(data, self._center_search_window)
        intensity = np.sum(windows, axis=(4, 5, 6), dtype=np.float64)

        flat_max = intensity.reshape((n, -1)).argmax(axis=1)
        max_idx = np.column_stack(
            np.unravel_index(flat_max, data[0, ...].shape)
        )
        assert len(max_idx) == n

        max_idx += self._center_search_offset
        return max_idx

    def get_center_2d_falloff_line(
        self,
        data: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        axial_axis = self.axial_dim
        n_points = self.lateral_max_radius_voxels
        lat_axes = [i for i in range(3) if i != axial_axis]
        n = len(data)
        n_range = np.arange(n)

        # center is Nx3. Convert to N by getting the axial center value for
        # each batch item
        axial_center = center.take(axial_axis, axis=1)
        # these are the center values for the 1st and 2nd lat axes
        lat_c1 = center[:, lat_axes[0]]
        lat_c2 = center[:, lat_axes[1]]

        # data is 4-d with first axis batch. Use centers to index the axial dim
        # to get the center plane for each batch item. We end up with 3-d array
        planes = data[
            _arr_index(4, [0, axial_axis + 1], [n_range, axial_center])
        ]
        ax2_line = planes[_arr_index(3, [0, 1], [n_range, lat_c1])]
        ax1_line = planes[_arr_index(3, [0, 2], [n_range, lat_c2])]

        line1 = ax2_line[
            n_range[:, None], lat_c2[:, None] + np.arange(n_points)[None, :]
        ]
        line2 = ax2_line[
            n_range[:, None],
            lat_c2[:, None] + np.arange(0, -n_points, -1)[None, :],
        ]
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
        np.divide(line, max_val, out=out_line, where=max_val)

        return out_line

    def get_center_1d_falloff_line(
        self,
        data: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        axial_axis = self.axial_dim
        n_points = self.axial_max_radius_voxels
        lat_axes = [i for i in range(3) if i != axial_axis]
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
        lines = planes[_arr_index(3, [0, lat_axes[1]], [n_range, lat_c2])]

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
        np.divide(line, max_val, out=out_line, where=max_val)

        return out_line

    def _make_circle_masks(
        self,
    ) -> None:
        axial_axis = self.axial_dim
        max_r = self.lateral_max_radius_voxels
        r_off = self.initial_center_search_radius_voxels
        r_off1, r_off2 = [v for i, v in enumerate(r_off) if i != axial_axis]
        cube = self.cube_voxels
        dim1, dim2 = [c for i, c in enumerate(cube) if i != axial_axis]
        c1, c2 = dim1 // 2, dim2 // 2

        if c1 - max_r - r_off1 < 0 or c1 + max_r + r_off1 >= dim1:
            raise ValueError
        if c2 - max_r - r_off2 < 0 or c2 + max_r + r_off2 >= dim2:
            raise ValueError

        # dims are batch, c1, c2, dim1, dim2, mask_r
        masks = np.zeros(
            (
                1,
                r_off1 * 2 + 1,
                r_off2 * 2 + 1,
                max_r + 1,
                max_r * 2 + 1 + r_off1 * 2,
                max_r * 2 + 1 + r_off2 * 2,
            )
        )
        dist1 = np.arange(-max_r - r_off1, max_r + r_off1 + 1)[:, None]
        dist2 = np.arange(-max_r - r_off2, max_r + r_off2 + 1)[None, :]

        for off1 in range(-r_off1, r_off1 + 1):
            for off2 in range(-r_off2, r_off2 + 1):
                for r in range(max_r + 1):
                    dist = np.sqrt(
                        np.square(dist1 - off1) + np.square(dist2 - off2)
                    )
                    masks[0, off1 + r_off1, off2 + r_off2, r, :, :] = dist <= r

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
                max_r_lat * 2 + 1 + r_off_lat * 2,
                max_r_lat * 2 + 1 + r_off_lat * 2,
                max_r_ax * 2 + 1 + r_off_ax * 2,
            )
        )

        dist1 = np.arange(-max_r_lat - r_off_lat, max_r_lat + r_off_lat + 1)[
            :, None
        ]
        dist2 = np.arange(-max_r_lat - r_off_lat, max_r_lat + r_off_lat + 1)[
            None, :
        ]
        plane_dims = [i for i in range(3) if i != axial_axis]
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
        axial_axis = self.axial_dim
        max_r = self.lateral_max_radius_voxels
        r_off = self.initial_center_search_radius_voxels
        r_off1, r_off2 = [v for i, v in enumerate(r_off) if i != axial_axis]
        cube = self.cube_voxels
        dim1, dim2 = [c for i, c in enumerate(cube) if i != axial_axis]
        c1, c2 = dim1 // 2, dim2 // 2

        axial_axis = self.axial_dim
        lat_axes = [i for i in range(3) if i != axial_axis]
        # these are the center values for the 1st and 2nd lat axes
        lat_c1 = center[:, lat_axes[0]]
        lat_c2 = center[:, lat_axes[1]]
        ax_c = center[:, axial_axis]

        n = len(data)
        n_zeros = np.zeros(n, dtype=np.int_)

        # get axial center plane for each batch item
        data_idx = _arr_index(4, [0, axial_axis], [np.arange(n), ax_c])
        data = data[data_idx]
        # reduce planes to mask size, add final dim for radius dim
        data = data[
            :,
            c1 - max_r - r_off1 : c1 + max_r + r_off1 + 1,
            c2 - max_r - r_off2 : c2 + max_r + r_off2 + 1,
            None,
        ]

        # select masks for the offset
        c1_rel = lat_c1 - c1 + r_off1
        c2_rel = lat_c2 - c2 + r_off2
        masks_idx = _arr_index(6, [0, 1, 2], [n_zeros, c1_rel, c2_rel])
        masks = self._circle_masks[masks_idx]

        masks_flat = np.reshape(masks, (n, max_r + 1, -1))
        data_flat = np.reshape(data, (n, max_r + 1, -1))

        intensity_sum = np.sum(
            masks_flat * data_flat, axis=2, dtype=np.float64
        )
        mask_size = np.sum(masks_flat, axis=2, dtype=np.float64)

        intensity = intensity_sum / mask_size
        if margin:
            intensity[:, 1:] = (
                intensity_sum[:, 1:] - intensity_sum[:, :-1]
            ) / (mask_size[:, 1:] - mask_size[:, :-1])

        intensity -= intensity.min(axis=1, keepdims=True)

        out_line = np.zeros_like(intensity)
        max_val = intensity.max(axis=1, keepdims=True)
        np.divide(intensity, max_val, out=out_line, where=max_val)

        return out_line

    def get_radius_from_gaussian(
        self,
        data: np.ndarray,
        decay_fraction: float,
        max_n: int,
    ) -> tuple[float, float, float, float, float]:
        data = data[:max_n]
        n = len(data)
        bounds = (
            [0.1, -3, 0.1, -1],
            [1.25, 3, 10, 1],
        )

        try:
            (a, offset, sigma, c), _ = curve_fit(
                gaussian_func,
                np.arange(n),
                data,
                p0=[1, 0, 0.5 * (max_n - 1), 0],
                bounds=bounds,
            )
        except (RuntimeError, ValueError):
            return -1, 0, 0, 0, 0

        desired_val = c + decay_fraction * a
        r = math.sqrt(-2 * sigma**2 * math.log((desired_val - c) / a))

        # r is relative to "offset" from center
        return r, a, offset, sigma, c

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
