import math
from functools import lru_cache
from numbers import Number
from typing import Literal

import numpy as np
from scipy.optimize import curve_fit


def _gaussian_func(x, a, offset, sigma, c):
    return a * np.exp(-np.square(x - offset) / (2 * sigma**2)) + c


class CellSizeCalc:

    axial_dim: int
    voxel_size: tuple[float, float, float]
    cube_voxels: tuple[int, int, int]

    initial_center_search_voxels: tuple[int, int, int]
    initial_center_search_volume_voxels: tuple[int, int, int]

    lateral_intensity_algorithm: Literal["center_line", "area", "area_margin"]
    lateral_max_radius_voxels: int
    lateral_decay_algorithm: Literal["gaussian", "manual"]
    lateral_decay_len_voxels: int
    lateral_decay_fraction: float

    axial_intensity_algorithm: Literal["center_line", "area", "area_margin"]
    axial_max_radius_voxels: int
    axial_decay_algorithm: Literal["gaussian", "manual"]
    axial_decay_len_voxels: int
    axial_decay_fraction: float

    def __init__(
        self,
        axial_dim: int = 0,
        voxel_size: tuple[float, float, float] = (5, 1, 1),
        cube_size: float | tuple[float, float, float] = (100, 50, 50),
        initial_center_search_size: float | tuple[float, float, float] = (
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
            "center_line", "area", "area_margin"
        ] = "center_line",
        axial_max_radius: float = 40,
        axial_decay_length: float = 35,
        axial_decay_fraction: float = 1 / math.e,
        axial_decay_algorithm: Literal["gaussian", "manual"] = "gaussian",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._make_circle_masks = lru_cache()(self.make_circle_masks)
        self._make_sphere_masks = lru_cache()(self.make_sphere_masks)
        self.axial_dim = axial_dim
        self.lateral_intensity_algorithm = lateral_intensity_algorithm
        self.lateral_decay_algorithm = lateral_decay_algorithm
        self.lateral_decay_fraction = lateral_decay_fraction
        self.axial_intensity_algorithm = axial_intensity_algorithm
        self.axial_decay_algorithm = axial_decay_algorithm
        self.axial_decay_fraction = axial_decay_fraction

        if isinstance(cube_size, Number):
            cube_size = cube_size, cube_size, cube_size
        self.cube_voxels = tuple(
            int(round(c / v))
            for c, v in zip(cube_size, voxel_size, strict=False)
        )

        if isinstance(initial_center_search_size, Number):
            initial_center_search_size = (
                initial_center_search_size,
                initial_center_search_size,
                initial_center_search_size,
            )
        self.initial_center_search_voxels = tuple(
            int(round(c / v))
            for c, v in zip(
                initial_center_search_size, voxel_size, strict=False
            )
        )

        if isinstance(initial_center_search_volume, Number):
            initial_center_search_volume = (
                initial_center_search_volume,
                initial_center_search_volume,
                initial_center_search_volume,
            )
        self.initial_center_search_volume_voxels = tuple(
            int(round(c / v))
            for c, v in zip(
                initial_center_search_volume, voxel_size, strict=False
            )
        )

        lat_voxels = [r for i, r in enumerate(voxel_size) if i != axial_dim]
        lat_vox_mean = sum(lat_voxels) / 2
        self.lateral_max_radius_voxels = int(
            round(lateral_max_radius / lat_vox_mean)
        )
        self.lateral_decay_len_voxels = int(
            round(lateral_decay_length / lat_vox_mean)
        )

        axial_vox = voxel_size[axial_dim]
        self.axial_max_radius_voxels = int(round(axial_max_radius / axial_vox))
        self.axial_decay_len_voxels = int(
            round(axial_decay_length / axial_vox)
        )

    def __call__(
        self, data: np.ndarray
    ) -> tuple[tuple[int, int, int], int, int]:
        center = self.find_pos_center_max(data)

        match self.lateral_intensity_algorithm:
            case "center_line":
                line = self.get_center_2d_falloff_line(data, center)
            case "area":
                index = [slice(None)] * 3
                index[self.axial_dim] = center[self.axial_dim]
                plane = data[tuple(index)]

                masks = self._make_circle_masks(center)
                line = self.get_shape_falloff_line(plane, masks)
            case "area_margin":
                index = [slice(None)] * 3
                index[self.axial_dim] = center[self.axial_dim]
                plane = data[tuple(index)]

                masks = self._make_circle_masks(center)
                line = self.get_marginal_shape_falloff_line(plane, masks)
            case _:
                raise ValueError

        off_lat = 0
        match self.lateral_decay_algorithm:
            case "gaussian":
                r_lat, (_, off_lat, _, _) = self.get_radius_from_gaussian(
                    line,
                    self.lateral_decay_fraction,
                    self.lateral_decay_len_voxels,
                )
            case "manual":
                r_lat = self.get_radius_from_decay(
                    line,
                    self.lateral_decay_fraction,
                    self.lateral_decay_len_voxels,
                )
            case _:
                raise ValueError

        match self.axial_intensity_algorithm:
            case "center_line":
                line = self.get_center_1d_falloff_line(data, center)
            case "area":
                masks = self._make_sphere_masks(center)
                line = self.get_shape_falloff_line(data, masks)
            case "area_margin":
                masks = self._make_sphere_masks(center)
                line = self.get_marginal_shape_falloff_line(data, masks)
            case _:
                raise ValueError

        off_axial = None
        match self.axial_decay_algorithm:
            case "gaussian":
                r_axial, (_, off_axial, _, _) = self.get_radius_from_gaussian(
                    line,
                    self.axial_decay_fraction,
                    self.axial_decay_len_voxels,
                )
            case "manual":
                r_axial = self.get_radius_from_decay(
                    line,
                    self.axial_decay_fraction,
                    self.axial_decay_len_voxels,
                )
            case _:
                raise ValueError

        ax_i = self.axial_dim
        off_lat = int(round(off_lat))
        off_axial = int(round(off_axial))
        center = tuple(
            c + (off_axial if i == ax_i else off_lat)
            for i, c in enumerate(center)
        )
        return center, int(round(r_lat)), int(round(r_axial))

    def find_pos_center_max(
        self,
        data: np.ndarray,
    ) -> tuple[int, int, int]:
        dim1, dim2, dim3 = data.shape
        c1, c2, c3 = dim1 // 2, dim2 // 2, dim3 // 2
        n1, n2, n3 = self.initial_center_search_voxels
        v1, v2, v3 = self.initial_center_search_volume_voxels

        intensity = np.zeros(
            (n1 * 2 + 1, n2 * 2 + 1, n3 * 2 + 1), dtype=np.float64
        )
        for i in range(-n1, n1 + 1):
            for j in range(-n2, n2 + 1):
                for k in range(-n3, n3 + 1):
                    offset1 = c1 + i - v1 // 2
                    offset2 = c2 + j - v2 // 2
                    offset3 = c3 + k - v3 // 2
                    intensity[i + n1, j + n2, k + n3] = np.sum(
                        data[
                            offset1 : offset1 + v1,
                            offset2 : offset2 + v2,
                            offset3 : offset3 + v3,
                        ]
                    )

        flat_max = np.argmax(intensity)
        m1, m2, m3 = np.unravel_index(flat_max, intensity.shape)
        return c1 - n1 + m1, c2 - n2 + m2, c3 - n3 + m3

    def get_center_2d_falloff_line(
        self,
        data: np.ndarray,
        center: tuple[int, int, int],
    ) -> np.ndarray:
        vol_axis = self.axial_dim
        n_points = self.lateral_max_radius_voxels
        c1, c2 = [s for i, s in enumerate(center) if i != vol_axis]
        index = [slice(None)] * 3
        index[vol_axis] = center[vol_axis]
        plane = data[tuple(index)]

        # plus shaped
        data1 = plane[:, c2]
        data2 = plane[c1, :]
        line1 = data1[c1 : c1 + n_points]
        line2 = data1[c1 : c1 - n_points : -1]
        line3 = data2[c2 : c2 + n_points]
        line4 = data2[c2 : c2 - n_points : -1]

        n = min([len(line1), len(line2), len(line3), len(line4)])
        line = line1[:n] + line2[:n] + line3[:n] + line4[:n]
        line -= line.min()
        line /= line.max()

        return line

    def get_center_1d_falloff_line(
        self,
        data: np.ndarray,
        center: tuple[int, int, int],
    ) -> np.ndarray:
        vol_axis = self.axial_dim
        n_points = self.axial_max_radius_voxels
        index = tuple(
            slice(None) if i == vol_axis else c for i, c in enumerate(center)
        )
        line_data = data[index]
        vol_center = center[vol_axis]

        line1 = line_data[vol_center : vol_center + n_points]
        line2 = line_data[vol_center : vol_center - n_points : -1]

        n = min(len(line1), len(line2))
        line = line1[:n] + line2[:n]
        line -= line.min()
        line /= line.max()

        return line

    def make_circle_masks(
        self,
        center: tuple[int, int, int],
    ) -> np.ndarray:
        vol_axis = self.axial_dim
        max_r = self.lateral_max_radius_voxels
        shape = self.cube_voxels

        dim1, dim2 = [s for i, s in enumerate(shape) if i != vol_axis]
        c1, c2 = [s for i, s in enumerate(center) if i != vol_axis]
        dist1 = np.arange(dim1)[:, None] - c1
        dist2 = np.arange(dim2)[None, :] - c2
        dist = np.sqrt(np.square(dist1) + np.square(dist2))

        masks = np.zeros((max_r + 1, dim1, dim2))
        for i in range(max_r + 1):
            masks[i, :, :] = dist <= i
        return masks

    def make_sphere_masks(
        self,
        center: tuple[int, int, int],
        plane_r: int,
    ) -> np.ndarray:
        vol_axis = self.axial_dim
        max_r = self.axial_max_radius_voxels
        dim1, dim2, dim3 = self.cube_voxels
        c1, c2, c3 = center

        plane_dims = [i for i in range(3) if i != vol_axis]
        dist1 = np.expand_dims(np.abs(np.arange(dim1) - c1), plane_dims)
        dist2 = np.arange(dim2)[:, None] - c2
        dist3 = np.arange(dim3)[None, :] - c3

        dist_23 = 1 - (np.square(dist2) + np.square(dist3)) / plane_r**2
        valid_23 = dist_23 >= 0

        masks = np.zeros((max_r + 1, dim1, dim2, dim3))
        for i in range(max_r + 1):
            dist1_max = np.ones_like(dist_23) * -1
            dist1_max[valid_23] = np.sqrt(dist_23[valid_23] * i**2)
            masks[i, ...] = dist1 <= np.expand_dims(dist1_max, vol_axis)
        return masks

    def get_shape_falloff_line(
        self,
        data: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        n_masks = len(masks)

        intensity = np.sum(
            (data[None, ...] * masks).reshape((n_masks, -1)), axis=1
        )
        mask_size = np.sum(masks.reshape((n_masks, -1)), axis=1)

        intensity /= mask_size
        intensity -= np.min(intensity)
        intensity /= intensity.max()

        return intensity

    def get_marginal_shape_falloff_line(
        self,
        data: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        n_masks = len(masks)

        intensity = np.sum(
            (data[None, ...] * masks).reshape((n_masks, -1)), axis=1
        )
        mask_size = np.sum(masks.reshape((n_masks, -1)), axis=1)

        margin_intensity = intensity / mask_size
        margin_intensity[1:] = (intensity[1:] - intensity[:-1]) / (
            mask_size[1:] - mask_size[:-1]
        )
        margin_intensity -= np.min(margin_intensity)
        margin_intensity /= margin_intensity.max()

        return margin_intensity

    def get_radius_from_gaussian(
        self,
        data: np.ndarray,
        decay_fraction: float = 1 / math.e,
        max_n: int = 7,
    ) -> tuple[float, tuple[float, float, float, float]]:
        data = data[:max_n]
        n = len(data)
        bounds = (
            [0.1, -3, 0.1, -1],
            [1.25, 3, 10, 1],
        )

        try:
            (a, offset, sigma, c), _ = curve_fit(
                _gaussian_func,
                np.arange(n),
                data,
                p0=[1, 0, 0.5 * (max_n - 1), 0],
                bounds=bounds,
                # sigma=[1, 1, 1, 1, .2, .2] + [.2] * (max_n - 6),
            )
        except (RuntimeError, ValueError):
            return 0, (1, 0, 1, 0)

        desired_val = c + decay_fraction * a
        r = math.sqrt(-2 * sigma**2 * math.log((desired_val - c) / a))

        return r, (a, offset, sigma, c)

    def get_radius_from_decay(
        self,
        data: np.ndarray,
        decay_fraction: float = 1 / math.e,
        max_n: int = 20,
        argmax_range: int = 4,
    ) -> int:
        data = data[:max_n]
        offset_max = np.argmax(data[:argmax_range])

        less_mask = data[offset_max:] <= decay_fraction
        if not np.any(less_mask):
            return 0

        offset_min = np.argmax(less_mask)
        return offset_max + offset_min
