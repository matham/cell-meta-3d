import math

import numpy as np
from scipy.optimize import curve_fit


def find_pos_center_max(
    data: np.ndarray,
    search_num: int | tuple[int, int, int],
    volume_size: int | tuple[int, int, int],
) -> tuple[int, int, int]:
    dim1, dim2, dim3 = data.shape
    c1, c2, c3 = dim1 // 2, dim2 // 2, dim3 // 2
    if isinstance(search_num, int):
        n1 = n2 = n3 = search_num
    else:
        n1, n2, n3 = search_num
    if isinstance(volume_size, int):
        v1 = v2 = v3 = volume_size
    else:
        v1, v2, v3 = volume_size

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


def make_circle_masks(
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
    vol_axis: int,
    max_r: int,
) -> np.ndarray:
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
    shape: tuple[int, int, int],
    center: tuple[int, int, int],
    plane_r: int,
    vol_axis: int,
    max_r: int,
) -> np.ndarray:
    dim1, dim2, dim3 = shape
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


def get_2d_falloff_center_line(
    data: np.ndarray,
    center: tuple[int, int, int],
    vol_axis: int,
    n_points: int,
) -> np.ndarray:
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


def get_1d_falloff_center_line(
    data: np.ndarray,
    center: tuple[int, int, int],
    vol_axis: int,
    n_points: int,
) -> np.ndarray:
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


def get_content_falloff_line(
    masks: np.ndarray,
    data: np.ndarray,
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


def get_shape_marginal_falloff_line(
    masks: np.ndarray, data: np.ndarray
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


def _gaussian_func(x, a, offset, sigma, c):
    return a * np.exp(-np.square(x - offset) / (2 * sigma**2)) + c


def get_radius_from_exp(
    data: np.ndarray,
    decay_fraction: float = 1 / math.e,
    max_n: int = 7,
    use_bounds: bool = True,
) -> tuple[int, tuple[float, float, float, float]]:
    data = data[:max_n]
    n = len(data)
    if not use_bounds:
        bounds = -np.inf, np.inf
    else:
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
    r = offset + math.sqrt(-2 * sigma**2 * math.log((desired_val - c) / a))
    r = int(round(r))

    return r, (a, offset, sigma, c)


def get_radius_from_decay(
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
