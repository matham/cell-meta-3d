import logging
import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import datetime
from functools import wraps
from numbers import Number
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import tqdm
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.general.system import get_num_processes
from brainglobe_utils.IO.cells import get_cells, save_cells
from brainglobe_utils.IO.image.load import read_z_stack
from cellfinder.core import types
from cellfinder.core.classify.cube_generator import (
    CuboidBatchSampler,
    get_data_cuboid_range,
)
from fancylog import fancylog
from torch.utils.data import DataLoader

import cell_meta_3d
from cell_meta_3d.arg_parse import cell_meta_3d_parser
from cell_meta_3d.dataset import (
    CellMeasureStackDataset,
    CellMeasureTiffDataset,
)
from cell_meta_3d.measure import CellSizeCalc, gaussian_func

T = TypeVar("T")
P = ParamSpec("P")


def _set_torch_threads(worker: int):
    torch.set_num_threads(4)


def _set_torch_threads_dec(f: Callable[P, T]) -> Callable[P, T]:
    @wraps(f)
    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        threads = torch.get_num_threads()
        _set_torch_threads(0)
        try:
            return f(*args, **kwargs)
        finally:
            torch.set_num_threads(threads)

    return inner


def get_cuboid_center(axis: str, size: int) -> int:
    # use a point at zero, this will give us the start of the cube relative
    # to zero. Then, abs of that will be the distance from start to center
    start, end = get_data_cuboid_range(size, size, axis)
    return size - start


def _get_cuboid_center_by_index(ax: int, size: int) -> int:
    # z, y, x
    match ax:
        case 0:
            return get_cuboid_center("z", size)
        case 1:
            return get_cuboid_center("y", size)
        case 2:
            return get_cuboid_center("x", size)
        case _:
            raise ValueError


def _interpolate(
    values: np.ndarray, index: float, default_value: float
) -> float | np.ndarray:
    index = float(index)

    if index < 0 or index > len(values) - 1:
        return default_value

    if index.is_integer():
        return values[int(index)].item()

    lower_i = int(index)
    ratio = values[lower_i + 1] - values[lower_i]
    value = ratio * (index - lower_i) + values[lower_i]
    return value


def _get_dataset(
    cells: list[Cell],
    points_filenames: Sequence[str] | None,
    signal_array: types.array | None,
    voxel_size: tuple[float, float, float],
    cube_voxels: tuple[int, int, int],
    batch_size: int,
    n_free_cpus: int,
    max_workers: int,
    cell_calc: CellSizeCalc,
) -> tuple[
    DataLoader,
    CellMeasureStackDataset | CellMeasureTiffDataset,
    CuboidBatchSampler,
    int,
]:
    # data and network voxel size are the same b/c we're not rescaling the cube
    if signal_array is not None:
        dataset = CellMeasureStackDataset(
            cell_calc=cell_calc,
            signal_array=signal_array,
            background_array=None,
            points=cells,
            data_voxel_sizes=voxel_size,
            network_voxel_sizes=voxel_size,
            network_cuboid_voxels=cube_voxels,
            axis_order=("z", "y", "x"),
            output_axis_order=("z", "y", "x", "c"),
            target_output="index",
        )
    elif points_filenames:
        dataset = CellMeasureTiffDataset(
            cell_calc=cell_calc,
            points_filenames=[[f] for f in points_filenames],
            points=cells,
            data_voxel_sizes=voxel_size,
            network_voxel_sizes=voxel_size,
            network_cuboid_voxels=cube_voxels,
            axis_order=("z", "y", "x"),
            output_axis_order=("z", "y", "x", "c"),
            target_output="index",
        )
    else:
        raise ValueError

    sampler = CuboidBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        sort_by_axis="z",
    )

    workers = get_num_processes(min_free_cpu_cores=n_free_cpus)
    workers = min(workers, max_workers)
    workers = min(workers, len(cells) // batch_size)

    # this will sample the dataset in the given sampler order (sorted by z)
    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=workers,
        worker_init_fn=_set_torch_threads,
    )

    return data_loader, dataset, sampler, workers


def _debug_display(
    cell: Cell,
    r_lat_data: dict[str, float],
    r_axial_data: dict[str, float],
    lat_line: np.ndarray,
    ax_line: np.ndarray,
    plot_output_path: Path,
    cell_calc: CellSizeCalc,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2)

    z, y, x = cell.z, cell.y, cell.x
    vz, vy, vx = cell_calc.voxel_size
    r_lat_data = {k: v for k, v in r_lat_data.items() if not k.endswith("std")}
    r_axial_data = {
        k: v for k, v in r_axial_data.items() if not k.endswith("std")
    }

    ax1.plot(np.arange(len(lat_line)) * vy, lat_line, "k--", label="Measured")

    func = ""
    model_label = "Modeled"
    if cell_calc.lateral_decay_algorithm == "gaussian":
        n = cell_calc.lateral_line_length
        radii = np.linspace(-n / 4, n, 200)
        ax1.plot(
            radii * vy,
            gaussian_func(radii * vy, **r_lat_data),
            "g-.",
            label=model_label,
        )
        model_label = None

        func = (
            "\n${a:0.2f}*e^{{-\\frac{{(\\frac{{r_um}}-"
            "{offset:0.2f})^{{2}}}}{{2*{sigma:0.2f}^{{2}}}}}}+{c:0.2f}$"
        ).format(**r_lat_data)

    p_hor = cell.metadata["r_xy_um"] + r_lat_data["offset"]
    val = _interpolate(lat_line, p_hor, 0)
    ax1.plot(
        [p_hor],
        [val],
        "ro",
        label=f"{100 * cell_calc.lateral_decay_fraction:0.2f}% threshold",
    )

    ax1.set_xlabel("Distance from point (microns)")
    ax1.set_ylabel("Normalized intensity")
    std = int(cell.metadata["r_xy_um_max_std"])
    ax1.set_title(f"Lateral radius (std={std}){func}")

    ax2.plot(np.arange(len(ax_line)) * vz, ax_line, "k--")

    func = ""
    if cell_calc.axial_decay_algorithm == "gaussian":
        n = cell_calc.axial_line_length
        radii = np.linspace(-n / 4, n, 200)
        ax2.plot(
            radii * vz,
            gaussian_func(radii * vz, **r_axial_data),
            "g-.",
            label=model_label,
        )

        func = (
            "\n${a:0.2f}*e^{{-\\frac{{(\\frac{{r}}-"
            "{offset:0.2f})^{{2}}}}{{2*{sigma:0.2f}^{{2}}}}}}+{c:0.2f}$"
        ).format(**r_axial_data)

    p_hor = cell.metadata["r_z_um"] + r_axial_data["offset"]
    val = _interpolate(ax_line, p_hor, 0)
    ax2.plot(
        [p_hor],
        [val],
        "mo",
        label=f"{100 * cell_calc.axial_decay_fraction:0.2f}% threshold",
    )

    ax2.set_xlabel("Distance from point (microns)")
    ax2.set_ylabel("Normalized intensity")
    std = int(cell.metadata["r_z_um_max_std"])
    ax2.set_title(f"Axial radius (std={std}){func}")

    fig.legend(loc="lower center", ncols=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    if plot_output_path:
        name = f"radius_cell_z{z:05}y{y:05}x{x:05}.jpg"
        fig.savefig(plot_output_path / name, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def create_segmentation_datasets(
    cell_calc: CellSizeCalc,
    points_filenames: Sequence[str] | None,
    signal_array: types.array | None,
    batch_size: int,
    h5_file: h5py.File,
) -> dict[str, h5py.Dataset]:
    h5_file.attrs["cube_voxels"] = cell_calc.cube_voxels
    h5_file.attrs["voxel_size"] = cell_calc.voxel_size
    h5_file.attrs["upsampled_voxel_size"] = cell_calc.upsampled_voxel_size
    h5_file.attrs["super_voxel"] = cell_calc.seg_super_voxel
    super_voxel = cell_calc.seg_super_voxel

    mask_index_dtype = np.uint16
    if all(
        v * s <= 256
        for v, s in zip(cell_calc.cube_voxels, super_voxel, strict=False)
    ):
        mask_index_dtype = np.uint8

    intensity_index = h5_file.create_dataset(
        "upsampled_cuboid_intensity_index",
        shape=(0, 3),
        maxshape=(None, 3),
        chunks=(5 * 5 * 5 * batch_size * 10, 3),
        dtype=mask_index_dtype,
        compression="gzip",
    )

    if signal_array is not None:
        dtype = signal_array.dtype
    else:
        dtype = tifffile.imread(points_filenames[0]).dtype
    intensity = h5_file.create_dataset(
        "upsampled_cuboid_intensity",
        shape=(0,),
        maxshape=(None,),
        chunks=(5 * 5 * 5 * batch_size * 10,),
        dtype=dtype,
        compression="gzip",
    )

    cell_index_range = h5_file.create_dataset(
        "cell_index_range",
        shape=(0, 2),
        maxshape=(None, 2),
        chunks=(batch_size * 10, 2),
        dtype=np.int64,
        # compression="gzip",
    )

    cell_corner = h5_file.create_dataset(
        "cell_corner_vox",
        shape=(0, 3),
        maxshape=(None, 3),
        chunks=(batch_size * 10, 3),
        dtype=np.uint32,
        # compression="gzip",
    )

    h5_datasets = {
        "cell_index_range": cell_index_range,
        "intensity": intensity,
        "intensity_index": intensity_index,
        "cell_corner": cell_corner,
    }

    return h5_datasets


def append_segmentation_data_h5(
    h5_datasets: dict[str, h5py.Dataset],
    upsampled_segmentation_mask: np.ndarray,
    upsampled_raw_intensity: np.ndarray,
    cells: list[Cell],
    cube_center_vox: tuple[int, int, int],
):
    n = upsampled_segmentation_mask.shape[0]
    old_size_flat = len(h5_datasets["intensity"])
    old_size_batch = len(h5_datasets["cell_index_range"])

    dset = h5_datasets["intensity_index"]
    _, zi, yi, xi = np.nonzero(upsampled_segmentation_mask)
    dset.resize(old_size_flat + len(zi), axis=0)
    dset[old_size_flat:, 0] = zi
    dset[old_size_flat:, 1] = yi
    dset[old_size_flat:, 2] = xi

    dset = h5_datasets["intensity"]
    flat = upsampled_raw_intensity[upsampled_segmentation_mask]
    dset.resize(old_size_flat + len(flat), axis=0)
    dset[old_size_flat:] = flat

    dset = h5_datasets["cell_index_range"]
    dset.resize(old_size_batch + n, axis=0)
    counts = upsampled_segmentation_mask.reshape((n, -1)).sum(axis=1)
    ends = np.cumsum(counts) + old_size_flat
    dset[old_size_batch:, 0] = ends - counts
    dset[old_size_batch:, 1] = ends

    dset = h5_datasets["cell_corner"]
    dset.resize(old_size_batch + n, axis=0)
    cell_centers_vox = np.array([[c.z, c.y, c.x] for c in cells])
    dset[old_size_batch:, :] = cell_centers_vox - [cube_center_vox]


def _run_batches(
    data_loader: DataLoader,
    dataset: CellMeasureTiffDataset | CellMeasureStackDataset,
    sampler: CuboidBatchSampler,
    cell_calc: CellSizeCalc,
    cells: list[Cell],
    plot_output_path: Path | None,
    debug_data: bool,
    h5_datasets: dict[str, h5py.Dataset],
    stop_after_n_cells: int | None,
    add_segmentation_to_metadata: bool,
    status_callback: Callable[[int], None] | None,
):
    output_cells = []
    # data order is always z, y, x. Units are voxels
    z_center = get_cuboid_center("z", cell_calc.cube_voxels[0])
    y_center = get_cuboid_center("y", cell_calc.cube_voxels[1])
    x_center = get_cuboid_center("x", cell_calc.cube_voxels[2])
    up_vox_size = cell_calc.upsampled_voxel_size
    up_vox_vol = up_vox_size[0] * up_vox_size[1] * up_vox_size[2]
    vox_size = cell_calc.voxel_size

    lat_params_names = cell_calc.lateral_parameters_names
    axial_params_names = cell_calc.axial_parameters_names
    lat_std_i = [
        i for i, name in enumerate(lat_params_names) if name.endswith("std")
    ]
    ax_std_i = [
        i for i, name in enumerate(axial_params_names) if name.endswith("std")
    ]

    total_cells = 0
    for batch, indices in tqdm.tqdm(data_loader, total=len(data_loader)):
        # data comes in as batches of torch tensors
        (
            center,
            center_intensity,
            min_intensity,
            r_lat,
            lat_line,
            lat_params_data,
            r_axial,
            ax_line,
            axial_params_data,
            upsampled_data,
            segmentation_mask_upsampled,
            paor_vectors_intensity,
            paor_centroid_intensity,
            paor_moment2_intensity,
            paor_extent_intensity,
            paor_vectors_mask,
            paor_centroid_mask,
            paor_moment2_mask,
            paor_extent_mask,
        ) = [item.numpy() for item in batch]
        center = np.round(center).astype(int)
        indices = indices.numpy().astype(int)

        if h5_datasets:
            append_segmentation_data_h5(
                h5_datasets,
                segmentation_mask_upsampled,
                upsampled_data,
                [cells[point_i] for point_i in indices],
                (z_center, y_center, x_center),
            )

        # convert to list so items become native python type
        center = center.tolist()
        center_intensity = center_intensity.tolist()
        min_intensity = min_intensity.tolist()
        r_lat = r_lat.tolist()
        lat_params_data = lat_params_data.tolist()
        r_axial = r_axial.tolist()
        axial_params_data = axial_params_data.tolist()
        paor_vectors_intensity = paor_vectors_intensity.tolist()
        paor_centroid_intensity = paor_centroid_intensity.tolist()
        paor_moment2_intensity = paor_moment2_intensity.tolist()
        paor_extent_intensity = paor_extent_intensity.tolist()
        paor_vectors_mask = paor_vectors_mask.tolist()
        paor_centroid_mask = paor_centroid_mask.tolist()
        paor_moment2_mask = paor_moment2_mask.tolist()
        paor_extent_mask = paor_extent_mask.tolist()
        volume = (
            np.sum(segmentation_mask_upsampled, axis=(1, 2, 3)) * up_vox_vol
        ).tolist()

        for i, point_i in enumerate(indices):
            cell = deepcopy(cells[point_i])
            corner = cell.z - z_center, cell.y - y_center, cell.x - x_center
            corner_um = [
                c * vx for c, vx in zip(corner, vox_size, strict=True)
            ]

            z, y, x = center[i]
            # shift pos by the amount it shifted from center
            cell.z = corner[0] + z
            cell.y = corner[1] + y
            cell.x = corner[2] + x

            if not hasattr(cell, "metadata"):
                cell.metadata = {}
            cell.metadata.update(
                {
                    "intensity": center_intensity[i],
                    "min_intensity": min_intensity[i],
                    "r_xy_um": r_lat[i],
                    "r_z_um": r_axial[i],
                    "r_xy_um_max_std": -1,
                    "r_z_um_max_std": -1,
                    "seg_id": total_cells,
                    "volume_um3": volume[i],
                    "paor_xyz_um": paor_vectors_intensity[i],
                    "paor_shape_xyz_um": paor_vectors_mask[i],
                    "paor_centroid_xyz_um": [
                        cr + cn
                        for cr, cn in zip(
                            corner_um[::-1],
                            paor_centroid_intensity[i],
                            strict=True,
                        )
                    ],
                    "paor_centroid_shape_xyz_um": [
                        cr + cn
                        for cr, cn in zip(
                            corner_um[::-1], paor_centroid_mask[i], strict=True
                        )
                    ],
                    "paor_um5": paor_moment2_intensity[i],
                    "paor_shape_um5": paor_moment2_mask[i],
                    "paor_extent_um": paor_extent_intensity[i],
                    "paor_extent_shape_um": paor_extent_mask[i],
                }
            )
            total_cells += 1

            if add_segmentation_to_metadata:
                cell.metadata["segmentation_upsampled"] = {
                    "mask": segmentation_mask_upsampled[i],
                    "intensity": upsampled_data[i],
                    "corner_um": corner_um,
                    "upsampled_voxel_size": up_vox_size,
                }

            if len(lat_params_data[i]):
                cell.metadata["r_xy_um_max_std"] = max(
                    lat_params_data[i][k] for k in lat_std_i
                )
            if len(axial_params_data[i]):
                cell.metadata["r_z_um_max_std"] = max(
                    axial_params_data[i][k] for k in ax_std_i
                )

            if debug_data:
                # these are in um where applicable
                cell.metadata["r_xy_parameters"] = dict(
                    zip(lat_params_names, lat_params_data[i], strict=True)
                )
                cell.metadata["r_z_parameters"] = dict(
                    zip(axial_params_names, axial_params_data[i], strict=True)
                )
                cell.metadata["r_xy_radial_line"] = lat_line[i].tolist()
                cell.metadata["r_z_radial_line"] = ax_line[i].tolist()

            output_cells.append(cell)

            if plot_output_path:
                _debug_display(
                    cell,
                    dict(
                        zip(lat_params_names, lat_params_data[i], strict=True)
                    ),
                    dict(
                        zip(
                            axial_params_names,
                            axial_params_data[i],
                            strict=True,
                        )
                    ),
                    lat_line[i, :],
                    ax_line[i, :],
                    plot_output_path,
                    cell_calc,
                )

        if status_callback is not None:
            status_callback(total_cells)

        if stop_after_n_cells and total_cells >= stop_after_n_cells:
            break

    return output_cells


@_set_torch_threads_dec
def main(
    *,
    cells: list[Cell],
    points_filenames: Sequence[str] | None = None,
    signal_array: types.array | None = None,
    voxel_size: tuple[float, float, float] = (5, 1, 1),
    cube_size: float | tuple[float, float, float] = (100, 50, 50),
    initial_center_search_radius: float | tuple[float, float, float] = (
        10,
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
    axial_max_radius: float = 35,
    axial_decay_length: float = 35,
    axial_decay_fraction: float = 1 / math.e,
    axial_decay_algorithm: Literal["gaussian", "manual"] = "gaussian",
    decay_gaussian_bounds: Sequence[float] = (
        0.1,
        1.25,
        -0.25,
        3,
        0.1,
        10.0,
        -1,
        1,
    ),
    seg_decay_fraction: float = 1 / math.e,
    seg_super_voxel: tuple[int, int, int] = (1, 1, 1),
    output_cells_path: Path | None = None,
    batch_size: int = 32,
    stop_after_n_cells: int | None = None,
    n_free_cpus: int = 2,
    max_workers: int = 6,
    plot_output_path: Path | str | None = None,
    segmentation_path: Path | str | None = None,
    debug_data: bool = False,
    status_callback: Callable[[int], None] | None = None,
    add_segmentation_to_metadata: bool = False,
) -> list[Cell]:
    """
    We expect the input data to have dimension order of z, y, x. All the
    parameters (voxel_size etc.) are specified in this order.

    cube_size, initial_center_search_radius etc are all in microns.

    :param cells:
    :param points_filenames:
    :param signal_array:
    :param voxel_size:
    :param cube_size:
    :param initial_center_search_radius:
    :param lateral_intensity_algorithm:
    :param lateral_max_radius:
    :param lateral_decay_length:
    :param lateral_decay_fraction:
    :param lateral_decay_algorithm:
    :param axial_intensity_algorithm:
    :param axial_max_radius:
    :param axial_decay_length:
    :param axial_decay_fraction:
    :param axial_decay_algorithm:
    :param decay_gaussian_bounds:
    :param output_cells_path:
    :param batch_size:
    :param n_free_cpus:
    :param max_workers:
    :param plot_output_path:
    :param debug_data:
    :param status_callback:
    :return:
    """
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    ts = datetime.now()
    cube_size_um = cube_size

    if isinstance(cube_size_um, Number):
        cube_size_um = cube_size_um, cube_size_um, cube_size_um
    # convert cube size to real size by ensuring it's a multiple of voxel size
    cube_size_voxels = tuple(
        int(round(c / v))
        for c, v in zip(cube_size_um, voxel_size, strict=True)
    )
    # make sure it'll be even voxels to avoid rounding issues when getting the
    # cube data, which displace center unpredictably
    cube_size_voxels = [(v + 1) if v % 2 else v for v in cube_size_voxels]
    cube_size_um = [
        c * v for c, v in zip(cube_size_voxels, voxel_size, strict=True)
    ]

    cell_calc = CellSizeCalc(
        axial_dim=0,  # axis_order below is z, y, x
        voxel_size=voxel_size,
        cube_size_um=cube_size_um,
        cuboid_center_func=_get_cuboid_center_by_index,
        initial_center_search_radius_um=initial_center_search_radius,
        lateral_intensity_algorithm=lateral_intensity_algorithm,
        lateral_max_radius_um=lateral_max_radius,
        lateral_decay_length_um=lateral_decay_length,
        lateral_decay_fraction=lateral_decay_fraction,
        lateral_decay_algorithm=lateral_decay_algorithm,
        axial_intensity_algorithm=axial_intensity_algorithm,
        axial_max_radius_um=axial_max_radius,
        axial_decay_length_um=axial_decay_length,
        axial_decay_fraction=axial_decay_fraction,
        axial_decay_algorithm=axial_decay_algorithm,
        decay_gaussian_bounds=decay_gaussian_bounds,
        seg_decay_fraction=seg_decay_fraction,
        seg_super_voxel=seg_super_voxel,
    )

    logging.info(f"cell_meta_3d: Starting analysis of {len(cells)} cells")

    data_loader, dataset, sampler, workers = _get_dataset(
        cells,
        points_filenames,
        signal_array,
        voxel_size,
        cube_size_voxels,
        batch_size,
        n_free_cpus,
        max_workers,
        cell_calc,
    )

    if plot_output_path:
        plot_output_path = Path(plot_output_path)
        plot_output_path.parent.mkdir(parents=True, exist_ok=True)

    h5_file = None
    h5_datasets = {}
    if segmentation_path:
        segmentation_path = Path(segmentation_path)
        segmentation_path.parent.mkdir(parents=True, exist_ok=True)
        h5_file = h5py.File(segmentation_path, "w")

        h5_datasets = create_segmentation_datasets(
            cell_calc, points_filenames, signal_array, batch_size, h5_file
        )

    if workers:
        dataset.start_dataset_thread(workers)
    try:
        output_cells = _run_batches(
            data_loader,
            dataset,
            sampler,
            cell_calc,
            cells,
            plot_output_path,
            debug_data,
            h5_datasets,
            stop_after_n_cells,
            add_segmentation_to_metadata,
            status_callback,
        )
    finally:
        try:
            dataset.stop_dataset_thread()
        finally:
            if h5_file is not None:
                h5_file.close()

    # remove duplicate cells - can happen if moving center shifted multiple
    # cells to same center
    dedup_cells = []
    seen = set()
    for cell in output_cells:
        key = cell.x, cell.y, cell.z
        if key in seen:
            continue

        dedup_cells.append(cell)
        seen.add(key)

    # temporarily remove segmentation data so it doesn't get saved to yaml
    segmentations = []
    if add_segmentation_to_metadata:
        segmentations = [
            c.metadata.pop("segmentation_upsampled") for c in dedup_cells
        ]
    save_cells(dedup_cells, str(output_cells_path))
    logging.info(f"cell_meta_3d: Analysis took {datetime.now() - ts}")

    if add_segmentation_to_metadata:
        for cell, segmentation in zip(dedup_cells, segmentations, strict=True):
            cell.metadata["segmentation_upsampled"] = segmentation

    return dedup_cells


def run_main():
    args = cell_meta_3d_parser().parse_args()

    signal = read_z_stack(args.signal_planes_path)
    cells = get_cells(args.cells_path, cells_only=True)
    output_cells = Path(args.output_cells_path)
    output_cells.parent.mkdir(parents=True, exist_ok=True)

    fancylog.start_logging(
        output_cells.parent,
        cell_meta_3d,
        variables=[
            args,
        ],
        verbose=args.debug_data,
        log_header="CellMeta3D Log",
        multiprocessing_aware=True,
    )

    main(
        cells=cells,
        signal_array=signal,
        voxel_size=args.voxel_size,
        cube_size=args.cube_size,
        initial_center_search_radius=args.initial_center_search_radius,
        lateral_intensity_algorithm=args.lateral_intensity_algorithm,
        lateral_max_radius=args.lateral_max_radius,
        lateral_decay_length=args.lateral_decay_length,
        lateral_decay_fraction=args.lateral_decay_fraction,
        lateral_decay_algorithm=args.lateral_decay_algorithm,
        axial_intensity_algorithm=args.axial_intensity_algorithm,
        axial_max_radius=args.axial_max_radius,
        axial_decay_length=args.axial_decay_length,
        axial_decay_fraction=args.axial_decay_fraction,
        axial_decay_algorithm=args.axial_decay_algorithm,
        decay_gaussian_bounds=args.decay_gaussian_bounds,
        seg_decay_fraction=args.seg_decay_fraction,
        seg_super_voxel=args.seg_super_voxel,
        batch_size=args.batch_size,
        stop_after_n_cells=args.stop_after_n_cells,
        output_cells_path=output_cells,
        n_free_cpus=args.n_free_cpus,
        max_workers=args.max_workers,
        plot_output_path=args.plot_output_path,
        segmentation_path=args.segmentation_path,
        debug_data=args.debug_data,
    )


if __name__ == "__main__":
    run_main()
