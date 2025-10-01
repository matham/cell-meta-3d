import logging
import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import datetime
from functools import wraps
from numbers import Number
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar

import matplotlib.pyplot as plt
import numpy as np
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
    start, _ = get_data_cuboid_range(0, size, axis)
    return abs(start)


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
            gaussian_func(radii, **r_lat_data),
            "g-.",
            label=model_label,
        )
        model_label = None

        func = (
            "\n${a:0.2f}*e^{{-\\frac{{(\\frac{{r}}{{{vy:0.2f}}}-"
            "{offset:0.2f})^{{2}}}}{{2*{sigma:0.2f}^{{2}}}}}}+{c:0.2f}$"
        ).format(vy=vy, **r_lat_data)

    p_hor = cell.metadata["r_xy"] + r_lat_data["offset"]
    val = _interpolate(lat_line, p_hor, 0)
    ax1.plot(
        [p_hor * vy],
        [val],
        "ro",
        label=f"{100 * cell_calc.lateral_decay_fraction:0.2f}% threshold",
    )

    ax1.set_xlabel("Distance from point (microns)")
    ax1.set_ylabel("Normalized intensity")
    std = int(cell.metadata["r_xy_max_std"])
    ax1.set_title(f"Lateral radius (std={std}){func}")

    ax2.plot(np.arange(len(ax_line)) * vz, ax_line, "k--")

    func = ""
    if cell_calc.axial_decay_algorithm == "gaussian":
        n = cell_calc.axial_line_length
        radii = np.linspace(-n / 4, n, 200)
        ax2.plot(
            radii * vz,
            gaussian_func(radii, **r_axial_data),
            "g-.",
            label=model_label,
        )

        func = (
            "\n${a:0.2f}*e^{{-\\frac{{(\\frac{{r}}{{{vz:0.2f}}}-"
            "{offset:0.2f})^{{2}}}}{{2*{sigma:0.2f}^{{2}}}}}}+{c:0.2f}$"
        ).format(vz=vz, **r_axial_data)

    p_hor = cell.metadata["r_z"] + r_axial_data["offset"]
    val = _interpolate(ax_line, p_hor, 0)
    ax2.plot(
        [p_hor * vz],
        [val],
        "mo",
        label=f"{100 * cell_calc.axial_decay_fraction:0.2f}% threshold",
    )

    ax2.set_xlabel("Distance from point (microns)")
    ax2.set_ylabel("Normalized intensity")
    std = int(cell.metadata["r_z_max_std"])
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


def _run_batches(
    data_loader: DataLoader,
    dataset: CellMeasureTiffDataset | CellMeasureStackDataset,
    sampler: CuboidBatchSampler,
    cell_calc: CellSizeCalc,
    cells: list[Cell],
    plot_output_path: Path | None,
    debug_data: bool,
    status_callback: Callable[[int], None] | None,
):
    output_cells = []
    count = 0
    # data order is always z, y, x
    z_center = get_cuboid_center("z", cell_calc.cube_voxels[0])
    y_center = get_cuboid_center("y", cell_calc.cube_voxels[1])
    x_center = get_cuboid_center("x", cell_calc.cube_voxels[2])

    lat_params_names = cell_calc.lateral_parameters_names
    axial_params_names = cell_calc.axial_parameters_names
    lat_std_i = [
        i for i, name in enumerate(lat_params_names) if name.endswith("std")
    ]
    ax_std_i = [
        i for i, name in enumerate(axial_params_names) if name.endswith("std")
    ]
    lat_line_len = cell_calc.lateral_line_length
    axial_line_len = cell_calc.axial_line_length

    # data format as it comes out from the dataset is NxK, where K is flattened
    # measured data of the cube/point. This happens in the dataset instance.
    # We have to split it back into individual measurement objects
    splits = [
        3,  # 3d indices in the cube
        4,  # intensity
        5,  # lateral radius
        5 + lat_line_len,  # lateral average line
    ]
    # params may be zero, e.g. if it's manual not Gaussian
    splits.append(splits[-1] + len(lat_params_names))  # the lateral parameters
    splits.append(splits[-1] + 1)  # axial radius
    splits.append(splits[-1] + axial_line_len)  # axial average line
    # remaining is len(axial_params_names) for the axial parameters

    for data, indices in tqdm.tqdm(data_loader, total=len(data_loader)):
        # data comes in as batches of torch tensors
        data = data.numpy()

        (
            center,
            intensity,
            r_lat,
            lat_line,
            lat_params_data,
            r_axial,
            ax_line,
            axial_params_data,
        ) = np.split(data, splits, axis=1)
        center = center.tolist()
        intensity = intensity.tolist()
        r_lat = r_lat.tolist()
        lat_params_data = lat_params_data.tolist()
        r_axial = r_axial.tolist()
        axial_params_data = axial_params_data.tolist()

        for i, point_i in enumerate(indices.tolist()):
            cell = deepcopy(cells[int(point_i)])

            z, y, x = [int(round(c)) for c in center[i]]
            # shift pos by the amount it shifted from center
            cell.z += z - z_center
            cell.y += y - y_center
            cell.x += x - x_center

            if not hasattr(cell, "metadata"):
                cell.metadata = {}
            cell.metadata.update(
                {
                    "center_intensity": intensity[i][0],
                    "r_xy": r_lat[i][0],
                    "r_z": r_axial[i][0],
                    "r_xy_max_std": -1,
                    "r_z_max_std": -1,
                }
            )
            if lat_params_data[i]:
                cell.metadata["r_xy_max_std"] = max(
                    lat_params_data[i][k] for k in lat_std_i
                )
            if axial_params_data[i]:
                cell.metadata["r_z_max_std"] = max(
                    axial_params_data[i][k] for k in ax_std_i
                )

            if debug_data:
                cell.metadata["r_xy_parameters"] = dict(
                    zip(lat_params_names, lat_params_data[i], strict=False)
                )
                cell.metadata["r_z_parameters"] = dict(
                    zip(axial_params_names, axial_params_data[i], strict=False)
                )
                cell.metadata["r_xy_radial_line"] = lat_line[i].tolist()
                cell.metadata["r_z_radial_line"] = ax_line[i].tolist()

            output_cells.append(cell)

            if plot_output_path:
                _debug_display(
                    cell,
                    dict(
                        zip(lat_params_names, lat_params_data[i], strict=False)
                    ),
                    dict(
                        zip(
                            axial_params_names,
                            axial_params_data[i],
                            strict=False,
                        )
                    ),
                    lat_line[i, :],
                    ax_line[i, :],
                    plot_output_path,
                    cell_calc,
                )

        count += len(indices)
        if status_callback is not None:
            status_callback(count)

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
    output_cells_path: Path | None = None,
    batch_size: int = 32,
    n_free_cpus: int = 2,
    max_workers: int = 6,
    plot_output_path: Path | str | None = None,
    debug_data: bool = False,
    status_callback: Callable[[int], None] | None = None,
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
    :param initial_center_search_volume:
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

    if isinstance(cube_size, Number):
        cube_size = cube_size, cube_size, cube_size
    # convert cube size to real size by ensuring it's a multiple of voxel size
    cube_voxels = tuple(
        int(round(c / v)) for c, v in zip(cube_size, voxel_size, strict=False)
    )
    cube_size_um = [
        c * v for c, v in zip(cube_voxels, voxel_size, strict=False)
    ]

    cell_calc = CellSizeCalc(
        axial_dim=0,  # axis_order below is z, y, x
        voxel_size=voxel_size,
        cube_size_um=cube_size_um,
        cuboid_center_func=_get_cuboid_center_by_index,
        initial_center_search_radius_um=initial_center_search_radius,
        initial_center_search_volume_um=initial_center_search_volume,
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
    )

    logging.info(f"cell_meta_3d: Starting analysis of {len(cells)} cells")

    data_loader, dataset, sampler, workers = _get_dataset(
        cells,
        points_filenames,
        signal_array,
        voxel_size,
        cube_voxels,
        batch_size,
        n_free_cpus,
        max_workers,
        cell_calc,
    )

    if plot_output_path:
        plot_output_path = Path(plot_output_path)
        plot_output_path.parent.mkdir(parents=True, exist_ok=True)

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
            status_callback,
        )
    finally:
        dataset.stop_dataset_thread()

    save_cells(output_cells, str(output_cells_path))
    logging.info(f"cell_meta_3d: Analysis took {datetime.now() - ts}")

    return output_cells


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
        initial_center_search_volume=args.initial_center_search_volume,
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
        batch_size=args.batch_size,
        output_cells_path=output_cells,
        n_free_cpus=args.n_free_cpus,
        max_workers=args.max_workers,
        plot_output_path=args.plot_output_path,
        debug_data=args.debug_data,
    )


if __name__ == "__main__":
    run_main()
