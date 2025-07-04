import logging
import math
from collections.abc import Callable, Sequence
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.general.system import get_num_processes
from brainglobe_utils.IO.cells import get_cells, save_cells
from brainglobe_utils.IO.image.load import read_z_stack
from cellfinder.core import types
from cellfinder.core.classify.cube_generator import CuboidBatchSampler
from torch.utils.data import DataLoader

from cell_meta_3d.arg_parse import cell_meta_3d_parser
from cell_meta_3d.dataset import (
    CellMeasureStackDataset,
    CellMeasureTiffDataset,
)
from cell_meta_3d.measure import CellSizeCalc, gaussian_func


def _get_dataset(
    cells: list[Cell],
    points_filenames: Sequence[str] | None,
    signal_array: types.array | None,
    voxel_size: tuple[float, float, float],
    cube_size: float | tuple[float, float, float],
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
    if isinstance(cube_size, Number):
        cube_size = cube_size, cube_size, cube_size
    cube_voxels = tuple(
        int(round(c / v)) for c, v in zip(cube_size, voxel_size, strict=True)
    )
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

    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=workers,
    )

    return data_loader, dataset, sampler, workers


def _debug_display(
    cell: Cell,
    r_lat_data,
    r_axial_data,
    lat_line,
    ax_line,
    plot_output_path,
    cell_calc,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2)

    z, y, x = cell.z, cell.y, cell.x
    vz, vy, vx = cell_calc.voxel_size

    ax1.plot(np.arange(len(lat_line)) * vy, lat_line, "k--", label="Measured")

    func = ""
    model_label = "Modeled"
    if cell_calc.lateral_decay_algorithm == "gaussian":
        n = cell_calc.lateral_decay_len_voxels
        radii = np.linspace(-n / 4, n, 200)
        ax1.plot(
            radii * vy,
            gaussian_func(radii, *r_lat_data[1:]),
            "g-.",
            label=model_label,
        )
        model_label = None

        a, off, sig, c = r_lat_data[1:]
        func = (
            f"\n${a:0.2f}*e^{{-\\frac{{(\\frac{{r}}{{{vy:0.2f}}}-"
            f"{off:0.2f})^{{2}}}}{{2*{sig:0.2f}^{{2}}}}}}+{c:0.2f}$"
        )

    p_hor = int(round(cell.metadata["r_y"] + r_lat_data[2]))
    val = 0
    if 0 <= p_hor < len(lat_line):
        val = lat_line[p_hor]
    ax1.plot(
        [p_hor * vy],
        [val],
        "ro",
        label=f"{100 * cell_calc.lateral_decay_fraction:0.2f}% threshold",
    )

    ax1.set_xlabel("Distance from point (microns)")
    ax1.set_ylabel("Normalized intensity")
    std = int(cell.metadata["r_y_std"])
    ax1.set_title(f"Lateral radius (std={std}){func}")

    ax2.plot(np.arange(len(ax_line)) * vz, ax_line, "k--")

    func = ""
    if cell_calc.axial_decay_algorithm == "gaussian":
        n = cell_calc.axial_decay_len_voxels
        radii = np.linspace(-n / 4, n, 200)
        ax2.plot(
            radii * vz,
            gaussian_func(radii, *r_axial_data[1:]),
            "g-.",
            label=model_label,
        )

        a, off, sig, c = r_axial_data[1:]
        func = (
            f"\n${a:0.2f}*e^{{-\\frac{{(\\frac{{r}}{{{vz:0.2f}}}-"
            f"{off:0.2f})^{{2}}}}{{2*{sig:0.2f}^{{2}}}}}}+{c:0.2f}$"
        )

    p_hor = int(round(cell.metadata["r_z"] + r_axial_data[2]))
    val = 0
    if 0 <= p_hor < len(ax_line):
        val = ax_line[p_hor]
    ax2.plot(
        [p_hor * vz],
        [val],
        "mo",
        label=f"{100 * cell_calc.axial_decay_fraction:0.2f}% threshold",
    )

    ax2.set_xlabel("Distance from point (microns)")
    ax2.set_ylabel("Normalized intensity")
    std = int(cell.metadata["r_z_std"])
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
    points: list[Cell],
    plot_output_path: Path | None,
    debug_data: bool,
    status_callback: Callable[[int], None] | None,
):
    output_cells = []
    count = 0
    z_center = dataset.get_cuboid_center("z", cell_calc.cube_voxels[0])
    y_center = dataset.get_cuboid_center("y", cell_calc.cube_voxels[1])
    x_center = dataset.get_cuboid_center("x", cell_calc.cube_voxels[2])

    for data, batch in tqdm.tqdm(
        zip(data_loader, sampler, strict=True), total=len(sampler)
    ):
        data = data.numpy()

        splits = [
            3,
            4,
            9,
            13,
            18,
            22,
            22 + cell_calc.lateral_max_radius_voxels + 1,
        ]
        (
            center,
            intensity,
            r_lat_data,
            lat_debug,
            r_axial_data,
            ax_debug,
            lat_line,
            ax_line,
        ) = np.split(data, splits, axis=1)
        center = center.tolist()
        intensity = intensity.tolist()
        r_lat_data = r_lat_data.tolist()
        r_axial_data = r_axial_data.tolist()

        for i, point_i in enumerate(batch):
            cell = points[point_i]
            z, y, x = map(int, center[i])
            r_lat, a_lat, offset_lat, sigma_lat, c_lat = r_lat_data[i]
            r_ax, a_ax, offset_ax, sigma_ax, c_ax = r_axial_data[i]

            cell.z += z - z_center
            cell.y += y - y_center
            cell.x += x - x_center

            if not hasattr(cell, "metadata"):
                cell.metadata = {}
            r_lat_std = lat_debug[i].max().item()
            cell.metadata.update(
                {
                    "r_z": int(round(r_ax)),
                    "r_y": int(round(r_lat)),
                    "r_x": int(round(r_lat)),
                    "center_intensity": intensity[i],
                    "r_z_std": ax_debug[i].max().item(),
                    "r_y_std": r_lat_std,
                    "r_x_std": r_lat_std,
                }
            )
            if debug_data:
                cell.metadata["lateral_parameters"] = {
                    "a": a_lat,
                    "offset": offset_lat,
                    "sigma": sigma_lat,
                    "c": c_lat,
                }
                cell.metadata["lateral_parameters_std"] = lat_debug[i].tolist()
                cell.metadata["lateral_line"] = lat_line[i].tolist()
                cell.metadata["axial_parameters"] = {
                    "a": a_ax,
                    "offset": offset_ax,
                    "sigma": sigma_ax,
                    "c": c_ax,
                }
                cell.metadata["axial_parameters_std"] = ax_debug[i].tolist()
                cell.metadata["axial_line"] = ax_line[i].tolist()

            output_cells.append(cell)

            if plot_output_path:
                _debug_display(
                    cell,
                    r_lat_data[i],
                    r_axial_data[i],
                    lat_line[i, :],
                    ax_line[i, :],
                    plot_output_path,
                    cell_calc,
                )

        count += len(batch)
        if status_callback is not None:
            status_callback(count)

    return output_cells


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
    output_cells_path: Path | None = None,
    batch_size: int = 32,
    n_free_cpus: int = 2,
    max_workers: int = 6,
    plot_output_path: Path | str | None = None,
    debug_data: bool = False,
    status_callback: Callable[[int], None] | None = None,
) -> list[Cell]:
    ts = datetime.now()
    cell_calc = CellSizeCalc(
        axial_dim=0,  # axis_order below is z, y, x
        voxel_size=voxel_size,
        cube_size=cube_size,
        initial_center_search_radius=initial_center_search_radius,
        initial_center_search_volume=initial_center_search_volume,
        lateral_intensity_algorithm=lateral_intensity_algorithm,
        lateral_max_radius=lateral_max_radius,
        lateral_decay_length=lateral_decay_length,
        lateral_decay_fraction=lateral_decay_fraction,
        lateral_decay_algorithm=lateral_decay_algorithm,
        axial_intensity_algorithm=axial_intensity_algorithm,
        axial_max_radius=axial_max_radius,
        axial_decay_length=axial_decay_length,
        axial_decay_fraction=axial_decay_fraction,
        axial_decay_algorithm=axial_decay_algorithm,
    )

    logging.info(f"cell_meta_3d: Starting analysis of {len(cells)} cells")

    data_loader, dataset, sampler, workers = _get_dataset(
        cells,
        points_filenames,
        signal_array,
        voxel_size,
        cube_size,
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
            dataset.points,
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
        batch_size=args.batch_size,
        output_cells_path=output_cells,
        n_free_cpus=args.n_free_cpus,
        max_workers=args.max_workers,
        plot_output_path=args.plot_output_path,
        debug_data=args.debug_data,
    )


if __name__ == "__main__":
    run_main()
