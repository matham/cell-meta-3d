import logging
import math
from collections.abc import Callable, Sequence
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import Literal

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
from cell_meta_3d.measure import CellSizeCalc


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
    axial_max_radius: float = 40,
    axial_decay_length: float = 35,
    axial_decay_fraction: float = 1 / math.e,
    axial_decay_algorithm: Literal["gaussian", "manual"] = "gaussian",
    output_cells_path: Path | None = None,
    batch_size: int = 32,
    n_free_cpus: int = 2,
    max_workers: int = 6,
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

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=workers,
    )

    logging.info(f"cell_meta_3d: Starting analysis of {len(cells)} cells")

    output_cells = []
    i = 0

    if workers:
        dataset.start_dataset_thread(workers)
    try:
        results = list(tqdm.tqdm(data_loader, unit="batch"))
    finally:
        dataset.stop_dataset_thread()

    assert len(results) == len(sampler)
    points = dataset.points
    for data, batch in zip(results, sampler, strict=False):
        data = data.numpy()

        splits = [3, 4, 9, 14, 14 + cell_calc.lateral_decay_len_voxels]
        center, intensity, r_lat_data, r_axial_data, lat_line, ax_line = (
            np.split(data, splits)
        )

        for i in batch:
            cell = points[i]
            # z, y, x, r_z, r_y, r_x, intensity = item.tolist()
            # cell.z = z
            # cell.y = y
            # cell.x = x
            # cell.metadata = {
            #     "r_z": r_z,
            #     "r_y": r_y,
            #     "r_x": r_x,
            #     "center_intensity": intensity,
            # }
            output_cells.append(cell)

        i += len(batch)
        if status_callback is not None:
            status_callback(i)

    save_cells(output_cells, str(output_cells_path))
    logging.info(f"cell_meta_3d: Analysis took {datetime.now() - ts}")

    return output_cells


def run_main():
    args = cell_meta_3d_parser().parse_args()

    signal = read_z_stack(args.signal_planes_path)
    cells = get_cells(args.cells_path, cells_only=True)
    output_cells = Path(args.output_cells_path)
    output_cells.parent.mkdir(parents=True, exist_ok=True)

    return main(
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
    )


if __name__ == "__main__":
    run_main()
