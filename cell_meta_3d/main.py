import math
from collections.abc import Callable
from numbers import Number
from pathlib import Path
from typing import Literal

import torch
import tqdm
from brainglobe_utils.cells.cells import Cell
from cellfinder.core import types
from cellfinder.core.classify.cube_generator import (
    CuboidBatchSampler,
    CuboidStackDataset,
)

from cell_meta_3d.measure import CellSizeCalc


class CellMeasureStackDataset(CuboidStackDataset):

    def __init__(self, cell_calc: CellSizeCalc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_calc = cell_calc

    def convert_to_output(self, data: torch.Tensor) -> torch.Tensor:
        if self.data_voxel_sizes != self.network_voxel_sizes:
            raise ValueError
        if len(data.shape) != 5:
            raise ValueError("Needs 5 dimensions: batch, channel and space")
        if self.output_axis_order[-1] != "c":
            raise ValueError("Channel should be last in data")

        # remove channel dim
        data = data[..., 0]
        output = torch.empty(
            (len(data), 7), dtype=torch.float32, device=data.device
        )

        cell_calc = self.cell_calc
        data = data.numpy()
        for i in range(len(data)):
            (z, y, x), r_lat, r_ax = cell_calc(data[i, ...])
            output[i, :] = torch.asarray(
                [z, y, x, r_ax, r_lat, r_lat, data[i, z, y, x]]
            )

        return output


def main(
    *,
    signal_array: types.array,
    cells: list[Cell],
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
    batch_size: int = 1,
    output_path: Path | None = None,
    status_callback: Callable[[int], None] | None = None,
) -> list[Cell]:
    cell_calc = CellSizeCalc(
        axial_dim=0,
        voxel_size=voxel_size,
        cube_size=cube_size,
        initial_center_search_size=initial_center_search_size,
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
        int(round(c / v)) for c, v in zip(cube_size, voxel_size, strict=False)
    )
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
        target_output="cell",
    )

    sampler = CuboidBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        sort_by_axis="z",
    )

    output_cells = []
    i = 0
    for batch in tqdm.tqdm(sampler):
        data, cells = dataset[batch]
        for item, cell in zip(data, cells, strict=False):
            z, y, x, r_z, r_y, r_x, intensity = item.tolist()
            cell.z = z
            cell.y = y
            cell.x = x
            cell.metadata = {
                "r_z": r_z,
                "r_y": r_y,
                "r_x": r_x,
                "center_intensity": intensity,
            }
            output_cells.append(cell)

        i += len(batch)
        if status_callback is not None:
            status_callback(i)

    return output_cells
