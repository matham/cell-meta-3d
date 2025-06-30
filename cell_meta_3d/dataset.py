from typing import Union

import numpy as np
import torch
from cellfinder.core.classify.cube_generator import (
    CuboidDatasetBase,
    CuboidStackDataset,
    CuboidTiffDataset,
)

from cell_meta_3d.measure import CellSizeCalc


class CellMeasureDatasetBase:
    """Output order is same as `output_axis_order`, excluding channel."""

    def __init__(self, cell_calc: CellSizeCalc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_calc = cell_calc

    def convert_to_output(
        self: Union["CuboidDatasetBase", "CellMeasureDatasetBase"],
        data: torch.Tensor,
    ) -> torch.Tensor:
        if self.data_voxel_sizes != self.network_voxel_sizes:
            raise ValueError
        if len(data.shape) != 5:
            raise ValueError("Needs 5 dimensions: batch, channel and space")
        if self.output_axis_order[-1] != "c":
            raise ValueError("Channel should be last in data")

        # remove channel dim
        np_data = data[..., 0].numpy()
        cell_calc = self.cell_calc
        center, r_lat_data, r_axial_data, lat_line, ax_line = cell_calc(
            np_data
        )

        offsets = np.zeros_like(center)
        for i in range(3):
            r_data = r_axial_data if i == cell_calc.axial_dim else r_lat_data
            offsets[:, i] = np.round(r_data[:, 2]).astype(np.int_)

        idx = center + offsets
        intensity = np_data[
            np.arange(len(np_data)), idx[:, 0], idx[:, 1], idx[:, 2]
        ]

        output = np.concatenate(
            (
                center,
                intensity[:, None],
                r_lat_data,
                r_axial_data,
                lat_line,
                ax_line,
            ),
            axis=1,
        )

        return torch.from_numpy(output).to(device=data.device)


class CellMeasureStackDataset(CellMeasureDatasetBase, CuboidStackDataset):
    pass


class CellMeasureTiffDataset(CellMeasureDatasetBase, CuboidTiffDataset):
    pass
