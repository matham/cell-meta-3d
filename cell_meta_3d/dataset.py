from typing import Union

import numpy as np
import torch
from cellfinder.core.classify.cube_generator import (
    CuboidDatasetBase,
    CuboidStackDataset,
    CuboidTiffDataset,
)
from numpy.lib import recfunctions as rfn

from cell_meta_3d.measure import CellSizeCalc


class CellMeasureDatasetBase:
    """Output order is same as `output_axis_order`, excluding channel.

    Basic idea is that cube extraction is executed in the workers, and we want
    the calculations to also happen in the worker. So we get the cube, process
    it, and return the calculated parameters. This gets collated by the
    data loader and returned in the main function that reads the data loader.
    """

    def __init__(self, cell_calc: CellSizeCalc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_calc = cell_calc

    def convert_to_output(
        self: Union["CuboidDatasetBase", "CellMeasureDatasetBase"],
        data: torch.Tensor,
    ) -> torch.Tensor:
        """
        We do our own conversion to output. By default, it converts it so the
        network can process the data. Instead, we pass it through the
        `CellSizeCalc` and return its result.
        """
        if self.data_voxel_sizes != self.network_voxel_sizes:
            # we don't do any scaling so data and output (network) size are
            # expected to have been set the same
            raise ValueError
        if len(data.shape) != 5:
            raise ValueError("Needs 5 dimensions: batch, channel and space")
        if self.output_axis_order[-1] != "c":
            # this is also set when creating the instance
            raise ValueError("Channel should be last in data")

        # data comes in/returned as torch tensors. But calc needs numpy arrays
        # remove channel dim because we just have data for the signal channel
        np_data = data[..., 0].numpy()
        cell_calc = self.cell_calc
        # process batch of the cubes and get back their measured data
        (
            center,
            r_lat,
            lat_line,
            r_lat_params,
            r_axial,
            ax_line,
            r_axial_params,
        ) = cell_calc(np_data)
        if len(lat_line.shape) != 2:
            lat_line = lat_line[:, None]
        if len(ax_line.shape) != 2:
            ax_line = ax_line[:, None]

        # get the center intensity of the points
        intensity = np_data[
            np.arange(len(np_data)), center[:, 0], center[:, 1], center[:, 2]
        ]

        # convert it to a flat NxK array so we can convert it to torch
        arrays = [
            center,
            intensity[:, None],
            r_lat[:, None],
            lat_line,
        ]
        if r_lat_params is not None:
            # structured_to_unstructured will always create 2d array
            arrays.append(rfn.structured_to_unstructured(r_lat_params))
        arrays.extend([r_axial[:, None], ax_line])
        if r_axial_params is not None:
            arrays.append(rfn.structured_to_unstructured(r_axial_params))

        output = np.concatenate(arrays, axis=1)

        return torch.from_numpy(output).to(device=data.device)


class CellMeasureStackDataset(CellMeasureDatasetBase, CuboidStackDataset):
    pass


class CellMeasureTiffDataset(CellMeasureDatasetBase, CuboidTiffDataset):
    pass
