from typing import Union

import numpy as np
import torch
from cellfinder.core.classify.cube_generator import (
    CuboidArrayDataset,
    CuboidDatasetBase,
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
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
            segmentation_mask,
            paor_vectors_intensity,
            paor_centroid_intensity,
            paor_moment2_intensity,
            paor_extent_intensity,
            paor_vectors_mask,
            paor_centroid_mask,
            paor_moment2_mask,
            paor_extent_mask,
        ) = cell_calc(np_data)

        # get the center intensity of the points
        intensity = np_data[
            np.arange(len(np_data)), center[:, 0], center[:, 1], center[:, 2]
        ]
        min_intensity = np.min(np_data, axis=(1, 2, 3))

        arrays = (
            center,
            intensity,
            min_intensity,
            r_lat,
            lat_line,
            (
                np.array([])
                if r_lat_params is None
                else rfn.structured_to_unstructured(r_lat_params)
            ),
            r_axial,
            ax_line,
            (
                np.array([])
                if r_axial_params is None
                else rfn.structured_to_unstructured(r_axial_params)
            ),
            segmentation_mask,
            paor_vectors_intensity,
            paor_centroid_intensity,
            paor_moment2_intensity,
            paor_extent_intensity,
            paor_vectors_mask,
            paor_centroid_mask,
            paor_moment2_mask,
            paor_extent_mask,
        )

        arrays = tuple(
            torch.from_numpy(arr).to(device=data.device) for arr in arrays
        )
        arrays = *arrays, data[..., 0]

        return arrays


class CellMeasureStackDataset(CellMeasureDatasetBase, CuboidArrayDataset):
    pass


class CellMeasureTiffDataset(CellMeasureDatasetBase, CuboidTiffDataset):
    pass
