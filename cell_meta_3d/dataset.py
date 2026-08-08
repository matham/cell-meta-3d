import math
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

        keys = (
            "center",
            "r_lat",
            "lat_line",
            "r_lat_params",
            "r_axial",
            "ax_line",
            "r_axial_params",
            "center_values",
            "data_min",
            "paor_vectors_intensity",
            "paor_centroid_intensity",
            "paor_moment2_intensity",
            "paor_extent_intensity",
            "paor_vectors_mask",
            "paor_centroid_mask",
            "paor_moment2_mask",
            "paor_extent_mask",
        )
        data_keys = (
            "slice_start",
            "sliced_upsampled_shape",
            "sliced_upsampled_flat_data",
            "sliced_seg_mask_upsampled_flat_idx",
            "sliced_upsampled_n_non_zeros",
        )

        sub_size = 16
        micro_batches = {k: [] for k in keys + data_keys}
        n = np_data.shape[0]
        total = 0
        for i in range(int(math.ceil(n / sub_size))):
            # process batch of the cubes and get back their measured data
            res_data = cell_calc(
                np_data[i * sub_size : (i + 1) * sub_size, ...]
            )
            for key, arr in res_data.items():
                # could be none for params that are available only if selected
                if arr is None or key not in keys or key in data_keys:
                    continue

                if arr.dtype.names is not None:
                    arr = rfn.structured_to_unstructured(arr)

                micro_batches[key].append(arr)

            # flatten out the intensity values to not copy the full cube
            mask = res_data["sliced_segmentation_mask_upsampled"]
            intensity = res_data["sliced_upsampled_data"]
            micro_n = mask.shape[0]

            # needs batch dim
            micro_batches["slice_start"].append(
                np.repeat(res_data["slice_start"][None, :], micro_n, axis=0)
            )
            micro_batches["sliced_upsampled_shape"].append(
                np.repeat([mask.shape[1:]], micro_n, axis=0)
            )
            micro_batches["sliced_upsampled_flat_data"].append(intensity[mask])

            bi, d1i, d2i, d3i = np.nonzero(mask)
            bi += total
            mask_i = np.array([bi, d1i, d2i, d3i], dtype=np.intp).transpose()
            micro_batches["sliced_seg_mask_upsampled_flat_idx"].append(mask_i)

            non_zero_counts = mask.reshape((micro_n, -1)).sum(axis=1)
            micro_batches["sliced_upsampled_n_non_zeros"].append(
                non_zero_counts
            )

            total += micro_n

        assert total == n

        arrays = []
        for key in keys + data_keys:
            item_arrays = micro_batches[key]
            if len(item_arrays):
                arrays.append(np.concatenate(item_arrays, axis=0))
            else:
                arrays.append(np.array([]))

        arrays = tuple(
            torch.from_numpy(arr).to(device=data.device) for arr in arrays
        )
        return arrays


class CellMeasureStackDataset(CellMeasureDatasetBase, CuboidArrayDataset):
    pass


class CellMeasureTiffDataset(CellMeasureDatasetBase, CuboidTiffDataset):
    pass
