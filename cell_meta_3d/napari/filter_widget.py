from pathlib import Path
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import napari
import numpy as np
from magicgui import widgets
from magicgui.widgets import FunctionGui
from napari.layers import Points

from cell_meta_3d.napari.widget import _add_vol_from_multi_masks


class FilterWidget:

    def __init__(self):
        self._layer_widgets = []
        self.show_hist_widget = None

    def _calculate_measure(
        self,
        available_cells: Points,
        measure: Literal[""],
        indexing: list[int],
        measure2: Literal[None],
        indexing2: list[int],
        measure2_op: Literal["divide", "multiply"] = "divide",
        x_log_scale: bool = False,
        x_log_min: float = 1e-5,
    ) -> np.ndarray:
        data = np.array([v for v in available_cells.features[measure]])
        for i in indexing:
            data = data[:, i]

        if measure2:
            data2 = np.array([v for v in available_cells.features[measure2]])
            for i in indexing2:
                data2 = data2[:, i]

            if measure2_op == "divide":
                data = np.divide(
                    data, data2, out=np.zeros_like(data2), where=data2 != 0
                )
            elif measure2_op == "multiply":
                data = data * data2
            else:
                raise ValueError

        data = np.maximum(data, x_log_min)
        if x_log_scale:
            data = np.log10(data)

        return data

    def show_hist(
        self,
        available_cells: Points,
        measure: Literal[""],
        indexing: list[int],
        measure2: Literal[None],
        indexing2: list[int],
        measure2_op: Literal["divide", "multiply"] = "divide",
        x_log_scale: bool = False,
        x_log_min: float = 1e-5,
        y_log_scale: bool = False,
        num_bins: int = 100,
    ) -> None:
        if not available_cells:
            raise ValueError("Cells must be provided")
        if not measure:
            raise ValueError("Measure must be provided")

        data = self._calculate_measure(
            available_cells,
            measure,
            indexing,
            measure2,
            indexing2,
            measure2_op,
            x_log_scale,
            x_log_min,
        )

        plt.hist(data, bins=num_bins, log=y_log_scale)
        plt.show()

    def apply_filter(
        self,
        displayed_cells: Points,
        cutoff: float = 0.0,
        keep_above: bool = True,
        reset_visible: bool = True,
        application: Literal["filter", "color"] = "filter",
    ):
        if not displayed_cells:
            raise ValueError("Cells must be provided")

        w = self.show_hist_widget
        data = self._calculate_measure(
            displayed_cells,
            w.measure.value,
            w.indexing.value,
            w.measure2.value,
            w.indexing2.value,
            w.measure2_op.value,
            w.x_log_scale.value,
            w.x_log_min.value,
        )

        if application == "filter":
            if keep_above:
                mask = data >= cutoff
            else:
                mask = data <= cutoff

            if reset_visible:
                displayed_cells.shown = mask
            else:
                displayed_cells.shown = np.logical_and(
                    mask, displayed_cells.shown
                )
        else:
            displayed_cells.shown = True
            data -= data.min()
            data /= data.max()
            displayed_cells.face_color = plt.cm.plasma(data)

    def show_segmentation(
        self,
        cells: Points,
        h5_segmentation_path: Path,
        selected_cells_only: bool = False,
    ):
        seg_ids = cells.features["seg_id"]
        if selected_cells_only:
            seg_ids = [seg_ids[i] for i in cells.selected_data]

        sz, sy, sx = cells.scale
        masks = []
        with h5py.File(h5_segmentation_path, "r") as h5_file:
            super_voxel = tuple(v.item() for v in h5_file.attrs["super_voxel"])
            intensity_index_dset: h5py.Dataset = h5_file[
                "upsampled_cuboid_intensity_index"
            ]
            intensity_dset: h5py.Dataset = h5_file[
                "upsampled_cuboid_intensity"
            ]
            cell_index_range_dset: h5py.Dataset = h5_file["cell_index_range"]
            cell_corner_dset: h5py.Dataset = h5_file["cell_corner_vox"]
            cuboid_voxels_dset: h5py.Dataset = h5_file[
                "upsampled_cuboid_voxels"
            ]

            for data_index in seg_ids:
                intensity_s, intensity_e = cell_index_range_dset[data_index, :]
                corner_up = [
                    int(round(v))
                    for v in cell_corner_dset[data_index, :] * super_voxel
                ]
                intensity_index = np.asarray(
                    intensity_index_dset[intensity_s:intensity_e, :],
                    dtype=np.intp,
                )
                intensity = np.asarray(intensity_dset[intensity_s:intensity_e])
                cuboid_voxels = cuboid_voxels_dset[data_index, :]

                volume = np.zeros(cuboid_voxels, dtype=bool)
                volume[
                    intensity_index[:, 0],
                    intensity_index[:, 1],
                    intensity_index[:, 2],
                ] = True

                masks.append((corner_up, volume.shape, volume, intensity))

        if not masks:
            raise ValueError("No points available to show")

        _add_vol_from_multi_masks(
            napari.current_viewer(),
            masks,
            (
                sz / super_voxel[0],
                sy / super_voxel[1],
                sx / super_voxel[2],
            ),
            "Segmentation",
        )

    def _update_layer_widgets_new_layers(self, *args):
        for widget in self._layer_widgets:
            widget.reset_choices()

    def _update_measure_choices(self, *args):
        cells = self.show_hist_widget.available_cells
        if cells is None:
            return

        names = cells.value.features.columns
        self.show_hist_widget.measure.choices = list(names)
        self.show_hist_widget.measure2.choices = list(names)

    def build(self) -> widgets.Container:
        show_hist_widget = FunctionGui(
            self.show_hist,
            call_button="Show hist",
            auto_call=False,
            persist=True,
            param_options={
                "measure": {"nullable": True},
                "indexing": {"options": {"min": 0}},
                "measure2": {"nullable": True},
                "indexing2": {"options": {"min": 0}},
                "x_log_min": {"step": 1e-5},
                "num_bins": {"min": 2},
            },
        )
        self.show_hist_widget = show_hist_widget
        apply_filter_widget = FunctionGui(
            self.apply_filter,
            call_button="Apply filter",
            auto_call=False,
            persist=True,
            param_options={
                "cutoff": {"step": 1e-4, "max": 100_000},
            },
        )
        show_seg_widget = FunctionGui(
            self.show_segmentation,
            call_button="Show segmentation",
            auto_call=False,
            persist=True,
        )

        container = widgets.Container(
            widgets=[
                show_hist_widget,
                apply_filter_widget,
                show_seg_widget,
            ],
            layout="vertical",
            labels=False,
            scrollable=True,
        )

        show_hist_widget.available_cells.changed.connect(
            self._update_measure_choices
        )

        self._layer_widgets = [
            show_hist_widget.available_cells,
            apply_filter_widget.displayed_cells,
            show_seg_widget.cells,
        ]
        layers_update = self._update_layer_widgets_new_layers

        viewer = napari.current_viewer()
        viewer.layers.events.inserted.connect(layers_update)
        viewer.layers.events.removed.connect(layers_update)
        viewer.layers.events.renamed.connect(layers_update)

        # needed for enabling scrolling
        return container.root_native_widget


def filter_widget():
    return FilterWidget().build()
