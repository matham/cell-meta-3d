from pathlib import Path
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import napari
import numpy as np
from magicgui import widgets
from magicgui.widgets import FunctionGui
from napari.layers import Points
from scipy.optimize import curve_fit

from cell_meta_3d.napari.widget import _add_vol_from_multi_masks


def gaussian_func(x, a, offset, sigma, c):
    return a * np.exp(-np.square(x - offset) / (2 * sigma**2)) + c


class FilterWidget:

    def __init__(self):
        self._layer_widgets = []
        self.show_hist_widget = None

    def _get_metadata_value(self, cells: Points, measure: str) -> np.ndarray:
        if measure == "paor_cuboid_um3":
            extent = np.array([v for v in cells.features["paor_extent_um"]])
            extent = -extent[:, :, 0] + extent[:, :, 1]
            data = extent[:, 0] * extent[:, 1] * extent[:, 2]
        elif measure == "paor_cuboid_shape_um3":
            extent = np.array(
                [v for v in cells.features["paor_extent_shape_um"]]
            )
            extent = -extent[:, :, 0] + extent[:, :, 1]
            data = extent[:, 0] * extent[:, 1] * extent[:, 2]
        else:
            data = np.array([v for v in cells.features[measure]])

        return data

    def _calculate_measure(
        self,
        available_cells: Points,
        measure: Literal[""],
        indexing: list[int],
        measure2: Literal[None],
        indexing2: list[int],
        measure2_op: Literal["divide", "multiply"] = "divide",
        x_log_scale: bool = False,
        restrict_range: bool = False,
        x_range: tuple[float, float] = (1e-5, 1e5),
    ) -> np.ndarray:
        data = self._get_metadata_value(available_cells, measure)
        for i in indexing:
            data = data[:, i]

        if measure2:
            data2 = self._get_metadata_value(available_cells, measure2)
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

        if restrict_range:
            mask = np.logical_and(x_range[0] <= data, data <= x_range[1])
            masked_data = np.zeros_like(data)
            if x_log_scale:
                masked_data[mask] = np.log10(data[mask])
            else:
                masked_data[mask] = data[mask]
        else:
            if x_log_scale:
                masked_data = np.log10(data)
            else:
                masked_data = data

        return masked_data

    def _get_histogram(
        self,
        data: np.ndarray,
        y_log_scale: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        hist, edges = np.histogram(data, bins="auto")
        center = (edges[1:] - edges[:-1]) / 2 + edges[:-1]

        if y_log_scale:
            min_val = np.min(hist)
            if np.isclose(min_val, 0):
                hist = (
                    hist
                    + hist[np.logical_not(np.isclose(hist, 0))].min() / 100
                )

            hist = np.log10(hist)

        return center, hist

    def _get_gaussian_params(
        self,
        center: np.ndarray,
        hist: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        max_hist_i = np.argmax(hist)
        max_c_n = np.sum(center <= center[max_hist_i])
        (a, offset, sigma, c), pcov = curve_fit(
            gaussian_func,
            center[:max_c_n],
            hist[:max_c_n],
            p0=[hist[max_hist_i], center[max_hist_i], 1, 0],
        )
        perr = np.sqrt(np.diag(pcov))

        return a, offset, np.abs(sigma), c, perr

    def show_hist(
        self,
        available_cells: Points,
        measure: Literal[""],
        indexing: list[int],
        measure2: Literal[None],
        indexing2: list[int],
        measure2_op: Literal["divide", "multiply"] = "divide",
        restrict_range: bool = False,
        x_range: tuple[float, float] = (1e-5, 1e5),
        x_log_scale: bool = False,
        y_log_scale: bool = False,
        fit_gaussian: bool = False,
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
            restrict_range,
            x_range,
        )

        center, hist = self._get_histogram(data, y_log_scale)
        plt.plot(center, hist, label="histogram")

        if fit_gaussian:
            a, offset, sigma, c, perr = self._get_gaussian_params(center, hist)
            print(
                "Got values of a, offset, sigma, c, perr:",
                a,
                offset,
                sigma,
                c,
                perr,
            )

            pred = gaussian_func(center, a, offset, sigma, c)
            plt.plot(center, pred, label="Gaussian")
            plt.plot(center, hist - pred, label="Residual")

        plt.legend()
        plt.tight_layout()
        plt.show()

    def apply_filter(
        self,
        displayed_cells: Points,
        gaussian_cutoff: bool = False,
        cutoff: float = 0.0,
        residual_mean: bool = False,
        residual_percentile: float = 0,
        keep_above: bool = True,
        reset_visible: bool = True,
        application: Literal["filter", "color"] = "filter",
    ):
        w = self.show_hist_widget
        if not displayed_cells:
            raise ValueError("Cells must be provided")
        if not w.available_cells.value:
            raise ValueError("Cells must be provided")

        data = self._calculate_measure(
            displayed_cells,
            w.measure.value,
            w.indexing.value,
            w.measure2.value,
            w.indexing2.value,
            w.measure2_op.value,
            w.x_log_scale.value,
            w.restrict_range.value,
            w.x_range.value,
        )
        data_full = self._calculate_measure(
            w.available_cells.value,
            w.measure.value,
            w.indexing.value,
            w.measure2.value,
            w.indexing2.value,
            w.measure2_op.value,
            w.x_log_scale.value,
            w.restrict_range.value,
            w.x_range.value,
        )

        if application == "filter":
            if gaussian_cutoff:
                center, hist = self._get_histogram(
                    data_full, w.y_log_scale.value
                )
                a, offset, sigma, c, perr = self._get_gaussian_params(
                    center, hist
                )
                cutoff = offset + cutoff * sigma

                if residual_mean:
                    cutoff = np.mean(data_full[data_full > cutoff])
                elif residual_percentile > 0:
                    cutoff = np.percentile(
                        data_full[data_full > cutoff], residual_percentile
                    )
            print(f"filter cutoff is {cutoff}")

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
            min_val = min(data_full.min(), data.min())
            data -= min_val
            data_full -= min_val
            data /= max(data_full.max(), data.max())
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
        self._update_measure_choices()

    def _update_measure_choices(self, *args):
        cells = self.show_hist_widget.available_cells
        if cells is None:
            return

        names = cells.value.features.columns
        self.show_hist_widget.measure.choices = list(names) + [
            "paor_cuboid_um3",
            "paor_cuboid_shape_um3",
        ]
        self.show_hist_widget.measure2.choices = list(names) + [
            "paor_cuboid_um3",
            "paor_cuboid_shape_um3",
        ]

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
                "x_range": {
                    "options": {"step": 1e-5, "max": 100_000, "min": -100_000}
                },
            },
        )
        self.show_hist_widget = show_hist_widget
        apply_filter_widget = FunctionGui(
            self.apply_filter,
            call_button="Apply filter",
            auto_call=False,
            persist=True,
            param_options={
                "cutoff": {"step": 1e-4, "max": 100_000, "min": -100_000},
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
        layers_update()

        # needed for enabling scrolling
        return container.root_native_widget


def filter_widget():
    return FilterWidget().build()
