""" """

import math
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal

import napari
import napari.layers
import numpy as np
from brainglobe_utils.cells.cells import Cell
from cellfinder.napari.utils import (
    brainglobe_points_axis_order,
    napari_array_to_cells,
)
from magicgui import magicgui, widgets
from magicgui.widgets import FunctionGui, ProgressBar
from napari.qt.threading import WorkerBase, WorkerBaseSignals
from napari.utils.notifications import show_info
from qtpy.QtCore import Signal

from cell_meta_3d.main import main


class MyWorkerSignals(WorkerBaseSignals):
    """
    Signals used by the Worker class below.
    """

    # Emits (label, max, value) for the progress bar
    update_progress_bar = Signal(str, int, int)


class Worker(WorkerBase):
    """
    Runs cellfinder in a separate thread, to prevent GUI blocking.

    Also handles callbacks between the worker thread and main napari GUI thread
    to update a progress bar.
    """

    def __init__(
        self,
        signal_array: napari.layers.Image,
        cells: list[Cell],
        **main_args,
    ):
        super().__init__(SignalsClass=MyWorkerSignals)
        self.signal_array = signal_array
        self.cells = cells
        self.main_args = main_args

    def connect_progress_bar_callback(self, progress_bar: ProgressBar):
        """
        Connects the progress bar to the work so that updates are shown on
        the bar.
        """

        def update_progress_bar(label: str, max_val: int, value: int):
            progress_bar.label = label
            progress_bar.max = max_val
            progress_bar.value = value

        self.update_progress_bar.connect(update_progress_bar)

    def work(self) -> list:
        self.update_progress_bar.emit("Setting up...", len(self.cells), 0)

        def status_callback(count: int) -> None:
            self.update_progress_bar.emit(
                "Analysing cells",
                len(self.cells),
                count,
            )

        cells = main(
            signal_array=self.signal_array.data,
            cells=self.cells,
            status_callback=status_callback,
            **self.main_args,
        )

        self.update_progress_bar.emit(
            "Finished analysis", len(self.cells), len(self.cells)
        )
        return cells


def _add_sphere_layers(
    cells: list[Cell], viewer: napari.Viewer, data_layer: napari.layers.Image
):
    sz, sy, sx = data_layer.scale
    s_lat = (sx + sy) / 2
    for cell in cells:
        z = cell.z
        y = cell.y
        x = cell.x

        r_xy = max(cell.metadata["r_xy_um"] / s_lat, 1)
        r_z = max(cell.metadata["r_z_um"] / sz, 1)
        size_xy = max(int(round(2 * cell.metadata["r_xy_um"] / s_lat)), 1) + 2
        size_z = max(int(round(2 * cell.metadata["r_z_um"] / sz)), 1) + 2
        c_xy = size_xy / 2
        c_z = size_z / 2

        zg, yg, xg = np.mgrid[0:size_z:1, 0:size_xy:1, 0:size_xy:1]
        r = np.sqrt(
            ((zg - c_z) / r_z) ** 2
            + ((yg - c_xy) / r_xy) ** 2
            + ((xg - c_xy) / r_xy) ** 2
        )
        sphere = np.zeros((*r.shape, 4))
        sphere[r <= 1, :] = 1

        viewer.add_image(
            sphere,
            name=f"{z}z{y}y{x}x sphere",
            scale=[sz, sy, sx],
            translate=((z - c_z) * sz, (y - c_xy) * sy, (x - c_xy) * sx),
            rgb=True,
        )


def _add_segmentation_layers(
    cells: list[Cell],
    viewer: napari.Viewer,
    data_layer: napari.layers.Image,
    voxel_size: tuple[float, float, float],
    seg_super_voxel: tuple[int, int, int],
):
    sz, sy, sx = data_layer.scale
    for cell in cells:
        seg_data = cell.metadata["segmentation_upsampled"]
        mask = seg_data["mask"][:, :, :, None]
        mask = np.repeat(mask, 4, axis=3)
        zcn, ycn, xcn = (
            c / s
            for c, s in zip(seg_data["corner_um"], voxel_size, strict=True)
        )

        z, y, x = cell.z, cell.y, cell.x

        viewer.add_image(
            mask,
            name=f"{z}z{y}y{x}x segmentation",
            scale=(
                sz / seg_super_voxel[0],
                sy / seg_super_voxel[1],
                sx / seg_super_voxel[2],
            ),
            translate=(zcn * sz, ycn * sy, xcn * sx),
            rgb=True,
        )

        for tp, name in (("", "intensity"), ("_shape", "shape")):
            # convert all to zyx
            vectors_um = np.flip(
                np.array(cell.metadata[f"paor{tp}_xyz_um"]), axis=1
            )
            vectors_vox = vectors_um / [voxel_size]

            centroid = np.array(cell.metadata[f"paor_centroid{tp}_xyz_um"])[
                ::-1
            ]
            centroid_vox = centroid / voxel_size

            extent_um = np.array(cell.metadata[f"paor_extent{tp}_um"])

            # vectors_vox * extent_um is vectors_um * extent_um / voxel_size,
            # which is extent_vector_vox
            lines = [
                np.array(
                    [
                        centroid_vox + vectors_vox[ax, :] * extent_um[ax, 0],
                        centroid_vox + vectors_vox[ax, :] * extent_um[ax, 1],
                    ]
                )
                for ax in range(3)
            ]

            viewer.add_shapes(
                lines,
                name=f"{z}z{y}y{x}x PAOR {name}",
                shape_type="line",
                edge_color=["red", "green", "blue"],
                scale=(sz, sy, sx),
                edge_width=0.25,
            )


def process_worker_result(
    cells: list[Cell],
    viewer: napari.Viewer,
    data_layer: napari.layers.Image,
    add_sphere_layers: bool,
    add_segmentation_layers: bool,
    voxel_size: tuple[float, float, float],
    seg_super_voxel: tuple[int, int, int],
):
    if add_sphere_layers:
        _add_sphere_layers(cells, viewer, data_layer)

    if add_segmentation_layers:
        _add_segmentation_layers(
            cells, viewer, data_layer, voxel_size, seg_super_voxel
        )


def get_heavy_widgets(
    options: dict[str, Any],
) -> tuple[Callable, Callable]:
    # heavy widgets are updated only when they update because they are slower
    @magicgui(
        call_button=False,
        persist=False,
        scrollable=False,
        labels=False,
        auto_call=True,
    )
    def signal_image_opt(
        viewer: napari.Viewer,
        signal_image: napari.layers.Image,
    ):
        """
        magicgui widget for setting the signal_image parameter.

        Parameters
        ----------
        signal_image : napari.layers.Image
             Image layer containing the cells
        """
        options["signal_image"] = signal_image
        options["viewer"] = viewer

    @magicgui(
        call_button=False,
        persist=False,
        scrollable=False,
        labels=False,
        auto_call=True,
    )
    def cell_layer_opt(
        cell_layer: napari.layers.Points,
    ):
        """
        magicgui widget for setting the cell layer.

        Parameters
        ----------
        cell_layer : napari.layers.Points
            The cell layer containing the detected cells to analyse.
        """
        options["cell_layer"] = cell_layer

    return signal_image_opt, cell_layer_opt


def add_heavy_widgets(
    root: FunctionGui,
    widgets: tuple[FunctionGui, ...],
    new_names: tuple[str, ...],
    insertions: tuple[str, ...],
) -> None:
    for widget, new_name, insertion in zip(
        widgets, new_names, insertions, strict=True
    ):
        # make it look as if it's directly in the root container
        widget.margins = 0, 0, 0, 0
        # the parameters of these widgets are updated using `auto_call` only.
        # If False, magicgui passes these as args to root() when the root's
        # function runs. But that doesn't list them as args of its function
        widget.gui_only = True
        root.insert(root.index(insertion), widget)
        getattr(root, widget.name).label = new_name


def reraise(e: Exception) -> None:
    """Re-raises the exception."""
    raise Exception from e


def analyse_widget() -> widgets.Container:
    progress_bar = ProgressBar()

    # options that is filled in from the gui
    options = {
        "signal_image": None,
        "viewer": None,
        "cell_layer": None,
    }
    signal_image_opt, cell_layer_opt = get_heavy_widgets(options)

    @magicgui(
        lateral_decay_fraction={"max": 1, "step": 0.0001},
        axial_decay_fraction={"max": 1, "step": 0.0001},
        plot_output_path={"mode": "d"},
        output_cells_path={"mode": "w"},
        segmentation_path={"mode": "w"},
        call_button=True,
        persist=True,
    )
    def widget(
        selected_cells_only: bool = False,
        voxel_size: tuple[float, float, float] = (5, 1, 1),
        cube_size: tuple[float, float, float] = (100, 50, 50),
        initial_center_search_radius: tuple[float, float, float] = (10, 3, 3),
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
        seg_decay_fraction: float = 1 / math.e,
        seg_super_voxel: tuple[int, int, int] = (1, 1, 1),
        batch_size: int = 32,
        n_free_cpus: int = 2,
        max_workers: int = 3,
        output_cells_path: Path = None,
        segmentation_path: Path | None = None,
        plot_output_path: Path | None = None,
        save_segmentation: bool = False,
        save_plots: bool = False,
        debug_data: bool = False,
        add_sphere_layers: bool = False,
        add_segmentation_layers: bool = False,
    ) -> None:
        """
        Run analysis.

        Parameters
        ----------
        """
        # we must manually call so that the parameters of these functions are
        # initialized and updated. Because, if the images are open in napari
        # before we open cellfinder, then these functions may never be called
        signal_image_opt()
        cell_layer_opt()
        signal_image = options["signal_image"]
        cell_layer = options["cell_layer"]

        if signal_image is None or cell_layer is None:
            show_info("Both signal image and cells must be provided.")
            return

        if save_plots and not plot_output_path:
            raise ValueError
        if not save_plots:
            plot_output_path = None
        if save_segmentation and not segmentation_path:
            raise ValueError
        if not save_segmentation:
            segmentation_path = None

        if selected_cells_only:
            selection = np.asarray(list(cell_layer.selected_data))
            data = np.asarray(cell_layer.data)[selection, :]
            data = data[:, brainglobe_points_axis_order].tolist()
            cells = []
            for row in data:
                cells.append(Cell(pos=row, cell_type=Cell.UNKNOWN))
        else:
            cells = napari_array_to_cells(cell_layer, Cell.CELL)

        worker = Worker(
            signal_array=signal_image,
            cells=cells,
            voxel_size=voxel_size,
            cube_size=cube_size,
            initial_center_search_radius=initial_center_search_radius,
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
            seg_decay_fraction=seg_decay_fraction,
            seg_super_voxel=seg_super_voxel,
            batch_size=batch_size,
            output_cells_path=output_cells_path,
            n_free_cpus=n_free_cpus,
            max_workers=max_workers,
            plot_output_path=plot_output_path,
            debug_data=debug_data,
            segmentation_path=segmentation_path,
            add_segmentation_to_metadata=True,
        )

        # Make sure if the worker emits an error, it is propagated to this
        # thread
        worker.errored.connect(reraise)
        worker.connect_progress_bar_callback(progress_bar)
        worker.returned.connect(
            partial(
                process_worker_result,
                viewer=options["viewer"],
                data_layer=signal_image,
                add_sphere_layers=add_sphere_layers,
                add_segmentation_layers=add_segmentation_layers,
                voxel_size=voxel_size,
                seg_super_voxel=seg_super_voxel,
            )
        )

        worker.start()

    add_heavy_widgets(
        widget,
        (signal_image_opt, cell_layer_opt),
        ("Signal image", "Cell layer"),
        ("voxel_size", "voxel_size"),
    )
    widget.insert(widget.index("add_segmentation_layers") + 1, progress_bar)

    container = widgets.Container(
        widgets=[widget],
        layout="vertical",
        labels=False,
        scrollable=True,
    )

    # needed for enabling scrolling
    return container.root_native_widget
