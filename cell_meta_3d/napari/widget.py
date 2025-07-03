""" """

import math
from collections.abc import Callable
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
from magicgui import magicgui
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


def analyse_widget():
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
        call_button=True,
        persist=True,
    )
    def widget(
        selected_cells_only: bool = False,
        voxel_size: tuple[float, float, float] = (5, 1, 1),
        cube_size: tuple[float, float, float] = (100, 50, 50),
        initial_center_search_radius: tuple[float, float, float] = (10, 3, 3),
        initial_center_search_volume: tuple[float, float, float] = (15, 3, 3),
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
        output_cells_path: Path = None,
        batch_size: int = 32,
        n_free_cpus: int = 2,
        max_workers: int = 3,
        plot_output_path: Path | None = None,
        save_plots: bool = False,
        debug_data: bool = False,
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
            batch_size=batch_size,
            output_cells_path=output_cells_path,
            n_free_cpus=n_free_cpus,
            max_workers=max_workers,
            plot_output_path=plot_output_path,
            debug_data=debug_data,
        )

        # Make sure if the worker emits an error, it is propagated to this
        # thread
        worker.errored.connect(reraise)
        worker.connect_progress_bar_callback(progress_bar)

        worker.start()

    add_heavy_widgets(
        widget,
        (signal_image_opt, cell_layer_opt),
        ("Signal image", "Cell layer"),
        ("voxel_size", "voxel_size"),
    )
    widget.insert(widget.index("output_cells_path") + 1, progress_bar)

    return widget
