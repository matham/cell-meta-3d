""" """

from collections.abc import Callable
from typing import Any

import napari
import napari.layers
import numpy as np
from brainglobe_utils.cells.cells import Cell
from cellfinder.napari.utils import (
    add_single_layer,
)
from magicgui import magicgui
from magicgui.widgets import FunctionGui, ProgressBar
from napari.qt.threading import WorkerBase, WorkerBaseSignals
from napari.utils.notifications import show_info
from qtpy.QtCore import Signal


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
        voxel_size_z: float,
        voxel_size_y: float,
        voxel_size_x: float,
        cells: np.ndarray,
    ):
        super().__init__(SignalsClass=MyWorkerSignals)
        self.cells = cells

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
        self.update_progress_bar.emit("Setting up...", 1, 0)

        def update_callback(count: int) -> None:
            self.update_progress_bar.emit(
                "Analysing cells",
                1,
                count / len(self.cells),
            )

        self.update_progress_bar.emit("Finished analysis", 1, 1)
        return []


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
        widgets, new_names, insertions, strict=False
    ):
        # make it look as if it's directly in the root container
        widget.margins = 0, 0, 0, 0
        # the parameters of these widgets are updated using `auto_call` only.
        # If False, magicgui passes these as args to root() when the root's
        # function runs. But that doesn't list them as args of its function
        widget.gui_only = True
        root.insert(root.index(insertion), widget)
        getattr(root, widget.name).label = new_name


def get_results_callback(viewer: napari.Viewer) -> Callable:
    """
    Returns the callback that is connected to output of the pipeline.
    It returns the detected points that we have to visualize.
    """

    def done_func(points):
        add_single_layer(
            points,
            viewer=viewer,
            name="Analyzed cells",
            cell_type=Cell.CELL,
        )

    return done_func


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
        voxel_size_z={"value": 5, "label": "Voxel size (z)"},
        voxel_size_y={"value": 1, "label": "Voxel size (y)"},
        voxel_size_x={"value": 1, "label": "Voxel size (x)"},
        call_button=True,
        persist=True,
        scrollable=False,
    )
    def widget(
        voxel_size_z: float,
        voxel_size_y: float,
        voxel_size_x: float,
    ) -> None:
        """
        Run analysis.

        Parameters
        ----------
        voxel_size_z : float
            Size of your voxels in the axial dimension (microns)
        voxel_size_y : float
            Size of your voxels in the y direction (top to bottom) (microns)
        voxel_size_x : float
            Size of your voxels in the x direction (left to right) (microns)
        """
        # we must manually call so that the parameters of these functions are
        # initialized and updated. Because, if the images are open in napari
        # before we open cellfinder, then these functions may never be called
        signal_image_opt()
        cell_layer_opt()
        signal_image = options["signal_image"]
        cell_layer = options["cell_layer"]
        viewer = options["viewer"]

        if signal_image is None or cell_layer is None:
            show_info("Both signal image and cells must be provided.")
            return

        worker = Worker(
            voxel_size_x=voxel_size_x,
            voxel_size_y=voxel_size_y,
            voxel_size_z=voxel_size_z,
            cells=cell_layer.data,
        )

        worker.returned.connect(get_results_callback(viewer))
        # Make sure if the worker emits an error, it is propagated to this
        # thread
        worker.errored.connect(reraise)
        worker.connect_progress_bar_callback(progress_bar)

        worker.start()

    add_heavy_widgets(
        widget,
        (signal_image_opt, cell_layer_opt),
        ("Signal image", "Candidate cell layer"),
        ("voxel_size_z", "voxel_size_z"),
    )

    return widget
