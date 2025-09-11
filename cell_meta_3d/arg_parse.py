import math
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
)
from functools import partial

from brainglobe_utils.general.numerical import (
    check_positive_float,
    check_positive_int,
)
from brainglobe_utils.general.string import check_str

from cell_meta_3d import __version__


def cell_meta_3d_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-s",
        "--signal-planes-path",
        dest="signal_planes_path",
        type=str,
        required=True,
        help="Path to the directory of the signal files. Can also be a text"
        "file pointing to the files. For a 3d tiff, data is in z, y, x order",
    )
    parser.add_argument(
        "-c",
        "--cells-path",
        dest="cells_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-cells-path",
        dest="output_cells_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v",
        "--voxel-size",
        dest="voxel_size",
        required=True,
        nargs=3,
        type=partial(check_positive_float, none_allowed=False),
        help="Voxel sizes in microns, in the order of data orientation "
        "(z, y, x). E.g. '5 2 2'",
    )
    parser.add_argument(
        "--cube-size",
        dest="cube_size",
        nargs=3,
        type=partial(check_positive_float, none_allowed=False),
        default=[100, 50, 50],
    )
    parser.add_argument(
        "-initial-r",
        "--initial-center-search-radius",
        dest="initial_center_search_radius",
        nargs=3,
        type=partial(check_positive_float, none_allowed=False),
        default=[10, 3, 3],
    )
    parser.add_argument(
        "-initial-v",
        "--initial-center-search-volume",
        dest="initial_center_search_volume",
        nargs=3,
        type=partial(check_positive_float, none_allowed=False),
        default=[15, 3, 3],
    )
    parser.add_argument(
        "-lat-int-algo",
        "--lateral-intensity-algorithm",
        dest="lateral_intensity_algorithm",
        choices=["center_line", "area", "area_margin"],
        default="area_margin",
    )
    parser.add_argument(
        "-lat-r",
        "--lateral-max-radius",
        dest="lateral_max_radius",
        type=partial(check_positive_float, none_allowed=False),
        default=20,
    )
    parser.add_argument(
        "-lat-len",
        "--lateral-decay-length",
        dest="lateral_decay_length",
        type=partial(check_positive_float, none_allowed=False),
        default=12,
    )
    parser.add_argument(
        "-lat-frac",
        "--lateral-decay-fraction",
        dest="lateral_decay_fraction",
        type=partial(check_positive_float, none_allowed=False),
        default=1 / math.e,
    )
    parser.add_argument(
        "-lat-dec-algo",
        "--lateral-decay-algorithm",
        dest="lateral_decay_algorithm",
        choices=["gaussian", "manual"],
        default="gaussian",
    )
    parser.add_argument(
        "-ax-int-algo",
        "--axial-intensity-algorithm",
        dest="axial_intensity_algorithm",
        choices=["center_line", "volume", "volume_margin"],
        default="center_line",
    )
    parser.add_argument(
        "-ax-r",
        "--axial-max-radius",
        dest="axial_max_radius",
        type=partial(check_positive_float, none_allowed=False),
        default=35,
    )
    parser.add_argument(
        "-ax-len",
        "--axial-decay-length",
        dest="axial_decay_length",
        type=partial(check_positive_float, none_allowed=False),
        default=35,
    )
    parser.add_argument(
        "-ax-frac",
        "--axial-decay-fraction",
        dest="axial_decay_fraction",
        type=partial(check_positive_float, none_allowed=False),
        default=1 / math.e,
    )
    parser.add_argument(
        "-ax-dec-algo",
        "--axial-decay-algorithm",
        dest="axial_decay_algorithm",
        choices=["gaussian", "manual"],
        default="gaussian",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=partial(check_positive_int, none_allowed=False),
        default=32,
    )
    parser.add_argument(
        "--n-free-cpus",
        dest="n_free_cpus",
        type=partial(check_positive_int, none_allowed=False),
        default=2,
    )
    parser.add_argument(
        "--max-workers",
        dest="max_workers",
        type=partial(check_positive_int, none_allowed=False),
        default=6,
    )
    parser.add_argument(
        "-p-path",
        "--plot-output-path",
        dest="plot_output_path",
        type=check_str,
        default=None,
    )
    parser.add_argument(
        "--debug-data",
        dest="debug_data",
        action="store_true",
    )

    return parser
