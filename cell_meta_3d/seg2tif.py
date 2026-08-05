from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
)
from pathlib import Path

import h5py
import numpy as np
import tifffile
import tqdm
from brainglobe_utils.cells.cells import Cell
from brainglobe_utils.IO.cells import get_cells
from skimage.transform import downscale_local_mean

from cell_meta_3d import __version__


def cell_meta_seg2tif_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-c",
        "--cells-path",
        dest="cells_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--segmentation-path",
        dest="segmentation_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-tif-path",
        dest="output_tif_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--tif-shape",
        dest="tif_shape",
        type=str,
        default="",
        help="",
    )

    return parser


def main(
    cells: list[Cell],
    segmentation_path: Path,
    output_tif_path: Path,
    tif_shape: tuple[int, int, int],
):
    flush_count_freq = 5000
    flush_z_freq = 50
    output_tif_path.parent.mkdir(parents=True, exist_ok=True)
    cells = sorted(cells, key=lambda c: c.z)

    with h5py.File(segmentation_path, "r") as h5_file:
        super_voxel = tuple(v.item() for v in h5_file.attrs["super_voxel"])
        is_super = any(v != 1 for v in super_voxel)

        intensity_index_dset: h5py.Dataset = h5_file[
            "upsampled_cuboid_intensity_index"
        ]
        intensity_dset: h5py.Dataset = h5_file["upsampled_cuboid_intensity"]
        cell_index_range_dset: h5py.Dataset = h5_file["cell_index_range"]
        cell_corner_dset: h5py.Dataset = h5_file["cell_corner_vox"]
        cuboid_voxels_dset: h5py.Dataset = h5_file["upsampled_cuboid_voxels"]

        def frames():
            blank = np.zeros(tif_shape[1:], dtype=intensity_dset.dtype)

            for _ in range(tif_shape[0]):
                yield blank

        tifffile.imwrite(
            output_tif_path,
            frames(),
            shape=tif_shape,
            dtype=intensity_dset.dtype,
            metadata={"axes": "ZYX"},
            bigtiff=True,
        )

        arr = tifffile.memmap(output_tif_path, mode="r+")
        last_z = 0
        last_flush = 0
        for i, cell in tqdm.tqdm(
            enumerate(cells), total=len(cells), unit="cell"
        ):
            data_index = cell.metadata["seg_id"]

            intensity_s, intensity_e = cell_index_range_dset[data_index, :]
            corner = cell_corner_dset[data_index, :]
            intensity_index = np.asarray(
                intensity_index_dset[intensity_s:intensity_e, :], dtype=np.intp
            )
            intensity = np.asarray(intensity_dset[intensity_s:intensity_e])

            if is_super:
                cuboid_voxels = cuboid_voxels_dset[i, :]
                down_sampled_cuboid_voxels = cuboid_voxels // super_voxel
                block = np.zeros(cuboid_voxels, dtype=intensity_dset.dtype)
                block[
                    intensity_index[:, 0],
                    intensity_index[:, 1],
                    intensity_index[:, 2],
                ] = intensity

                block_downsampled = downscale_local_mean(block, super_voxel)

                assert np.all(
                    np.equal(
                        block_downsampled.shape, cuboid_voxels / super_voxel
                    )
                )
                zcr, ycr, xcr = corner
                zn, yn, xn = down_sampled_cuboid_voxels
                arr[zcr : zcr + zn, ycr : ycr + yn, xcr : xcr + xn] = (
                    block_downsampled
                )
            else:
                intensity_index += corner[None, :]
                arr[
                    intensity_index[:, 0],
                    intensity_index[:, 1],
                    intensity_index[:, 2],
                ] = intensity

            if (
                cell.z - last_z > flush_z_freq
                and i - last_flush > flush_count_freq
            ):
                arr.flush()
                last_z = cell.z
                last_flush = i


def run_main():
    args = cell_meta_seg2tif_parser().parse_args()

    cells = get_cells(args.cells_path, cells_only=True)
    segmentation_path = Path(args.segmentation_path)
    output_tif_path = Path(args.output_tif_path)

    if args.tif_shape:
        z, y, x = map(int, args.tif_shape.split(","))
    else:
        with h5py.File(segmentation_path, "r") as h5_file:
            zs, ys, xs = [v.item() for v in h5_file.attrs["cube_voxels"]]

        z = y = x = 0
        for cell in cells:
            z = max(z, cell.z)
            y = max(y, cell.y)
            x = max(x, cell.x)

        z += zs
        y += ys
        x += xs

    main(
        cells=cells,
        segmentation_path=segmentation_path,
        output_tif_path=output_tif_path,
        tif_shape=(z, y, x),
    )


if __name__ == "__main__":
    run_main()
