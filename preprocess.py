#!/usr/bin/env python3
"""Build CowTalk point clouds from ground-truth and erroneous label volumes.

Example:
    python preprocess.py \\
        --error-dir data/error_volumes \\
        --gt-dir data/labels \\
        --output-dir data/pointclouds

Each error file ``topcow_ct_001_0001.nii.gz`` is paired with
``topcow_ct_001.nii.gz`` in the ground-truth directory. The written NPZ is the
point cloud training and inference load. The original volumes are not read again
after this step.

The crop, 128^3 resize, and halo are the recipe recovered from the last training
cache (``preprocessed_volume_pair_v2``). See ``pointcloud.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from pointcloud import build_pointcloud, find_gt_volume, iter_error_volumes, save_pointcloud


def _load_nifti(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert label volumes into CowTalk point clouds.")
    parser.add_argument("--error-dir", type=Path, required=True, help="Directory of erroneous label NIfTI files.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory of ground-truth label NIfTI files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write point-cloud NPZ files.")
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.05,
        help="Fractional padding of the full volume size around the error foreground (default: 0.05).",
    )
    parser.add_argument("--limit", type=int, default=0, help="If > 0, only process this many volumes.")
    args = parser.parse_args()

    if not args.error_dir.is_dir():
        raise SystemExit(f"Error directory not found: {args.error_dir}")
    if not args.gt_dir.is_dir():
        raise SystemExit(f"Ground-truth directory not found: {args.gt_dir}")

    error_paths = list(iter_error_volumes(args.error_dir))
    if args.limit > 0:
        error_paths = error_paths[: args.limit]
    if not error_paths:
        raise SystemExit(f"No NIfTI files found in {args.error_dir}")

    written = 0
    skipped = 0
    for error_path in error_paths:
        gt_path = find_gt_volume(args.gt_dir, error_path)
        if gt_path is None:
            print(f"[skip] {error_path.name}: no ground-truth volume")
            skipped += 1
            continue
        error_volume = _load_nifti(error_path)
        gt_volume = _load_nifti(gt_path)
        if not np.any(error_volume > 0):
            print(f"[skip] {error_path.name}: empty error foreground")
            skipped += 1
            continue
        payload = build_pointcloud(gt_volume, error_volume, margin_ratio=args.margin_ratio)
        stem = error_path.name
        for suffix in (".nii.gz", ".nii"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        out_path = args.output_dir / f"{stem}.npz"
        save_pointcloud(out_path, payload)
        written += 1
        print(
            f"[ok] {out_path.name}  fg={len(payload['fg_coords'])}  "
            f"halo={len(payload['halo_coords'])}  crop={tuple(payload['volume_shape'])}"
        )

    print(f"Wrote {written} point clouds to {args.output_dir} ({skipped} skipped).")


if __name__ == "__main__":
    main()
