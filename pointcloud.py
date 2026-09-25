"""Shape volumes to the point cloud CowTalk actually trains on.

The released training set is not a mesh and not the older surface-point export
(``*_surface_points.npz`` with signed distances). The model reads a point cloud
built from a pair of label volumes:

* ground-truth segmentation
* erroneous segmentation of the same scan

The crop, resize, and halo below were recovered by matching
``preprocessed_volume_pair_v2`` (the files the current trainer loaded) back to
the NIfTI label volumes. Sampling ratios are what ``main.py`` requests and what
``dataset.py`` implemented, including the one-quarter halo split (the comment
in the old loader said "half", but the code used ``fg_count // 4``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, zoom

POINTCLOUD_FORMAT = "cowtalk-pointcloud-v1"
RESIZE_SHAPE = (128, 128, 128)
NUM_CLASSES = 14  # background + labels 1..13
MARGIN_RATIO = 0.05
HALO_ITERATIONS = 2


def subject_id_from_name(path: Path) -> str:
    """``topcow_ct_001_0001`` -> ``topcow_ct_001``."""
    stem = path.name
    for suffix in (".nii.gz", ".nii", ".npz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected file name: {path.name}")
    return "_".join(parts[:3])


def gt_stem_for_error(error_path: Path) -> str:
    """Drop the instance suffix: ``topcow_ct_001_0001`` -> ``topcow_ct_001``."""
    return subject_id_from_name(error_path)


def crop_box_from_error(
    error_volume: np.ndarray,
    margin_ratio: float = MARGIN_RATIO,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bounding box used to crop both the error volume and the ground truth.

    The box is the error-foreground extent, padded by ``round(margin_ratio * full_shape)``
    voxels on each axis. The stored end index is exclusive and equals
    ``fg_max_inclusive + margin`` (clipped to the volume size), so the included
    padding on the far side is one voxel thinner than ``margin`` whenever the
    box does not hit the volume border. That matches the ``bbox`` arrays saved
    with the training volumes.
    """
    if error_volume.ndim != 3:
        raise ValueError("error volume must be 3D")
    coords = np.argwhere(error_volume > 0)
    if coords.size == 0:
        raise ValueError("error volume has no foreground")
    shape = np.asarray(error_volume.shape, dtype=np.int64)
    margin = np.rint(shape.astype(np.float64) * margin_ratio).astype(np.int64)
    start = np.maximum(coords.min(axis=0) - margin, 0)
    end = np.minimum(coords.max(axis=0) + margin, shape)
    if np.any(end <= start):
        raise ValueError("crop box is empty")
    return start.astype(np.int64), end.astype(np.int64)


def crop_volume(volume: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return volume[int(start[0]) : int(end[0]), int(start[1]) : int(end[1]), int(start[2]) : int(end[2])]


def load_medical_image(medical_images_dir: Path, npz_name: str) -> np.ndarray:
    """Load the 128³ scan saved next to a case. Key is ``medical_image``."""
    path = Path(medical_images_dir) / npz_name
    if not path.exists():
        raise FileNotFoundError(f"Medical image file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "medical_image" not in data:
            raise KeyError(f"{path.name} has no 'medical_image' array")
        image = np.asarray(data["medical_image"], dtype=np.float32)
    if tuple(image.shape) != RESIZE_SHAPE:
        raise ValueError(f"{path.name} medical image shape {image.shape}, expected {RESIZE_SHAPE}")
    return image


def stack_cnn_input(segmentation: np.ndarray, medical_image: Optional[np.ndarray] = None) -> np.ndarray:
    """CNN volume: ``[1, 128, 128, 128]`` shape only, or ``[2, ...]`` with the scan on channel 1."""
    seg = np.asarray(segmentation, dtype=np.float32)
    if seg.ndim == 4:
        seg = seg[0]
    channels = [seg[None]]
    if medical_image is not None:
        med = np.asarray(medical_image, dtype=np.float32)
        if med.ndim == 4:
            med = med[0]
        channels.append(med[None])
    return np.ascontiguousarray(np.concatenate(channels, axis=0))


def resize_labels(volume: np.ndarray, size: Sequence[int] = RESIZE_SHAPE) -> np.ndarray:
    """Nearest-neighbor resize. Matches ``scipy.ndimage.zoom(..., order=0)``."""
    factors = [float(size[i]) / float(volume.shape[i]) for i in range(3)]
    resized = zoom(volume, factors, order=0)
    if tuple(resized.shape) != tuple(size):
        raise RuntimeError(f"resize produced {resized.shape}, expected {tuple(size)}")
    return resized.astype(np.float32, copy=False)


def foreground_halo(gt_cropped: np.ndarray, error_cropped: np.ndarray, iterations: int = HALO_ITERATIONS) -> np.ndarray:
    """Two-voxel 6-connected shell around the union of the two label volumes.

    This is ``dilated_region`` in the training files: dilate the union twice
    with face connectivity, then drop the original foreground. It does not
    overlap the union. ``modified_cropped_dilated`` in those files is the same
    dilation before the foreground is removed; the loader never reads that array.
    """
    union = (gt_cropped != 0) | (error_cropped != 0)
    structure = generate_binary_structure(3, 1)
    dilated = binary_dilation(union, structure=structure, iterations=iterations)
    return (dilated & ~union).astype(np.uint8)


def remap_label_15(labels: np.ndarray) -> np.ndarray:
    """TopCoW label 15 is treated as class 13. Applied to ground truth only."""
    if not np.any(labels == 15):
        return labels
    remapped = np.array(labels, copy=True)
    remapped[labels == 15] = 13
    return remapped


def build_pointcloud(
    gt_volume: np.ndarray,
    error_volume: np.ndarray,
    margin_ratio: float = MARGIN_RATIO,
) -> Dict[str, np.ndarray]:
    """Convert one GT/error label pair into the point cloud the model loads.

    Foreground points are every voxel where either label is non-zero, inside
    the crop, with both the ground-truth label and the error label kept so
    training can still mix error classes. Halo points are the near-surface
    shell used for part of the query cloud. ``modified_resized`` is the 128^3
    nearest-neighbor error segmentation consumed by the CNN branch.
    """
    if gt_volume.shape != error_volume.shape:
        raise ValueError(f"shape mismatch: gt {gt_volume.shape} vs error {error_volume.shape}")
    start, end = crop_box_from_error(error_volume, margin_ratio=margin_ratio)
    gt_cropped = remap_label_15(crop_volume(gt_volume, start, end))
    error_cropped = crop_volume(error_volume, start, end)
    modified_resized = resize_labels(error_cropped)
    halo = foreground_halo(gt_cropped, error_cropped)

    union = (gt_cropped != 0) | (error_cropped != 0)
    fg_coords = np.argwhere(union).astype(np.int32, copy=False)
    if len(fg_coords) == 0:
        raise ValueError("cropped pair has no foreground")
    fg_gt = gt_cropped[fg_coords[:, 0], fg_coords[:, 1], fg_coords[:, 2]].astype(np.int16, copy=False)
    fg_error = error_cropped[fg_coords[:, 0], fg_coords[:, 1], fg_coords[:, 2]].astype(np.int16, copy=False)
    halo_coords = np.argwhere(halo > 0).astype(np.int32, copy=False)
    bbox = np.array(
        [start[0], end[0], start[1], end[1], start[2], end[2]],
        dtype=np.int32,
    )
    return {
        "fg_coords": np.ascontiguousarray(fg_coords),
        "fg_gt_labels": np.ascontiguousarray(fg_gt),
        "fg_error_labels": np.ascontiguousarray(fg_error),
        "halo_coords": np.ascontiguousarray(halo_coords),
        "volume_shape": np.asarray(gt_cropped.shape, dtype=np.int32),
        "bbox": bbox,
        "modified_resized": np.ascontiguousarray(modified_resized),
        "format": np.array(POINTCLOUD_FORMAT),
    }


def save_pointcloud(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_pointcloud(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.keys())
        required = {"fg_coords", "fg_gt_labels", "fg_error_labels", "halo_coords", "volume_shape", "modified_resized"}
        if not required.issubset(keys):
            raise KeyError(
                f"{path.name} is not a {POINTCLOUD_FORMAT} file (missing {sorted(required - keys)}). "
                "Run preprocess.py on the label volumes instead of loading the raw shapes."
            )
        loaded = {key: data[key] for key in data.keys()}
    return loaded


def scatter_labels(shape: Sequence[int], coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    volume = np.zeros(tuple(int(v) for v in shape), dtype=np.int64)
    if len(coords) == 0:
        return volume
    volume[coords[:, 0], coords[:, 1], coords[:, 2]] = labels
    return volume


def mix_error_labels(
    gt_labels: np.ndarray,
    error_labels: np.ndarray,
    remaining_error_classes: Set[int],
    manipulated_error_classes: Set[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Same class mixing the trainer applied to the dense crops.

    ``current`` keeps ground truth except for remaining error classes, which are
    replaced by the erroneous labels. ``input`` then also paints the manipulated
    classes (the errors the text instruction is allowed to mention).
    """
    current = gt_labels.copy()
    if remaining_error_classes:
        remaining = np.fromiter(remaining_error_classes, dtype=np.int64)
        current[np.isin(current, remaining)] = 0
        for class_id in remaining_error_classes:
            current[error_labels == class_id] = class_id
    inputs = current.copy()
    if manipulated_error_classes:
        manipulated = np.fromiter(manipulated_error_classes, dtype=np.int64)
        inputs[np.isin(current, manipulated)] = 0
        for class_id in manipulated_error_classes:
            inputs[error_labels == class_id] = class_id
    return current, inputs


def normalize_coordinates(coords: np.ndarray, volume_shape: Sequence[int]) -> np.ndarray:
    """Map crop indices to [-1, 1] with one scale for all axes (aspect preserved)."""
    shape = np.asarray(volume_shape, dtype=np.float32)
    mins = np.zeros(3, dtype=np.float32)
    maxs = shape - 1.0
    spans = np.clip(maxs - mins, 1e-6, None)
    max_span = np.max(spans)
    center = (mins + maxs) / 2.0
    normalized = (coords.astype(np.float32, copy=False) - center) / (max_span / 2.0)
    return normalized.astype(np.float32, copy=False)


def resize_query_coordinates(coords: np.ndarray, volume_shape: Sequence[int]) -> np.ndarray:
    """Per-axis [-1, 1] coordinates in (x, y, z) for 3D ``grid_sample``."""
    native = np.asarray(volume_shape, dtype=np.float32)
    coords_f = coords.astype(np.float32, copy=False)
    norm = (coords_f / np.maximum(native - 1.0, 1.0)) * 2.0 - 1.0
    return np.flip(norm, axis=-1).astype(np.float32, copy=False)


def sample_points_from_volume(
    volume: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
    fg_ratio: float = 0.0,
    random_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample foreground (``volume > 0``) and background (``volume == 0``) voxels."""
    total_ratio = fg_ratio + random_ratio
    if total_ratio > 0:
        fg_ratio /= total_ratio
        random_ratio /= total_ratio

    z_max, y_max, x_max = volume.shape
    all_coords = []
    all_labels = []

    if fg_ratio > 0 and num_points > 0:
        num_fg = int(num_points * fg_ratio)
        if num_fg > 0:
            fg_coords = np.argwhere(volume > 0)
            if len(fg_coords) > 0:
                if len(fg_coords) <= num_fg:
                    selected_fg = fg_coords
                else:
                    indices = rng.choice(len(fg_coords), size=num_fg, replace=False)
                    selected_fg = fg_coords[indices]
                all_coords.append(selected_fg)
                all_labels.append(volume[selected_fg[:, 0], selected_fg[:, 1], selected_fg[:, 2]])

    remaining = num_points - sum(len(coords) for coords in all_coords) if all_coords else num_points
    if random_ratio > 0 and remaining > 0:
        num_random = int(num_points * random_ratio)
        if remaining < num_random:
            num_random = remaining
        if num_random > 0:
            bg_coords = np.argwhere(volume == 0)
            if len(bg_coords) > 0:
                if len(bg_coords) <= num_random:
                    selected_bg = bg_coords
                else:
                    indices = rng.choice(len(bg_coords), size=num_random, replace=False)
                    selected_bg = bg_coords[indices]
                all_coords.append(selected_bg)
                all_labels.append(volume[selected_bg[:, 0], selected_bg[:, 1], selected_bg[:, 2]])

    if not all_coords:
        z_coords = rng.integers(0, z_max, size=num_points)
        y_coords = rng.integers(0, y_max, size=num_points)
        x_coords = rng.integers(0, x_max, size=num_points)
        coords = np.stack([z_coords, y_coords, x_coords], axis=1)
        labels = volume[z_coords, y_coords, x_coords]
        return coords, labels

    coords = np.concatenate(all_coords, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if len(coords) < num_points:
        num_needed = num_points - len(coords)
        z_coords = rng.integers(0, z_max, size=num_needed)
        y_coords = rng.integers(0, y_max, size=num_needed)
        x_coords = rng.integers(0, x_max, size=num_needed)
        extra = np.stack([z_coords, y_coords, x_coords], axis=1)
        coords = np.concatenate([coords, extra], axis=0)
        labels = np.concatenate([labels, volume[z_coords, y_coords, x_coords]], axis=0)

    perm = rng.permutation(len(coords))
    return coords[perm], labels[perm]


def sample_query_coordinates(
    current_volume: np.ndarray,
    gt_cropped: np.ndarray,
    error_cropped: np.ndarray,
    halo_volume: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
    fg_ratio: float,
    random_ratio: float,
) -> np.ndarray:
    """Query coordinates used in training.

    Of the foreground quota, one quarter is drawn from the halo and the rest
    from the union of the ground-truth and error shapes. The remaining quota is
    background in ``current_volume``.
    """
    fg_count = int(num_points * fg_ratio)
    rand_count = max(num_points - fg_count, 0)
    parts = []

    halo_count = fg_count // 4
    if halo_count > 0:
        halo_coords, _ = sample_points_from_volume(halo_volume, halo_count, rng, fg_ratio=1.0, random_ratio=0.0)
        parts.append(halo_coords)

    union_count = fg_count - halo_count
    if union_count > 0:
        union = ((gt_cropped != 0) | (error_cropped != 0)).astype(np.int64)
        union_coords, _ = sample_points_from_volume(union, union_count, rng, fg_ratio=1.0, random_ratio=0.0)
        parts.append(union_coords)

    if rand_count > 0:
        bg_coords = np.argwhere(current_volume == 0)
        if len(bg_coords) > 0:
            take = min(rand_count, len(bg_coords))
            choice = rng.choice(len(bg_coords), size=take, replace=False)
            parts.append(bg_coords[choice])

    if not parts:
        return np.empty((0, 3), dtype=np.int32)
    return np.concatenate(parts, axis=0)


def iter_error_volumes(error_dir: Path) -> Iterable[Path]:
    patterns = ("*.nii.gz", "*.nii")
    files = []
    for pattern in patterns:
        files.extend(error_dir.glob(pattern))
    return sorted(files)


def find_gt_volume(gt_dir: Path, error_path: Path) -> Optional[Path]:
    stem = gt_stem_for_error(error_path)
    for suffix in (".nii.gz", ".nii"):
        candidate = gt_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None
