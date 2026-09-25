"""Load preprocessed CowTalk point clouds and draw the training subsets.

The files on disk are point clouds produced by ``preprocess.py``, not the raw
label volumes. Each item still applies the training-time error-class dropout
and the foreground / background / halo sampling ratios on that point cloud.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from pointcloud import (
    load_medical_image,
    load_pointcloud,
    mix_error_labels,
    normalize_coordinates,
    resize_query_coordinates,
    sample_points_from_volume,
    sample_query_coordinates,
    scatter_labels,
    stack_cnn_input,
    subject_id_from_name,
)


def _sample_points_from_volume(volume, num_points, rng, fg_ratio=0.0, random_ratio=1.0):
    """Compatibility wrapper for inference scripts."""
    return sample_points_from_volume(volume, num_points, rng, fg_ratio=fg_ratio, random_ratio=random_ratio)


def _normalize_coordinates(coords: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Compatibility wrapper. ``bbox`` is ``[[mins], [maxs]]`` in crop index space."""
    volume_shape = np.asarray(bbox[1], dtype=np.float32) - np.asarray(bbox[0], dtype=np.float32) + 1.0
    return normalize_coordinates(coords - np.asarray(bbox[0], dtype=np.float32), volume_shape)

NO_ERROR_PHRASES = [
    "No errors detected in this segmentation.",
    "The shape prediction is correct and requires no correction.",
    "This segmentation is accurate and needs no modifications.",
    "All predictions are correct with no errors to fix.",
    "The segmentation is error-free.",
    "There are no mistakes present in this segmentation result.",
    "Segmentation is perfect; no errors require correction.",
    "All regions have been labeled correctly.",
    "No further edits or changes are necessary.",
    "There are no erroneous regions in this prediction.",
    "Everything appears correct; nothing to fix.",
    "All anatomical structures are segmented correctly.",
    "No missing or mislabeled segments detected.",
    "Segmentation is flawless; no discrepancies observed.",
    "Everything is as expected—no corrections needed.",
    "The output is correct and needs no update.",
    "No segmentation errors were found.",
    "All regions match the reference annotation.",
    "No differences from ground truth—segmentation is accurate.",
    "There are no visible inaccuracies in the segmentation.",
    "No errors present; you can proceed.",
    "Full accuracy achieved in this segmentation; no action required.",
]


def load_split_subjects(split_path: Path) -> Dict[str, List[str]]:
    with open(split_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    subjects = payload.get("subjects")
    if not isinstance(subjects, dict):
        raise ValueError(f"Split file {split_path} missing 'subjects' dictionary")
    for subset in ("train", "val", "test"):
        if subset not in subjects:
            raise ValueError(f"Split file {split_path} missing subset '{subset}'")
    return subjects


def index_pointclouds(root_dir: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(root_dir.glob("*.npz")):
        if path.name.endswith("_surface_points.npz"):
            continue
        mapping[subject_id_from_name(path)].append(path)
    if not mapping:
        raise RuntimeError(f"No point-cloud NPZ files found in {root_dir}")
    return mapping


def _validate_ratios(fg: float, background: float, name: str) -> Tuple[float, float]:
    ratios = [max(0.0, min(1.0, value)) for value in (fg, background)]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"{name} ratios must sum to 1.0")
    return ratios[0], ratios[1]


class CowPointCloudDataset(Dataset):
    """Input points carry a label feature. Query points are supervised by the remaining errors."""

    def __init__(
        self,
        root_dir: str,
        split_subjects: Sequence[str],
        input_points: int,
        seed: int = 0,
        input_fg_ratio: float = 0.7,
        input_random_ratio: float = 0.3,
        query_points: int = None,
        query_fg_ratio: float = 0.9,
        query_random_ratio: float = 0.1,
        use_gt_labels: bool = True,
        instructions_dir: str = None,
        inference_mode: bool = False,
        use_class_hint: bool = False,
        medical_images_dir: str = None,
    ) -> None:
        # None: multi-class shape only. A directory: also feed the aligned 128³ scan.
        self.use_image = medical_images_dir is not None
        self.medical_images_dir = Path(medical_images_dir) if self.use_image else None
        root_path = Path(root_dir)
        subject_to_files = index_pointclouds(root_path)
        selected: List[Path] = []
        for subject in split_subjects:
            files = subject_to_files.get(subject)
            if not files:
                raise ValueError(f"No point clouds found for subject {subject} in {root_dir}")
            selected.extend(files)
        if not selected:
            raise RuntimeError("Dataset split did not yield any files.")

        self.files = selected
        self.input_points = input_points
        self.query_points = query_points if query_points is not None else input_points
        self.seed = seed
        self.use_class_hint = use_class_hint
        self.use_gt_labels = use_gt_labels
        self.inference_mode = inference_mode
        self._rng = np.random.default_rng(seed)
        self._worker_rngs: Dict[int, np.random.Generator] = {}
        self.input_ratios = _validate_ratios(input_fg_ratio, input_random_ratio, "input")
        self.query_ratios = _validate_ratios(query_fg_ratio, query_random_ratio, "query")
        if instructions_dir is None:
            instructions_dir = root_path.parent / "instructions_generated"
        self.instructions_dir = Path(instructions_dir)

    def __len__(self) -> int:
        return len(self.files)

    def _get_rng(self) -> np.random.Generator:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self._rng
        if worker_info.id not in self._worker_rngs:
            self._worker_rngs[worker_info.id] = np.random.default_rng(self.seed + worker_info.id)
        return self._worker_rngs[worker_info.id]

    def _instruction_record(self, pointcloud_path: Path) -> dict:
        instruction_file = self.instructions_dir / f"{pointcloud_path.stem}.json"
        if not instruction_file.exists():
            return {}
        try:
            with open(instruction_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _phrase_for_class(self, record: dict, class_id: int) -> Optional[str]:
        # The previous loader's coin flip used ``random.uniform(0, 1) > 1``, which
        # never selected the long text. Short instructions are the version that ran.
        info = record.get(str(class_id))
        if not isinstance(info, dict):
            return None
        text = info.get("short") or ""
        lowered = text.lower()
        if not text.strip() or "n/a" in lowered or "no error" in lowered:
            return None
        return text

    def _error_classes(self, record: dict) -> Set[int]:
        classes = set()
        for class_id in range(1, 14):
            if self._phrase_for_class(record, class_id):
                classes.add(class_id)
        return classes

    def _load_text(self, record: dict, classes_to_include: Optional[Set[int]], rng: np.random.Generator) -> str:
        texts = []
        for class_id in range(1, 14):
            if classes_to_include is not None and class_id not in classes_to_include:
                continue
            phrase = self._phrase_for_class(record, class_id)
            if phrase:
                texts.append(phrase)
        rng.shuffle(texts)
        return " ".join(texts)

    def _volumes_from_cloud(self, cloud: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = cloud["volume_shape"]
        gt = scatter_labels(shape, cloud["fg_coords"], cloud["fg_gt_labels"])
        error = scatter_labels(shape, cloud["fg_coords"], cloud["fg_error_labels"])
        halo = scatter_labels(shape, cloud["halo_coords"], np.ones(len(cloud["halo_coords"]), dtype=np.int64))
        resized = cloud["modified_resized"].astype(np.float32, copy=False)
        return gt, error, halo, resized

    def __getitem__(self, index: int):
        rng = self._get_rng()
        path = self.files[index]
        cloud = load_pointcloud(path)
        gt_cropped, error_cropped, halo_volume, modified_resized = self._volumes_from_cloud(cloud)
        volume_shape = np.asarray(gt_cropped.shape, dtype=np.float32)
        record = self._instruction_record(path)

        use_fully_corrected = rng.random() < 0.05
        if use_fully_corrected:
            current_volume = gt_cropped.copy()
            input_volume = gt_cropped.copy()
            text_instruction = rng.choice(NO_ERROR_PHRASES)
            input_error_classes: Set[int] = set()
            remaining_error_classes: Set[int] = set()
            class_hint = np.zeros(13, dtype=np.float32) if self.use_class_hint else None
        else:
            initial_error_classes = self._error_classes(record)
            input_error_classes = {class_id for class_id in initial_error_classes if rng.random() >= 0.20}
            remaining_error_classes = {class_id for class_id in input_error_classes if rng.random() >= 0.50}
            manipulated_error_classes = input_error_classes - remaining_error_classes
            if self.use_class_hint:
                class_hint = np.zeros(13, dtype=np.float32)
                for class_id in input_error_classes:
                    if 1 <= class_id <= 13:
                        class_hint[class_id - 1] = 1.0
            else:
                class_hint = None

            # Mix on the stored foreground points, then scatter. Same result as
            # editing the dense crops, without keeping the raw shapes around.
            current_fg, input_fg = mix_error_labels(
                cloud["fg_gt_labels"].astype(np.int64, copy=False),
                cloud["fg_error_labels"].astype(np.int64, copy=False),
                remaining_error_classes,
                manipulated_error_classes,
            )
            current_volume = scatter_labels(cloud["volume_shape"], cloud["fg_coords"], current_fg)
            input_volume = scatter_labels(cloud["volume_shape"], cloud["fg_coords"], input_fg)
            text_instruction = self._load_text(record, manipulated_error_classes, rng)
            if not text_instruction.strip():
                text_instruction = rng.choice(NO_ERROR_PHRASES)

        input_coords, input_labels = sample_points_from_volume(
            input_volume,
            self.input_points,
            rng,
            fg_ratio=self.input_ratios[0],
            random_ratio=self.input_ratios[1],
        )

        query_coords_original = None
        if self.inference_mode:
            z_coords, y_coords, x_coords = np.meshgrid(
                np.arange(volume_shape[0], dtype=np.float32),
                np.arange(volume_shape[1], dtype=np.float32),
                np.arange(volume_shape[2], dtype=np.float32),
                indexing="ij",
            )
            query_coords = np.stack([z_coords.ravel(), y_coords.ravel(), x_coords.ravel()], axis=1)
            query_coords_original = query_coords.copy()
        else:
            query_coords = sample_query_coordinates(
                current_volume,
                gt_cropped,
                error_cropped,
                halo_volume,
                self.query_points,
                rng,
                fg_ratio=self.query_ratios[0],
                random_ratio=self.query_ratios[1],
            )

        query_idx = query_coords.astype(np.int64, copy=False)
        query_labels = current_volume[query_idx[:, 0], query_idx[:, 1], query_idx[:, 2]]
        query_labels_input = input_volume[query_idx[:, 0], query_idx[:, 1], query_idx[:, 2]]

        input_coords_norm = normalize_coordinates(input_coords, volume_shape)
        query_coords_norm = normalize_coordinates(query_coords, volume_shape)
        query_coords_resized = resize_query_coordinates(query_coords, volume_shape)
        input_features = np.concatenate(
            [input_coords_norm, input_labels.astype(np.float32).reshape(-1, 1)],
            axis=1,
        )

        result = {
            "points": torch.from_numpy(np.ascontiguousarray(input_features, dtype=np.float32)),
            "labels": torch.from_numpy(np.ascontiguousarray(input_labels)),
            "query_points": torch.from_numpy(np.ascontiguousarray(query_coords_norm, dtype=np.float32)),
            "query_labels": torch.from_numpy(np.ascontiguousarray(query_labels)),
            "query_labels_input": torch.from_numpy(np.ascontiguousarray(query_labels_input.astype(np.float32))),
            "text": text_instruction,
            "file_name": str(path),
        }
        if class_hint is not None:
            result["class_hint"] = torch.from_numpy(class_hint)

        medical = None
        if self.use_image:
            medical = load_medical_image(self.medical_images_dir, path.name)
        cnn_volume = stack_cnn_input(modified_resized, medical)
        result["modified_resized"] = torch.from_numpy(cnn_volume)
        result["query_coords_resized"] = torch.from_numpy(np.ascontiguousarray(query_coords_resized, dtype=np.float32))

        if self.inference_mode:
            result["native_volume_shape"] = torch.from_numpy(volume_shape.astype(np.int64))
            result["current_volume"] = torch.from_numpy(current_volume.astype(np.int64))
            result["input_volume"] = torch.from_numpy(input_volume.astype(np.int64))
            result["gt_volume"] = torch.from_numpy(gt_cropped.astype(np.int64))
            bbox = cloud.get("bbox")
            if bbox is None:
                bbox = np.array([0, volume_shape[0], 0, volume_shape[1], 0, volume_shape[2]], dtype=np.float32)
            result["bbox"] = torch.from_numpy(np.asarray(bbox, dtype=np.float32))
            if query_coords_original is not None:
                result["query_coords_original"] = torch.from_numpy(
                    np.ascontiguousarray(query_coords_original, dtype=np.float32)
                )
            result["input_error_classes"] = torch.from_numpy(np.array(sorted(input_error_classes), dtype=np.int64))
            result["remaining_error_classes"] = torch.from_numpy(
                np.array(sorted(remaining_error_classes), dtype=np.int64)
            )
        return result


def build_datasets(
    root_dir: str,
    split_path: Path,
    input_points: int,
    seed: int = 0,
    input_fg_ratio: float = 0.7,
    input_random_ratio: float = 0.3,
    query_points: int = None,
    query_fg_ratio: float = 0.9,
    query_random_ratio: float = 0.1,
    use_gt_labels: bool = False,
    instructions_dir: str = None,
    medical_images_dir: str = None,
    use_class_hint: bool = False,
):
    subjects = load_split_subjects(split_path)
    kwargs = {
        "root_dir": root_dir,
        "input_points": input_points,
        "input_fg_ratio": input_fg_ratio,
        "input_random_ratio": input_random_ratio,
        "query_points": query_points,
        "query_fg_ratio": query_fg_ratio,
        "query_random_ratio": query_random_ratio,
        "use_gt_labels": use_gt_labels,
        "instructions_dir": instructions_dir,
        "medical_images_dir": medical_images_dir,
        "use_class_hint": use_class_hint,
    }
    return tuple(
        CowPointCloudDataset(split_subjects=subjects[split], seed=seed + i, **kwargs)
        for i, split in enumerate(["train", "val", "test"])
    )
