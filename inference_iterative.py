"""
Iterative inference script for volume-based point cloud transformer.

This script performs iterative inference with pass 4 only for each test sample:
- Pass 4: Input error classes randomly divided into 4 sets, corrected iteratively (4 iterations)

For each sample:
- Saves input volume and pass 4 output volume as NIfTI files
- Calculates macro-averaged Dice, F1, Chamfer distance, and normalized surface dice (input vs gt, pass 4 vs gt)
- Saves results to CSV file
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

import dataset
import models


def compute_macro_avg_dice(pred_volume: np.ndarray, target_volume: np.ndarray, num_classes: int) -> float:
    """
    Compute macro-average Dice score between predicted and target volumes.
    Only considers classes present in the target volume.
    
    Args:
        pred_volume: Predicted volume [Z, Y, X]
        target_volume: Target volume [Z, Y, X]
        num_classes: Number of classes
        
    Returns:
        Macro-average Dice score
    """
    pred_flat = pred_volume.flatten()
    target_flat = target_volume.flatten()
    
    # Find classes present in target
    present_classes = np.unique(target_flat)
    
    dice_scores = []
    for c in present_classes:
        if c < 0 or c >= num_classes:
            continue
        pred_c = (pred_flat == c).astype(np.float32)
        target_c = (target_flat == c).astype(np.float32)
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = 2.0 * intersection / union
            dice_scores.append(dice)
    
    if len(dice_scores) == 0:
        return 0.0
    
    return float(np.mean(dice_scores))


def compute_macro_avg_f1(pred_volume: np.ndarray, target_volume: np.ndarray, num_classes: int) -> float:
    """
    Compute macro-average F1 score between predicted and target volumes.
    Only considers classes present in the target volume.
    
    Args:
        pred_volume: Predicted volume [Z, Y, X]
        target_volume: Target volume [Z, Y, X]
        num_classes: Number of classes
        
    Returns:
        Macro-average F1 score
    """
    pred_flat = pred_volume.flatten()
    target_flat = target_volume.flatten()
    
    # Find classes present in target
    present_classes = np.unique(target_flat)
    
    f1_scores = []
    for c in present_classes:
        if c < 0 or c >= num_classes:
            continue
        pred_c = (pred_flat == c).astype(np.float32)
        target_c = (target_flat == c).astype(np.float32)
        
        tp = (pred_c * target_c).sum()
        fp = (pred_c * (1 - target_c)).sum()
        fn = ((1 - pred_c) * target_c).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if (precision + recall) > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
    
    if len(f1_scores) == 0:
        return 0.0
    
    return float(np.mean(f1_scores))


def _get_surface_coords(mask_3d: np.ndarray) -> np.ndarray:
    """
    Get coordinates of surface voxels (boundary) of a binary 3D mask.
    Surface = voxels that have at least one 6-neighbor outside the mask.
    
    Args:
        mask_3d: Binary mask [Z, Y, X]
        
    Returns:
        Array of shape [N, 3] with (z, y, x) coordinates.
    """
    eroded = ndimage.binary_erosion(mask_3d.astype(np.uint8), structure=ndimage.generate_binary_structure(3, 1))
    surface = mask_3d.astype(np.uint8) & (~eroded).astype(np.uint8)
    coords = np.argwhere(surface > 0)  # [N, 3] in (z, y, x)
    return coords.astype(np.float64)


def _chamfer_symmetric(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """
    Symmetric Chamfer distance between two point sets in voxel space.
    Chamfer = (1/|A|) sum_{a in A} min_{b in B} ||a-b|| + (1/|B|) sum_{b in B} min_{a in A} ||b-a||
    Lower is better.
    
    Args:
        coords_a: [N_a, 3]
        coords_b: [N_b, 3]
        
    Returns:
        Symmetric Chamfer distance (float). Returns 0.0 if either set is empty.
    """
    if coords_a.size == 0 or coords_b.size == 0:
        return 0.0
    tree_a = cKDTree(coords_a)
    tree_b = cKDTree(coords_b)
    d_a_to_b, _ = tree_b.query(coords_a, k=1)  # [N_a]
    d_b_to_a, _ = tree_a.query(coords_b, k=1)  # [N_b]
    term_a = float(np.mean(d_a_to_b))
    term_b = float(np.mean(d_b_to_a))
    return term_a + term_b


def compute_macro_avg_chamfer(pred_volume: np.ndarray, target_volume: np.ndarray, num_classes: int) -> float:
    """
    Compute macro-average symmetric Chamfer distance (in voxels) between predicted and target volumes.
    Per class: surface of pred vs surface of gt, then Chamfer. Only classes present in target.
    
    Args:
        pred_volume: Predicted volume [Z, Y, X]
        target_volume: Target volume [Z, Y, X]
        num_classes: Number of classes
        
    Returns:
        Macro-average Chamfer distance (lower is better).
    """
    present_classes = np.unique(target_volume.flatten())
    chamfer_scores = []
    for c in present_classes:
        if c < 0 or c >= num_classes:
            continue
        pred_c = (pred_volume == c)
        target_c = (target_volume == c)
        if not np.any(target_c):
            continue
        surf_pred = _get_surface_coords(pred_c)
        surf_gt = _get_surface_coords(target_c)
        ch = _chamfer_symmetric(surf_pred, surf_gt)
        chamfer_scores.append(ch)
    if len(chamfer_scores) == 0:
        return 0.0
    return float(np.mean(chamfer_scores))


def _surface_dice_one_class(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    threshold_voxels: float = 1.0,
) -> float:
    """
    Normalized surface dice for one class: fraction of boundary points within threshold of the other boundary.
    NSD = (|D'_pred_gt| + |D'_gt_pred|) / (|D_pred_gt| + |D_gt_pred|)
    where D' are distances <= threshold. Returns 0 if both boundaries empty; 1 if both empty (no surface).
    
    Args:
        pred_mask: Binary mask [Z, Y, X]
        target_mask: Binary mask [Z, Y, X]
        threshold_voxels: Distance threshold in voxels (default 1.0).
        
    Returns:
        NSD in [0, 1], or np.nan if class not present in either.
    """
    surf_pred = _get_surface_coords(pred_mask)
    surf_gt = _get_surface_coords(target_mask)
    if surf_pred.size == 0 and surf_gt.size == 0:
        return np.nan  # class absent in both
    if surf_pred.size == 0 or surf_gt.size == 0:
        return 0.0
    tree_gt = cKDTree(surf_gt)
    tree_pred = cKDTree(surf_pred)
    d_pred_to_gt, _ = tree_gt.query(surf_pred, k=1)
    d_gt_to_pred, _ = tree_pred.query(surf_gt, k=1)
    boundary_complete = len(d_pred_to_gt) + len(d_gt_to_pred)
    boundary_correct = np.sum(d_pred_to_gt <= threshold_voxels) + np.sum(d_gt_to_pred <= threshold_voxels)
    if boundary_complete == 0:
        return np.nan
    return float(boundary_correct / boundary_complete)


def compute_macro_avg_surface_dice(
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
    num_classes: int,
    threshold_voxels: float = 1.0,
) -> float:
    """
    Compute macro-average normalized surface dice between predicted and target volumes.
    Only considers classes present in the target volume.
    
    Args:
        pred_volume: Predicted volume [Z, Y, X]
        target_volume: Target volume [Z, Y, X]
        num_classes: Number of classes
        threshold_voxels: Distance threshold in voxels for NSD (default 1.0).
        
    Returns:
        Macro-average NSD in [0, 1].
    """
    present_classes = np.unique(target_volume.flatten())
    nsd_scores = []
    for c in present_classes:
        if c < 0 or c >= num_classes:
            continue
        pred_c = (pred_volume == c)
        target_c = (target_volume == c)
        if not np.any(target_c):
            continue
        nsd = _surface_dice_one_class(pred_c, target_c, threshold_voxels=threshold_voxels)
        if not np.isnan(nsd):
            nsd_scores.append(nsd)
    if len(nsd_scores) == 0:
        return 0.0
    return float(np.mean(nsd_scores))


def resize_volume_to_128(volume: np.ndarray) -> np.ndarray:
    """
    Resize volume to 128x128x128 using nearest neighbor interpolation.
    
    Args:
        volume: Input volume [Z, Y, X]
        
    Returns:
        Resized volume [128, 128, 128]
    """
    target_size = (128, 128, 128)
    zoom_factors = np.array(target_size) / np.array(volume.shape)
    resized = ndimage.zoom(volume, zoom_factors, order=0, mode='nearest').astype(volume.dtype)
    return resized


def load_text_instructions_for_classes(
    instruction_file: Path,
    classes_to_include: set,
) -> str:
    """
    Load text instructions from JSON file for specified classes.
    
    Args:
        instruction_file: Path to instruction JSON file
        classes_to_include: Set of class IDs (as strings) to include
        
    Returns:
        Concatenated text string, or empty string if no valid text found
    """
    if not instruction_file.exists():
        return ""
    
    try:
        with open(instruction_file, "r", encoding="utf-8") as f:
            instruction_data = json.load(f)
        
        texts = []
        for label_id in range(1, 14):  # Labels 1 to 13
            label_key = str(label_id)
            
            # Skip if this class should not be included
            if classes_to_include is not None and label_key not in classes_to_include:
                continue
            
            if label_key in instruction_data:
                label_info = instruction_data[label_key]
                if isinstance(label_info, dict):
                    text_str = label_info.get("long", None)
                    
                    if text_str:
                        text_lower = text_str.lower()
                        if "n/a" not in text_lower and "no error" not in text_lower:
                            texts.append(text_str)
        
        # Concatenate texts in random order
        if texts:
            random.shuffle(texts)
            return " ".join(texts)
        else:
            return ""
    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"  Warning: Failed to load instruction file {instruction_file}: {e}", flush=True)
        return ""


def get_no_error_text() -> str:
    """Get a random 'no error' phrase."""
    no_error_phrases = [
        "No errors detected in this segmentation.",
        "The shape prediction is correct and requires no correction.",
        "This segmentation is accurate and needs no modifications.",
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
    return random.choice(no_error_phrases)


def reconstruct_volume_from_predictions(
    query_coords: np.ndarray,
    predictions: np.ndarray,
    volume_shape: tuple,
) -> np.ndarray:
    """
    Reconstruct volume from query point predictions.
    
    Args:
        query_coords: Query point coordinates [N, 3] in (z, y, x) format
        predictions: Predicted labels [N]
        volume_shape: Shape of volume (Z, Y, X)
        
    Returns:
        Reconstructed volume [Z, Y, X]
    """
    volume = np.zeros(volume_shape, dtype=np.int64)
    
    # Convert coordinates to integer indices
    coords_int = query_coords.astype(int)
    
    # Map predictions to volume
    for i, (z, y, x) in enumerate(coords_int):
        if 0 <= z < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= x < volume_shape[2]:
            volume[int(z), int(y), int(x)] = predictions[i]
    
    return volume


def save_volume_as_nifti(
    volume: np.ndarray,
    output_path: Path,
    affine: np.ndarray = None,
):
    """
    Save a volume as a NIfTI (.nii.gz) file.
    
    Args:
        volume: 3D volume array [Z, Y, X]
        output_path: Path to save the NIfTI file
        affine: Optional 4x4 affine transformation matrix. If None, uses identity matrix.
    """
    # Convert to int32 (standard for NIfTI label files)
    volume_int32 = volume.astype(np.int32)
    
    # Use identity affine if not provided
    if affine is None:
        affine = np.eye(4)
        
    # Create NIfTI image
    nii_img = nib.Nifti1Image(volume_int32, affine, dtype=np.int32)
    
    # Save the file
    nib.save(nii_img, str(output_path))


def run_inference_on_volume(
    model: torch.nn.Module,
    input_volume: np.ndarray,
    volume_shape: tuple,
    query_coords_original: np.ndarray,
    query_points: torch.Tensor,
    query_coords_resized: torch.Tensor,
    input_points: torch.Tensor,
    text_instruction: str,
    device: torch.device,
    batch_size: int = 10000,
    modified_resized: torch.Tensor = None,
    query_labels_input: torch.Tensor = None,
    class_hint: torch.Tensor = None,
) -> np.ndarray:
    """
    Run inference on a volume using the model.
    
    Args:
        model: PointTransformer model
        input_volume: Input volume [Z, Y, X] (used to extract query_labels_input if not provided)
        volume_shape: Shape of the volume (Z, Y, X)
        query_coords_original: Query coordinates in original space [N, 3] (z, y, x)
        query_points: Query point coordinates (normalized) [N, 3] for point encoding
        query_coords_resized: Query coordinates in resized space [N, 3] (x, y, z) for CNN
        input_points: Input point cloud [M, 4] (coords + label)
        text_instruction: Text instruction string
        device: Device to run on
        batch_size: Batch size for inference
        modified_resized: Resized volume [1, 128, 128, 128] for CNN (if None, created from input_volume)
        query_labels_input: Query point labels from input volume [N] (if None, will be extracted)
        class_hint: Class hint vector [13] (optional)
        
    Returns:
        Predicted volume [Z, Y, X]
    """
    model.eval()
    
    # Extract query_labels_input from input_volume if not provided
    if query_labels_input is None:
        z_indices = query_coords_original[:, 0].astype(int)
        y_indices = query_coords_original[:, 1].astype(int)
        x_indices = query_coords_original[:, 2].astype(int)
        query_labels_input_np = input_volume[z_indices, y_indices, x_indices]  # [N]
        query_labels_input = torch.from_numpy(query_labels_input_np).to(device)
    
    # Create modified_resized from input_volume if not provided
    if modified_resized is None:
        input_resized = resize_volume_to_128(input_volume).astype(np.float32)
        modified_resized = torch.from_numpy(input_resized[None]).to(device)
    
    # Add batch dimensions
    if input_points.dim() == 2:
        input_points = input_points.unsqueeze(0)  # [1, M, 4]
    if query_points.dim() == 2:
        query_points = query_points.unsqueeze(0)  # [1, N, 3]
    if query_coords_resized.dim() == 2:
        query_coords_resized = query_coords_resized.unsqueeze(0)  # [1, N, 3]
    if modified_resized.dim() == 4:
        modified_resized = modified_resized.unsqueeze(0)  # [1, 1, 128, 128, 128]
    if query_labels_input.dim() == 1:
        query_labels_input = query_labels_input.unsqueeze(0)  # [1, N]
    if class_hint is not None and class_hint.dim() == 1:
        class_hint = class_hint.unsqueeze(0)  # [1, 13]
    
    texts = [text_instruction]
    num_query_points = query_points.shape[1]
    all_predictions = []
    
    # Extract shared features once
    with torch.no_grad():
        shared_features = model.extract_shared_features(
            input_points=input_points,
            texts=texts,
            class_hint=class_hint,
            modified_resized=modified_resized,
        )
        latents = shared_features["latents"]  # [1, num_latents, dim]
        feature_pyramid = shared_features["feature_pyramid"]  # List of 3 feature maps
        
        # Process query points in batches
        for start_idx in range(0, num_query_points, batch_size):
            end_idx = min(start_idx + batch_size, num_query_points)
            
            # Extract batch
            batch_query_points = query_points[:, start_idx:end_idx]  # [1, B, 3]
            batch_query_coords_resized = query_coords_resized[:, start_idx:end_idx]  # [1, B, 3]
            batch_query_labels_input = query_labels_input[:, start_idx:end_idx]  # [1, B]
            
            # Run optimized forward pass
            outputs = model.forward_query_points(
                query_points=batch_query_points,
                query_coords_resized=batch_query_coords_resized,
                latents=latents,
                feature_pyramid=feature_pyramid,
                query_labels_input=batch_query_labels_input,
            )
            
            # Get predictions
            logits = outputs["logits"]  # [1, B, num_classes]
            predictions = torch.argmax(logits, dim=-1)  # [1, B]
            predictions_np = predictions.cpu().numpy()[0]  # [B]
            all_predictions.append(predictions_np)
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)  # [N]
    
    # Reconstruct volume
    pred_volume = reconstruct_volume_from_predictions(
        query_coords=query_coords_original,
        predictions=all_predictions,
        volume_shape=volume_shape,
    )
    
    return pred_volume


def randomly_divide_classes(error_classes: List[int], num_sets: int, rng: np.random.Generator) -> List[List[int]]:
    """
    Randomly divide error classes into num_sets groups.
    
    Args:
        error_classes: List of error class IDs
        num_sets: Number of sets to divide into
        rng: Random number generator
        
    Returns:
        List of num_sets lists, each containing a subset of error_classes
    """
    if len(error_classes) == 0:
        return [[] for _ in range(num_sets)]
    
    # Shuffle error classes
    shuffled = error_classes.copy()
    rng.shuffle(shuffled)
    
    # Divide into num_sets groups
    sets = [[] for _ in range(num_sets)]
    for i, class_id in enumerate(shuffled):
        sets[i % num_sets].append(class_id)
    
    return sets


def main():
    parser = argparse.ArgumentParser(description="Run iterative volume inference on test set.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default="checkpoints/checkpoint_best.pth",
        help="Path to model checkpoint file.",
    )
    parser.add_argument(
        "--pointcloud-data-root",
        type=str,
        required=False,
        default="data/pointclouds",
        help="Root directory for point cloud NPZ files.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        required=False,
        default="data/splits/fold_1.json",
        help="Path to split JSON file.",
    )
    parser.add_argument(
        "--instructions-dir",
        type=str,
        required=False,
        default="data/instructions_generated",
        help="Directory containing text instruction JSON files.",
    )
    parser.add_argument(
        "--medical-images-dir",
        type=str,
        required=False,
        default="data/preprocessed_volume_images",
        help="128³ scan NPZs. Used when the checkpoint name contains _img or --use-image is set.",
    )
    parser.add_argument(
        "--use-image",
        action="store_true",
        help="Force shape+image CNN input. Also enabled when the checkpoint name contains _img.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="results/iterative",
        help="Directory to save output volumes and CSV results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for query point inference (default: 10000).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold number for checkpoint matching (default: 1).",
    )
    parser.add_argument(
        "--input-points",
        type=int,
        default=10000,
        help="Number of input points (for checkpoint matching, default: 10000).",
    )
    parser.add_argument(
        "--query-points",
        type=int,
        default=8000,
        help="Number of query points (for checkpoint matching, default: 8000).",
    )
    parser.add_argument(
        "--input-fg-ratio",
        type=float,
        default=0.7,
        help="Input foreground ratio (for checkpoint matching, default: 0.7).",
    )
    parser.add_argument(
        "--input-random-ratio",
        type=float,
        default=0.3,
        help="Input random ratio (for checkpoint matching, default: 0.3).",
    )
    parser.add_argument(
        "--query-fg-ratio",
        type=float,
        default=0.9,
        help="Query foreground ratio (for checkpoint matching, default: 0.9).",
    )
    parser.add_argument(
        "--query-random-ratio",
        type=float,
        default=0.1,
        help="Query random ratio (for checkpoint matching, default: 0.1).",
    )
    parser.add_argument(
        "--no-cnn",
        action="store_true",
        help="Disable CNN encoder (for checkpoint matching).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}", flush=True)
        return
    
    print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    
    # Detect CNN usage from checkpoint name
    checkpoint_name = checkpoint_path.name
    if "_nocnn" in checkpoint_name:
        use_cnn = False
        print(f"Detected non-CNN checkpoint from name: {checkpoint_name}", flush=True)
    else:
        use_cnn = True
        print(f"Detected CNN checkpoint from name: {checkpoint_name}", flush=True)
    use_image = use_cnn and (args.use_image or "_img" in checkpoint_path.stem)
    print(f"CNN input: {'shape+image' if use_image else 'shape only'}", flush=True)
    
    # Load test set
    print("Loading test set with inference_mode=True...", flush=True)
    from dataset import load_split_subjects, CowPointCloudDataset
    split_file = Path(args.split_file)
    pointcloud_data_root = Path(args.pointcloud_data_root)
    instructions_dir = Path(args.instructions_dir)
    medical_images_dir = Path(args.medical_images_dir)
    
    subjects = load_split_subjects(split_file)
    test_set = CowPointCloudDataset(
        root_dir=str(pointcloud_data_root),
        split_subjects=subjects["test"],
        input_points=args.input_points,
        seed=42,
        input_fg_ratio=args.input_fg_ratio,
        input_random_ratio=args.input_random_ratio,
        query_points=args.query_points,
        query_fg_ratio=args.query_fg_ratio,
        query_random_ratio=args.query_random_ratio,
        use_gt_labels=False,
        instructions_dir=str(instructions_dir),
        medical_images_dir=str(medical_images_dir) if use_image else None,
        inference_mode=True,
    )
    
    print(f"Test set size: {len(test_set)}", flush=True)
    if len(test_set) == 0:
        print("ERROR: Test set is empty! Exiting.", flush=True)
        return
    
    # Build model
    print("Building model...", flush=True)
    sample = test_set[0]
    point_feature_dim = sample["points"].shape[-1]
    model = models.build_point_transformer(
        input_points=args.input_points,
        query_points=args.query_points,
        point_feature_dim=point_feature_dim,
        num_classes=14,
        num_latents=512,
        depth=6,
        dim=512,
        heads=8,
        dim_head=64,
        decoder_ff=True,
        use_cnn=use_cnn,
        use_image=use_image,
    )
    
    # Load checkpoint
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    if "epoch" in checkpoint:
        print(f"  Checkpoint from epoch: {checkpoint['epoch']}", flush=True)
    if "best_acc" in checkpoint:
        print(f"  Best accuracy: {checkpoint['best_acc']:.4f}", flush=True)
    
    model.to(args.device)
    model.eval()
    
    device = torch.device(args.device)
    num_classes = 14
    
    # CSV file for results
    csv_path = output_dir / "iterative_inference_results.csv"
    
    # Check if CSV exists and load existing sample names
    existing_samples = set()
    csv_exists = csv_path.exists()
    if csv_exists:
        print(f"Found existing CSV file: {csv_path}", flush=True)
        print("  Loading existing results to skip already processed samples...", flush=True)
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header
            for row in reader:
                if row:  # Non-empty row
                    existing_samples.add(row[0])  # First column is sample_name
        print(f"  Found {len(existing_samples)} already processed samples", flush=True)
    
    # Open CSV file in append mode if it exists, otherwise create new
    if csv_exists:
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
    else:
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'sample_name',
            'input_vs_gt_dice',
            'input_vs_gt_f1',
            'input_vs_gt_chamfer',
            'input_vs_gt_nsd',
            'pass4_dice',
            'pass4_f1',
            'pass4_chamfer',
            'pass4_nsd',
        ])
    
    # Process each test sample
    print(f"\nRunning iterative inference on {len(test_set)} test samples...", flush=True)
    print(f"Results will be saved to: {output_dir}", flush=True)
    
    rng = np.random.Generator(np.random.PCG64(args.seed))
    
    for idx in range(len(test_set)):
        print(f"\n{'='*60}", flush=True)
        print(f"Processing sample {idx + 1}/{len(test_set)}", flush=True)
        print(f"{'='*60}", flush=True)
        sample = test_set[idx]
        
        # Get sample name
        sample_name = sample.get("file_name", f"sample_{idx}")
        if isinstance(sample_name, Path):
            sample_name = sample_name.stem
        print(f"  Sample: {sample_name}", flush=True)
        
        # Create output folder for this sample
        sample_output_dir = output_dir / sample_name
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if inference was already done for this sample (existing predictions)
        pass4_path = sample_output_dir / "pass4_volume.nii.gz"
        input_volume_path = sample_output_dir / "input_volume.nii.gz"
        has_pass4 = pass4_path.exists()
        in_csv = sample_name in existing_samples

        if has_pass4:
            # Prediction already exists - skip inference
            if in_csv:
                print(f"  ⏭️  Skipping sample {sample_name} - existing prediction found (in CSV)", flush=True)
            else:
                print(f"  ⏭️  Skipping sample {sample_name} - existing prediction found (computing metrics from files for CSV)", flush=True)
                # Load existing volumes and compute metrics so we can add row to CSV
                gt_volume = sample["gt_volume"].numpy()
                pass4_nii = nib.load(str(pass4_path))
                pass4_volume = np.asarray(pass4_nii.dataobj).astype(np.int64)
                if input_volume_path.exists():
                    input_nii = nib.load(str(input_volume_path))
                    input_volume = np.asarray(input_nii.dataobj).astype(np.int64)
                else:
                    input_volume = sample["input_volume"].numpy()
                input_vs_gt_dice = compute_macro_avg_dice(input_volume, gt_volume, num_classes)
                input_vs_gt_f1 = compute_macro_avg_f1(input_volume, gt_volume, num_classes)
                input_vs_gt_chamfer = compute_macro_avg_chamfer(input_volume, gt_volume, num_classes)
                input_vs_gt_nsd = compute_macro_avg_surface_dice(input_volume, gt_volume, num_classes)
                pass4_dice = compute_macro_avg_dice(pass4_volume, gt_volume, num_classes)
                pass4_f1 = compute_macro_avg_f1(pass4_volume, gt_volume, num_classes)
                pass4_chamfer = compute_macro_avg_chamfer(pass4_volume, gt_volume, num_classes)
                pass4_nsd = compute_macro_avg_surface_dice(pass4_volume, gt_volume, num_classes)
                csv_writer.writerow([
                    sample_name,
                    f"{input_vs_gt_dice:.6f}",
                    f"{input_vs_gt_f1:.6f}",
                    f"{input_vs_gt_chamfer:.6f}",
                    f"{input_vs_gt_nsd:.6f}",
                    f"{pass4_dice:.6f}",
                    f"{pass4_f1:.6f}",
                    f"{pass4_chamfer:.6f}",
                    f"{pass4_nsd:.6f}",
                ])
                csv_file.flush()
                existing_samples.add(sample_name)
            continue
        elif in_csv:
            print(f"  ⚠️  Warning: {sample_name} found in CSV but pass4_volume.nii.gz missing. Will reprocess.", flush=True)
        
        # Get volume data
        volume_shape = tuple(sample["native_volume_shape"].numpy().astype(int))  # (Z, Y, X)
        input_volume = sample["input_volume"].numpy()  # [Z, Y, X]
        gt_volume = sample["gt_volume"].numpy()  # [Z, Y, X]
        query_coords_original = sample["query_coords_original"].numpy()  # [N, 3] in (z, y, x)
        query_points = sample["query_points"]  # [N, 3] - normalized coordinates for point encoding
        query_coords_resized = sample["query_coords_resized"]  # [N, 3] in (x, y, z) for CNN
        input_points = sample["points"]  # [M, 4]
        input_error_classes = sample["input_error_classes"].numpy()  # [K] - Class IDs with input errors
        modified_resized = sample["modified_resized"]  # [1 or 2, 128, 128, 128]
        medical = modified_resized[1].numpy() if modified_resized.shape[0] == 2 else None
        
        # Get instruction file path
        pointcloud_path = sample.get("file_name")
        if pointcloud_path is None:
            # Try to get from dataset
            pointcloud_path = test_set.files[idx]
        if isinstance(pointcloud_path, Path):
            base_name = pointcloud_path.stem
        else:
            base_name = str(pointcloud_path).split('/')[-1].replace('.npz', '')
        instruction_file = instructions_dir / f"{base_name}.json"
        
        print(f"  Volume shape: {volume_shape}", flush=True)
        print(f"  Input error classes: {sorted(input_error_classes.tolist())}", flush=True)
        
        # Calculate baseline metrics: input vs gt (Dice, F1, Chamfer, NSD)
        input_vs_gt_dice = compute_macro_avg_dice(input_volume, gt_volume, num_classes)
        input_vs_gt_f1 = compute_macro_avg_f1(input_volume, gt_volume, num_classes)
        input_vs_gt_chamfer = compute_macro_avg_chamfer(input_volume, gt_volume, num_classes)
        input_vs_gt_nsd = compute_macro_avg_surface_dice(input_volume, gt_volume, num_classes)
        print(f"  Input vs GT - Dice: {input_vs_gt_dice:.4f}, F1: {input_vs_gt_f1:.4f}, Chamfer: {input_vs_gt_chamfer:.4f}, NSD: {input_vs_gt_nsd:.4f}", flush=True)
        
        # Save input volume
        input_volume_path = sample_output_dir / "input_volume.nii.gz"
        save_volume_as_nifti(input_volume, input_volume_path)
        
        # PASS 4 only: Iterative correction with 4 sets
        pass_num = 4
        print(f"\n  --- PASS {pass_num}: Iterative Correction (4 Sets) ---", flush=True)
        
        error_classes_list = sorted(input_error_classes.tolist())
        error_sets = randomly_divide_classes(error_classes_list, pass_num, rng)
        print(f"  Divided into {pass_num} sets: {[len(s) for s in error_sets]}", flush=True)
        
        current_volume = input_volume.copy()
        current_volume_resized = modified_resized
        
        for iter_idx, error_set in enumerate(error_sets):
            print(f"    Iteration {iter_idx + 1}/{pass_num}: {len(error_set)} error classes", flush=True)
            
            if len(error_set) > 0:
                classes_str_set = {str(cid) for cid in error_set}
                text_instruction = load_text_instructions_for_classes(instruction_file, classes_str_set)
                if not text_instruction.strip():
                    text_instruction = get_no_error_text()
                class_hint = torch.zeros(13, dtype=torch.float32)
                for class_id in error_set:
                    if 1 <= class_id <= 13:
                        class_hint[class_id - 1] = 1.0
            else:
                text_instruction = get_no_error_text()
                class_hint = torch.zeros(13, dtype=torch.float32)
            
            from pointcloud import stack_cnn_input
            current_volume_resized = torch.from_numpy(
                stack_cnn_input(resize_volume_to_128(current_volume), medical)
            ).to(device)
            
            z_indices = query_coords_original[:, 0].astype(int)
            y_indices = query_coords_original[:, 1].astype(int)
            x_indices = query_coords_original[:, 2].astype(int)
            query_labels_input = torch.from_numpy(current_volume[z_indices, y_indices, x_indices]).to(device)
            
            pred_volume = run_inference_on_volume(
                model=model,
                input_volume=current_volume,
                volume_shape=volume_shape,
                query_coords_original=query_coords_original,
                query_points=query_points.to(device),
                query_coords_resized=query_coords_resized.to(device),
                input_points=input_points.to(device),
                text_instruction=text_instruction,
                device=device,
                batch_size=args.batch_size,
                modified_resized=current_volume_resized,
                query_labels_input=query_labels_input,
                class_hint=class_hint.to(device),
            )
            current_volume = pred_volume.copy()
        
        pass4_dice = compute_macro_avg_dice(current_volume, gt_volume, num_classes)
        pass4_f1 = compute_macro_avg_f1(current_volume, gt_volume, num_classes)
        pass4_chamfer = compute_macro_avg_chamfer(current_volume, gt_volume, num_classes)
        pass4_nsd = compute_macro_avg_surface_dice(current_volume, gt_volume, num_classes)
        print(f"  Pass 4 vs GT - Dice: {pass4_dice:.4f}, F1: {pass4_f1:.4f}, Chamfer: {pass4_chamfer:.4f}, NSD: {pass4_nsd:.4f}", flush=True)
        
        pass4_volume_path = sample_output_dir / "pass4_volume.nii.gz"
        save_volume_as_nifti(current_volume, pass4_volume_path)
        
        # Write CSV row
        csv_writer.writerow([
            sample_name,
            f"{input_vs_gt_dice:.6f}",
            f"{input_vs_gt_f1:.6f}",
            f"{input_vs_gt_chamfer:.6f}",
            f"{input_vs_gt_nsd:.6f}",
            f"{pass4_dice:.6f}",
            f"{pass4_f1:.6f}",
            f"{pass4_chamfer:.6f}",
            f"{pass4_nsd:.6f}",
        ])
        csv_file.flush()
        
        print(f"\n  ✓ Completed sample {sample_name}", flush=True)
    
    csv_file.close()
    print(f"\n{'='*60}", flush=True)
    print(f"All samples processed. Results saved to: {csv_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
