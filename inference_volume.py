"""
Inference script for volume-based point cloud transformer.

This script:
1. Loads test set with inference_mode=True
2. For each sample, performs dual-pass inference:
   - Pass 1: "no error" text
   - Pass 2: actual text instruction
3. Uses optimized inference: extract_shared_features once, then forward_query_points in batches
4. Reconstructs full volume predictions
5. Calculates metrics (Dice, F1) and saves results as compressed NPZ files
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
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


def extract_error_instances(
    input_volume: np.ndarray,
    input_error_classes: np.ndarray,
) -> List[Dict]:
    """
    Extract error instances from input_volume for classes in input_error_classes.
    Each instance represents one class that has an error in the input volume.
    
    Includes all classes in input_error_classes, even if they're not present in input_volume
    (e.g., "missing_prediction" errors where the class is missing from input).
    
    Args:
        input_volume: Input volume with errors [Z, Y, X]
        input_error_classes: Array of class IDs that have input errors
        
    Returns:
        List of dictionaries, each containing:
        - 'class_id': int - Class label
        - 'mask': np.ndarray[bool] - Boolean mask for this class in input_volume
          (empty mask if class is not in input_volume, e.g., missing_prediction)
    """
    instances = []
    volume_shape = input_volume.shape
    
    # Each class in input_error_classes is one instance
    for class_id in input_error_classes:
        class_mask = (input_volume == class_id)
        
        # Include all classes, even if not present in input_volume (missing_prediction case)
        # For missing_prediction, class_mask will be all False, which is fine
        instances.append({
            'class_id': int(class_id),
            'mask': class_mask,  # Will be empty (all False) for missing_prediction
        })
    
    return instances


def compute_instance_metrics(
    instance_mask: np.ndarray,
    instance_class: int,
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Dice and F1 scores for a single instance (class).
    
    Evaluates the entire class mask for instance_class across the whole volume,
    comparing predicted mask vs target mask.
    
    Args:
        instance_mask: Boolean mask [Z, Y, X] - used only to identify the class being evaluated
        instance_class: Class ID of the instance
        pred_volume: Predicted volume [Z, Y, X]
        target_volume: Target volume [Z, Y, X] (current_volume with remaining errors)
        
    Returns:
        Dictionary with 'dice' and 'f1' scores
    """
    # Evaluate the entire class mask for instance_class across the whole volume
    # Target: entire mask for instance_class in target_volume
    target_mask = (target_volume == instance_class)
    
    # Prediction: entire mask for instance_class in pred_volume
    pred_mask = (pred_volume == instance_class)
    
    # Dice score: how well does predicted class mask match target class mask
    intersection = (pred_mask & target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    dice = (2.0 * intersection / union) if union > 0 else 0.0
    
    # F1 score
    tp = intersection
    fp = (pred_mask & ~target_mask).sum()
    fn = (~pred_mask & target_mask).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {'dice': float(dice), 'f1': float(f1)}


def compute_per_class_recall_precision(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-class recall and precision metrics.
    
    Args:
        predictions: Predicted labels [N]
        ground_truth: Ground truth labels [N]
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping class_id to {'recall': float, 'precision': float, 'support': int}
    """
    results = {}
    
    for c in range(num_classes):
        pred_c = (predictions == c).astype(np.float32)
        gt_c = (ground_truth == c).astype(np.float32)
        
        tp = (pred_c * gt_c).sum()
        fp = (pred_c * (1 - gt_c)).sum()
        fn = ((1 - pred_c) * gt_c).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        support = int(gt_c.sum())
        
        results[c] = {
            'recall': float(recall),
            'precision': float(precision),
            'support': support,
        }
    
    return results


def compute_per_class_dice_f1(
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-class Dice and F1 scores.
    
    Args:
        pred_volume: Predicted volume [Z, Y, X]
        target_volume: Target volume [Z, Y, X]
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping class_id to {'dice': float, 'f1': float, 'support': int}
    """
    pred_flat = pred_volume.flatten()
    target_flat = target_volume.flatten()
    
    results = {}
    for c in range(num_classes):
        pred_c = (pred_flat == c).astype(np.float32)
        target_c = (target_flat == c).astype(np.float32)
        
        # Dice score
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2.0 * intersection / union) if union > 0 else 0.0
        
        # F1 score
        tp = intersection
        fp = (pred_c * (1 - target_c)).sum()
        fn = ((1 - pred_c) * target_c).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        support = int(target_c.sum())
        
        results[c] = {
            'dice': float(dice),
            'f1': float(f1),
            'support': support,
        }
    
    return results


def load_report_file(report_path: Path) -> Dict[int, str]:
    """
    Load and parse report file to extract error types for each class.
    
    Args:
        report_path: Path to the report JSON file
        
    Returns:
        Dictionary mapping class_id (int) to error_type (str)
        Returns empty dict if file doesn't exist or parsing fails
    """
    if not report_path.exists():
        return {}
    
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        report = data.get('report', {})
        error_type_map = {}
        
        for class_str, class_data in report.items():
            try:
                class_id = int(class_str)
                error_type = class_data.get('error_type', '')
                if error_type:
                    error_type_map[class_id] = error_type
            except (ValueError, TypeError):
                continue
        
        return error_type_map
    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"  Warning: Failed to load report file {report_path}: {e}", flush=True)
        return {}


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
            volume[z, y, x] = predictions[i]
    
    return volume


def save_volume_as_nifti(
    volume: np.ndarray,
    output_path: Path,
    affine: np.ndarray = None,
):
    """
    Save a volume as a NIfTI (.nii.gz) file.
    
    Args:
        volume: 3D volume array [Z, Y, X] (or [H, W, D])
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


def run_inference_on_sample(
    model: torch.nn.Module,
    sample: Dict,
    device: torch.device,
    batch_size: int = 10000,
    text_override: str = None,
    query_labels_gt: np.ndarray = None,
) -> tuple:
    """
    Run inference on all query points for a single sample.
    
    Args:
        model: PointTransformer model
        sample: Sample dictionary from dataset
        device: Device to run on
        batch_size: Batch size for inference (default 10000)
        text_override: Optional text to override the sample's text instruction
        query_labels_gt: Optional ground truth labels for query points [N]. If provided, accuracy will be printed after each 100k points.
        
    Returns:
        Tuple of:
        - Predicted labels for all query points [N]
        - Total number of query points (int)
        - Number of batches processed (int)
        - Total points processed (int)
    """
    model.eval()
    
    # Get query points and related data
    # Note: dataset returns tensors without batch dimension, so we need to add it
    query_points = sample["query_points"].to(device)  # [N, 3] -> [1, N, 3]
    if query_points.dim() == 2:
        query_points = query_points.unsqueeze(0)  # Add batch dimension
    
    query_coords_resized = sample["query_coords_resized"].to(device)  # [N, 3] -> [1, N, 3]
    if query_coords_resized.dim() == 2:
        query_coords_resized = query_coords_resized.unsqueeze(0)  # Add batch dimension
    
    modified_resized = sample["modified_resized"].to(device)  # [1, 128, 128, 128] from dataset
    if modified_resized.dim() == 4:
        modified_resized = modified_resized.unsqueeze(0)  # [1, 1, 128, 128, 128]
    
    input_points = sample["points"].to(device)  # [M, 4] -> [1, M, 4]
    if input_points.dim() == 2:
        input_points = input_points.unsqueeze(0)  # Add batch dimension
    
    # Get query_labels_input if available (labels from input_volume at query points)
    query_labels_input = sample.get("query_labels_input")
    if query_labels_input is not None:
        query_labels_input = query_labels_input.to(device)  # [N] or [1, N]
        if query_labels_input.dim() == 1:
            query_labels_input = query_labels_input.unsqueeze(0)  # [1, N]
    
    # Use text_override if provided, otherwise use sample's text
    texts = [text_override if text_override is not None else sample["text"]]
    
    # Get class_hint if available (optional)
    class_hint = sample.get("class_hint")
    if class_hint is not None:
        class_hint = class_hint.to(device)  # [13] -> [1, 13]
        if class_hint.dim() == 1:
            class_hint = class_hint.unsqueeze(0)  # Add batch dimension
    
    num_query_points = query_points.shape[1]
    all_predictions = []
    num_batches_processed = 0
    total_points_processed = 0
    
    # Track accuracy if ground truth labels are provided
    track_accuracy = query_labels_gt is not None
    if track_accuracy:
        total_correct = 0
        last_printed_milestone = 0  # Track last printed 100k milestone
    
    # OPTIMIZATION: Extract shared features once (multimodal + CNN)
    # These don't depend on query points, so we compute them once and reuse
    # This significantly speeds up inference when processing many query points in batches
    with torch.no_grad():
        shared_features = model.extract_shared_features(
            input_points=input_points,
            texts=texts,
            class_hint=class_hint,
            modified_resized=modified_resized,
        )
        latents = shared_features["latents"]  # [1, num_latents, dim]
        feature_pyramid = shared_features["feature_pyramid"]  # List of 3 feature maps (None if --no-cnn)
        # Process query points in batches using pre-computed shared features
        for start_idx in range(0, num_query_points, batch_size):
            end_idx = min(start_idx + batch_size, num_query_points)
            batch_size_actual = end_idx - start_idx
            
            # Extract batch
            batch_query_points = query_points[:, start_idx:end_idx]  # [1, B, 3]
            batch_query_coords_resized = query_coords_resized[:, start_idx:end_idx]  # [1, B, 3]
            batch_query_labels_input = None
            if query_labels_input is not None:
                batch_query_labels_input = query_labels_input[:, start_idx:end_idx]  # [1, B]
            
            # Run optimized forward pass using pre-computed features
            outputs = model.forward_query_points(
                query_points=batch_query_points,
                query_coords_resized=batch_query_coords_resized,
                latents=latents,
                feature_pyramid=feature_pyramid,
                query_labels_input=batch_query_labels_input,
            )
            
            # Get predictions (argmax of logits)
            logits = outputs["logits"]  # [1, B, num_classes]
            predictions = torch.argmax(logits, dim=-1)  # [1, B]
            predictions_np = predictions.cpu().numpy()[0]  # [B]
            all_predictions.append(predictions_np)
            
            # Track accuracy if ground truth is provided
            if track_accuracy:
                batch_gt = query_labels_gt[start_idx:end_idx]  # [B]
                batch_correct = (predictions_np == batch_gt).sum()
                total_correct += batch_correct
                total_points_processed += batch_size_actual
                
                # Print overall accuracy once after reaching each 100k milestone
                # Calculate which 100k milestone we're currently at
                current_milestone = (total_points_processed // 100000) * 100000
                
                # Only print if we've crossed into a new milestone (and it's >= 100k)
                '''
                if current_milestone > last_printed_milestone and current_milestone >= 100000:
                    overall_accuracy = total_correct / total_points_processed
                    print(f"    [OVERALL ACCURACY] After {total_points_processed:,} query points: {overall_accuracy:.4f} ({total_correct:,}/{total_points_processed:,} correct)", flush=True)
                    # Update last printed milestone
                    last_printed_milestone = current_milestone
                '''
            else:
                total_points_processed += batch_size_actual
            
            # Track processing
            num_batches_processed += 1
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)  # [N]
    
    # Return predictions and processing info
    return all_predictions, num_query_points, num_batches_processed, total_points_processed  # [N], int, int, int


def main():
    parser = argparse.ArgumentParser(description="Run volume inference on test set.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file.",
    )
    parser.add_argument(
        "--pointcloud-data-root",
        type=str,
        required=True,
        help="Root directory for point cloud NPZ files.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help="Path to split JSON file.",
    )
    parser.add_argument(
        "--instructions-dir",
        type=str,
        required=True,
        help="Directory containing text instruction JSON files.",
    )
    parser.add_argument(
        "--correction-result-dir",
        type=str,
        required=True,
        help="Directory to save correction results (NPZ files).",
    )
    parser.add_argument(
        "--medical-images-dir",
        type=str,
        default=None,
        help="128³ scan NPZs. Used when the checkpoint name contains _img or --use-image is set.",
    )
    parser.add_argument(
        "--use-image",
        action="store_true",
        help="Force shape+image CNN input. Also enabled when the checkpoint name contains _img.",
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
        help="Fold number (for checkpoint auto-generation).",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory containing report JSON files. Defaults to 'reports' folder relative to data root.",
    )
    parser.add_argument(
        "--input-points",
        type=int,
        default=8000,
        help="Number of input points (for checkpoint auto-generation).",
    )
    parser.add_argument(
        "--query-points",
        type=int,
        default=8000,
        help="Number of query points (only used for checkpoint name matching and model initialization; inference uses all volume points).",
    )
    parser.add_argument(
        "--input-fg-ratio",
        type=float,
        default=0.6,
        help="Input foreground ratio (for checkpoint name matching, default: 0.6).",
    )
    parser.add_argument(
        "--input-random-ratio",
        type=float,
        default=0.4,
        help="Input random ratio (for checkpoint name matching, default: 0.4).",
    )
    parser.add_argument(
        "--query-fg-ratio",
        type=float,
        default=0.6,
        help="Query foreground ratio (only used for checkpoint name matching; inference uses all volume points, default: 0.6).",
    )
    parser.add_argument(
        "--query-random-ratio",
        type=float,
        default=0.4,
        help="Query random ratio (only used for checkpoint name matching; inference uses all volume points, default: 0.4).",
    )
    parser.add_argument(
        "--no-cnn",
        action="store_true",
        help="If set, disable CNN volume encoder. Query points will use only coordinates through MLP before cross-attention (no self-attention).",
    )
    args = parser.parse_args()
    
    # Setup paths
    pointcloud_data_root = Path(args.pointcloud_data_root)
    
    # Set up report directory
    if args.report_dir is not None:
        report_dir = Path(args.report_dir)
    else:
        # Default to 'reports' folder relative to data root
        report_dir = pointcloud_data_root.parent / "reports"
    print(f"Report directory: {report_dir}", flush=True)
    split_file = Path(args.split_file)
    instructions_dir = Path(args.instructions_dir)
    correction_result_dir = Path(args.correction_result_dir)
    correction_result_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup checkpoint path
    checkpoint_path = Path(args.checkpoint)
    
    # Define variables needed for checkpoint search (used in both if and else blocks)
    repo_root = Path(__file__).resolve().parent
    ckp_dir = repo_root / "cross_attention_checkpoints"
    
    # New format with ratios
    input_fg_int = int(args.input_fg_ratio * 100)
    input_random_int = int(args.input_random_ratio * 100)
    query_fg_int = int(args.query_fg_ratio * 100)
    query_random_int = int(args.query_random_ratio * 100)
    points_str_new = f"inp{args.input_points}_ifg{input_fg_int}_ir{input_random_int}_qry{args.query_points}_qfg{query_fg_int}_qr{query_random_int}"
    
    # Old format (without ratios) for backward compatibility
    points_str_old = f"inp{args.input_points}_qry{args.query_points}"
    
    # Determine CNN suffix based on --no-cnn flag
    cnn_suffix = "_nocnn" if args.no_cnn else ""
    
    if not checkpoint_path.exists():
        # Try auto-generated paths (try new format first, then old format for backward compatibility)

        # Try new format first (with CNN suffix matching the flag)
        checkpoint_path = ckp_dir / f"checkpoint_best_{args.fold}_{points_str_new}{cnn_suffix}.pth"
        if not checkpoint_path.exists():
            # Try new format with L1 (with CNN suffix)
            for l1_weight in [0.1, 0.01, 0.5, 1.0]:
                l1_points_str = f"{points_str_new}_l1{l1_weight}{cnn_suffix}"
                checkpoint_path = ckp_dir / f"checkpoint_best_{args.fold}_{l1_points_str}.pth"
                if checkpoint_path.exists():
                    break
            else:
                # Try without CNN suffix (for backward compatibility - old checkpoints default to CNN=True)
                if not args.no_cnn:  # Only try without suffix if we're looking for CNN checkpoints
                    checkpoint_path = ckp_dir / f"checkpoint_best_{args.fold}_{points_str_new}.pth"
                    if not checkpoint_path.exists():
                        for l1_weight in [0.1, 0.01, 0.5, 1.0]:
                            l1_points_str = f"{points_str_new}_l1{l1_weight}"
                            checkpoint_path = ckp_dir / f"checkpoint_best_{args.fold}_{l1_points_str}.pth"
                            if checkpoint_path.exists():
                                break

                if not checkpoint_path.exists():
                    # Try old format (without ratios, with CNN suffix)
                    checkpoint_path = ckp_dir / f"checkpoint_best_{args.fold}_{points_str_old}{cnn_suffix}.pth"
                    if not checkpoint_path.exists():
                        # Try old format with L1 (with CNN suffix)
                        for l1_weight in [0.1, 0.01, 0.5, 1.0]:
                            l1_points_str = f"{points_str_old}_l1{l1_weight}{cnn_suffix}"
                            checkpoint_path = ckp_dir / f"checkpoint_best_{args.fold}_{l1_points_str}.pth"
                            if checkpoint_path.exists():
                                break
        else:
            # Try old format without CNN suffix (backward compatibility)
            if not args.no_cnn:
                checkpoint_path = ckp_dir / f"checkpoint_best_{args.fold}_{points_str_old}.pth"
                if not checkpoint_path.exists():
                    for l1_weight in [0.1, 0.01, 0.5, 1.0]:
                        l1_points_str = f"{points_str_old}_l1{l1_weight}"
                        checkpoint_path = ckp_dir / f"checkpoint_best_{args.fold}_{l1_points_str}.pth"
                        if checkpoint_path.exists():
                            break

            if not checkpoint_path.exists():
                # If still not found, try to find any checkpoint matching the pattern
                pattern = f"checkpoint_best_{args.fold}_{points_str_new}*{cnn_suffix}.pth"
                matching_checkpoints = list(ckp_dir.glob(pattern))
                if not matching_checkpoints and not args.no_cnn:
                    # Try without CNN suffix for backward compatibility
                    pattern = f"checkpoint_best_{args.fold}_{points_str_new}*.pth"
                    matching_checkpoints = list(ckp_dir.glob(pattern))
                    # Filter out _nocnn checkpoints when looking for CNN checkpoints
                    matching_checkpoints = [c for c in matching_checkpoints if "_nocnn" not in c.name]
                if not matching_checkpoints:
                    pattern = f"checkpoint_best_{args.fold}_{points_str_old}*{cnn_suffix}.pth"
                    matching_checkpoints = list(ckp_dir.glob(pattern))
                    if not matching_checkpoints and not args.no_cnn:
                        pattern = f"checkpoint_best_{args.fold}_{points_str_old}*.pth"
                        matching_checkpoints = list(ckp_dir.glob(pattern))
                        matching_checkpoints = [c for c in matching_checkpoints if "_nocnn" not in c.name]
                if matching_checkpoints:
                    checkpoint_path = matching_checkpoints[0]
                    print(f"Found checkpoint: {checkpoint_path.name}", flush=True)
        
        # Check if checkpoint was found after all searching
        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found at {args.checkpoint} or any matching pattern in {ckp_dir}", flush=True)
            print(f"  Tried new format: checkpoint_best_{args.fold}_{points_str_new}*{cnn_suffix}.pth", flush=True)
            print(f"  Tried old format: checkpoint_best_{args.fold}_{points_str_old}*{cnn_suffix}.pth", flush=True)
            return
    
    print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    
    # Detect CNN usage from checkpoint name (backward compatible: default to CNN=True if not specified)
    checkpoint_name = checkpoint_path.name
    if "_nocnn" in checkpoint_name:
        detected_use_cnn = False
        print(f"Detected non-CNN checkpoint from name: {checkpoint_name}", flush=True)
    else:
        detected_use_cnn = True
        print(f"Detected CNN checkpoint from name: {checkpoint_name}", flush=True)
    
    # Use detected value from checkpoint name (backward compatible: default to CNN=True)
    # The --no-cnn flag should match the checkpoint, but we'll use the checkpoint name as source of truth
    use_cnn = detected_use_cnn
    if args.no_cnn != (not detected_use_cnn):
        print(f"WARNING: --no-cnn flag ({args.no_cnn}) doesn't match checkpoint name (CNN={'enabled' if detected_use_cnn else 'disabled'}). Using checkpoint setting.", flush=True)
    use_image = use_cnn and (args.use_image or "_img" in checkpoint_path.stem)
    print(f"CNN input: {'shape+image' if use_image else 'shape only'}", flush=True)
    medical_images_dir = None
    if use_image:
        medical_images_dir = args.medical_images_dir or str(pointcloud_data_root.parent / "preprocessed_volume_images")
    
    # Load test set with inference_mode=True
    print("Loading test set with inference_mode=True...", flush=True)
    from dataset import load_split_subjects, CowPointCloudDataset
    subjects = load_split_subjects(split_file)
    test_set = CowPointCloudDataset(
        root_dir=str(pointcloud_data_root),
        split_subjects=subjects["test"],
        input_points=args.input_points,
        seed=42,
        input_fg_ratio=args.input_fg_ratio,
        input_random_ratio=args.input_random_ratio,
        query_points=args.query_points,  # Only used for checkpoint name matching; ignored in inference_mode=True
        query_fg_ratio=args.query_fg_ratio,  # Only used for checkpoint name matching; ignored in inference_mode=True
        query_random_ratio=args.query_random_ratio,  # Only used for checkpoint name matching; ignored in inference_mode=True
        use_gt_labels=False,
        instructions_dir=str(instructions_dir),
        medical_images_dir=medical_images_dir,
        inference_mode=True,  # In inference mode, all volume points are used as query points regardless of these parameters
    )
    
    print(f"Test set size: {len(test_set)}", flush=True)
    if len(test_set) == 0:
        print("ERROR: Test set is empty! Exiting.", flush=True)
        return
    
    # Build model
    print("Building model...", flush=True)
    sample = test_set[0]
    point_feature_dim = sample["points"].shape[-1]
    # use_cnn is already set from checkpoint name detection above
    print(f"Building model with use_cnn={use_cnn} (detected from checkpoint name)", flush=True)
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
    no_error_text = "No errors detected in this segmentation."
        
    # Process each test sample
    print(f"\nRunning inference on {len(test_set)} test samples...", flush=True)
    print(f"Results will be saved to: {correction_result_dir}", flush=True)
    
    for idx in range(len(test_set)):
        print(f"\n{'='*60}", flush=True)
        print(f"Processing sample {idx + 1}/{len(test_set)}", flush=True)
        print(f"{'='*60}", flush=True)
        sample = test_set[idx]
        
        # Get volume shapes and data
        volume_shape = tuple(sample["native_volume_shape"].numpy().astype(int))  # (Z, Y, X)
        current_volume = sample["current_volume"].numpy()  # [Z, Y, X]
        input_volume = sample["input_volume"].numpy()  # [Z, Y, X]
        gt_volume = sample["gt_volume"].numpy()  # [Z, Y, X]
        query_coords_original_all = sample["query_coords_original"].numpy()  # [N, 3] in (z, y, x)
        actual_text = sample["text"]
        input_error_classes = sample["input_error_classes"].numpy()  # [M] - Class IDs with input errors from dataset
        remaining_error_classes = sample["remaining_error_classes"].numpy()  # [K] - Class IDs with remaining errors from dataset
        
        # Use all voxels in the volume for query points (no foreground filtering)
        # query_coords_original_all already contains all voxels from the dataset's inference_mode
        query_coords_original = query_coords_original_all  # [N, 3] - all voxels in volume
        
        # Use all query points without filtering - use original sample as-is
        # No need to create filtered version since we're querying all voxels
        
        # Get ground truth labels for all query points
        query_labels_gt = current_volume[query_coords_original[:, 0].astype(int), 
                                         query_coords_original[:, 1].astype(int), 
                                         query_coords_original[:, 2].astype(int)]  # [N]
        
        print(f"  Volume shape: {volume_shape}", flush=True)
        print(f"  Total volume points: {volume_shape[0] * volume_shape[1] * volume_shape[2]:,}", flush=True)
        print(f"  Query points (all voxels): {len(query_coords_original):,}", flush=True)
        text_preview = f"  Actual text instruction: {actual_text[:100]}..." if len(actual_text) > 100 else f"  Actual text instruction: {actual_text}"
        print(text_preview, flush=True)
        
        # PASS 1: No error text
        print(f"\n  --- PASS 1: No Error Text ---", flush=True)
        print(f"  Text: {no_error_text}", flush=True)
        print("  Running inference...", flush=True)
        predictions_pass1, num_query_total, num_batches_pass1, total_processed_pass1 = run_inference_on_sample(
            model=model,
            sample=sample,
            device=device,
            batch_size=args.batch_size,
            text_override=no_error_text,
            query_labels_gt=query_labels_gt,
        )
        print(f"  Processed {total_processed_pass1} query points in {num_batches_pass1} batches", flush=True)
        
        # Reconstruct predicted volume for pass 1
        print("  Reconstructing predicted volume...", flush=True)
        pred_volume_pass1 = reconstruct_volume_from_predictions(
            query_coords=query_coords_original,
            predictions=predictions_pass1,
            volume_shape=volume_shape,
        )
        
        # PASS 2: Actual text instruction
        print(f"\n  --- PASS 2: Actual Text Instruction ---", flush=True)
        text_preview_pass2 = f"  Text: {actual_text[:100]}..." if len(actual_text) > 100 else f"  Text: {actual_text}"
        print(text_preview_pass2, flush=True)
        print("  Running inference...", flush=True)
        predictions_pass2, num_query_total_pass2, num_batches_pass2, total_processed_pass2 = run_inference_on_sample(
            model=model,
            sample=sample,
            device=device,
            batch_size=args.batch_size,
            text_override=None,  # Use actual text from sample
            query_labels_gt=query_labels_gt,
        )
        print(f"  Processed {total_processed_pass2} query points in {num_batches_pass2} batches", flush=True)
        
        # Reconstruct predicted volume for pass 2
        print("  Reconstructing predicted volume...", flush=True)
        pred_volume_pass2 = reconstruct_volume_from_predictions(
            query_coords=query_coords_original,
            predictions=predictions_pass2,
            volume_shape=volume_shape,
            )
        
        # Load report file
        base_name = test_set.files[idx].stem
        # Convert filename format: topcow_ct_001_0005 -> topcow_ct_001_metadata_0005
        parts = base_name.rsplit('_', 1)
        if len(parts) == 2:
            report_base_name = f"{parts[0]}_metadata_{parts[1]}"
        else:
            report_base_name = f"{base_name}_metadata"
        report_path = report_dir / f"{report_base_name}.json"
        error_type_map = load_report_file(report_path)
        if error_type_map:
            print(f"  Loaded report file: {report_path.name} ({len(error_type_map)} classes with error types)", flush=True)
        else:
            print(f"  No report file found or empty: {report_path.name}", flush=True)
        
        # Compute overall Dice and F1 scores
        print("\n  --- Overall Metrics ---", flush=True)
        dice_pred_vs_current_pass1 = compute_macro_avg_dice(pred_volume_pass1, current_volume, num_classes)
        f1_pred_vs_current_pass1 = compute_macro_avg_f1(pred_volume_pass1, current_volume, num_classes)
        dice_pred_vs_current_pass2 = compute_macro_avg_dice(pred_volume_pass2, current_volume, num_classes)
        f1_pred_vs_current_pass2 = compute_macro_avg_f1(pred_volume_pass2, current_volume, num_classes)
        
        print(f"  Pass 1 (No Error) vs Remaining Error - Dice: {dice_pred_vs_current_pass1:.4f}, F1: {f1_pred_vs_current_pass1:.4f}", flush=True)
        print(f"  Pass 2 (Actual Text) vs Remaining Error - Dice: {dice_pred_vs_current_pass2:.4f}, F1: {f1_pred_vs_current_pass2:.4f}", flush=True)
        
        # Extract error instances from input_volume for classes in input_error_classes
        print("\n  --- Extracting Error Instances ---", flush=True)
        print(f"  Input error classes: {sorted(input_error_classes.tolist())}", flush=True)
        print(f"  Remaining error classes: {sorted(remaining_error_classes.tolist())}", flush=True)
        error_instances = extract_error_instances(input_volume, input_error_classes)
        print(f"  Found {len(error_instances)} error instances", flush=True)
        
        # Evaluate each instance
        instance_results = []
        for instance in error_instances:
            instance_mask = instance['mask']
            instance_class = instance['class_id']
            
            # Check if instance is in remaining error
            # Simply check if the class is in remaining_error_classes from the dataset
            is_remaining_error = instance_class in remaining_error_classes
            
            # Get error type from report
            error_type = error_type_map.get(instance_class, 'unknown')
            
            # Compute metrics for this instance (Pass 1)
            metrics_pass1 = compute_instance_metrics(
                instance_mask, instance_class, pred_volume_pass1, current_volume
            )
            
            # Compute metrics for this instance (Pass 2)
            metrics_pass2 = compute_instance_metrics(
                instance_mask, instance_class, pred_volume_pass2, current_volume
            )
            
            instance_results.append({
                'class_id': instance_class,
                'error_type': error_type,
                'is_remaining_error': is_remaining_error,
                'dice_pass1': metrics_pass1['dice'],
                'f1_pass1': metrics_pass1['f1'],
                'dice_pass2': metrics_pass2['dice'],
                'f1_pass2': metrics_pass2['f1'],
            })
        
        # Print instance-level results
        print("\n  --- Instance-Level Results ---", flush=True)
        print(f"  Total instances: {len(instance_results)}", flush=True)
        print("  Class | Error Type              | Remaining | Dice (P1) | F1 (P1)  | Dice (P2) | F1 (P2)", flush=True)
        print("  ------|-------------------------|-----------|-----------|-----------|-----------|----------", flush=True)
        for result in instance_results:
            remaining_str = "Yes" if result['is_remaining_error'] else "No"
            print(f"  {result['class_id']:5d} | {result['error_type']:23s} | {remaining_str:9s} | {result['dice_pass1']:9.4f} | {result['f1_pass1']:9.4f} | {result['dice_pass2']:9.4f} | {result['f1_pass2']:9.4f}", flush=True)
        print(f"  Printed {len(instance_results)} instance(s)", flush=True)
        
        # Save results
        # Create subfolder for this test case
        sample_result_dir = correction_result_dir / base_name
        sample_result_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare instance-level data for saving
        num_instances = len(instance_results)
        instance_classes = np.array([r['class_id'] for r in instance_results], dtype=np.int32)
        instance_error_types = np.array([r['error_type'] for r in instance_results], dtype=object)
        instance_is_remaining = np.array([r['is_remaining_error'] for r in instance_results], dtype=bool)
        instance_dice_pass1 = np.array([r['dice_pass1'] for r in instance_results], dtype=np.float32)
        instance_f1_pass1 = np.array([r['f1_pass1'] for r in instance_results], dtype=np.float32)
        instance_dice_pass2 = np.array([r['dice_pass2'] for r in instance_results], dtype=np.float32)
        instance_f1_pass2 = np.array([r['f1_pass2'] for r in instance_results], dtype=np.float32)
        
        # Save all results as compressed NPZ file
        result_file = sample_result_dir / "results.npz"
        np.savez_compressed(
            result_file,
            corrected_volume_pass1=pred_volume_pass1.astype(np.int32),
            corrected_volume_pass2=pred_volume_pass2.astype(np.int32),
            error_volume=input_volume.astype(np.int32),
            remaining_error_volume=current_volume.astype(np.int32),
            text=actual_text,
            # Overall metrics
            dice_pred_vs_current_pass1=dice_pred_vs_current_pass1,
            f1_pred_vs_current_pass1=f1_pred_vs_current_pass1,
            dice_pred_vs_current_pass2=dice_pred_vs_current_pass2,
            f1_pred_vs_current_pass2=f1_pred_vs_current_pass2,
            # Instance-level information
            num_instances=num_instances,
            instance_classes=instance_classes,
            instance_error_types=instance_error_types,
            instance_is_remaining_error=instance_is_remaining,
            instance_dice_pass1=instance_dice_pass1,
            instance_f1_pass1=instance_f1_pass1,
            instance_dice_pass2=instance_dice_pass2,
            instance_f1_pass2=instance_f1_pass2,
        )
        print(f"  ✓ Saved results to {result_file}", flush=True)
        
        # Save volumes as NIfTI (.nii.gz) files
        # Use identity affine matrix (default, since original affine is not available from NPZ)
        default_affine = np.eye(4)
            
        print("  Saving volumes as NIfTI files...", flush=True)
        save_volume_as_nifti(
            volume=pred_volume_pass1,
            output_path=sample_result_dir / "corrected_volume_pass1.nii.gz",
            affine=default_affine,
        )
        save_volume_as_nifti(
            volume=pred_volume_pass2,
            output_path=sample_result_dir / "corrected_volume_pass2.nii.gz",
            affine=default_affine,
        )
        save_volume_as_nifti(
            volume=input_volume,
            output_path=sample_result_dir / "input_volume.nii.gz",
            affine=default_affine,
        )
        save_volume_as_nifti(
            volume=current_volume,
            output_path=sample_result_dir / "desired_output_volume.nii.gz",
            affine=default_affine,
                    )
        print(f"  ✓ Saved NIfTI files to {sample_result_dir}", flush=True)
            
    print(f"\n{'='*60}", flush=True)
    print("Inference completed!", flush=True)
    print(f"All results saved to: {correction_result_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
