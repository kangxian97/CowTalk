#!/usr/bin/env python3
"""
Test script to validate dataset loading and visualize input volume point clouds
and remaining error point clouds side by side (background points omitted).
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib
# Use non-interactive backend if no display is available
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import dataset


def get_color_for_label(label_id, num_classes=14):
    """Get a distinct color for each label."""
    cmap = plt.cm.get_cmap("tab20", max(num_classes, 1))
    return cmap(label_id % cmap.N)


def visualize_point_clouds_side_by_side(
    input_coords: np.ndarray,
    input_labels: np.ndarray,
    query_coords: np.ndarray,
    query_labels: np.ndarray,
    title_prefix: str = "Sample",
    save_path: Path = None,
    subsample: int = 5000,
):
    """
    Visualize input and query point clouds side by side, omitting background points.
    
    Args:
        input_coords: (N, 3) input point coordinates (normalized)
        input_labels: (N,) input point labels
        query_coords: (M, 3) query point coordinates (normalized)
        query_labels: (M,) query point labels
        title_prefix: Prefix for plot titles
        save_path: Optional path to save the figure
        subsample: Maximum number of points to visualize per cloud
    """
    # Filter out background points (label 0)
    input_fg_mask = input_labels != 0
    query_fg_mask = query_labels != 0
    
    input_coords_fg = input_coords[input_fg_mask]
    input_labels_fg = input_labels[input_fg_mask]
    query_coords_fg = query_coords[query_fg_mask]
    query_labels_fg = query_labels[query_fg_mask]
    
    print(f"  Input points: {len(input_coords)} total, {len(input_coords_fg)} foreground")
    print(f"  Query points: {len(query_coords)} total, {len(query_labels_fg)} foreground")
    
    # Subsample if needed
    if len(input_coords_fg) > subsample:
        indices = np.random.choice(len(input_coords_fg), size=subsample, replace=False)
        input_coords_fg = input_coords_fg[indices]
        input_labels_fg = input_labels_fg[indices]
        print(f"  Subsampled input to {subsample} points")
    
    if len(query_coords_fg) > subsample:
        indices = np.random.choice(len(query_coords_fg), size=subsample, replace=False)
        query_coords_fg = query_coords_fg[indices]
        query_labels_fg = query_labels_fg[indices]
        print(f"  Subsampled query to {subsample} points")
    
    # Get unique labels for color mapping
    all_labels = np.concatenate([input_labels_fg, query_labels_fg])
    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)
    
    # Create color map
    label_to_color = {label: get_color_for_label(i, num_classes) for i, label in enumerate(unique_labels)}
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Left subplot: Input points (from current_volume with remaining errors)
    ax1 = fig.add_subplot(121, projection="3d")
    if len(input_coords_fg) > 0:
        input_colors = np.array([label_to_color[label] for label in input_labels_fg])
        ax1.scatter(
            input_coords_fg[:, 0],
            input_coords_fg[:, 1],
            input_coords_fg[:, 2],
            c=input_colors,
            s=5,
            alpha=0.8,
            edgecolors="none",
        )
    ax1.set_xlabel("X (normalized)")
    ax1.set_ylabel("Y (normalized)")
    ax1.set_zlabel("Z (normalized)")
    ax1.set_title(f"{title_prefix}: Input Points\n(Labels: All Input Errors)", fontsize=12)
    
    # Right subplot: Query points (sampled from input_volume, but labels from current_volume for supervision)
    ax2 = fig.add_subplot(122, projection="3d")
    if len(query_coords_fg) > 0:
        query_colors = np.array([label_to_color[label] for label in query_labels_fg])
        ax2.scatter(
            query_coords_fg[:, 0],
            query_coords_fg[:, 1],
            query_coords_fg[:, 2],
            c=query_colors,
            s=5,
            alpha=0.8,
            edgecolors="none",
        )
    ax2.set_xlabel("X (normalized)")
    ax2.set_ylabel("Y (normalized)")
    ax2.set_zlabel("Z (normalized)")
    ax2.set_title(f"{title_prefix}: Query Points\n(Supervision: Remaining Errors Only)", fontsize=12)
    
    # Set equal aspect ratio for both plots
    for ax in [ax1, ax2]:
        if len(input_coords_fg) > 0 or len(query_coords_fg) > 0:
            all_coords = np.concatenate([input_coords_fg, query_coords_fg], axis=0) if len(input_coords_fg) > 0 and len(query_coords_fg) > 0 else (input_coords_fg if len(input_coords_fg) > 0 else query_coords_fg)
            if len(all_coords) > 0:
                max_range = np.array([
                    all_coords[:, 0].max() - all_coords[:, 0].min(),
                    all_coords[:, 1].max() - all_coords[:, 1].min(),
                    all_coords[:, 2].max() - all_coords[:, 2].min(),
                ]).max() / 2.0
                mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
                mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
                mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    # Always save if save_path is provided, otherwise try to show (if display available)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved visualization to {save_path}")
    
    # Try to show if display is available and no save_path was provided
    if not save_path:
        try:
            plt.show(block=False)
            print(f"  Displayed visualization (close window to continue)")
            # Give a moment for the window to appear
            import time
            time.sleep(0.5)
        except Exception as e:
            print(f"  Warning: Could not display visualization: {e}")
            print(f"  Consider using --output_dir to save visualizations instead")
    
    plt.close(fig)


def validate_label_differences(input_labels: np.ndarray, query_labels: np.ndarray, 
                                class_hint: torch.Tensor, remaining_error_classes: torch.Tensor,
                                index: int):
    """
    Validate the difference between input point labels and query point labels.
    
    Expected:
    - Input labels: from input_volume (all input errors = remaining + manipulated)
    - Query labels: from current_volume (remaining errors only)
    - Difference: manipulated errors should be present in input but not in query
    """
    print(f"\n  {'-'*76}")
    print(f"  Label Difference Validation:")
    print(f"  {'-'*76}")
    
    # Convert to numpy if needed
    if isinstance(input_labels, torch.Tensor):
        input_labels = input_labels.numpy()
    if isinstance(query_labels, torch.Tensor):
        query_labels = query_labels.numpy()
    if isinstance(class_hint, torch.Tensor):
        class_hint = class_hint.numpy()
    if isinstance(remaining_error_classes, torch.Tensor):
        remaining_error_classes = remaining_error_classes.numpy()
    
    # Get unique labels
    unique_input_labels = np.unique(input_labels)
    unique_query_labels = np.unique(query_labels)
    
    # Get error class information
    input_error_classes = np.where(class_hint > 0.5)[0] + 1  # Convert to 1-indexed
    remaining_classes = np.where(remaining_error_classes > 0.5)[0] + 1
    manipulated_classes = np.setdiff1d(input_error_classes, remaining_classes)
    
    print(f"    Input error classes (from class_hint): {input_error_classes.tolist()}")
    print(f"    Remaining error classes (prediction target): {remaining_classes.tolist()}")
    print(f"    Manipulated error classes (should be removed): {manipulated_classes.tolist()}")
    
    # Check 1: Input labels should contain all input error classes
    input_labels_set = set(unique_input_labels)
    input_error_classes_set = set(input_error_classes)
    missing_in_input = input_error_classes_set - input_labels_set
    if missing_in_input:
        print(f"    ⚠️  WARNING: Input error classes {list(missing_in_input)} not found in input labels")
    else:
        print(f"    ✓ All input error classes present in input labels")
    
    # Check 2: Query labels should only contain remaining error classes (and background/other non-error classes)
    # Query labels should NOT contain manipulated error classes
    query_labels_set = set(unique_query_labels)
    manipulated_classes_set = set(manipulated_classes)
    manipulated_in_query = manipulated_classes_set & query_labels_set
    if manipulated_in_query:
        print(f"    ⚠️  WARNING: Manipulated error classes {list(manipulated_in_query)} found in query labels!")
        print(f"       Query labels should only contain remaining errors, not manipulated errors.")
    else:
        print(f"    ✓ No manipulated error classes in query labels (correct)")
    
    # Check 3: Remaining error classes should be in both input and query labels
    remaining_classes_set = set(remaining_classes)
    remaining_in_input = remaining_classes_set & input_labels_set
    remaining_in_query = remaining_classes_set & query_labels_set
    if remaining_classes_set:
        if remaining_in_input == remaining_classes_set:
            print(f"    ✓ All remaining error classes present in input labels")
        else:
            missing = remaining_classes_set - remaining_in_input
            print(f"    ⚠️  WARNING: Remaining error classes {list(missing)} missing in input labels")
        
        if remaining_in_query == remaining_classes_set:
            print(f"    ✓ All remaining error classes present in query labels")
        else:
            missing = remaining_classes_set - remaining_in_query
            print(f"    ⚠️  WARNING: Remaining error classes {list(missing)} missing in query labels")
    else:
        print(f"    ℹ️  No remaining error classes (all errors are manipulated)")
    
    # Check 4: Label distribution comparison
    print(f"\n    Label Distribution Comparison:")
    all_labels = sorted(set(unique_input_labels) | set(unique_query_labels))
    for label in all_labels:
        input_count = np.sum(input_labels == label)
        query_count = np.sum(query_labels == label)
        input_pct = input_count / len(input_labels) * 100
        query_pct = query_count / len(query_labels) * 100
        
        status = ""
        if label in manipulated_classes_set:
            status = " [MANIPULATED - should be removed]"
        elif label in remaining_classes_set:
            status = " [REMAINING - should stay]"
        elif label in input_error_classes_set:
            status = " [INPUT ERROR]"
        
        print(f"      Label {label:2d}: Input={input_count:5d} ({input_pct:5.1f}%) | "
              f"Query={query_count:5d} ({query_pct:5.1f}%){status}")
    
    # Check 5: Summary statistics
    print(f"\n    Summary:")
    print(f"      Total input points: {len(input_labels)}")
    print(f"      Total query points: {len(query_labels)}")
    print(f"      Unique input labels: {len(unique_input_labels)}")
    print(f"      Unique query labels: {len(unique_query_labels)}")
    
    # Check if there are differences
    labels_only_in_input = input_labels_set - query_labels_set
    labels_only_in_query = query_labels_set - input_labels_set
    if labels_only_in_input:
        print(f"      Labels only in input (not in query): {sorted(labels_only_in_input)}")
    if labels_only_in_query:
        print(f"      Labels only in query (not in input): {sorted(labels_only_in_query)}")
    
    return {
        "input_error_classes": input_error_classes,
        "remaining_classes": remaining_classes,
        "manipulated_classes": manipulated_classes,
        "validation_passed": len(missing_in_input) == 0 and len(manipulated_in_query) == 0
    }


def print_sample_info(sample: dict, index: int):
    """Print information about a dataset sample."""
    print(f"\n{'='*80}")
    print(f"Sample {index}:")
    print(f"{'='*80}")
    
    points = sample["points"]
    labels = sample["labels"]
    query_points = sample["query_points"]
    query_labels = sample["query_labels"]
    text = sample["text"]
    class_hint = sample.get("class_hint")
    remaining_error_classes = sample.get("remaining_error_classes")
    
    print(f"  Input points shape: {points.shape}")
    print(f"  Input labels shape: {labels.shape}")
    print(f"  Query points shape: {query_points.shape}")
    print(f"  Query labels shape: {query_labels.shape}")
    
    # Statistics
    unique_labels = torch.unique(labels).numpy()
    unique_query_labels = torch.unique(query_labels).numpy()
    print(f"  Unique input labels: {unique_labels}")
    print(f"  Unique query labels: {unique_query_labels}")
    print(f"  Input label distribution:")
    for label in unique_labels:
        count = (labels == label).sum().item()
        print(f"    Label {label}: {count} points ({count/len(labels)*100:.1f}%)")
    
    print(f"  Query label distribution:")
    for label in unique_query_labels:
        count = (query_labels == label).sum().item()
        print(f"    Label {label}: {count} points ({count/len(query_labels)*100:.1f}%)")
    
    # Error class information is present only when the dataset was built with those fields.
    if class_hint is None or remaining_error_classes is None:
        print("  Class-hint fields are not in this sample.")
        return None
    input_error_classes = torch.where(class_hint > 0.5)[0].numpy() + 1  # Convert to 1-indexed
    remaining_classes = torch.where(remaining_error_classes > 0.5)[0].numpy() + 1
    manipulated_classes = np.setdiff1d(input_error_classes, remaining_classes)
    print(f"  Input error classes: {input_error_classes.tolist()}")
    print(f"  Remaining error classes (prediction target): {remaining_classes.tolist()}")
    print(f"  Manipulated error classes: {manipulated_classes.tolist()}")
    
    # Text instruction
    if text:
        text_preview = text[:100] + "..." if len(text) > 100 else text
        print(f"  Text instruction: {text_preview}")
    else:
        print(f"  Text instruction: (empty)")
    
    # Validate label differences
    validation_result = validate_label_differences(
        input_labels=labels,
        query_labels=query_labels,
        class_hint=class_hint,
        remaining_error_classes=remaining_error_classes,
        index=index
    )
    
    return validation_result


def test_dataset_loading(
    root_dir: str,
    split_path: Path,
    num_samples: int = 5,
    input_points: int = 2048,
    query_points: int = 2048,
    input_surface_ratio: float = 0.2,
    input_fg_ratio: float = 0.5,
    input_random_ratio: float = 0.3,
    query_surface_ratio: float = 0.2,
    query_fg_ratio: float = 0.5,
    query_random_ratio: float = 0.3,
    instructions_dir: str = None,
    output_dir: Path = None,
    seed: int = 42,
):
    """Test dataset loading and visualize samples."""
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("="*80)
    print("Testing Dataset Loading")
    print("="*80)
    
    # Build datasets
    print("\nBuilding datasets...")
    train_set, val_set, test_set = dataset.build_datasets(
        root_dir=root_dir,
        split_path=split_path,
        input_points=input_points,
        seed=seed,
        input_fg_ratio=input_fg_ratio,
        input_random_ratio=input_random_ratio,
        query_points=query_points,
        query_fg_ratio=query_fg_ratio,
        query_random_ratio=query_random_ratio,
        instructions_dir=instructions_dir,
    )
    
    print(f"  Train set size: {len(train_set)}")
    print(f"  Val set size: {len(val_set)}")
    print(f"  Test set size: {len(test_set)}")
    
    # Test with validation set
    test_dataset = val_set
    print(f"\nTesting with validation set ({len(test_dataset)} samples)")
    
    # Test loading a few samples
    indices_to_test = list(range(min(num_samples, len(test_dataset))))
    
    for i, idx in enumerate(indices_to_test):
        try:
            print(f"\nLoading sample {idx}...")
            sample = test_dataset[idx]
            
            # Print sample information and validate label differences
            validation_result = print_sample_info(sample, idx)
            
            # Check validation result
            if validation_result is None:
                print(f"  Class-hint validation skipped for sample {idx}")
            elif not validation_result.get("validation_passed", False):
                print(f"  ⚠️  Validation warnings detected for sample {idx}")
            else:
                print(f"  ✓ Validation passed for sample {idx}")
            
            # Extract data for visualization
            input_points = sample["points"].numpy()  # (N, 4) - coords + label feature
            input_labels = sample["labels"].numpy()  # (N,)
            query_points_coords = sample["query_points"].numpy()  # (M, 3)
            query_labels = sample["query_labels"].numpy()  # (M,)
            
            # Extract coordinates from input points (first 3 dimensions)
            input_coords = input_points[:, :3]
            
            # Visualize
            title = f"Sample_{idx}"
            # Always save visualizations (default to current directory if output_dir not specified)
            if output_dir:
                save_path = output_dir / f"sample_{idx}_visualization.png"
            else:
                # Default: save to current directory
                save_path = Path(f"sample_{idx}_visualization.png")
            
            visualize_point_clouds_side_by_side(
                input_coords=input_coords,
                input_labels=input_labels,
                query_coords=query_points_coords,
                query_labels=query_labels,
                title_prefix=title,
                save_path=save_path,
                subsample=5000,
            )
            
            print(f"  ✓ Sample {idx} loaded and visualized successfully")
            
        except Exception as e:
            print(f"  ✗ Error loading sample {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Dataset loading test completed!")
    print(f"{'='*80}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Test dataset loading and visualize point clouds")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/pointclouds",
        help="Root directory containing volume NPZ files",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default="data/splits/fold_1.json",
        help="Path to split JSON file",
    )
    parser.add_argument(
        "--instructions_dir",
        type=str,
        default="data/instructions_generated",
        help="Directory containing instruction JSON files (default: root_dir.parent / instructions_generated)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to test and visualize",
    )
    parser.add_argument(
        "--input_points",
        type=int,
        default=7000,
        help="Number of input points to sample",
    )
    parser.add_argument(
        "--query_points",
        type=int,
        default=7000,
        help="Number of query points to sample",
    )
    parser.add_argument(
        "--input_surface_ratio",
        type=float,
        default=0,
        help="Ratio of surface points in input",
    )
    parser.add_argument(
        "--input_fg_ratio",
        type=float,
        default=1,
        help="Ratio of foreground points in input",
    )
    parser.add_argument(
        "--input_random_ratio",
        type=float,
        default=0,
        help="Ratio of random points in input",
    )
    parser.add_argument(
        "--query_surface_ratio",
        type=float,
        default=0,
        help="Ratio of surface points in query",
    )
    parser.add_argument(
        "--query_fg_ratio",
        type=float,
        default=1,
        help="Ratio of foreground points in query",
    )
    parser.add_argument(
        "--query_random_ratio",
        type=float,
        default=0,
        help="Ratio of random points in query",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: save to current directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def main():
    args = _parse_args()
    
    # Validate ratios
    input_ratios_sum = args.input_surface_ratio + args.input_fg_ratio + args.input_random_ratio
    query_ratios_sum = args.query_surface_ratio + args.query_fg_ratio + args.query_random_ratio
    
    if abs(input_ratios_sum - 1.0) > 1e-6:
        raise ValueError(f"Input ratios must sum to 1.0, got {input_ratios_sum}")
    if abs(query_ratios_sum - 1.0) > 1e-6:
        raise ValueError(f"Query ratios must sum to 1.0, got {query_ratios_sum}")
    
    # Setup paths
    root_dir = args.root_dir
    split_path = Path(args.split_path)
    instructions_dir = args.instructions_dir
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if not split_path.exists():
        raise FileNotFoundError(f"Split file {split_path} not found")
    
    # Run test
    test_dataset_loading(
        root_dir=root_dir,
        split_path=split_path,
        num_samples=args.num_samples,
        input_points=args.input_points,
        query_points=args.query_points,
        input_surface_ratio=args.input_surface_ratio,
        input_fg_ratio=args.input_fg_ratio,
        input_random_ratio=args.input_random_ratio,
        query_surface_ratio=args.query_surface_ratio,
        query_fg_ratio=args.query_fg_ratio,
        query_random_ratio=args.query_random_ratio,
        instructions_dir=instructions_dir,
        output_dir=output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
