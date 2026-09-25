"""
Analyze corrected error instances from inference_volume.py results.

This script:
1. Loads all results.npz files from the result directory
2. Filters instances where is_remaining_error == False (successfully corrected)
3. Groups by error_type (error category)
4. Computes average Dice and F1 scores for pass 1 and pass 2 for each category
5. Outputs aggregated results
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_results_from_directory(result_dir: Path) -> List[Dict]:
    """
    Load all results.npz files from subdirectories in result_dir.
    
    Args:
        result_dir: Root directory containing sample subfolders
        
    Returns:
        List of dictionaries, each containing data from one results.npz file
    """
    results = []
    
    # Find all subdirectories
    subdirs = [d for d in result_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(subdirs)} sample directories in {result_dir}", flush=True)
    
    for subdir in subdirs:
        result_file = subdir / "results.npz"
        if result_file.exists():
            try:
                data = np.load(result_file, allow_pickle=True)
                
                # Check if instance arrays exist
                required_keys = ['instance_classes', 'instance_error_types', 'instance_is_remaining_error',
                               'instance_dice_pass1', 'instance_f1_pass1', 'instance_dice_pass2', 'instance_f1_pass2']
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    print(f"  Warning: Missing keys in {result_file}: {missing_keys}", flush=True)
                    continue
                
                # Extract instance-level data
                instance_classes = data['instance_classes']
                instance_error_types = data['instance_error_types']
                instance_is_remaining = data['instance_is_remaining_error']
                instance_dice_pass1 = data['instance_dice_pass1']
                instance_f1_pass1 = data['instance_f1_pass1']
                instance_dice_pass2 = data['instance_dice_pass2']
                instance_f1_pass2 = data['instance_f1_pass2']
                
                # Get num_instances if available, otherwise compute from array length
                if 'num_instances' in data:
                    num_instances = int(data['num_instances'])
                else:
                    # Compute from array length (handle both scalars and arrays)
                    if instance_classes.ndim == 0:
                        num_instances = 1
                    else:
                        num_instances = len(instance_classes)
                
                # Handle case where there are no instances
                if num_instances == 0:
                    continue
                
                # Convert numpy arrays to lists for easier handling
                # Handle both 1D arrays and 0D scalars
                def to_list(arr, converter):
                    """Convert numpy array to list, handling scalars and arrays."""
                    if arr.ndim == 0:
                        return [converter(arr.item())]
                    else:
                        return [converter(x) for x in arr]
                
                instance_classes = to_list(instance_classes, int)
                instance_error_types = to_list(instance_error_types, str)
                instance_is_remaining = to_list(instance_is_remaining, bool)
                instance_dice_pass1 = to_list(instance_dice_pass1, float)
                instance_f1_pass1 = to_list(instance_f1_pass1, float)
                instance_dice_pass2 = to_list(instance_dice_pass2, float)
                instance_f1_pass2 = to_list(instance_f1_pass2, float)
                
                results.append({
                    'sample_name': subdir.name,
                    'num_instances': num_instances,
                    'instance_classes': instance_classes,
                    'instance_error_types': instance_error_types,
                    'instance_is_remaining': instance_is_remaining,
                    'instance_dice_pass1': instance_dice_pass1,
                    'instance_f1_pass1': instance_f1_pass1,
                    'instance_dice_pass2': instance_dice_pass2,
                    'instance_f1_pass2': instance_f1_pass2,
                })
            except Exception as e:
                print(f"  Warning: Failed to load {result_file}: {e}", flush=True)
                continue
        else:
            print(f"  Warning: No results.npz found in {subdir}", flush=True)
    
    print(f"Successfully loaded {len(results)} result files", flush=True)
    return results


def aggregate_corrected_instances(results: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate corrected instances (is_remaining_error == False) by error type.
    
    Args:
        results: List of result dictionaries from load_results_from_directory
        
    Returns:
        Dictionary mapping error_type to aggregated statistics:
        {
            'error_type': {
                'count': int,
                'dice_pass1': List[float],
                'f1_pass1': List[float],
                'dice_pass2': List[float],
                'f1_pass2': List[float],
            }
        }
    """
    aggregated = defaultdict(lambda: {
        'count': 0,
        'dice_pass1': [],
        'f1_pass1': [],
        'dice_pass2': [],
        'f1_pass2': [],
    })
    
    total_instances = 0
    corrected_instances = 0
    
    for result in results:
        num_instances = result['num_instances']
        total_instances += num_instances
        
        for i in range(num_instances):
            is_remaining = result['instance_is_remaining'][i]
            
            # Only aggregate instances that are NOT remaining errors (i.e., successfully corrected)
            if not is_remaining:
                corrected_instances += 1
                error_type = result['instance_error_types'][i]
                
                aggregated[error_type]['count'] += 1
                aggregated[error_type]['dice_pass1'].append(result['instance_dice_pass1'][i])
                aggregated[error_type]['f1_pass1'].append(result['instance_f1_pass1'][i])
                aggregated[error_type]['dice_pass2'].append(result['instance_dice_pass2'][i])
                aggregated[error_type]['f1_pass2'].append(result['instance_f1_pass2'][i])
    
    print(f"\nTotal instances across all samples: {total_instances}", flush=True)
    print(f"Corrected instances (is_remaining_error == False): {corrected_instances}", flush=True)
    print(f"Remaining instances (is_remaining_error == True): {total_instances - corrected_instances}", flush=True)
    
    return dict(aggregated)


def compute_statistics(aggregated: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compute average statistics for each error type.
    
    Args:
        aggregated: Dictionary from aggregate_corrected_instances
        
    Returns:
        Dictionary mapping error_type to statistics:
        {
            'error_type': {
                'count': int,
                'avg_dice_pass1': float,
                'avg_f1_pass1': float,
                'avg_dice_pass2': float,
                'avg_f1_pass2': float,
                'std_dice_pass1': float,
                'std_f1_pass1': float,
                'std_dice_pass2': float,
                'std_f1_pass2': float,
            }
        }
    """
    statistics = {}
    
    for error_type, data in aggregated.items():
        count = data['count']
        
        if count == 0:
            continue
        
        dice_pass1 = np.array(data['dice_pass1'])
        f1_pass1 = np.array(data['f1_pass1'])
        dice_pass2 = np.array(data['dice_pass2'])
        f1_pass2 = np.array(data['f1_pass2'])
        
        statistics[error_type] = {
            'count': count,
            'avg_dice_pass1': float(np.mean(dice_pass1)),
            'avg_f1_pass1': float(np.mean(f1_pass1)),
            'avg_dice_pass2': float(np.mean(dice_pass2)),
            'avg_f1_pass2': float(np.mean(f1_pass2)),
            'std_dice_pass1': float(np.std(dice_pass1)),
            'std_f1_pass1': float(np.std(f1_pass1)),
            'std_dice_pass2': float(np.std(dice_pass2)),
            'std_f1_pass2': float(np.std(f1_pass2)),
        }
    
    return statistics


def print_results(statistics: Dict[str, Dict]):
    """
    Print aggregated results in a formatted table.
    
    Args:
        statistics: Dictionary from compute_statistics
    """
    print("\n" + "="*100, flush=True)
    print("AGGREGATED RESULTS FOR CORRECTED INSTANCES (is_remaining_error == False)", flush=True)
    print("="*100, flush=True)
    print(f"\n{'Error Type':<30} | {'Count':<8} | {'Avg Dice P1':<12} | {'Avg F1 P1':<12} | {'Avg Dice P2':<12} | {'Avg F1 P2':<12}", flush=True)
    print("-"*100, flush=True)
    
    # Sort by error type for consistent output
    sorted_types = sorted(statistics.keys())
    
    for error_type in sorted_types:
        stats = statistics[error_type]
        print(f"{error_type:<30} | {stats['count']:>8} | {stats['avg_dice_pass1']:>12.4f} | {stats['avg_f1_pass1']:>12.4f} | {stats['avg_dice_pass2']:>12.4f} | {stats['avg_f1_pass2']:>12.4f}", flush=True)
    
    print("-"*100, flush=True)
    
    # Print summary statistics
    total_count = sum(s['count'] for s in statistics.values())
    if total_count > 0:
        overall_avg_dice_pass1 = np.mean([s['avg_dice_pass1'] for s in statistics.values()])
        overall_avg_f1_pass1 = np.mean([s['avg_f1_pass1'] for s in statistics.values()])
        overall_avg_dice_pass2 = np.mean([s['avg_dice_pass2'] for s in statistics.values()])
        overall_avg_f1_pass2 = np.mean([s['avg_f1_pass2'] for s in statistics.values()])
        
        print(f"\n{'OVERALL AVERAGE':<30} | {total_count:>8} | {overall_avg_dice_pass1:>12.4f} | {overall_avg_f1_pass1:>12.4f} | {overall_avg_dice_pass2:>12.4f} | {overall_avg_f1_pass2:>12.4f}", flush=True)
    
    print("="*100, flush=True)


def save_results_to_csv(statistics: Dict[str, Dict], output_path: Path):
    """
    Save aggregated results to a CSV file.
    
    Args:
        statistics: Dictionary from compute_statistics
        output_path: Path to save CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Error Type',
            'Count',
            'Avg Dice Pass1',
            'Std Dice Pass1',
            'Avg F1 Pass1',
            'Std F1 Pass1',
            'Avg Dice Pass2',
            'Std Dice Pass2',
            'Avg F1 Pass2',
            'Std F1 Pass2',
        ])
        
        # Sort by error type for consistent output
        sorted_types = sorted(statistics.keys())
        
        # Write data rows
        for error_type in sorted_types:
            stats = statistics[error_type]
            writer.writerow([
                error_type,
                stats['count'],
                f"{stats['avg_dice_pass1']:.6f}",
                f"{stats['std_dice_pass1']:.6f}",
                f"{stats['avg_f1_pass1']:.6f}",
                f"{stats['std_f1_pass1']:.6f}",
                f"{stats['avg_dice_pass2']:.6f}",
                f"{stats['std_dice_pass2']:.6f}",
                f"{stats['avg_f1_pass2']:.6f}",
                f"{stats['std_f1_pass2']:.6f}",
            ])
    
    print(f"\nResults saved to CSV: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze corrected error instances from inference_volume.py results."
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=False,
        default="results/instances",
        help="Directory containing correction results (subfolders with results.npz files).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/instances/res.csv",
        help="Optional path to save results as CSV file.",
    )
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"ERROR: Result directory does not exist: {result_dir}", flush=True)
        return
    
    print(f"Loading results from: {result_dir}", flush=True)
    
    # Load all results
    results = load_results_from_directory(result_dir)
    
    if len(results) == 0:
        print("ERROR: No results found!", flush=True)
        return
    
    # Aggregate corrected instances by error type
    aggregated = aggregate_corrected_instances(results)
    
    if len(aggregated) == 0:
        print("WARNING: No corrected instances found (all instances have is_remaining_error == True)", flush=True)
        return
    
    # Compute statistics
    statistics = compute_statistics(aggregated)
    
    # Print results
    print_results(statistics)
    
    # Save to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        save_results_to_csv(statistics, output_path)
    
    print("\nAnalysis completed!", flush=True)


if __name__ == "__main__":
    main()
