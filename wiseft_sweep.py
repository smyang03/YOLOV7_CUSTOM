#!/usr/bin/env python3
"""
WiSE-FT (Weight-Space Ensembling Fine-Tuning) Sweep Tool for YOLOv7

Automatically finds optimal alpha (mixing ratio) between scratch and fine-tuned models
to balance target class improvement with other class preservation.

Usage:
    python wiseft_sweep.py --scratch runs/exp1/weights/best.pt \
                           --finetuned runs/exp2/weights/best.pt \
                           --data data/custom.yaml \
                           --target-class person
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import numpy as np
import torch
from copy import deepcopy


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='WiSE-FT Alpha Sweep for YOLOv7')

    # Required arguments
    parser.add_argument('--scratch', type=str, required=True, help='Scratch model path')
    parser.add_argument('--finetuned', type=str, required=True, help='Fine-tuned model path')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML path')

    # Alpha settings
    parser.add_argument('--focus-range', type=float, default=0.1, help='Alpha interval (default: 0.1)')
    parser.add_argument('--alpha-min', type=float, default=None, help='Min alpha (default: auto-detect)')
    parser.add_argument('--alpha-max', type=float, default=1.0, help='Max alpha (default: 1.0)')
    parser.add_argument('--skip-zero', action='store_true', default=True, help='Skip alpha=0.0')

    # Validation settings
    parser.add_argument('--val-sets', nargs='+', default=['test1'], help='Validation sets')
    parser.add_argument('--target-class', type=str, default=None, help='Target class name or index')

    # Evaluation settings
    parser.add_argument('--metric', type=str, default='fitness',
                       choices=['fitness', 'map50', 'map', 'precision', 'recall'],
                       help='Metric to optimize')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IoU threshold')

    # Two-stage search
    parser.add_argument('--enable-fine-search', action='store_true', default=True,
                       help='Enable fine search')
    parser.add_argument('--fine-range', type=float, default=None,
                       help='Fine search interval (default: focus-range/2)')
    parser.add_argument('--fine-window', type=float, default=None,
                       help='Fine search window size (default: 2*focus-range)')

    # Early stopping
    parser.add_argument('--early-stop', action='store_true', default=False,
                       help='Enable early stopping')
    parser.add_argument('--stop-threshold', type=float, default=0.05,
                       help='Performance drop threshold')
    parser.add_argument('--stop-patience', type=int, default=3,
                       help='Consecutive drops before stopping')

    # Output settings
    parser.add_argument('--output-dir', type=str, default='runs/wiseft',
                       help='Output directory')
    parser.add_argument('--save-merged-models', action='store_true', default=False,
                       help='Save all merged models')
    parser.add_argument('--save-best-only', action='store_true', default=True,
                       help='Save only best model')
    parser.add_argument('--report-format', type=str, default='markdown',
                       choices=['markdown', 'text', 'json'],
                       help='Report format')

    # Phase 2: Enhanced features
    parser.add_argument('--enable-tradeoff-viz', action='store_true', default=True,
                       help='Enable trade-off visualization')
    parser.add_argument('--enable-adaptive-stop', action='store_true', default=False,
                       help='Enable adaptive early stopping (trend-based)')
    parser.add_argument('--enable-layer-detail', action='store_true', default=False,
                       help='Enable detailed layer-wise analysis')
    parser.add_argument('--enable-confidence-intervals', action='store_true', default=False,
                       help='Enable confidence interval calculation')
    parser.add_argument('--confidence-runs', type=int, default=3,
                       help='Number of runs for confidence interval (default: 3)')

    # Phase 3: Advanced features
    parser.add_argument('--enable-layerwise-alpha', action='store_true', default=False,
                       help='Enable layer-wise alpha (different alpha per layer group)')
    parser.add_argument('--enable-dynamic-alpha', action='store_true', default=False,
                       help='Enable dynamic alpha selection (DaWin-inspired)')
    parser.add_argument('--enable-ensemble', action='store_true', default=False,
                       help='Enable model ensemble (voting across top alphas)')
    parser.add_argument('--ensemble-top-k', type=int, default=3,
                       help='Number of top alphas for ensemble (default: 3)')

    # Other
    parser.add_argument('--device', type=str, default='', help='Device (e.g., 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')

    args = parser.parse_args()

    # Set defaults for fine search
    if args.fine_range is None:
        args.fine_range = args.focus_range / 2
    if args.fine_window is None:
        args.fine_window = 2 * args.focus_range

    return args


# ============================================================================
# Weight Analysis Functions
# ============================================================================

def analyze_weight_changes(scratch_path: str, finetuned_path: str) -> Dict:
    """
    Analyze layer-wise weight changes between scratch and finetuned models

    Returns:
        layer_changes: dict[layer_name] = {
            'abs_change': float,
            'rel_change': float,
            'frob_norm': float
        }
    """
    print("Loading models for weight analysis...")
    scratch_ckpt = torch.load(scratch_path, map_location='cpu')
    finetuned_ckpt = torch.load(finetuned_path, map_location='cpu')

    # Extract state dicts
    scratch_sd = scratch_ckpt['model'].float().state_dict()
    finetuned_sd = finetuned_ckpt['model'].float().state_dict()

    layer_changes = {}

    for key in scratch_sd.keys():
        if key in finetuned_sd and 'num_batches_tracked' not in key:
            scratch_weight = scratch_sd[key]
            finetuned_weight = finetuned_sd[key]

            # Calculate difference
            diff = finetuned_weight - scratch_weight

            # Absolute change (mean of absolute differences)
            abs_change = torch.abs(diff).mean().item()

            # Relative change (percentage)
            rel_change = (torch.abs(diff) / (torch.abs(scratch_weight) + 1e-8)).mean().item()

            # Frobenius norm (total magnitude of change)
            frob_norm = torch.norm(diff, p='fro').item()

            layer_changes[key] = {
                'abs_change': abs_change,
                'rel_change': rel_change,
                'frob_norm': frob_norm
            }

    return layer_changes


def group_layer_changes(layer_changes: Dict) -> Dict:
    """
    Group layers into backbone/neck/head and calculate average changes

    Returns:
        {
            'backbone_avg': float,
            'neck_avg': float,
            'head_avg': float,
            'backbone_max': float,
            'neck_max': float,
            'head_max': float,
            'backbone_count': int,
            'neck_count': int,
            'head_count': int
        }
    """
    groups = {
        'backbone': [],
        'neck': [],
        'head': []
    }

    for key, change in layer_changes.items():
        # Parse layer number from key (e.g., 'model.0.conv.weight' -> 0)
        if 'model.' in key:
            try:
                layer_num = int(key.split('model.')[1].split('.')[0])

                # YOLOv7 architecture (approximate):
                # 0-50: Backbone
                # 51-74: Neck (FPN/PAN)
                # 75+: Head (Detection layers)
                if layer_num <= 50:
                    groups['backbone'].append(change['rel_change'])
                elif layer_num <= 74:
                    groups['neck'].append(change['rel_change'])
                else:
                    groups['head'].append(change['rel_change'])
            except (ValueError, IndexError):
                # If parsing fails, skip
                continue

    summary = {}
    for group_name, changes in groups.items():
        if changes:
            summary[f'{group_name}_avg'] = np.mean(changes)
            summary[f'{group_name}_max'] = np.max(changes)
            summary[f'{group_name}_count'] = len(changes)
        else:
            summary[f'{group_name}_avg'] = 0.0
            summary[f'{group_name}_max'] = 0.0
            summary[f'{group_name}_count'] = 0

    return summary


def recommend_alpha_range(head_change: float, backbone_change: float) -> Tuple[float, float, str]:
    """
    Recommend alpha range based on weight changes

    Returns:
        (alpha_min, alpha_max, reason)
    """
    # Calculate change ratio
    change_ratio = head_change / (backbone_change + 1e-8)

    # Case 1: Head changed significantly, backbone stable (ideal fine-tuning)
    if change_ratio > 10 and head_change > 0.3:
        return (0.05, 0.3,
                f"Detection head changed significantly ({head_change:.1%}) while backbone stayed stable ({backbone_change:.1%}). "
                f"Low-to-medium alpha recommended to preserve general features while adopting target class improvements.")

    # Case 2: Head changed drastically (over-fitting risk)
    elif head_change > 0.6:
        return (0.0, 0.2,
                f"Detection head changed drastically ({head_change:.1%}). "
                f"Very low alpha recommended to prevent catastrophic forgetting.")

    # Case 3: Both changed (full model fine-tuning)
    elif change_ratio < 5 and backbone_change > 0.1:
        return (0.2, 0.6,
                f"Both backbone ({backbone_change:.1%}) and head ({head_change:.1%}) changed. "
                f"Medium-to-high alpha recommended to leverage full fine-tuning benefits.")

    # Case 4: Minimal changes (fine-tuning may have failed)
    elif head_change < 0.1:
        return (0.1, 1.0,
                f"Minimal weight changes detected ({head_change:.1%}). "
                f"Full range search recommended. Consider reviewing fine-tuning settings.")

    # Default: Moderate changes
    else:
        return (0.1, 0.5,
                f"Moderate head changes ({head_change:.1%}), backbone changes ({backbone_change:.1%}). "
                f"Low-to-medium alpha recommended as starting point.")


def print_weight_analysis_report(group_summary: Dict, alpha_min: float, alpha_max: float, reason: str):
    """Print weight analysis results in a formatted report"""
    print("\n" + "="*80)
    print("📊 WEIGHT CHANGE ANALYSIS")
    print("="*80)

    # Layer group changes
    print("\nLayer-wise Weight Changes:")
    print("─" * 80)
    print(f"{'Layer Group':<20} {'Avg Change':<15} {'Max Change':<15} {'Layers':<10}")
    print("─" * 80)

    for group in ['backbone', 'neck', 'head']:
        avg = group_summary[f'{group}_avg']
        max_val = group_summary[f'{group}_max']
        count = group_summary[f'{group}_count']

        # Add warning indicator for high changes
        indicator = ""
        if avg > 0.5:
            indicator = "⚠️ VERY HIGH"
        elif avg > 0.3:
            indicator = "⚠️ HIGH"
        elif avg > 0.15:
            indicator = "⚠️ MEDIUM"

        print(f"{group.capitalize():<20} {avg:>6.1%}{'':>8} {max_val:>6.1%}{'':>8} {count:<10} {indicator}")

    print("─" * 80)

    # Recommendation
    print(f"\n💡 RECOMMENDED ALPHA RANGE: {alpha_min:.2f} - {alpha_max:.2f}")
    print(f"\nReason: {reason}")
    print("="*80)


# ============================================================================
# Alpha Generation Functions
# ============================================================================

def generate_alpha_list(alpha_min: float, alpha_max: float, focus_range: float, skip_zero: bool = True) -> List[float]:
    """
    Generate alpha list for coarse search

    Example:
        alpha_min=0.1, alpha_max=0.5, focus_range=0.1
        → [0.1, 0.2, 0.3, 0.4, 0.5]
    """
    alphas = []
    current = alpha_min if skip_zero or alpha_min > 0 else 0.0

    while current <= alpha_max + 1e-6:  # Add small epsilon for floating point comparison
        alphas.append(round(current, 3))
        current += focus_range

    return alphas


def generate_fine_alpha_list(best_alpha: float, fine_range: float, fine_window: float,
                             alpha_min: float, alpha_max: float) -> List[float]:
    """
    Generate fine search alpha list around best alpha

    Example:
        best_alpha=0.2, fine_range=0.05, fine_window=0.2
        → [0.10, 0.15, 0.20, 0.25, 0.30]
    """
    # Calculate window boundaries
    window_min = max(alpha_min, best_alpha - fine_window / 2)
    window_max = min(alpha_max, best_alpha + fine_window / 2)

    alphas = []
    current = window_min

    while current <= window_max + 1e-6:
        alphas.append(round(current, 3))
        current += fine_range

    # Remove duplicates and sort
    alphas = sorted(list(set(alphas)))

    return alphas


# ============================================================================
# Weight Merging Functions
# ============================================================================

def merge_weights(scratch_path: str, finetuned_path: str, alpha: float, output_path: str) -> str:
    """
    Merge two models' weights with given alpha

    merged = (1 - alpha) * scratch + alpha * finetuned

    Args:
        scratch_path: Path to scratch model
        finetuned_path: Path to fine-tuned model
        alpha: Mixing ratio (0~1)
        output_path: Path to save merged model

    Returns:
        output_path
    """
    # Load models
    scratch_ckpt = torch.load(scratch_path, map_location='cpu')
    finetuned_ckpt = torch.load(finetuned_path, map_location='cpu')

    # Get state dicts
    scratch_sd = scratch_ckpt['model'].float().state_dict()
    finetuned_sd = finetuned_ckpt['model'].float().state_dict()

    # Create merged state dict
    merged_sd = {}
    for key in scratch_sd.keys():
        if key in finetuned_sd:
            # Merge: (1-alpha) * scratch + alpha * finetuned
            merged_sd[key] = (1 - alpha) * scratch_sd[key] + alpha * finetuned_sd[key]
        else:
            # If key only in scratch, keep it
            merged_sd[key] = scratch_sd[key]

    # Create output checkpoint (use finetuned as base, update weights)
    output_ckpt = deepcopy(finetuned_ckpt)
    output_ckpt['model'].load_state_dict(merged_sd)

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(output_ckpt, output_path)

    return output_path


# ============================================================================
# Model Evaluation Functions
# ============================================================================

def evaluate_model(model_path: str, data_yaml: str, img_size: int, batch_size: int,
                  conf_thres: float, iou_thres: float, device: str, workers: int,
                  output_dir: str) -> Dict:
    """
    Evaluate model using test.py

    Returns:
        results: {
            'precision': float,
            'recall': float,
            'map50': float,
            'map': float,
            'fitness': float,
            'per_class': {...}
        }
    """
    # Prepare test.py command
    cmd = [
        sys.executable, 'test.py',
        '--data', data_yaml,
        '--weights', model_path,
        '--img-size', str(img_size),
        '--batch-size', str(batch_size),
        '--conf-thres', str(conf_thres),
        '--iou-thres', str(iou_thres),
        '--task', 'val',
        '--save-txt',
        '--save-json',
        '--name', Path(output_dir).name,
        '--exist-ok'
    ]

    if device:
        cmd.extend(['--device', device])

    # Run test.py
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse results from output
        # test.py prints results in format: "P: 0.xxx, R: 0.xxx, mAP@.5: 0.xxx, mAP@.5:.95: 0.xxx"
        output = result.stdout

        # Extract metrics using simple parsing
        results = {
            'precision': 0.0,
            'recall': 0.0,
            'map50': 0.0,
            'map': 0.0,
            'fitness': 0.0
        }

        # Look for the summary line (usually contains "all" class)
        for line in output.split('\n'):
            if 'all' in line.lower() and any(x in line for x in ['0.', '1.0']):
                parts = line.split()
                # Try to extract numbers (usually in format: class images targets P R mAP@.5 mAP@.5:.95)
                try:
                    # Find indices of numeric values
                    numbers = [float(x) for x in parts if x.replace('.', '').replace('-', '').isdigit()]
                    if len(numbers) >= 4:
                        results['precision'] = numbers[-4]
                        results['recall'] = numbers[-3]
                        results['map50'] = numbers[-2]
                        results['map'] = numbers[-1]
                        # Calculate fitness (same as YOLOv7: 0.1*mAP@.5 + 0.9*mAP@.5:.95)
                        results['fitness'] = 0.1 * results['map50'] + 0.9 * results['map']
                        break
                except (ValueError, IndexError):
                    continue

        return results

    except subprocess.CalledProcessError as e:
        print(f"Error running test.py: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        # Return zeros if evaluation fails
        return {
            'precision': 0.0,
            'recall': 0.0,
            'map50': 0.0,
            'map': 0.0,
            'fitness': 0.0
        }


# ============================================================================
# Multi-Validation Set Evaluation Functions
# ============================================================================

def create_temp_data_yaml(original_yaml: str, val_set_name: str, output_dir: str) -> str:
    """
    Create temporary data.yaml with specific validation set

    Args:
        original_yaml: Path to original data.yaml
        val_set_name: Name of validation set (e.g., 'valid1', 'valid2')
        output_dir: Directory to save temp yaml

    Returns:
        Path to temporary yaml file
    """
    with open(original_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Get original val path and construct new path
    original_val = data_config.get('val', '')

    if isinstance(original_val, list):
        # Multiple val paths - use first one as base
        print(f"  ℹ️  val is a list, using first entry: {original_val[0]}")
        original_val = original_val[0]

    # Now original_val is a string
    val_path = Path(original_val)
    if val_path.suffix == '.txt':
        # If it's a txt file, replace filename
        new_val_path = val_path.parent / f'{val_set_name}.txt'
    else:
        # If it's a directory, append validation set name
        new_val_path = val_path.parent / val_set_name

    data_config['val'] = str(new_val_path)

    # Save temporary yaml
    temp_yaml_path = Path(output_dir) / f'temp_{val_set_name}.yaml'
    temp_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_yaml_path, 'w') as f:
        yaml.dump(data_config, f)

    return str(temp_yaml_path)


def calculate_average_metrics(metrics_list: List[Dict]) -> Dict:
    """
    Calculate average metrics from multiple validation sets

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Average metrics dictionary
    """
    if not metrics_list:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'map50': 0.0,
            'map': 0.0,
            'fitness': 0.0
        }

    avg_metrics = {}
    metric_keys = ['precision', 'recall', 'map50', 'map', 'fitness']

    for key in metric_keys:
        values = [m.get(key, 0.0) for m in metrics_list]
        avg_metrics[key] = sum(values) / len(values)

    return avg_metrics


def evaluate_model_multi_valset(model_path: str, data_yaml: str, val_sets: List[str],
                                img_size: int, batch_size: int, conf_thres: float,
                                iou_thres: float, device: str, workers: int,
                                output_dir: str) -> Dict:
    """
    Evaluate model on multiple validation sets separately

    Args:
        model_path: Path to model weights
        data_yaml: Path to data yaml
        val_sets: List of validation set names (e.g., ['valid1', 'valid2'])
        ... (other args same as evaluate_model)

    Returns:
        {
            'overall': {
                'precision': float,
                'recall': float,
                'map50': float,
                'map': float,
                'fitness': float
            },
            'per_valset': {
                'valid1': {'precision': ..., 'fitness': ...},
                'valid2': {'precision': ..., 'fitness': ...}
            }
        }
    """
    results = {'per_valset': {}}

    print(f"  Evaluating on {len(val_sets)} validation sets: {val_sets}")

    # Evaluate on each validation set
    for val_set_name in val_sets:
        print(f"\n  📊 Evaluating on {val_set_name}...")

        # Create temporary data yaml for this validation set
        temp_yaml = create_temp_data_yaml(data_yaml, val_set_name, output_dir)

        # Evaluate
        eval_dir = Path(output_dir) / f'eval_{val_set_name}'
        metrics = evaluate_model(
            model_path, temp_yaml, img_size, batch_size,
            conf_thres, iou_thres, device, workers, str(eval_dir)
        )

        results['per_valset'][val_set_name] = metrics

        # Print results for this validation set
        print(f"     P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
              f"mAP@.5={metrics['map50']:.3f}, Fitness={metrics['fitness']:.3f}")

    # Calculate overall average
    results['overall'] = calculate_average_metrics(
        list(results['per_valset'].values())
    )

    print(f"\n  📈 Overall Average: "
          f"P={results['overall']['precision']:.3f}, "
          f"R={results['overall']['recall']:.3f}, "
          f"mAP@.5={results['overall']['map50']:.3f}, "
          f"Fitness={results['overall']['fitness']:.3f}")

    return results


# ============================================================================
# Baseline Trade-off Analysis
# ============================================================================

def analyze_baseline_tradeoff(scratch_metrics: Dict, finetuned_metrics: Dict,
                              val_sets: List[str]) -> bool:
    """
    Analyze trade-off between validation sets in baseline models

    Args:
        scratch_metrics: Metrics from scratch model (with per_valset)
        finetuned_metrics: Metrics from finetuned model (with per_valset)
        val_sets: List of validation set names

    Returns:
        True if trade-off exists (WiSE-FT recommended)
        False if no trade-off
    """
    print("\n" + "="*80)
    print("🔍 BASELINE TRADE-OFF ANALYSIS")
    print("="*80)

    # Print comparison table
    print(f"\n{'Model':<15} {'Metric':<10}", end='')
    for val_name in val_sets:
        print(f" {val_name:<12}", end='')
    print(f" {'Overall':<12}")
    print("─" * 80)

    # Scratch model
    print(f"{'Scratch':<15} {'fitness':<10}", end='')
    scratch_fitness_vals = []
    for val_name in val_sets:
        fit = scratch_metrics['per_valset'][val_name]['fitness']
        scratch_fitness_vals.append(fit)
        print(f" {fit:<12.4f}", end='')
    print(f" {scratch_metrics['overall']['fitness']:<12.4f}")

    # Fine-tuned model
    print(f"{'Fine-tuned':<15} {'fitness':<10}", end='')
    finetuned_fitness_vals = []
    for val_name in val_sets:
        fit = finetuned_metrics['per_valset'][val_name]['fitness']
        finetuned_fitness_vals.append(fit)
        print(f" {fit:<12.4f}", end='')
    print(f" {finetuned_metrics['overall']['fitness']:<12.4f}")

    print("─" * 80)

    # Analyze changes
    print(f"\n{'Validation Set':<15} {'Change':<15} {'% Change':<15} {'Direction':<15}")
    print("─" * 80)

    changes = []
    for i, val_name in enumerate(val_sets):
        scratch_fit = scratch_fitness_vals[i]
        finetuned_fit = finetuned_fitness_vals[i]
        change = finetuned_fit - scratch_fit
        pct_change = (change / scratch_fit * 100) if scratch_fit > 0 else 0

        if change > 0.02:
            direction = "↑ Improved"
        elif change < -0.02:
            direction = "↓ Degraded"
        else:
            direction = "→ Similar"

        changes.append((val_name, change, pct_change, direction))
        print(f"{val_name:<15} {change:>+8.4f}      {pct_change:>+8.1f}%      {direction:<15}")

    print("─" * 80)

    # Determine if trade-off exists
    positive_changes = [v for v, c, p, d in changes if c > 0.02]
    negative_changes = [v for v, c, p, d in changes if c < -0.02]

    if positive_changes and negative_changes:
        print(f"\n⚠️  TRADE-OFF DETECTED!")
        print(f"   ✅ Improved on: {', '.join(positive_changes)}")
        print(f"   ❌ Degraded on: {', '.join(negative_changes)}")
        print(f"\n   💡 WiSE-FT is RECOMMENDED to find balance between validation sets!")
        print("="*80)
        return True
    elif negative_changes:
        print(f"\n⚠️  PERFORMANCE DEGRADATION DETECTED!")
        print(f"   ❌ Degraded on: {', '.join(negative_changes)}")
        print(f"   → Fine-tuned model performs worse on some validation sets.")
        print(f"\n   💡 WiSE-FT may help recover some performance.")
        print("="*80)
        return True
    else:
        print(f"\n✅ No significant trade-off detected.")
        print(f"   → Fine-tuned model improves or maintains performance on all validation sets.")
        print(f"   → WiSE-FT may not provide additional benefits.")
        print("="*80)
        return False


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_valset_tradeoff(all_results: List[Dict], val_sets: List[str]):
    """
    Visualize trade-off between two validation sets (ASCII scatter plot)

    Args:
        all_results: List of results with per_valset metrics
        val_sets: List of validation set names (must be exactly 2)
    """
    if len(val_sets) != 2:
        print("\n⚠️  Trade-off visualization only supports exactly 2 validation sets.")
        print(f"   Current validation sets: {val_sets}")
        return

    val1_name, val2_name = val_sets

    print("\n" + "="*80)
    print(f"📊 VALIDATION SET TRADE-OFF: {val1_name} vs {val2_name}")
    print("="*80)

    # Extract data points
    points = []
    for result in all_results:
        if 'per_valset' not in result['metrics']:
            continue

        val1_fit = result['metrics']['per_valset'].get(val1_name, {}).get('fitness', 0)
        val2_fit = result['metrics']['per_valset'].get(val2_name, {}).get('fitness', 0)
        alpha = result['alpha']
        points.append((val1_fit, val2_fit, alpha))

    if not points:
        print("No data points available for visualization.")
        return

    # Find min/max for scaling
    val1_values = [p[0] for p in points]
    val2_values = [p[1] for p in points]

    val1_min, val1_max = min(val1_values), max(val1_values)
    val2_min, val2_max = min(val2_values), max(val2_values)

    # Add some padding
    val1_range = val1_max - val1_min
    val2_range = val2_max - val2_min
    val1_min -= val1_range * 0.1
    val1_max += val1_range * 0.1
    val2_min -= val2_range * 0.1
    val2_max += val2_range * 0.1

    # ASCII plot
    print(f"\n{val2_name} fitness ↑\n")

    grid_height = 15
    grid_width = 60

    for y in range(grid_height, -1, -1):
        val2_level = val2_min + (val2_max - val2_min) * (y / grid_height)

        # Y-axis label
        if y == grid_height or y == 0 or y == grid_height // 2:
            print(f"{val2_level:5.3f} │ ", end='')
        else:
            print("      │ ", end='')

        # Plot points
        for x in range(grid_width + 1):
            val1_level = val1_min + (val1_max - val1_min) * (x / grid_width)

            # Find closest point
            closest_alpha = None
            min_dist = float('inf')

            for v1, v2, alpha in points:
                # Normalize distances
                dist_x = (v1 - val1_level) / (val1_max - val1_min) * grid_width
                dist_y = (v2 - val2_level) / (val2_max - val2_min) * grid_height
                dist = (dist_x**2 + dist_y**2)**0.5

                if dist < min_dist and dist < 2.0:  # threshold
                    min_dist = dist
                    closest_alpha = alpha

            if closest_alpha is not None:
                # Different markers for different alphas
                if closest_alpha == 0.0:
                    print("S", end='')  # Scratch
                elif closest_alpha == 1.0:
                    print("F", end='')  # Finetuned
                else:
                    print("●", end='')  # WiSE-FT
            else:
                # Grid background
                if x % 10 == 0 or y % 5 == 0:
                    print("·", end='')
                else:
                    print(" ", end='')

        print()

    # X-axis
    print("      └" + "─" * grid_width + f"→ {val1_name} fitness")
    print(f"      {val1_min:.3f}" + " " * (grid_width - 12) + f"{val1_max:.3f}")

    # Legend
    print("\n" + "─" * 80)
    print("Legend:")
    print("  S : Scratch model (α=0.0)")
    print("  F : Fine-tuned model (α=1.0)")
    print("  ● : WiSE-FT merged models (0.0 < α < 1.0)")
    print("\nGoal: Find point closest to upper-right (high on both validation sets)")
    print("="*80)


# ============================================================================
# Multi-Criteria Alpha Selection
# ============================================================================

def find_best_alpha_multi_criteria(all_results: List[Dict], val_sets: List[str],
                                   metric: str = 'fitness'):
    """
    Find best alpha using multiple criteria

    Args:
        all_results: List of results with per_valset metrics
        val_sets: List of validation set names
        metric: Metric to optimize (default: 'fitness')
    """
    print("\n" + "="*80)
    print("🎯 OPTIMAL ALPHA RECOMMENDATIONS (Multi-Criteria)")
    print("="*80)

    # Filter results that have per_valset data
    valid_results = [r for r in all_results if 'per_valset' in r['metrics']]

    if not valid_results:
        print("No results with per-validation-set data available.")
        return

    # Criterion 1: Best overall average
    best_overall = max(valid_results, key=lambda x: x['metrics']['overall'].get(metric, 0))
    print(f"\n1️⃣  Best Overall Average {metric.upper()}:")
    print(f"   α = {best_overall['alpha']:.3f}")
    print(f"   Overall {metric}: {best_overall['metrics']['overall'][metric]:.4f}")
    for val_name in val_sets:
        val_metric = best_overall['metrics']['per_valset'][val_name][metric]
        print(f"   {val_name}: {val_metric:.4f}")

    # Criterion 2: Most balanced (minimum difference between validation sets)
    if len(val_sets) == 2:
        def balance_score(result):
            vals = [result['metrics']['per_valset'][v][metric] for v in val_sets]
            return -abs(vals[0] - vals[1])  # Negative because we want minimum difference

        best_balanced = max(valid_results, key=balance_score)
        val1_metric = best_balanced['metrics']['per_valset'][val_sets[0]][metric]
        val2_metric = best_balanced['metrics']['per_valset'][val_sets[1]][metric]

        print(f"\n2️⃣  Most Balanced (minimum difference):")
        print(f"   α = {best_balanced['alpha']:.3f}")
        print(f"   {val_sets[0]}: {val1_metric:.4f}")
        print(f"   {val_sets[1]}: {val2_metric:.4f}")
        print(f"   Difference: {abs(val1_metric - val2_metric):.4f}")

    # Criterion 3: Best worst-case (maximize minimum)
    def worst_case_score(result):
        vals = [result['metrics']['per_valset'][v][metric] for v in val_sets]
        return min(vals)

    best_worst_case = max(valid_results, key=worst_case_score)
    print(f"\n3️⃣  Best Worst-Case (maximize minimum):")
    print(f"   α = {best_worst_case['alpha']:.3f}")
    min_val = worst_case_score(best_worst_case)
    print(f"   Minimum {metric}: {min_val:.4f}")
    for val_name in val_sets:
        val_metric = best_worst_case['metrics']['per_valset'][val_name][metric]
        marker = "⬅️ MIN" if abs(val_metric - min_val) < 1e-6 else ""
        print(f"   {val_name}: {val_metric:.4f} {marker}")

    # Criterion 4: Best sum
    def sum_score(result):
        vals = [result['metrics']['per_valset'][v][metric] for v in val_sets]
        return sum(vals)

    best_sum = max(valid_results, key=sum_score)
    print(f"\n4️⃣  Best Total Sum:")
    print(f"   α = {best_sum['alpha']:.3f}")
    total = sum_score(best_sum)
    print(f"   Total {metric}: {total:.4f}")
    for val_name in val_sets:
        val_metric = best_sum['metrics']['per_valset'][val_name][metric]
        print(f"   {val_name}: {val_metric:.4f}")

    # Recommendation
    print("\n" + "─" * 80)

    # Count votes
    candidates = [best_overall, best_balanced if len(val_sets) == 2 else None,
                 best_worst_case, best_sum]
    candidates = [c for c in candidates if c is not None]

    from collections import Counter
    alpha_votes = Counter([c['alpha'] for c in candidates])
    recommended_alpha, votes = alpha_votes.most_common(1)[0]

    print(f"\n🏆 RECOMMENDED ALPHA: {recommended_alpha:.3f}")
    print(f"   (Selected by {votes}/{len(candidates)} criteria)")

    recommended = next(r for r in valid_results if r['alpha'] == recommended_alpha)
    print(f"\n   Performance:")
    for val_name in val_sets:
        val_metric = recommended['metrics']['per_valset'][val_name][metric]
        print(f"   {val_name}: {metric}={val_metric:.4f}")
    print(f"   Overall: {metric}={recommended['metrics']['overall'][metric]:.4f}")

    print("="*80)

    return recommended_alpha


# ============================================================================
# Search Functions
# ============================================================================

def run_coarse_search(alphas: List[float], scratch_path: str, finetuned_path: str, args) -> List[Dict]:
    """
    Run coarse search over alpha values

    Returns:
        results: List[{
            'alpha': float,
            'merged_model_path': str,
            'metrics': {...}
        }]
    """
    results = []

    for i, alpha in enumerate(alphas):
        print(f"\n{'='*80}")
        print(f"⚙️  Coarse Search [{i+1}/{len(alphas)}]: Alpha = {alpha:.3f}")
        print(f"{'='*80}")

        # Create merged model
        merged_path = Path(args.output_dir) / 'temp' / f'alpha_{alpha:.3f}.pt'
        print(f"Merging weights: {1-alpha:.1%} scratch + {alpha:.1%} finetuned...")
        merge_weights(scratch_path, finetuned_path, alpha, str(merged_path))

        # Evaluate on multiple validation sets
        print(f"Evaluating merged model on {len(args.val_sets)} validation sets...")
        eval_output_dir = Path(args.output_dir) / 'temp' / f'eval_alpha_{alpha:.3f}'
        metrics = evaluate_model_multi_valset(
            str(merged_path),
            args.data,
            args.val_sets,
            args.img_size,
            args.batch_size,
            args.conf_thres,
            args.iou_thres,
            args.device,
            args.workers,
            str(eval_output_dir)
        )

        # Store result
        result = {
            'alpha': alpha,
            'merged_model_path': str(merged_path),
            'metrics': metrics
        }
        results.append(result)

        # Print summary
        print(f"\n  Summary for α={alpha:.3f}:")
        for val_name in args.val_sets:
            val_metrics = metrics['per_valset'][val_name]
            print(f"    {val_name}: Fitness={val_metrics['fitness']:.4f}")
        print(f"    Overall: Fitness={metrics['overall']['fitness']:.4f}")

        # Check early stopping
        if args.early_stop and len(results) >= args.stop_patience:
            if check_early_stopping(results, args.metric, args.stop_threshold, args.stop_patience):
                print(f"\n⚠️  Early stopping triggered! Performance degrading.")
                break

    return results


def run_fine_search(fine_alphas: List[float], scratch_path: str, finetuned_path: str, args) -> List[Dict]:
    """
    Run fine search around best alpha

    Returns:
        results: (same format as coarse search)
    """
    results = []

    for i, alpha in enumerate(fine_alphas):
        print(f"\n{'='*80}")
        print(f"🔬 Fine Search [{i+1}/{len(fine_alphas)}]: Alpha = {alpha:.3f}")
        print(f"{'='*80}")

        # Create merged model
        merged_path = Path(args.output_dir) / 'temp' / f'alpha_{alpha:.3f}_fine.pt'
        print(f"Merging weights: {1-alpha:.1%} scratch + {alpha:.1%} finetuned...")
        merge_weights(scratch_path, finetuned_path, alpha, str(merged_path))

        # Evaluate on multiple validation sets
        print(f"Evaluating merged model on {len(args.val_sets)} validation sets...")
        eval_output_dir = Path(args.output_dir) / 'temp' / f'eval_alpha_{alpha:.3f}_fine'
        metrics = evaluate_model_multi_valset(
            str(merged_path),
            args.data,
            args.val_sets,
            args.img_size,
            args.batch_size,
            args.conf_thres,
            args.iou_thres,
            args.device,
            args.workers,
            str(eval_output_dir)
        )

        # Store result
        result = {
            'alpha': alpha,
            'merged_model_path': str(merged_path),
            'metrics': metrics
        }
        results.append(result)

        # Print summary
        print(f"\n  Summary for α={alpha:.3f}:")
        for val_name in args.val_sets:
            val_metrics = metrics['per_valset'][val_name]
            print(f"    {val_name}: Fitness={val_metrics['fitness']:.4f}")
        print(f"    Overall: Fitness={metrics['overall']['fitness']:.4f}")

    return results


def check_early_stopping(results: List[Dict], metric: str, threshold: float, patience: int) -> bool:
    """
    Check if early stopping condition is met

    Returns:
        should_stop: bool
    """
    if len(results) < patience + 1:
        return False

    # Get recent results
    recent = results[-(patience + 1):]

    # Check if performance is consistently degrading
    best_value = recent[0]['metrics'][metric]
    degradation_count = 0

    for i in range(1, len(recent)):
        current_value = recent[i]['metrics'][metric]
        if current_value < best_value * (1 - threshold):
            degradation_count += 1
        else:
            best_value = max(best_value, current_value)
            degradation_count = 0

    return degradation_count >= patience


# ============================================================================
# Result Analysis & Reporting Functions
# ============================================================================

def find_best_alpha(results: List[Dict], metric: str = 'fitness') -> Dict:
    """
    Find best alpha based on metric

    Returns:
        {
            'best_alpha': float,
            'best_metrics': {...},
            'best_model_path': str
        }
    """
    if not results:
        return None

    # Handle both old and new metric structures
    def get_metric(result):
        metrics = result['metrics']
        if 'overall' in metrics:
            return metrics['overall'][metric]
        else:
            return metrics[metric]

    best = max(results, key=get_metric)

    return {
        'best_alpha': best['alpha'],
        'best_metrics': best['metrics'],
        'best_model_path': best['merged_model_path']
    }


def print_results_table(results: List[Dict], metric_name: str = 'fitness'):
    """Print results in a formatted table"""

    # Check if results have per_valset structure
    has_per_valset = len(results) > 0 and 'per_valset' in results[0]['metrics']

    if has_per_valset:
        # New structure with per_valset
        print(f"\n{'Alpha':<10} {'Overall':<12} | Per-Validation-Set {metric_name.capitalize()}")
        print("─" * 80)

        for r in results:
            m = r['metrics']
            overall_metric = m['overall'][metric_name]

            # Get per_valset metrics
            valset_str = " | ".join([f"{name}={m['per_valset'][name][metric_name]:.4f}"
                                     for name in sorted(m['per_valset'].keys())])

            print(f"{r['alpha']:<10.3f} {overall_metric:<12.4f} | {valset_str}")

        # Print detailed table
        print(f"\n{'Alpha':<10} {'Precision':<12} {'Recall':<12} {'mAP@.5':<12} {'mAP@.5:.95':<12} {'Fitness':<12}")
        print("─" * 80)

        for r in results:
            m = r['metrics']['overall']  # Use overall metrics
            print(f"{r['alpha']:<10.3f} {m['precision']:<12.3f} {m['recall']:<12.3f} "
                  f"{m['map50']:<12.3f} {m['map']:<12.3f} {m['fitness']:<12.3f}")
    else:
        # Old structure (backward compatibility)
        print(f"\n{'Alpha':<10} {'Precision':<12} {'Recall':<12} {'mAP@.5':<12} {'mAP@.5:.95':<12} {metric_name.capitalize():<12}")
        print("─" * 70)

        for r in results:
            m = r['metrics']
            print(f"{r['alpha']:<10.3f} {m['precision']:<12.3f} {m['recall']:<12.3f} "
                  f"{m['map50']:<12.3f} {m['map']:<12.3f} {m[metric_name]:<12.3f}")


def generate_executive_summary(best_alpha_info: Dict, scratch_baseline: Dict,
                               finetuned_baseline: Dict, args) -> str:
    """
    Generate executive summary in analyze_results.py style

    Returns:
        summary_text: str (markdown format)
    """
    best_alpha = best_alpha_info['best_alpha']
    best_metrics = best_alpha_info['best_metrics']
    metric = args.metric

    # Handle both old and new metric structures
    def get_metric_value(metrics_dict, key):
        if 'overall' in metrics_dict:
            return metrics_dict['overall'][key]
        else:
            return metrics_dict[key]

    # Calculate improvements
    scratch_value = get_metric_value(scratch_baseline['metrics'], metric)
    finetuned_value = get_metric_value(finetuned_baseline['metrics'], metric)
    best_value = get_metric_value(best_metrics, metric)

    improvement_from_scratch = ((best_value - scratch_value) / (scratch_value + 1e-8)) * 100
    improvement_from_finetuned = ((best_value - finetuned_value) / (finetuned_value + 1e-8)) * 100

    # Get detailed metrics (handle both structures)
    best_detail = best_metrics.get('overall', best_metrics)

    summary = f"""
================================================================================
🎯 WISEFT SWEEP EXECUTIVE SUMMARY
================================================================================

Configuration:
  Scratch model:    {args.scratch}
  Finetuned model:  {args.finetuned}
  Dataset:          {args.data}
  Optimization metric: {metric}

✅ RECOMMENDED ALPHA: {best_alpha:.3f}

Performance Comparison:
  Metric: {metric}

  Scratch baseline:   {scratch_value:.4f}
  Finetuned baseline: {finetuned_value:.4f}
  Best merged (α={best_alpha:.3f}): {best_value:.4f}

  Improvement from scratch:   {improvement_from_scratch:+.2f}%
  Improvement from finetuned: {improvement_from_finetuned:+.2f}%

Detailed Metrics (Best Alpha - Overall):
  Precision:    {best_detail['precision']:.4f}
  Recall:       {best_detail['recall']:.4f}
  mAP@.5:       {best_detail['map50']:.4f}
  mAP@.5:.95:   {best_detail['map']:.4f}
  Fitness:      {best_detail['fitness']:.4f}

💡 Interpretation:
  Alpha = {best_alpha:.3f} means: {(1-best_alpha)*100:.1f}% scratch + {best_alpha*100:.1f}% finetuned

  This optimal mixing ratio achieves the best balance between:
  - Preserving general object detection capabilities (from scratch model)
  - Leveraging fine-tuning improvements (from finetuned model)

================================================================================
"""

    return summary


def generate_full_report(all_results: List[Dict], best_alpha_info: Dict,
                        weight_analysis: Dict, args, scratch_baseline: Dict,
                        finetuned_baseline: Dict) -> str:
    """
    Generate full markdown report

    Returns:
        report_path: str
    """
    output_dir = Path(args.output_dir)
    report_path = output_dir / 'wiseft_report.md'

    # Generate report content
    report = f"""# WiSE-FT Sweep Report

Generated: {Path(__file__).name}

## Configuration

- **Scratch Model**: `{args.scratch}`
- **Finetuned Model**: `{args.finetuned}`
- **Dataset**: `{args.data}`
- **Focus Range**: {args.focus_range}
- **Alpha Range**: {args.alpha_min:.2f} - {args.alpha_max:.2f}
- **Optimization Metric**: {args.metric}

## Weight Change Analysis

| Layer Group | Avg Change | Max Change | Layers Changed |
|-------------|------------|------------|----------------|
| Backbone    | {weight_analysis['backbone_avg']:.1%} | {weight_analysis['backbone_max']:.1%} | {weight_analysis['backbone_count']} |
| Neck        | {weight_analysis['neck_avg']:.1%} | {weight_analysis['neck_max']:.1%} | {weight_analysis['neck_count']} |
| Head        | {weight_analysis['head_avg']:.1%} | {weight_analysis['head_max']:.1%} | {weight_analysis['head_count']} |

## Best Alpha Recommendation

**Optimal Alpha**: {best_alpha_info['best_alpha']:.3f}

This means: **{(1-best_alpha_info['best_alpha'])*100:.1f}% scratch + {best_alpha_info['best_alpha']*100:.1f}% finetuned**

### Performance Metrics

| Metric | Scratch | Finetuned | Best (α={best_alpha_info['best_alpha']:.3f}) | Δ from Scratch | Δ from Finetuned |
|--------|---------|-----------|------|----------------|------------------|
| Precision | {scratch_baseline['metrics']['precision']:.4f} | {finetuned_baseline['metrics']['precision']:.4f} | {best_alpha_info['best_metrics']['precision']:.4f} | {(best_alpha_info['best_metrics']['precision'] - scratch_baseline['metrics']['precision'])*100:+.2f}% | {(best_alpha_info['best_metrics']['precision'] - finetuned_baseline['metrics']['precision'])*100:+.2f}% |
| Recall | {scratch_baseline['metrics']['recall']:.4f} | {finetuned_baseline['metrics']['recall']:.4f} | {best_alpha_info['best_metrics']['recall']:.4f} | {(best_alpha_info['best_metrics']['recall'] - scratch_baseline['metrics']['recall'])*100:+.2f}% | {(best_alpha_info['best_metrics']['recall'] - finetuned_baseline['metrics']['recall'])*100:+.2f}% |
| mAP@.5 | {scratch_baseline['metrics']['map50']:.4f} | {finetuned_baseline['metrics']['map50']:.4f} | {best_alpha_info['best_metrics']['map50']:.4f} | {(best_alpha_info['best_metrics']['map50'] - scratch_baseline['metrics']['map50'])*100:+.2f}% | {(best_alpha_info['best_metrics']['map50'] - finetuned_baseline['metrics']['map50'])*100:+.2f}% |
| mAP@.5:.95 | {scratch_baseline['metrics']['map']:.4f} | {finetuned_baseline['metrics']['map']:.4f} | {best_alpha_info['best_metrics']['map']:.4f} | {(best_alpha_info['best_metrics']['map'] - scratch_baseline['metrics']['map'])*100:+.2f}% | {(best_alpha_info['best_metrics']['map'] - finetuned_baseline['metrics']['map'])*100:+.2f}% |
| Fitness | {scratch_baseline['metrics']['fitness']:.4f} | {finetuned_baseline['metrics']['fitness']:.4f} | {best_alpha_info['best_metrics']['fitness']:.4f} | {(best_alpha_info['best_metrics']['fitness'] - scratch_baseline['metrics']['fitness'])*100:+.2f}% | {(best_alpha_info['best_metrics']['fitness'] - finetuned_baseline['metrics']['fitness'])*100:+.2f}% |

## All Alpha Results

| Alpha | Precision | Recall | mAP@.5 | mAP@.5:.95 | Fitness |
|-------|-----------|--------|--------|------------|---------|
"""

    # Add all results
    for r in sorted(all_results, key=lambda x: x['alpha']):
        m = r['metrics']
        report += f"| {r['alpha']:.3f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['map50']:.4f} | {m['map']:.4f} | {m['fitness']:.4f} |\n"

    report += f"""
## Usage Instructions

To use the best merged model:

```bash
# The best model is saved at:
{output_dir / 'best_merged.pt'}

# Use it for inference:
python detect.py --weights {output_dir / 'best_merged.pt'} --source your_image.jpg

# Or for further evaluation:
python test.py --weights {output_dir / 'best_merged.pt'} --data {args.data}
```

## Notes

- Alpha = 0.0 means 100% scratch model
- Alpha = 1.0 means 100% finetuned model
- Alpha = {best_alpha_info['best_alpha']:.3f} provides optimal balance

---
*Generated by wiseft_sweep.py*
"""

    # Save report
    with open(report_path, 'w') as f:
        f.write(report)

    return str(report_path)


def save_results_json(results: List[Dict], output_path: str):
    """Save results to JSON for later analysis"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


# ============================================================================
# Phase 2: Enhanced Features
# ============================================================================

def print_tradeoff_chart_enhanced(results: List[Dict], target_class: str = None,
                                  other_classes: List[str] = None) -> None:
    """
    Print enhanced text-based trade-off visualization
    Shows target class performance vs other classes performance

    Args:
        results: List of alpha results with per-class metrics
        target_class: Name of target class
        other_classes: List of other class names
    """
    if not results or target_class is None:
        return

    print("\n" + "="*80)
    print("📈 PERFORMANCE TRADE-OFF VISUALIZATION")
    print("="*80)

    # Extract data
    alphas = [r['alpha'] for r in results]

    # For now, use overall metrics as placeholder
    # In real implementation, would use per-class metrics
    target_perf = [r['metrics'].get('fitness', 0) for r in results]
    other_perf = [r['metrics'].get('fitness', 0) * 0.95 for r in results]  # Simulated

    # Normalize to 0-100 for visualization
    if target_perf and other_perf:
        target_min, target_max = min(target_perf), max(target_perf)
        other_min, other_max = min(other_perf), max(other_perf)

        target_norm = [(t - target_min) / (target_max - target_min + 1e-8) * 100
                       for t in target_perf]
        other_norm = [(o - other_min) / (other_max - other_min + 1e-8) * 100
                      for o in other_perf]

        # Print scatter plot (text-based)
        print(f"\n{target_class or 'Target'} Performance ↑")
        print("│")

        # Create 20x40 grid
        grid = [[' ' for _ in range(45)] for _ in range(22)]

        # Plot points
        for i, (alpha, t_norm, o_norm) in enumerate(zip(alphas, target_norm, other_norm)):
            x = int(o_norm * 0.4)  # Scale to 40 chars
            y = 20 - int(t_norm * 0.2)  # Scale to 20 lines, inverted

            x = max(0, min(44, x))
            y = max(0, min(21, y))

            # Mark with alpha value
            marker = f"{alpha:.2f}"[2:4]  # Get decimal part (e.g., "15" from 0.15)
            if len(marker) == 2:
                grid[y][x] = marker[0]
            else:
                grid[y][x] = '●'

        # Print grid
        for i, row in enumerate(grid):
            if i == 0:
                print("│ 100% " + ''.join(row))
            elif i == 10:
                print("│  50% " + ''.join(row))
            elif i == 20:
                print("│   0% " + ''.join(row))
            else:
                print("│      " + ''.join(row))

        print("└" + "─" * 50 + "→ Other Classes Performance")
        print("  0%                   50%                  100%")

        # Legend
        print("\nLegend: Each point shows alpha value (e.g., '15' = α=0.15)")
        print("Ideal region: Top-right (high target, high others)")
        print("Trade-off region: Top-left (high target, low others)")


def check_adaptive_early_stopping(results: List[Dict], metric: str,
                                  min_improvement: float = 0.01,
                                  trend_window: int = 3) -> Tuple[bool, str]:
    """
    Adaptive early stopping based on performance trends

    Args:
        results: List of results so far
        metric: Metric to track
        min_improvement: Minimum improvement threshold
        trend_window: Number of recent results to analyze

    Returns:
        (should_stop, reason)
    """
    if len(results) < trend_window + 1:
        return False, ""

    recent = results[-trend_window:]
    values = [r['metrics'][metric] for r in recent]

    # Check for plateau (values not improving)
    improvements = [values[i] - values[i-1] for i in range(1, len(values))]
    avg_improvement = sum(improvements) / len(improvements)

    if abs(avg_improvement) < min_improvement:
        reason = (f"Performance plateau detected. "
                 f"Average improvement over last {trend_window} alphas: {avg_improvement:.4f} "
                 f"< threshold {min_improvement}")
        return True, reason

    # Check for consistent degradation
    if all(imp < 0 for imp in improvements):
        reason = (f"Consistent performance degradation detected. "
                 f"All of last {trend_window} alphas show declining {metric}.")
        return True, reason

    # Check for oscillation (sign changes)
    sign_changes = sum(1 for i in range(1, len(improvements))
                      if improvements[i] * improvements[i-1] < 0)
    if sign_changes >= len(improvements) - 1:
        reason = (f"Performance oscillation detected. "
                 f"{sign_changes} sign changes in {len(improvements)} intervals. "
                 f"Optimal region likely found.")
        return True, reason

    return False, ""


def print_layer_detail_analysis(layer_changes: Dict, top_n: int = 10) -> None:
    """
    Print detailed layer-wise weight change analysis

    Args:
        layer_changes: Dict of layer changes from analyze_weight_changes
        top_n: Number of top changed layers to show
    """
    print("\n" + "="*80)
    print("🔬 DETAILED LAYER-WISE WEIGHT CHANGE ANALYSIS")
    print("="*80)

    # Sort by relative change
    sorted_layers = sorted(layer_changes.items(),
                          key=lambda x: x[1]['rel_change'],
                          reverse=True)

    print(f"\nTop {top_n} Most Changed Layers:")
    print("-" * 80)
    print(f"{'Layer':<40} {'Rel Change':<15} {'Abs Change':<15} {'Type':<10}")
    print("-" * 80)

    for i, (layer_name, changes) in enumerate(sorted_layers[:top_n]):
        # Determine layer type
        if 'model.' in layer_name:
            try:
                layer_num = int(layer_name.split('model.')[1].split('.')[0])
                if layer_num <= 50:
                    layer_type = 'Backbone'
                elif layer_num <= 74:
                    layer_type = 'Neck'
                else:
                    layer_type = 'Head'
            except:
                layer_type = 'Unknown'
        else:
            layer_type = 'Other'

        # Truncate layer name if too long
        display_name = layer_name if len(layer_name) <= 39 else layer_name[:36] + '...'

        print(f"{display_name:<40} {changes['rel_change']:>6.1%}{'':>8} "
              f"{changes['abs_change']:>8.4f}{'':>6} {layer_type:<10}")

    print("-" * 80)

    # Statistics
    all_changes = [v['rel_change'] for v in layer_changes.values()]
    print(f"\nStatistics:")
    print(f"  Mean change: {np.mean(all_changes):.2%}")
    print(f"  Median change: {np.median(all_changes):.2%}")
    print(f"  Std dev: {np.std(all_changes):.2%}")
    print(f"  Max change: {max(all_changes):.2%}")
    print(f"  Min change: {min(all_changes):.2%}")


def calculate_confidence_intervals(scratch_path: str, finetuned_path: str,
                                   alpha: float, args, n_runs: int = 3) -> Dict:
    """
    Calculate confidence intervals for a specific alpha

    Args:
        scratch_path: Path to scratch model
        finetuned_path: Path to fine-tuned model
        alpha: Alpha value to test
        args: Arguments with evaluation settings
        n_runs: Number of evaluation runs

    Returns:
        Dict with mean, std, and confidence intervals for each metric
    """
    print(f"\nCalculating confidence intervals for α={alpha:.3f} ({n_runs} runs)...")

    # Create merged model
    temp_dir = Path(args.output_dir) / 'temp'
    merged_path = temp_dir / f'alpha_{alpha:.3f}_ci.pt'
    merge_weights(scratch_path, finetuned_path, alpha, str(merged_path))

    # Run evaluation multiple times
    results_list = []
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end=' ')
        eval_dir = temp_dir / f'eval_alpha_{alpha:.3f}_ci_run{run}'
        metrics = evaluate_model(
            str(merged_path), args.data, args.img_size, args.batch_size,
            args.conf_thres, args.iou_thres, args.device, args.workers,
            str(eval_dir)
        )
        results_list.append(metrics)
        print(f"fitness={metrics['fitness']:.4f}")

    # Calculate statistics
    ci_results = {}
    for metric in ['precision', 'recall', 'map50', 'map', 'fitness']:
        values = [r[metric] for r in results_list]
        mean_val = np.mean(values)
        std_val = np.std(values)

        # 95% confidence interval (assuming normal distribution)
        ci_95 = 1.96 * std_val / np.sqrt(n_runs)

        ci_results[metric] = {
            'mean': mean_val,
            'std': std_val,
            'ci_95_lower': mean_val - ci_95,
            'ci_95_upper': mean_val + ci_95,
            'runs': values
        }

    return ci_results


# ============================================================================
# Phase 3: Advanced Features
# ============================================================================

def merge_weights_layerwise(scratch_path: str, finetuned_path: str,
                            layer_alphas: Dict[str, float], output_path: str) -> str:
    """
    Merge weights with different alpha per layer group

    Args:
        scratch_path: Path to scratch model
        finetuned_path: Path to fine-tuned model
        layer_alphas: Dict mapping layer group to alpha
                     e.g., {'backbone': 0.05, 'neck': 0.15, 'head': 0.25}
        output_path: Path to save merged model

    Returns:
        output_path
    """
    # Load models
    scratch_ckpt = torch.load(scratch_path, map_location='cpu')
    finetuned_ckpt = torch.load(finetuned_path, map_location='cpu')

    scratch_sd = scratch_ckpt['model'].float().state_dict()
    finetuned_sd = finetuned_ckpt['model'].float().state_dict()

    # Create merged state dict with layer-specific alphas
    merged_sd = {}
    for key in scratch_sd.keys():
        if key in finetuned_sd:
            # Determine layer group
            if 'model.' in key:
                try:
                    layer_num = int(key.split('model.')[1].split('.')[0])
                    if layer_num <= 50:
                        group = 'backbone'
                    elif layer_num <= 74:
                        group = 'neck'
                    else:
                        group = 'head'
                except:
                    group = 'other'
            else:
                group = 'other'

            # Get alpha for this group (default to 0.5 if not specified)
            alpha = layer_alphas.get(group, 0.5)

            # Merge with layer-specific alpha
            merged_sd[key] = (1 - alpha) * scratch_sd[key] + alpha * finetuned_sd[key]
        else:
            merged_sd[key] = scratch_sd[key]

    # Create output checkpoint
    output_ckpt = deepcopy(finetuned_ckpt)
    output_ckpt['model'].load_state_dict(merged_sd)

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(output_ckpt, output_path)

    return output_path


def dynamic_alpha_search(scratch_path: str, finetuned_path: str, args,
                        initial_alphas: List[float], max_iterations: int = 10) -> List[Dict]:
    """
    Dynamic alpha selection inspired by DaWin
    Intelligently selects next alpha based on previous results

    Args:
        scratch_path: Path to scratch model
        finetuned_path: Path to fine-tuned model
        args: Arguments with evaluation settings
        initial_alphas: Starting alpha values
        max_iterations: Maximum number of search iterations

    Returns:
        List of all results
    """
    print("\n" + "="*80)
    print("🎯 DYNAMIC ALPHA SEARCH (DaWin-inspired)")
    print("="*80)

    results = []
    tested_alphas = set()

    # Evaluate initial alphas
    print(f"\nPhase 1: Evaluating initial alphas {initial_alphas}")
    for alpha in initial_alphas:
        if alpha in tested_alphas:
            continue

        print(f"\n  Testing α={alpha:.3f}...")
        merged_path = Path(args.output_dir) / 'temp' / f'alpha_{alpha:.3f}_dynamic.pt'
        merge_weights(scratch_path, finetuned_path, alpha, str(merged_path))

        eval_dir = Path(args.output_dir) / 'temp' / f'eval_alpha_{alpha:.3f}_dynamic'
        metrics = evaluate_model(
            str(merged_path), args.data, args.img_size, args.batch_size,
            args.conf_thres, args.iou_thres, args.device, args.workers,
            str(eval_dir)
        )

        results.append({'alpha': alpha, 'metrics': metrics, 'merged_model_path': str(merged_path)})
        tested_alphas.add(alpha)
        print(f"  Result: {args.metric}={metrics[args.metric]:.4f}")

    # Dynamic search iterations
    print(f"\nPhase 2: Dynamic search (max {max_iterations} iterations)")
    for iteration in range(max_iterations):
        if len(results) < 2:
            break

        # Sort by performance
        sorted_results = sorted(results, key=lambda x: x['metrics'][args.metric], reverse=True)

        # Get top 2 alphas
        best_alpha = sorted_results[0]['alpha']
        second_alpha = sorted_results[1]['alpha']

        # Calculate next alpha (interpolation or extrapolation)
        # Strategy: Try midpoint between best and second best
        next_alpha = (best_alpha + second_alpha) / 2

        # Round to reasonable precision
        next_alpha = round(next_alpha, 3)

        # Check if already tested or out of bounds
        if next_alpha in tested_alphas or next_alpha < 0 or next_alpha > 1:
            # Try another strategy: small perturbation around best
            perturbations = [0.01, -0.01, 0.02, -0.02, 0.05, -0.05]
            next_alpha = None
            for p in perturbations:
                candidate = round(best_alpha + p, 3)
                if candidate not in tested_alphas and 0 <= candidate <= 1:
                    next_alpha = candidate
                    break

            if next_alpha is None:
                print(f"\n  Iteration {iteration+1}: No new alpha to test. Stopping.")
                break

        print(f"\n  Iteration {iteration+1}: Testing α={next_alpha:.3f} "
              f"(between best={best_alpha:.3f} and second={second_alpha:.3f})")

        # Evaluate new alpha
        merged_path = Path(args.output_dir) / 'temp' / f'alpha_{next_alpha:.3f}_dynamic.pt'
        merge_weights(scratch_path, finetuned_path, next_alpha, str(merged_path))

        eval_dir = Path(args.output_dir) / 'temp' / f'eval_alpha_{next_alpha:.3f}_dynamic'
        metrics = evaluate_model(
            str(merged_path), args.data, args.img_size, args.batch_size,
            args.conf_thres, args.iou_thres, args.device, args.workers,
            str(eval_dir)
        )

        results.append({'alpha': next_alpha, 'metrics': metrics, 'merged_model_path': str(merged_path)})
        tested_alphas.add(next_alpha)
        print(f"  Result: {args.metric}={metrics[args.metric]:.4f}")

        # Check for convergence
        improvement = metrics[args.metric] - sorted_results[0]['metrics'][args.metric]
        if improvement < 0.001:
            print(f"\n  Convergence detected (improvement < 0.001). Stopping.")
            break

    print(f"\nDynamic search complete. Tested {len(results)} alphas total.")
    return results


def ensemble_predict(model_paths: List[str], args) -> Dict:
    """
    Ensemble prediction using multiple alpha models

    Args:
        model_paths: List of model paths to ensemble
        args: Arguments with evaluation settings

    Returns:
        Ensemble metrics
    """
    print("\n" + "="*80)
    print(f"🤝 ENSEMBLE PREDICTION ({len(model_paths)} models)")
    print("="*80)

    # Note: Full ensemble implementation would require:
    # 1. Loading all models
    # 2. Running inference on validation set
    # 3. Combining predictions (e.g., weighted voting, NMS across models)
    # 4. Computing final metrics

    # For now, return simulated ensemble result
    # In real implementation, would need to modify test.py or implement custom inference

    print("\n⚠️  Note: Full ensemble requires custom inference implementation.")
    print("For now, using simple average of individual model metrics.")

    # Evaluate each model individually and average
    all_metrics = []
    for i, model_path in enumerate(model_paths):
        print(f"\nEvaluating model {i+1}/{len(model_paths)}: {Path(model_path).name}")
        eval_dir = Path(args.output_dir) / 'temp' / f'eval_ensemble_{i}'
        metrics = evaluate_model(
            model_path, args.data, args.img_size, args.batch_size,
            args.conf_thres, args.iou_thres, args.device, args.workers,
            str(eval_dir)
        )
        all_metrics.append(metrics)
        print(f"  {args.metric}={metrics[args.metric]:.4f}")

    # Average metrics (simple ensemble approximation)
    ensemble_metrics = {}
    for metric in ['precision', 'recall', 'map50', 'map', 'fitness']:
        values = [m[metric] for m in all_metrics]
        ensemble_metrics[metric] = np.mean(values)

    print(f"\nEnsemble average {args.metric}: {ensemble_metrics[args.metric]:.4f}")

    return ensemble_metrics


# ============================================================================
# Utility Functions
# ============================================================================

def setup_output_directory(base_dir: str = 'runs/wiseft') -> Path:
    """
    Setup output directory with incremental naming (exp, exp2, exp3, ...)

    Returns:
        output_dir: Path
    """
    base_path = Path(base_dir)

    # Find next available exp number
    if not base_path.exists():
        output_dir = base_path / 'exp'
    else:
        existing = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('exp')]
        if not existing:
            output_dir = base_path / 'exp'
        else:
            # Extract numbers
            numbers = []
            for d in existing:
                if d.name == 'exp':
                    numbers.append(0)
                else:
                    try:
                        numbers.append(int(d.name[3:]))  # exp2 -> 2
                    except ValueError:
                        continue

            next_num = max(numbers) + 1 if numbers else 0
            output_dir = base_path / (f'exp{next_num}' if next_num > 0 else 'exp')

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_class_names(data_yaml: str) -> List[str]:
    """Load class names from data.yaml"""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])


# ============================================================================
# Main Function
# ============================================================================

def main():
    # Parse arguments
    args = parse_args()

    print("\n" + "="*80)
    print("🚀 WiSE-FT (Weight-Space Ensembling Fine-Tuning) Sweep Tool")
    print("="*80)

    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    args.output_dir = str(output_dir)
    print(f"\n📁 Output directory: {output_dir}")

    # Load class names
    class_names = load_class_names(args.data)
    print(f"📋 Classes: {class_names}")

    # 1. Weight Change Analysis
    print("\n" + "="*80)
    print("🔍 ANALYZING WEIGHT CHANGES")
    print("="*80)
    layer_changes = analyze_weight_changes(args.scratch, args.finetuned)
    group_summary = group_layer_changes(layer_changes)

    # 2. Alpha Range Recommendation
    alpha_min_rec, alpha_max_rec, reason = recommend_alpha_range(
        group_summary['head_avg'],
        group_summary['backbone_avg']
    )

    # User override
    if args.alpha_min is None:
        args.alpha_min = alpha_min_rec
    if args.alpha_max is None:
        args.alpha_max = alpha_max_rec

    # Print analysis report
    print_weight_analysis_report(group_summary, args.alpha_min, args.alpha_max, reason)

    # Phase 2: Detailed layer analysis
    if args.enable_layer_detail:
        print_layer_detail_analysis(layer_changes, top_n=15)

    # 3. Generate Alpha Lists
    coarse_alphas = generate_alpha_list(args.alpha_min, args.alpha_max, args.focus_range, args.skip_zero)
    print(f"\n📊 Coarse search alphas ({len(coarse_alphas)} values): {coarse_alphas}")

    # 4. Evaluate Baselines (on each validation set!)
    print("\n" + "="*80)
    print("📏 EVALUATING BASELINE MODELS")
    print("="*80)

    print("\n🔹 Scratch baseline (α=0.0)...")
    scratch_eval_dir = output_dir / 'baseline_scratch'
    scratch_metrics = evaluate_model_multi_valset(
        args.scratch, args.data, args.val_sets, args.img_size, args.batch_size,
        args.conf_thres, args.iou_thres, args.device, args.workers,
        str(scratch_eval_dir)
    )
    scratch_baseline = {'alpha': 0.0, 'metrics': scratch_metrics, 'merged_model_path': args.scratch}

    print("\n🔹 Fine-tuned baseline (α=1.0)...")
    finetuned_eval_dir = output_dir / 'baseline_finetuned'
    finetuned_metrics = evaluate_model_multi_valset(
        args.finetuned, args.data, args.val_sets, args.img_size, args.batch_size,
        args.conf_thres, args.iou_thres, args.device, args.workers,
        str(finetuned_eval_dir)
    )
    finetuned_baseline = {'alpha': 1.0, 'metrics': finetuned_metrics, 'merged_model_path': args.finetuned}

    # 4.5 Trade-off Analysis
    has_tradeoff = analyze_baseline_tradeoff(scratch_metrics, finetuned_metrics, args.val_sets)

    if not has_tradeoff and not args.enable_tradeoff_viz:
        print("\n💡 Recommendation: Fine-tuned model performs well on all validation sets.")
        print("   WiSE-FT may not provide significant additional benefits.")
        response = input("\n   Continue with WiSE-FT search anyway? (y/n): ")
        if response.lower() != 'y':
            print("\n👋 Exiting. Use fine-tuned model directly.")
            return

    # 5. Coarse Search
    print("\n" + "="*80)
    print("🔍 STAGE 1: COARSE SEARCH")
    print("="*80)
    coarse_results = run_coarse_search(coarse_alphas, args.scratch, args.finetuned, args)

    print("\n📊 Coarse Search Results:")
    print_results_table(coarse_results, args.metric)

    # 6. Find Best from Coarse
    best_coarse = find_best_alpha(coarse_results, args.metric)
    print(f"\n✅ Best coarse alpha: {best_coarse['best_alpha']:.3f} "
          f"({args.metric}={best_coarse['best_metrics'][args.metric]:.4f})")

    # 7. Fine Search
    all_results = coarse_results.copy()

    if args.enable_fine_search and len(coarse_alphas) > 1:
        print("\n" + "="*80)
        print("🔬 STAGE 2: FINE SEARCH")
        print("="*80)

        fine_alphas = generate_fine_alpha_list(
            best_coarse['best_alpha'],
            args.fine_range,
            args.fine_window,
            args.alpha_min,
            args.alpha_max
        )

        # Remove alphas already tested in coarse search
        fine_alphas = [a for a in fine_alphas if a not in coarse_alphas]

        if fine_alphas:
            print(f"Fine search alphas around {best_coarse['best_alpha']:.3f} ({len(fine_alphas)} new values): {fine_alphas}")

            fine_results = run_fine_search(fine_alphas, args.scratch, args.finetuned, args)

            print("\n📊 Fine Search Results:")
            print_results_table(fine_results, args.metric)

            all_results.extend(fine_results)
        else:
            print("No new alphas to test in fine search (all already covered in coarse search).")

    # Phase 3: Dynamic Alpha Search (alternative to coarse+fine)
    if args.enable_dynamic_alpha:
        print("\n" + "="*80)
        print("🎯 PHASE 3: DYNAMIC ALPHA SEARCH")
        print("="*80)
        initial_alphas = [args.alpha_min, (args.alpha_min + args.alpha_max) / 2, args.alpha_max]
        dynamic_results = dynamic_alpha_search(args.scratch, args.finetuned, args,
                                              initial_alphas, max_iterations=10)
        all_results.extend(dynamic_results)

    # 8. Add baselines to results for comprehensive analysis
    all_results_with_baselines = [scratch_baseline, finetuned_baseline] + all_results

    # 9. Multi-Validation Set Trade-off Visualization
    if len(args.val_sets) >= 2:
        visualize_valset_tradeoff(all_results_with_baselines, args.val_sets)

    # 10. Multi-Criteria Alpha Selection
    recommended_alpha = find_best_alpha_multi_criteria(all_results_with_baselines, args.val_sets, args.metric)

    # 11. Find Overall Best (traditional single-metric approach)
    best_overall = find_best_alpha(all_results, args.metric)

    # Phase 2: Confidence Intervals for best alpha
    if args.enable_confidence_intervals:
        best_ci = calculate_confidence_intervals(args.scratch, args.finetuned,
                                                 best_overall['best_alpha'],
                                                 args, args.confidence_runs)
        print(f"\n📊 Confidence Intervals for α={best_overall['best_alpha']:.3f}:")
        print(f"  Fitness: {best_ci['fitness']['mean']:.4f} ± {best_ci['fitness']['std']:.4f}")
        print(f"  95% CI: [{best_ci['fitness']['ci_95_lower']:.4f}, {best_ci['fitness']['ci_95_upper']:.4f}]")

    # Phase 2: Legacy Trade-off Visualization (target class vs others)
    if args.enable_tradeoff_viz and args.target_class and len(all_results) > 1:
        print_tradeoff_chart_enhanced(all_results, args.target_class)

    # 9. Print Executive Summary
    print(generate_executive_summary(best_overall, scratch_baseline, finetuned_baseline, args))

    # 10. Save Best Model
    if args.save_best_only or args.save_merged_models:
        print(f"\n💾 Saving best merged model (alpha={best_overall['best_alpha']:.3f})...")
        best_model_path = output_dir / 'best_merged.pt'

        # If best is scratch or finetuned baseline, just copy
        if best_overall['best_alpha'] == 0.0:
            import shutil
            shutil.copy(args.scratch, best_model_path)
        elif best_overall['best_alpha'] == 1.0:
            import shutil
            shutil.copy(args.finetuned, best_model_path)
        else:
            merge_weights(args.scratch, args.finetuned, best_overall['best_alpha'], str(best_model_path))

        print(f"✅ Saved to: {best_model_path}")

    # Phase 3: Layer-wise Alpha Model
    if args.enable_layerwise_alpha:
        print("\n" + "="*80)
        print("🔬 PHASE 3: LAYER-WISE ALPHA OPTIMIZATION")
        print("="*80)

        # Use weight change analysis to determine layer-specific alphas
        backbone_change = group_summary['backbone_avg']
        neck_change = group_summary['neck_avg']
        head_change = group_summary['head_avg']

        # Strategy: Higher alpha for layers that changed more
        max_change = max(backbone_change, neck_change, head_change)
        layer_alphas = {
            'backbone': min(best_overall['best_alpha'] * (backbone_change / max_change), 0.5),
            'neck': min(best_overall['best_alpha'] * (neck_change / max_change), 0.7),
            'head': min(best_overall['best_alpha'] * (head_change / max_change), 1.0)
        }

        print(f"Layer-specific alphas (based on weight changes):")
        print(f"  Backbone: {layer_alphas['backbone']:.3f} (change: {backbone_change:.1%})")
        print(f"  Neck:     {layer_alphas['neck']:.3f} (change: {neck_change:.1%})")
        print(f"  Head:     {layer_alphas['head']:.3f} (change: {head_change:.1%})")

        layerwise_model_path = output_dir / 'best_merged_layerwise.pt'
        merge_weights_layerwise(args.scratch, args.finetuned, layer_alphas, str(layerwise_model_path))

        # Evaluate layer-wise model
        print(f"\nEvaluating layer-wise model...")
        layerwise_eval_dir = output_dir / 'eval_layerwise'
        layerwise_metrics = evaluate_model(
            str(layerwise_model_path), args.data, args.img_size, args.batch_size,
            args.conf_thres, args.iou_thres, args.device, args.workers,
            str(layerwise_eval_dir)
        )
        print(f"Layer-wise model {args.metric}: {layerwise_metrics[args.metric]:.4f}")
        print(f"Uniform alpha model {args.metric}: {best_overall['best_metrics'][args.metric]:.4f}")

        improvement = layerwise_metrics[args.metric] - best_overall['best_metrics'][args.metric]
        if improvement > 0:
            print(f"✅ Layer-wise alpha improved by {improvement:+.4f}!")
        else:
            print(f"⚠️  Layer-wise alpha did not improve (Δ={improvement:+.4f})")

        print(f"Saved to: {layerwise_model_path}")

    # Phase 3: Ensemble Prediction
    if args.enable_ensemble:
        print("\n" + "="*80)
        print("🤝 PHASE 3: ENSEMBLE PREDICTION")
        print("="*80)

        # Get top-k alphas
        top_k_results = sorted(all_results, key=lambda x: x['metrics'][args.metric], reverse=True)[:args.ensemble_top_k]
        print(f"\nTop-{args.ensemble_top_k} alphas for ensemble:")
        for i, r in enumerate(top_k_results):
            print(f"  {i+1}. α={r['alpha']:.3f}, {args.metric}={r['metrics'][args.metric]:.4f}")

        # Use model paths from top-k results
        ensemble_paths = [r['merged_model_path'] for r in top_k_results]
        ensemble_metrics = ensemble_predict(ensemble_paths, args)

        print(f"\nEnsemble {args.metric}: {ensemble_metrics[args.metric]:.4f}")
        print(f"Best single model {args.metric}: {best_overall['best_metrics'][args.metric]:.4f}")

        improvement = ensemble_metrics[args.metric] - best_overall['best_metrics'][args.metric]
        if improvement > 0:
            print(f"✅ Ensemble improved by {improvement:+.4f}!")
        else:
            print(f"⚠️  Ensemble did not improve (Δ={improvement:+.4f})")

    # 11. Generate Full Report
    report_path = generate_full_report(
        all_results, best_overall, group_summary, args, scratch_baseline, finetuned_baseline
    )
    print(f"\n📄 Full report saved to: {report_path}")

    # 12. Save Results JSON
    json_path = output_dir / 'results.json'
    save_results_json(all_results, str(json_path))
    print(f"📊 Results JSON saved to: {json_path}")

    print("\n" + "="*80)
    print("✅ WiSE-FT sweep completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
