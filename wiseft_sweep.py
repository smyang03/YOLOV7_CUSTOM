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

        # Evaluate
        print(f"Evaluating merged model...")
        eval_output_dir = Path(args.output_dir) / 'temp' / f'eval_alpha_{alpha:.3f}'
        metrics = evaluate_model(
            str(merged_path),
            args.data,
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

        # Print metrics
        print(f"Results: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
              f"mAP@.5={metrics['map50']:.3f}, mAP@.5:.95={metrics['map']:.3f}, "
              f"Fitness={metrics['fitness']:.3f}")

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

        # Evaluate
        print(f"Evaluating merged model...")
        eval_output_dir = Path(args.output_dir) / 'temp' / f'eval_alpha_{alpha:.3f}_fine'
        metrics = evaluate_model(
            str(merged_path),
            args.data,
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

        # Print metrics
        print(f"Results: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
              f"mAP@.5={metrics['map50']:.3f}, mAP@.5:.95={metrics['map']:.3f}, "
              f"Fitness={metrics['fitness']:.3f}")

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

    best = max(results, key=lambda x: x['metrics'][metric])

    return {
        'best_alpha': best['alpha'],
        'best_metrics': best['metrics'],
        'best_model_path': best['merged_model_path']
    }


def print_results_table(results: List[Dict], metric_name: str = 'fitness'):
    """Print results in a formatted table"""
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

    # Calculate improvements
    scratch_value = scratch_baseline['metrics'][metric]
    finetuned_value = finetuned_baseline['metrics'][metric]
    best_value = best_metrics[metric]

    improvement_from_scratch = ((best_value - scratch_value) / (scratch_value + 1e-8)) * 100
    improvement_from_finetuned = ((best_value - finetuned_value) / (finetuned_value + 1e-8)) * 100

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

Detailed Metrics (Best Alpha):
  Precision:    {best_metrics['precision']:.4f}
  Recall:       {best_metrics['recall']:.4f}
  mAP@.5:       {best_metrics['map50']:.4f}
  mAP@.5:.95:   {best_metrics['map']:.4f}
  Fitness:      {best_metrics['fitness']:.4f}

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

    # 3. Generate Alpha Lists
    coarse_alphas = generate_alpha_list(args.alpha_min, args.alpha_max, args.focus_range, args.skip_zero)
    print(f"\n📊 Coarse search alphas ({len(coarse_alphas)} values): {coarse_alphas}")

    # 4. Evaluate Baselines
    print("\n" + "="*80)
    print("📏 EVALUATING BASELINE MODELS")
    print("="*80)

    print("\n🔹 Scratch baseline...")
    scratch_eval_dir = output_dir / 'baseline_scratch'
    scratch_metrics = evaluate_model(
        args.scratch, args.data, args.img_size, args.batch_size,
        args.conf_thres, args.iou_thres, args.device, args.workers,
        str(scratch_eval_dir)
    )
    scratch_baseline = {'alpha': 0.0, 'metrics': scratch_metrics, 'merged_model_path': args.scratch}
    print(f"Scratch: P={scratch_metrics['precision']:.3f}, R={scratch_metrics['recall']:.3f}, "
          f"mAP@.5={scratch_metrics['map50']:.3f}, mAP@.5:.95={scratch_metrics['map']:.3f}, "
          f"Fitness={scratch_metrics['fitness']:.3f}")

    print("\n🔹 Finetuned baseline...")
    finetuned_eval_dir = output_dir / 'baseline_finetuned'
    finetuned_metrics = evaluate_model(
        args.finetuned, args.data, args.img_size, args.batch_size,
        args.conf_thres, args.iou_thres, args.device, args.workers,
        str(finetuned_eval_dir)
    )
    finetuned_baseline = {'alpha': 1.0, 'metrics': finetuned_metrics, 'merged_model_path': args.finetuned}
    print(f"Finetuned: P={finetuned_metrics['precision']:.3f}, R={finetuned_metrics['recall']:.3f}, "
          f"mAP@.5={finetuned_metrics['map50']:.3f}, mAP@.5:.95={finetuned_metrics['map']:.3f}, "
          f"Fitness={finetuned_metrics['fitness']:.3f}")

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

    # 8. Find Overall Best
    best_overall = find_best_alpha(all_results, args.metric)

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
