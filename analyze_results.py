#!/usr/bin/env python3
"""
Result Analysis Tool for YOLOv7 Training

This script analyzes the results.txt file to find the best epoch based on:
- Validation set selection (test1, test2, Combined, or custom)
- Class selection (all classes or specific class)
- Metric selection (fitness, mAP@0.5, mAP@0.5:0.95, etc.)

Usage:
    python analyze_results.py --results results.txt
    python analyze_results.py --results results.txt --val-set test1 --class person
    python analyze_results.py --results results.txt --val-set Combined --metric map50
"""

import argparse
import re
import numpy as np
from pathlib import Path


def fitness(p, r, map50, map):
    """
    Calculate fitness score (same as training)
    weights: [P, R, mAP@0.5, mAP@0.5:0.95]
    """
    w = [0.0, 0.0, 0.1, 0.9]
    return w[0] * p + w[1] * r + w[2] * map50 + w[3] * map


def parse_results_file(results_file):
    """
    Parse results.txt file and extract all metrics

    Returns:
        dict: {
            epoch_num: {
                'train_metrics': {...},
                'val_sets': {
                    'test1': {
                        'overall': [P, R, mAP@.5, mAP@.5:.95, ...],
                        'per_class': {
                            'person': [P, R, mAP@.5, mAP@.5:.95, images],
                            ...
                        }
                    },
                    'test2': {...},
                    'Combined': {...}
                }
            }
        }
    """
    results = {}
    current_epoch = None

    with open(results_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()

        # Match epoch line (starts with epoch number)
        # Format: epoch, gpu_mem, box_loss, obj_loss, cls_loss, total_loss, targets, img_size
        epoch_match = re.match(r'^\s*(\d+)/\d+\s+', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            results[current_epoch] = {
                'train_metrics': {},
                'val_sets': {}
            }
            # Parse training metrics
            parts = line.split()
            if len(parts) >= 8:
                results[current_epoch]['train_metrics'] = {
                    'gpu_mem': parts[1],
                    'box_loss': float(parts[2]),
                    'obj_loss': float(parts[3]),
                    'cls_loss': float(parts[4]),
                    'total_loss': float(parts[5]),
                    'targets': int(parts[6]),
                    'img_size': int(parts[7])
                }
            continue

        if current_epoch is None:
            continue

        # Match validation set overall results
        # Format: [val_name] P R mAP@.5 mAP@.5:.95 val_box val_obj val_cls
        val_overall_match = re.match(r'\s*\[(\s*\w+)\]\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)', line)
        if val_overall_match:
            val_name = val_overall_match.group(1).strip()
            metrics = [float(val_overall_match.group(i)) for i in range(2, 9)]

            if val_name not in results[current_epoch]['val_sets']:
                results[current_epoch]['val_sets'][val_name] = {
                    'overall': metrics,
                    'per_class': {}
                }
            else:
                results[current_epoch]['val_sets'][val_name]['overall'] = metrics
            continue

        # Match per-class results
        # Format: [val_name][class_name] Images: N, P: X, R: X, mAP@.5: X, mAP@.5:.95: X
        class_match = re.match(r'\s*\[(\s*\w+)\]\[(\s*[\w\s]+)\]\s*Images:\s*(\d+),\s*P:\s*([\d.e+-]+),\s*R:\s*([\d.e+-]+),\s*mAP@\.5:\s*([\d.e+-]+),\s*mAP@\.5:.95:\s*([\d.e+-]+)', line)
        if class_match:
            val_name = class_match.group(1).strip()
            class_name = class_match.group(2).strip()
            images = int(class_match.group(3))
            p = float(class_match.group(4))
            r = float(class_match.group(5))
            map50 = float(class_match.group(6))
            map = float(class_match.group(7))

            if val_name not in results[current_epoch]['val_sets']:
                results[current_epoch]['val_sets'][val_name] = {
                    'overall': None,
                    'per_class': {}
                }

            results[current_epoch]['val_sets'][val_name]['per_class'][class_name] = {
                'P': p,
                'R': r,
                'mAP@.5': map50,
                'mAP@.5:.95': map,
                'images': images
            }

    return results


def find_best_epoch(results, val_set='test1', class_name='all', metric='fitness'):
    """
    Find the best epoch based on specified criteria

    Args:
        results: Parsed results dictionary
        val_set: Validation set name (test1, test2, Combined)
        class_name: Class name or 'all' for overall
        metric: Metric to optimize (fitness, map50, map, precision, recall)

    Returns:
        tuple: (best_epoch, best_value, all_scores)
    """
    scores = {}

    for epoch, data in results.items():
        if val_set not in data['val_sets']:
            continue

        val_data = data['val_sets'][val_set]

        if class_name == 'all':
            # Use overall metrics
            if val_data['overall'] is None:
                continue
            p, r, map50, map_val = val_data['overall'][0:4]
        else:
            # Use per-class metrics
            if class_name not in val_data['per_class']:
                continue
            class_data = val_data['per_class'][class_name]
            p = class_data['P']
            r = class_data['R']
            map50 = class_data['mAP@.5']
            map_val = class_data['mAP@.5:.95']

        # Calculate score based on metric
        if metric == 'fitness':
            score = fitness(p, r, map50, map_val)
        elif metric == 'map50' or metric == 'mAP@.5':
            score = map50
        elif metric == 'map' or metric == 'mAP@.5:.95':
            score = map_val
        elif metric == 'precision' or metric == 'P':
            score = p
        elif metric == 'recall' or metric == 'R':
            score = r
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores[epoch] = score

    if not scores:
        return None, None, {}

    best_epoch = max(scores, key=scores.get)
    best_value = scores[best_epoch]

    return best_epoch, best_value, scores


def print_analysis(results, val_set, class_name, metric):
    """Print analysis results"""
    best_epoch, best_value, scores = find_best_epoch(results, val_set, class_name, metric)

    if best_epoch is None:
        print(f"\n❌ No results found for val_set='{val_set}', class='{class_name}'")
        print("\nAvailable validation sets:")
        val_sets = set()
        for epoch_data in results.values():
            val_sets.update(epoch_data['val_sets'].keys())
        for vs in sorted(val_sets):
            print(f"  - {vs}")

        if val_set in val_sets:
            print(f"\nAvailable classes in '{val_set}':")
            classes = set()
            for epoch_data in results.values():
                if val_set in epoch_data['val_sets']:
                    classes.update(epoch_data['val_sets'][val_set]['per_class'].keys())
            print("  - all (overall metrics)")
            for cls in sorted(classes):
                print(f"  - {cls}")
        return

    print(f"\n{'='*80}")
    print(f"Analysis Results")
    print(f"{'='*80}")
    print(f"Results file: {args.results}")
    print(f"Validation set: {val_set}")
    print(f"Class: {class_name}")
    print(f"Metric: {metric}")
    print(f"\n🏆 Best Epoch: {best_epoch}")
    print(f"   Best {metric}: {best_value:.6f}")

    # Print detailed metrics for best epoch
    best_data = results[best_epoch]['val_sets'][val_set]

    if class_name == 'all':
        if best_data['overall']:
            p, r, map50, map_val = best_data['overall'][0:4]
            print(f"\n   Detailed Metrics:")
            print(f"   - Precision: {p:.6f}")
            print(f"   - Recall: {r:.6f}")
            print(f"   - mAP@.5: {map50:.6f}")
            print(f"   - mAP@.5:.95: {map_val:.6f}")
            print(f"   - Fitness: {fitness(p, r, map50, map_val):.6f}")
    else:
        if class_name in best_data['per_class']:
            class_data = best_data['per_class'][class_name]
            print(f"\n   Detailed Metrics:")
            print(f"   - Precision: {class_data['P']:.6f}")
            print(f"   - Recall: {class_data['R']:.6f}")
            print(f"   - mAP@.5: {class_data['mAP@.5']:.6f}")
            print(f"   - mAP@.5:.95: {class_data['mAP@.5:.95']:.6f}")
            print(f"   - Images: {class_data['images']}")
            print(f"   - Fitness: {fitness(class_data['P'], class_data['R'], class_data['mAP@.5'], class_data['mAP@.5:.95']):.6f}")

    # Print top 5 epochs
    print(f"\n📊 Top 5 Epochs:")
    sorted_epochs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for rank, (epoch, score) in enumerate(sorted_epochs, 1):
        print(f"   {rank}. Epoch {epoch:3d}: {metric} = {score:.6f}")

    # Show all validation sets performance at best epoch
    print(f"\n{'='*80}")
    print(f"📈 Best Model Performance (Epoch {best_epoch}) Across All Validation Sets")
    print(f"{'='*80}")

    best_epoch_data = results[best_epoch]['val_sets']

    for vs_name in sorted(best_epoch_data.keys()):
        vs_data = best_epoch_data[vs_name]

        print(f"\n🔹 {vs_name}:")

        # Overall metrics
        if vs_data['overall']:
            p, r, map50, map_val = vs_data['overall'][0:4]
            fit = fitness(p, r, map50, map_val)
            print(f"   Overall: P={p:.4f}, R={r:.4f}, mAP@.5={map50:.4f}, mAP@.5:.95={map_val:.4f}, fitness={fit:.4f}")

        # Per-class metrics
        if vs_data['per_class']:
            print(f"   Per-class:")
            for cls_name in sorted(vs_data['per_class'].keys()):
                cls_data = vs_data['per_class'][cls_name]
                cls_fit = fitness(cls_data['P'], cls_data['R'], cls_data['mAP@.5'], cls_data['mAP@.5:.95'])
                print(f"     • {cls_name:15s}: P={cls_data['P']:.4f}, R={cls_data['R']:.4f}, "
                      f"mAP@.5={cls_data['mAP@.5']:.4f}, mAP@.5:.95={cls_data['mAP@.5:.95']:.4f}, "
                      f"fitness={cls_fit:.4f} (images={cls_data['images']})")

    # Show statistics for selected validation set
    if val_set in best_epoch_data:
        print(f"\n{'='*80}")
        print(f"📊 Selected Validation Set Statistics: {val_set}")
        print(f"{'='*80}")

        # Calculate statistics across all epochs for this val set
        val_scores = {}
        for epoch, epoch_data in results.items():
            if val_set not in epoch_data['val_sets']:
                continue

            vs_data = epoch_data['val_sets'][val_set]

            if class_name == 'all':
                if vs_data['overall']:
                    p, r, map50, map_val = vs_data['overall'][0:4]
                    val_scores[epoch] = {
                        'P': p, 'R': r, 'mAP@.5': map50, 'mAP@.5:.95': map_val,
                        'fitness': fitness(p, r, map50, map_val)
                    }
            else:
                if class_name in vs_data['per_class']:
                    cls_data = vs_data['per_class'][class_name]
                    val_scores[epoch] = {
                        'P': cls_data['P'], 'R': cls_data['R'],
                        'mAP@.5': cls_data['mAP@.5'], 'mAP@.5:.95': cls_data['mAP@.5:.95'],
                        'fitness': fitness(cls_data['P'], cls_data['R'], cls_data['mAP@.5'], cls_data['mAP@.5:.95'])
                    }

        if val_scores:
            # Calculate statistics
            metrics_list = {
                'P': [v['P'] for v in val_scores.values()],
                'R': [v['R'] for v in val_scores.values()],
                'mAP@.5': [v['mAP@.5'] for v in val_scores.values()],
                'mAP@.5:.95': [v['mAP@.5:.95'] for v in val_scores.values()],
                'fitness': [v['fitness'] for v in val_scores.values()]
            }

            print(f"\nTotal epochs analyzed: {len(val_scores)}")
            print(f"Class: {class_name}")
            print(f"\nMetric Statistics:")
            print(f"{'Metric':<15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Best Epoch':>12}")
            print(f"{'-'*60}")

            for metric_name, values in metrics_list.items():
                min_val = min(values)
                max_val = max(values)
                mean_val = sum(values) / len(values)
                best_ep = max(val_scores, key=lambda e: val_scores[e][metric_name])
                print(f"{metric_name:<15} {min_val:>10.4f} {max_val:>10.4f} {mean_val:>10.4f} {best_ep:>12d}")

    print(f"\n{'='*80}\n")


def plot_metric_curve(results, val_set, class_name, metric):
    """Plot metric curve over epochs (optional, requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt

        _, _, scores = find_best_epoch(results, val_set, class_name, metric)

        if not scores:
            return

        epochs = sorted(scores.keys())
        values = [scores[e] for e in epochs]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, values, 'b-', linewidth=2, label=f'{metric}')

        # Mark best epoch
        best_epoch = max(scores, key=scores.get)
        best_value = scores[best_epoch]
        plt.plot(best_epoch, best_value, 'r*', markersize=15, label=f'Best (Epoch {best_epoch})')

        plt.xlabel('Epoch')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} over Epochs\nVal Set: {val_set}, Class: {class_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_file = f'analysis_{val_set}_{class_name}_{metric}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📈 Plot saved to: {output_file}")

    except ImportError:
        pass  # matplotlib not available


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze YOLOv7 training results')
    parser.add_argument('--results', type=str, default='runs/train/exp/results.txt',
                        help='Path to results.txt file')
    parser.add_argument('--val-set', type=str, default='test1',
                        help='Validation set to analyze (test1, test2, Combined, etc.)')
    parser.add_argument('--class', dest='class_name', type=str, default='all',
                        help='Class name to analyze or "all" for overall metrics')
    parser.add_argument('--metric', type=str, default='fitness',
                        choices=['fitness', 'map50', 'map', 'mAP@.5', 'mAP@.5:.95', 'precision', 'recall', 'P', 'R'],
                        help='Metric to optimize')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plot (requires matplotlib)')
    parser.add_argument('--list', action='store_true',
                        help='List available validation sets and classes')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.results).exists():
        print(f"❌ Error: Results file not found: {args.results}")
        exit(1)

    # Parse results file
    print(f"📖 Parsing results file: {args.results}")
    results = parse_results_file(args.results)
    print(f"✅ Parsed {len(results)} epochs")

    # List mode
    if args.list:
        print(f"\n{'='*80}")
        print("Available Validation Sets and Classes")
        print(f"{'='*80}")

        val_sets = {}
        for epoch_data in results.values():
            for val_set in epoch_data['val_sets'].keys():
                if val_set not in val_sets:
                    val_sets[val_set] = set()
                val_sets[val_set].update(epoch_data['val_sets'][val_set]['per_class'].keys())

        for val_set in sorted(val_sets.keys()):
            print(f"\n📊 {val_set}:")
            print(f"   - all (overall metrics)")
            for class_name in sorted(val_sets[val_set]):
                print(f"   - {class_name}")

        print(f"\n{'='*80}\n")
        exit(0)

    # Analyze
    print_analysis(results, args.val_set, args.class_name, args.metric)

    # Plot if requested
    if args.plot:
        plot_metric_curve(results, args.val_set, args.class_name, args.metric)
