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
    """Print analysis results with executive summary and recommendations"""
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

    # Get best epoch data
    best_epoch_data = results[best_epoch]['val_sets']
    best_data = best_epoch_data[val_set]

    # Extract metrics for the selected class
    if class_name == 'all':
        if best_data['overall']:
            p, r, map50, map_val = best_data['overall'][0:4]
            selected_fitness = fitness(p, r, map50, map_val)
        else:
            p, r, map50, map_val = 0, 0, 0, 0
            selected_fitness = 0
    else:
        if class_name in best_data['per_class']:
            class_data = best_data['per_class'][class_name]
            p = class_data['P']
            r = class_data['R']
            map50 = class_data['mAP@.5']
            map_val = class_data['mAP@.5:.95']
            selected_fitness = fitness(p, r, map50, map_val)
        else:
            p, r, map50, map_val = 0, 0, 0, 0
            selected_fitness = 0

    # Calculate statistics for insights
    val_scores = {}
    for epoch, epoch_data in results.items():
        if val_set not in epoch_data['val_sets']:
            continue
        vs_data = epoch_data['val_sets'][val_set]
        if class_name == 'all':
            if vs_data['overall']:
                p_t, r_t, map50_t, map_val_t = vs_data['overall'][0:4]
                val_scores[epoch] = {
                    'P': p_t, 'R': r_t, 'mAP@.5': map50_t, 'mAP@.5:.95': map_val_t,
                    'fitness': fitness(p_t, r_t, map50_t, map_val_t)
                }
        else:
            if class_name in vs_data['per_class']:
                cls_data = vs_data['per_class'][class_name]
                val_scores[epoch] = {
                    'P': cls_data['P'], 'R': cls_data['R'],
                    'mAP@.5': cls_data['mAP@.5'], 'mAP@.5:.95': cls_data['mAP@.5:.95'],
                    'fitness': fitness(cls_data['P'], cls_data['R'], cls_data['mAP@.5'], cls_data['mAP@.5:.95'])
                }

    # Check for overfitting
    fitness_values = [v['fitness'] for v in val_scores.values()]
    max_fitness = max(fitness_values)
    mean_fitness = sum(fitness_values) / len(fitness_values)
    is_overfit = best_epoch < len(results) * 0.7 and selected_fitness < max_fitness * 0.95

    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"🎯 EXECUTIVE SUMMARY")
    print(f"{'='*80}")
    print(f"Analysis: {val_set} / {class_name} / optimize for {metric}")
    print(f"\n✅ Best Epoch: {best_epoch}")
    print(f"   {metric}: {best_value:.4f} | fitness: {selected_fitness:.4f}")

    # Recommendation
    recommendation = "✅ RECOMMENDED"
    concerns = []

    if is_overfit:
        recommendation = "⚠️ CAUTION"
        concerns.append("Possible overfitting detected")

    # Check if early epoch is suspiciously good
    sorted_epochs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_epochs) > 1 and sorted_epochs[1][0] < len(results) * 0.2:
        concerns.append(f"Epoch {sorted_epochs[1][0]} (early) is 2nd best - verify data quality")

    # Check metric alignment
    fitness_best = max(val_scores, key=lambda e: val_scores[e]['fitness'])
    map_best = max(val_scores, key=lambda e: val_scores[e]['mAP@.5:.95'])
    if fitness_best != map_best:
        concerns.append(f"Best epoch differs: fitness→{fitness_best}, mAP@.5:.95→{map_best}")

    print(f"\n{recommendation}")
    if concerns:
        print(f"\n⚠️ Concerns:")
        for concern in concerns:
            print(f"   • {concern}")
    else:
        print(f"   ✓ No major concerns detected")
        print(f"   ✓ Stable training (improvement: {((selected_fitness/mean_fitness - 1) * 100):.1f}% vs mean)")

    print(f"\n{'='*80}")
    print(f"📊 BASIC INFO")
    print(f"{'='*80}")
    print(f"Results file: {args.results}")
    print(f"Validation set: {val_set}")
    print(f"Class: {class_name}")
    print(f"Metric: {metric}")
    print(f"Total epochs: {len(results)}")

    # Best epoch details
    print(f"\n🏆 Best Epoch Details (Epoch {best_epoch})")
    if class_name == 'all':
        print(f"   Overall metrics:")
    else:
        print(f"   Class '{class_name}' metrics:")
    print(f"   • Precision:    {p:.4f}")
    print(f"   • Recall:       {r:.4f}")
    print(f"   • mAP@.5:       {map50:.4f}")
    print(f"   • mAP@.5:.95:   {map_val:.4f}")
    print(f"   • Fitness:      {selected_fitness:.4f}")
    if class_name != 'all' and class_name in best_data['per_class']:
        print(f"   • Images:       {best_data['per_class'][class_name]['images']}")

    # Top 5 epochs
    print(f"\n📈 Top 5 Epochs (by {metric}):")
    for rank, (epoch, score) in enumerate(sorted_epochs[:5], 1):
        marker = "🏆" if epoch == best_epoch else "  "
        print(f"   {marker} {rank}. Epoch {epoch:3d}: {metric}={score:.4f}")

    # ========================================================================
    # VALIDATION SET COMPARISON
    # ========================================================================
    if class_name != 'all' and len(best_epoch_data) > 1:
        print(f"\n{'='*80}")
        print(f"📊 VALIDATION SET COMPARISON (Class: {class_name})")
        print(f"{'='*80}")

        # Collect metrics across validation sets
        vs_comparison = {}
        for vs_name in best_epoch_data.keys():
            vs_data = best_epoch_data[vs_name]
            if class_name in vs_data['per_class']:
                cls_data = vs_data['per_class'][class_name]
                vs_comparison[vs_name] = {
                    'P': cls_data['P'],
                    'R': cls_data['R'],
                    'mAP@.5': cls_data['mAP@.5'],
                    'mAP@.5:.95': cls_data['mAP@.5:.95'],
                    'fitness': fitness(cls_data['P'], cls_data['R'], cls_data['mAP@.5'], cls_data['mAP@.5:.95']),
                    'images': cls_data['images']
                }

        if len(vs_comparison) > 1:
            # Find best performing validation set
            best_vs = max(vs_comparison, key=lambda k: vs_comparison[k]['fitness'])

            print(f"\nPerformance at Epoch {best_epoch}:")
            print(f"{'Val Set':<15} {'Fitness':>10} {'Precision':>12} {'Recall':>10} {'mAP@.5':<10} {'mAP@.5:.95':>12} {'Images':>8}")
            print(f"{'-'*85}")

            for vs_name in sorted(vs_comparison.keys()):
                vs_metrics = vs_comparison[vs_name]
                marker = "⭐" if vs_name == best_vs else "  "
                print(f"{marker} {vs_name:<13} {vs_metrics['fitness']:>10.4f} {vs_metrics['P']:>12.4f} "
                      f"{vs_metrics['R']:>10.4f} {vs_metrics['mAP@.5']:<10.4f} {vs_metrics['mAP@.5:.95']:>12.4f} "
                      f"{vs_metrics['images']:>8}")

            # Calculate performance differences
            print(f"\n💡 Key Findings:")
            if best_vs:
                best_fitness = vs_comparison[best_vs]['fitness']
                for vs_name in vs_comparison.keys():
                    if vs_name != best_vs:
                        diff = ((vs_comparison[best_vs]['fitness'] / vs_comparison[vs_name]['fitness']) - 1) * 100
                        if abs(diff) > 5:
                            print(f"   • {best_vs} outperforms {vs_name} by {diff:.1f}%")

                # Check for large precision/recall gaps
                p_values = [v['P'] for v in vs_comparison.values()]
                r_values = [v['R'] for v in vs_comparison.values()]
                p_diff = max(p_values) - min(p_values)
                r_diff = max(r_values) - min(r_values)

                if p_diff > 0.15:
                    print(f"   ⚠️ Large Precision gap ({p_diff:.2f}) across validation sets")
                if r_diff > 0.15:
                    print(f"   ⚠️ Large Recall gap ({r_diff:.2f}) across validation sets")

    # ========================================================================
    # CLASS PERFORMANCE RANKING
    # ========================================================================
    if class_name != 'all':
        # Get all classes in the selected validation set
        all_classes_fitness = {}
        for cls in best_data['per_class'].keys():
            cls_data = best_data['per_class'][cls]
            all_classes_fitness[cls] = fitness(cls_data['P'], cls_data['R'], cls_data['mAP@.5'], cls_data['mAP@.5:.95'])

        if len(all_classes_fitness) > 1:
            print(f"\n{'='*80}")
            print(f"🏆 CLASS PERFORMANCE RANKING (Epoch {best_epoch}, {val_set})")
            print(f"{'='*80}")

            sorted_classes = sorted(all_classes_fitness.items(), key=lambda x: x[1], reverse=True)
            selected_class_rank = [i for i, (cls, _) in enumerate(sorted_classes, 1) if cls == class_name][0]
            best_class = sorted_classes[0][0]
            best_class_fitness = sorted_classes[0][1]

            print(f"\n{'Rank':<6} {'Class':<20} {'Fitness':>10} {'Bar':>20}")
            print(f"{'-'*60}")

            for rank, (cls, cls_fitness) in enumerate(sorted_classes, 1):
                bar_length = int((cls_fitness / best_class_fitness) * 20)
                bar = "█" * bar_length
                marker = "👉" if cls == class_name else "  "
                print(f"{marker} {rank:<4} {cls:<20} {cls_fitness:>10.4f}  {bar}")

            # Performance gap analysis
            if class_name != best_class:
                improvement_potential = ((best_class_fitness / selected_fitness) - 1) * 100
                print(f"\n💡 Your class '{class_name}' ranks #{selected_class_rank}/{len(sorted_classes)}")
                print(f"   Potential improvement: {improvement_potential:.1f}% (to match best class '{best_class}')")
            else:
                print(f"\n🎉 Your class '{class_name}' is the best performing class!")

    # ========================================================================
    # DETAILED PERFORMANCE
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"📋 DETAILED PERFORMANCE (Epoch {best_epoch}) - All Validation Sets")
    print(f"{'='*80}")

    for vs_name in sorted(best_epoch_data.keys()):
        vs_data = best_epoch_data[vs_name]

        print(f"\n🔹 {vs_name}:")

        # Overall metrics
        if vs_data['overall']:
            p_vs, r_vs, map50_vs, map_val_vs = vs_data['overall'][0:4]
            fit_vs = fitness(p_vs, r_vs, map50_vs, map_val_vs)
            print(f"   Overall: P={p_vs:.4f}, R={r_vs:.4f}, mAP@.5={map50_vs:.4f}, mAP@.5:.95={map_val_vs:.4f}, fitness={fit_vs:.4f}")

        # Per-class metrics
        if vs_data['per_class']:
            print(f"   Per-class:")
            for cls_name in sorted(vs_data['per_class'].keys()):
                cls_data = vs_data['per_class'][cls_name]
                cls_fit = fitness(cls_data['P'], cls_data['R'], cls_data['mAP@.5'], cls_data['mAP@.5:.95'])
                marker = "👉" if cls_name == class_name else "  "
                print(f"   {marker} • {cls_name:15s}: P={cls_data['P']:.4f}, R={cls_data['R']:.4f}, "
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

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"💡 NEXT STEPS & RECOMMENDATIONS")
    print(f"{'='*80}")

    recommendations = []

    # Recommendation 1: Early epoch warning
    if len(sorted_epochs) > 1 and sorted_epochs[1][0] < len(results) * 0.2:
        recommendations.append({
            'priority': '⚠️ HIGH',
            'action': f"Verify Epoch {sorted_epochs[1][0]} data quality",
            'reason': f"Early epoch ({sorted_epochs[1][0]}) is 2nd best - unusual pattern",
            'command': f"# Check epoch {sorted_epochs[1][0]} metrics:\npython analyze_results.py --results {args.results} --val-set {val_set} --class {class_name} --metric fitness"
        })

    # Recommendation 2: Validation set performance gap
    if class_name != 'all' and len(best_epoch_data) > 1:
        vs_fitness_vals = {}
        for vs_name in best_epoch_data.keys():
            vs_data = best_epoch_data[vs_name]
            if class_name in vs_data['per_class']:
                cls_data = vs_data['per_class'][class_name]
                vs_fitness_vals[vs_name] = fitness(cls_data['P'], cls_data['R'], cls_data['mAP@.5'], cls_data['mAP@.5:.95'])

        if len(vs_fitness_vals) > 1:
            best_vs_name = max(vs_fitness_vals, key=vs_fitness_vals.get)
            worst_vs_name = min(vs_fitness_vals, key=vs_fitness_vals.get)
            performance_gap = ((vs_fitness_vals[best_vs_name] / vs_fitness_vals[worst_vs_name]) - 1) * 100

            if performance_gap > 10 and val_set != best_vs_name:
                recommendations.append({
                    'priority': '📊 MEDIUM',
                    'action': f"Re-analyze with --val-set {best_vs_name}",
                    'reason': f"{best_vs_name} shows {performance_gap:.1f}% better performance",
                    'command': f"python analyze_results.py --results {args.results} --val-set {best_vs_name} --class {class_name} --metric {metric}"
                })

    # Recommendation 3: Class improvement potential
    if class_name != 'all' and len(all_classes_fitness) > 1:
        best_class_perf = max(all_classes_fitness.values())
        if selected_fitness < best_class_perf * 0.8:  # More than 20% gap
            improvement_pct = ((best_class_perf / selected_fitness) - 1) * 100
            recommendations.append({
                'priority': '💪 LOW',
                'action': f"Analyze best performing class '{best_class}'",
                'reason': f"'{class_name}' has {improvement_pct:.1f}% improvement potential",
                'command': f"python analyze_results.py --results {args.results} --val-set {val_set} --class {best_class} --metric fitness"
            })

    # Recommendation 4: Production deployment
    if not recommendations or len(recommendations) == 0:
        recommendations.append({
            'priority': '✅ READY',
            'action': f"Use Epoch {best_epoch} for deployment",
            'reason': "No significant concerns detected",
            'command': f"# Use weights: runs/train/exp/weights/epoch_{best_epoch}.pt"
        })
    else:
        recommendations.append({
            'priority': '✅ READY',
            'action': f"Epoch {best_epoch} can be used",
            'reason': f"Address concerns above if critical for your use case",
            'command': f"# Weights: runs/train/exp/weights/epoch_{best_epoch}.pt"
        })

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['action']}")
        print(f"   Why: {rec['reason']}")
        if 'command' in rec:
            print(f"   How: {rec['command']}")

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
