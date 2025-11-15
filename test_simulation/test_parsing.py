#!/usr/bin/env python3
"""Simple test to verify result parsing logic without numpy"""

import re
from pathlib import Path

def simple_parse_results(results_file):
    """Simplified parsing for verification"""
    results = {}
    current_epoch = None

    with open(results_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()

        # Match epoch line
        epoch_match = re.match(r'^\s*(\d+)/\d+\s+', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            results[current_epoch] = {
                'val_sets': {}
            }
            continue

        if current_epoch is None:
            continue

        # Match validation set overall results
        val_overall_match = re.match(r'\s*\[(\s*\w+)\]\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)', line)
        if val_overall_match:
            val_name = val_overall_match.group(1).strip()
            p = float(val_overall_match.group(2))
            r = float(val_overall_match.group(3))
            map50 = float(val_overall_match.group(4))
            map95 = float(val_overall_match.group(5))

            if val_name not in results[current_epoch]['val_sets']:
                results[current_epoch]['val_sets'][val_name] = {
                    'overall': {'P': p, 'R': r, 'mAP@.5': map50, 'mAP@.5:.95': map95},
                    'per_class': {}
                }
            continue

        # Match per-class results
        class_match = re.match(r'\s*\[(\s*\w+)\]\[(\s*[\w\s]+)\]\s*Images:\s*(\d+),\s*P:\s*([\d.e+-]+),\s*R:\s*([\d.e+-]+),\s*mAP@\.5:\s*([\d.e+-]+),\s*mAP@\.5:.95:\s*([\d.e+-]+)', line)
        if class_match:
            val_name = class_match.group(1).strip()
            class_name = class_match.group(2).strip()
            images = int(class_match.group(3))
            p = float(class_match.group(4))
            r = float(class_match.group(5))
            map50 = float(class_match.group(6))
            map95 = float(class_match.group(7))

            if val_name not in results[current_epoch]['val_sets']:
                results[current_epoch]['val_sets'][val_name] = {
                    'overall': None,
                    'per_class': {}
                }

            results[current_epoch]['val_sets'][val_name]['per_class'][class_name] = {
                'P': p, 'R': r, 'mAP@.5': map50, 'mAP@.5:.95': map95, 'images': images
            }

    return results

def calculate_fitness(p, r, map50, map95):
    """Calculate fitness score"""
    return 0.0 * p + 0.0 * r + 0.1 * map50 + 0.9 * map95

def find_best_epoch_simple(results, val_set, metric_name='fitness'):
    """Find best epoch"""
    scores = {}

    for epoch, data in results.items():
        if val_set not in data['val_sets']:
            continue

        val_data = data['val_sets'][val_set]
        if val_data['overall'] is None:
            continue

        overall = val_data['overall']
        p = overall['P']
        r = overall['R']
        map50 = overall['mAP@.5']
        map95 = overall['mAP@.5:.95']

        if metric_name == 'fitness':
            score = calculate_fitness(p, r, map50, map95)
        elif metric_name == 'map50':
            score = map50
        elif metric_name == 'map':
            score = map95
        elif metric_name == 'P':
            score = p
        elif metric_name == 'R':
            score = r
        else:
            score = calculate_fitness(p, r, map50, map95)

        scores[epoch] = score

    if not scores:
        return None, None

    best_epoch = max(scores, key=scores.get)
    best_score = scores[best_epoch]

    return best_epoch, best_score

# Test parsing
print("="*80)
print("Testing Result Parsing")
print("="*80)

results_file = "test_simulation/sample_results.txt"
results = simple_parse_results(results_file)

print(f"\n✅ Parsed {len(results)} epochs")

# Show available validation sets
all_val_sets = set()
all_classes = {}

for epoch_data in results.values():
    for val_set in epoch_data['val_sets'].keys():
        all_val_sets.add(val_set)
        if val_set not in all_classes:
            all_classes[val_set] = set()
        all_classes[val_set].update(epoch_data['val_sets'][val_set]['per_class'].keys())

print(f"\n📊 Available Validation Sets:")
for val_set in sorted(all_val_sets):
    print(f"  - {val_set}")
    if val_set in all_classes:
        print(f"    Classes: {', '.join(sorted(all_classes[val_set]))}")

# Test 1: Find best epoch for test1 (fitness)
print("\n" + "="*80)
print("Test 1: Best Epoch for test1 (fitness)")
print("="*80)
best_epoch, best_score = find_best_epoch_simple(results, 'test1', 'fitness')
if best_epoch is not None:
    print(f"🏆 Best Epoch: {best_epoch}")
    print(f"   Fitness: {best_score:.6f}")

    # Show details
    val_data = results[best_epoch]['val_sets']['test1']['overall']
    print(f"   P: {val_data['P']:.4f}")
    print(f"   R: {val_data['R']:.4f}")
    print(f"   mAP@.5: {val_data['mAP@.5']:.4f}")
    print(f"   mAP@.5:.95: {val_data['mAP@.5:.95']:.4f}")

# Test 2: Find best epoch for test2 (fitness)
print("\n" + "="*80)
print("Test 2: Best Epoch for test2 (fitness)")
print("="*80)
best_epoch, best_score = find_best_epoch_simple(results, 'test2', 'fitness')
if best_epoch is not None:
    print(f"🏆 Best Epoch: {best_epoch}")
    print(f"   Fitness: {best_score:.6f}")

    val_data = results[best_epoch]['val_sets']['test2']['overall']
    print(f"   P: {val_data['P']:.4f}")
    print(f"   R: {val_data['R']:.4f}")
    print(f"   mAP@.5: {val_data['mAP@.5']:.4f}")
    print(f"   mAP@.5:.95: {val_data['mAP@.5:.95']:.4f}")

# Test 3: Find best epoch for Combined (fitness)
print("\n" + "="*80)
print("Test 3: Best Epoch for Combined (fitness)")
print("="*80)
best_epoch, best_score = find_best_epoch_simple(results, 'Combined', 'fitness')
if best_epoch is not None:
    print(f"🏆 Best Epoch: {best_epoch}")
    print(f"   Fitness: {best_score:.6f}")

    val_data = results[best_epoch]['val_sets']['Combined']['overall']
    print(f"   P: {val_data['P']:.4f}")
    print(f"   R: {val_data['R']:.4f}")
    print(f"   mAP@.5: {val_data['mAP@.5']:.4f}")
    print(f"   mAP@.5:.95: {val_data['mAP@.5:.95']:.4f}")

# Test 4: Find best epoch for test1 (mAP@.5)
print("\n" + "="*80)
print("Test 4: Best Epoch for test1 (mAP@.5)")
print("="*80)
best_epoch, best_score = find_best_epoch_simple(results, 'test1', 'map50')
if best_epoch is not None:
    print(f"🏆 Best Epoch: {best_epoch}")
    print(f"   mAP@.5: {best_score:.6f}")

# Test 5: Per-class analysis
print("\n" + "="*80)
print("Test 5: Per-Class Analysis (test2, person)")
print("="*80)

person_scores = {}
for epoch, data in results.items():
    if 'test2' in data['val_sets']:
        per_class = data['val_sets']['test2']['per_class']
        if 'person' in per_class:
            class_data = per_class['person']
            fitness_score = calculate_fitness(
                class_data['P'],
                class_data['R'],
                class_data['mAP@.5'],
                class_data['mAP@.5:.95']
            )
            person_scores[epoch] = fitness_score

if person_scores:
    best_epoch = max(person_scores, key=person_scores.get)
    print(f"🏆 Best Epoch for person class: {best_epoch}")
    print(f"   Fitness: {person_scores[best_epoch]:.6f}")

    class_data = results[best_epoch]['val_sets']['test2']['per_class']['person']
    print(f"   P: {class_data['P']:.4f}")
    print(f"   R: {class_data['R']:.4f}")
    print(f"   mAP@.5: {class_data['mAP@.5']:.4f}")
    print(f"   mAP@.5:.95: {class_data['mAP@.5:.95']:.4f}")
    print(f"   Images: {class_data['images']}")

# Test 6: Show all epochs for Combined
print("\n" + "="*80)
print("Test 6: All Epochs Fitness Scores (Combined)")
print("="*80)

combined_scores = {}
for epoch, data in results.items():
    if 'Combined' in data['val_sets'] and data['val_sets']['Combined']['overall']:
        overall = data['val_sets']['Combined']['overall']
        fitness_score = calculate_fitness(
            overall['P'],
            overall['R'],
            overall['mAP@.5'],
            overall['mAP@.5:.95']
        )
        combined_scores[epoch] = fitness_score

for epoch in sorted(combined_scores.keys()):
    score = combined_scores[epoch]
    marker = "🏆" if epoch == max(combined_scores, key=combined_scores.get) else "  "
    print(f"{marker} Epoch {epoch}: {score:.6f}")

print("\n" + "="*80)
print("✅ All Tests Completed Successfully!")
print("="*80)
