#!/usr/bin/env python3
"""Test best-val-set selection logic from train.py"""

class MockOpt:
    def __init__(self, best_val_set):
        self.best_val_set = best_val_set

class MockLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)
        print(f"[INFO] {msg}")

    def warning(self, msg):
        self.messages.append(msg)
        print(f"[WARNING] {msg}")

def simulate_best_val_set_selection(all_val_results, best_val_set):
    """
    Simulate the best-val-set selection logic from train.py
    This is the same logic as in train.py lines 536-571
    """
    logger = MockLogger()

    print(f"\n{'='*80}")
    print(f"Testing --best-val-set={best_val_set}")
    print(f"{'='*80}")

    if best_val_set == 'first':
        results = all_val_results[0]['results']
        maps = all_val_results[0]['maps']
        logger.info(f'Using {all_val_results[0]["name"]} for best model selection')
    elif best_val_set == 'last':
        results = all_val_results[-1]['results']
        maps = all_val_results[-1]['maps']
        logger.info(f'Using {all_val_results[-1]["name"]} for best model selection')
    elif best_val_set == 'Combined' or best_val_set == 'combined':
        # Use average of all validation sets
        if len(all_val_results) > 1:
            # Simple average (without numpy for this test)
            avg_results = [0] * len(all_val_results[0]['results'])
            for vr in all_val_results:
                for i, val in enumerate(vr['results']):
                    avg_results[i] += val
            results = tuple([v / len(all_val_results) for v in avg_results])

            # For maps, average across all validation sets
            avg_maps = [0] * len(all_val_results[0]['maps'])
            for vr in all_val_results:
                for i, val in enumerate(vr['maps']):
                    avg_maps[i] += val
            maps = [v / len(all_val_results) for v in avg_maps]

            logger.info('Using Combined (average) results for best model selection')
        else:
            results = all_val_results[0]['results']
            maps = all_val_results[0]['maps']
            logger.info(f'Only one validation set available, using {all_val_results[0]["name"]}')
    else:
        # Find specific validation set by name
        found = False
        for val_result in all_val_results:
            if val_result['name'] == best_val_set:
                results = val_result['results']
                maps = val_result['maps']
                logger.info(f'Using {best_val_set} for best model selection')
                found = True
                break
        if not found:
            logger.warning(f'Validation set "{best_val_set}" not found. Using first validation set: {all_val_results[0]["name"]}')
            results = all_val_results[0]['results']
            maps = all_val_results[0]['maps']

    # Calculate fitness (same as train.py)
    # fitness = 0.0*P + 0.0*R + 0.1*mAP@.5 + 0.9*mAP@.5:.95
    p, r, map50, map95 = results[0:4]
    fitness = 0.0 * p + 0.0 * r + 0.1 * map50 + 0.9 * map95

    print(f"\nSelected Results:")
    print(f"  P: {p:.6f}")
    print(f"  R: {r:.6f}")
    print(f"  mAP@.5: {map50:.6f}")
    print(f"  mAP@.5:.95: {map95:.6f}")
    print(f"  Fitness: {fitness:.6f}")

    return results, maps, fitness, logger.messages

# Create mock validation results (simulating all_val_results from train.py)
all_val_results = [
    {
        'name': 'test1',
        'results': (0.5456, 0.4112, 0.4789, 0.3212, 0.0186, 0.0096, 0.0067),  # P, R, mAP@.5, mAP@.5:.95, ...
        'maps': [0.312, 0.316, 0.336, 0.0]  # per-class mAP
    },
    {
        'name': 'test2',
        'results': (0.6123, 0.4801, 0.5789, 0.4212, 0.0143, 0.0074, 0.0050),
        'maps': [0.433, 0.403, 0.0, 0.0]
    }
]

print("="*80)
print("Simulating Best Validation Set Selection Logic")
print("="*80)
print(f"\nAvailable validation sets:")
for vr in all_val_results:
    p, r, map50, map95 = vr['results'][0:4]
    fitness = 0.1 * map50 + 0.9 * map95
    print(f"  - {vr['name']}: P={p:.4f}, R={r:.4f}, mAP@.5={map50:.4f}, mAP@.5:.95={map95:.4f}, fitness={fitness:.4f}")

# Test 1: --best-val-set first (default)
results, maps, fitness, messages = simulate_best_val_set_selection(all_val_results, 'first')
assert 'test1' in messages[0], "Should use test1 for 'first'"
assert abs(fitness - 0.336970) < 0.0001, f"Fitness should be ~0.337, got {fitness}"

# Test 2: --best-val-set last
results, maps, fitness, messages = simulate_best_val_set_selection(all_val_results, 'last')
assert 'test2' in messages[0], "Should use test2 for 'last'"
assert abs(fitness - 0.436970) < 0.0001, f"Fitness should be ~0.437, got {fitness}"

# Test 3: --best-val-set Combined
results, maps, fitness, messages = simulate_best_val_set_selection(all_val_results, 'Combined')
assert 'Combined' in messages[0] or 'average' in messages[0].lower(), "Should use Combined"
# Average: P=(0.5456+0.6123)/2=0.57895, R=(0.4112+0.4801)/2=0.44565
# mAP@.5=(0.4789+0.5789)/2=0.5289, mAP@.5:.95=(0.3212+0.4212)/2=0.3712
# fitness=0.1*0.5289+0.9*0.3712=0.38697
expected_fitness = 0.1 * 0.5289 + 0.9 * 0.3712
assert abs(fitness - expected_fitness) < 0.0001, f"Fitness should be ~{expected_fitness:.4f}, got {fitness}"

# Test 4: --best-val-set test1 (specific name)
results, maps, fitness, messages = simulate_best_val_set_selection(all_val_results, 'test1')
assert 'test1' in messages[0], "Should use test1"
assert abs(fitness - 0.336970) < 0.0001, f"Fitness should be ~0.337, got {fitness}"

# Test 5: --best-val-set test2 (specific name)
results, maps, fitness, messages = simulate_best_val_set_selection(all_val_results, 'test2')
assert 'test2' in messages[0], "Should use test2"
assert abs(fitness - 0.436970) < 0.0001, f"Fitness should be ~0.437, got {fitness}"

# Test 6: --best-val-set invalid_name (should fallback to first)
results, maps, fitness, messages = simulate_best_val_set_selection(all_val_results, 'invalid_name')
assert any('not found' in msg.lower() for msg in messages), "Should show warning"
assert abs(fitness - 0.336970) < 0.0001, f"Should fallback to first, fitness should be ~0.337, got {fitness}"

print("\n" + "="*80)
print("✅ All Best-Val-Set Logic Tests Passed!")
print("="*80)

# Summary
print("\n📝 Summary of --best-val-set behavior:")
print(f"  • 'first'    → Uses test1 (fitness: 0.3370)")
print(f"  • 'last'     → Uses test2 (fitness: 0.4370)")
print(f"  • 'Combined' → Uses average (fitness: 0.3870)")
print(f"  • 'test1'    → Uses test1 (fitness: 0.3370)")
print(f"  • 'test2'    → Uses test2 (fitness: 0.4370)")
print(f"  • invalid    → Fallback to first with warning")

print("\n💡 Recommendation:")
print("  For balanced performance across datasets: --best-val-set Combined")
print("  For best performance on test2: --best-val-set test2")
print("  For best performance on test1: --best-val-set test1")
