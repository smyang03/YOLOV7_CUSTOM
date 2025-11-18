#!/usr/bin/env python3
"""
WiSE-FT Simulation & Verification Tool

실제 평가 없이 빠르게 동작을 검증하고 예상 시간 계산
"""

import time
import json
from pathlib import Path
from typing import List, Dict
import numpy as np


def simulate_evaluation(alpha: float, valset: str, delay: float = 0.5) -> Dict:
    """
    단일 평가 시뮬레이션 (실제 test.py 대신)

    Args:
        alpha: Alpha value
        valset: Validation set name
        delay: Simulated evaluation time (seconds)

    Returns:
        Simulated metrics
    """
    time.sleep(delay)  # 실제 평가 시간 시뮬레이션

    # 시뮬레이션된 성능 (실제 패턴 반영)
    # valid1: 스크래치에서 좋음, 파인튜닝에서 나쁨
    # valid2: 스크래치에서 나쁨, 파인튜닝에서 좋음

    if valset == 'valid1':
        # valid1: α 증가하면 성능 하락
        base_fitness = 0.75
        degradation = alpha * 0.25  # α=1.0일 때 0.50
        fitness = base_fitness - degradation
    else:  # valid2
        # valid2: α 증가하면 성능 향상
        base_fitness = 0.45
        improvement = alpha * 0.35  # α=1.0일 때 0.80
        fitness = base_fitness + improvement

    # 중간 알파에서 노이즈 추가 (실제 비선형성 반영)
    if 0.3 <= alpha <= 0.7:
        noise = np.random.uniform(-0.05, 0.05)
        fitness += noise

    # 다른 메트릭 생성
    precision = fitness + np.random.uniform(-0.05, 0.05)
    recall = fitness + np.random.uniform(-0.08, 0.02)
    map50 = fitness + np.random.uniform(-0.03, 0.03)

    return {
        'precision': max(0, min(1, precision)),
        'recall': max(0, min(1, recall)),
        'map50': max(0, min(1, map50)),
        'map': max(0, min(1, fitness * 0.8)),
        'fitness': max(0, min(1, fitness))
    }


def simulate_sequential(alphas: List[float], val_sets: List[str],
                       eval_time: float = 120.0) -> Dict:
    """
    순차 평가 시뮬레이션

    Args:
        alphas: List of alpha values
        val_sets: List of validation sets
        eval_time: Time per evaluation (seconds)

    Returns:
        {'results': [...], 'time': float}
    """
    print("\n" + "="*80)
    print("🐌 SEQUENTIAL EVALUATION (Current Implementation)")
    print("="*80)

    start_time = time.time()
    results_by_alpha = {}

    total_evals = len(alphas) * len(val_sets)
    completed = 0

    for alpha in alphas:
        print(f"\n⚙️  α={alpha:.3f}")

        if alpha not in results_by_alpha:
            results_by_alpha[alpha] = {
                'alpha': alpha,
                'metrics': {'per_valset': {}}
            }

        for valset in val_sets:
            print(f"  Evaluating {valset}...", end='', flush=True)

            metrics = simulate_evaluation(alpha, valset, delay=eval_time/60)

            results_by_alpha[alpha]['metrics']['per_valset'][valset] = metrics

            completed += 1
            progress = completed / total_evals * 100
            print(f" fitness={metrics['fitness']:.4f} [{progress:.1f}%]")

        # Calculate overall
        valset_metrics = list(results_by_alpha[alpha]['metrics']['per_valset'].values())
        avg_fitness = sum(m['fitness'] for m in valset_metrics) / len(valset_metrics)

        results_by_alpha[alpha]['metrics']['overall'] = {
            'fitness': avg_fitness,
            'precision': sum(m['precision'] for m in valset_metrics) / len(valset_metrics),
            'recall': sum(m['recall'] for m in valset_metrics) / len(valset_metrics),
            'map50': sum(m['map50'] for m in valset_metrics) / len(valset_metrics),
            'map': sum(m['map'] for m in valset_metrics) / len(valset_metrics),
        }

        print(f"  Overall: fitness={avg_fitness:.4f}")

    elapsed = time.time() - start_time

    print("\n" + "="*80)
    print(f"⏱️  Sequential time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"📊 Evaluations: {total_evals}")
    print(f"⚖️  Per evaluation: {elapsed/total_evals:.1f}s")
    print("="*80)

    return {
        'results': list(results_by_alpha.values()),
        'time': elapsed,
        'per_eval_time': elapsed / total_evals
    }


def simulate_parallel(alphas: List[float], val_sets: List[str],
                     num_gpus: int = 8, eval_time: float = 120.0) -> Dict:
    """
    병렬 평가 시뮬레이션

    Args:
        alphas: List of alpha values
        val_sets: List of validation sets
        num_gpus: Number of GPUs
        eval_time: Time per evaluation (seconds)

    Returns:
        {'results': [...], 'time': float}
    """
    print("\n" + "="*80)
    print(f"🚀 PARALLEL EVALUATION ({num_gpus} GPUs)")
    print("="*80)

    start_time = time.time()
    results_by_alpha = {}

    # Create task list
    tasks = []
    for alpha in alphas:
        for valset in val_sets:
            tasks.append({'alpha': alpha, 'valset': valset})

    total_evals = len(tasks)
    print(f"Total evaluations: {total_evals}")
    print(f"Parallel workers: {num_gpus}")
    print(f"Batches: {int(np.ceil(total_evals / num_gpus))}")

    # Simulate parallel execution
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def eval_task(task):
        alpha = task['alpha']
        valset = task['valset']
        metrics = simulate_evaluation(alpha, valset, delay=eval_time/60)
        return {'alpha': alpha, 'valset': valset, 'metrics': metrics}

    completed = 0

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(eval_task, task) for task in tasks]

        for future in as_completed(futures):
            result = future.result()
            alpha = result['alpha']
            valset = result['valset']
            metrics = result['metrics']

            if alpha not in results_by_alpha:
                results_by_alpha[alpha] = {
                    'alpha': alpha,
                    'metrics': {'per_valset': {}}
                }

            results_by_alpha[alpha]['metrics']['per_valset'][valset] = metrics

            completed += 1
            progress = completed / total_evals * 100
            print(f"[{completed}/{total_evals}] α={alpha:.3f}, {valset}: "
                  f"fitness={metrics['fitness']:.4f} [{progress:.1f}%]")

    # Calculate overall metrics
    for alpha, data in results_by_alpha.items():
        valset_metrics = list(data['metrics']['per_valset'].values())
        avg_fitness = sum(m['fitness'] for m in valset_metrics) / len(valset_metrics)

        data['metrics']['overall'] = {
            'fitness': avg_fitness,
            'precision': sum(m['precision'] for m in valset_metrics) / len(valset_metrics),
            'recall': sum(m['recall'] for m in valset_metrics) / len(valset_metrics),
            'map50': sum(m['map50'] for m in valset_metrics) / len(valset_metrics),
            'map': sum(m['map'] for m in valset_metrics) / len(valset_metrics),
        }

    elapsed = time.time() - start_time

    print("\n" + "="*80)
    print(f"⏱️  Parallel time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"📊 Evaluations: {total_evals}")
    print(f"⚖️  Per evaluation: {elapsed/total_evals:.1f}s")
    print(f"⚡ Speedup: {num_gpus:.1f}x (theoretical)")
    print("="*80)

    return {
        'results': list(results_by_alpha.values()),
        'time': elapsed,
        'per_eval_time': elapsed / total_evals
    }


def compare_strategies():
    """전략 비교"""
    print("\n" + "="*80)
    print("📈 WISEFT STRATEGY COMPARISON")
    print("="*80)

    # Configuration
    configs = {
        '전체 범위 (순차)': {
            'alphas': [round(x, 2) for x in np.arange(0.0, 1.1, 0.1)],
            'val_sets': ['valid1', 'valid2'],
            'parallel': False
        },
        '좁은 범위 (순차)': {
            'alphas': [round(x, 2) for x in np.arange(0.0, 0.6, 0.1)],
            'val_sets': ['valid1', 'valid2'],
            'parallel': False
        },
        '전체 범위 (병렬 8 GPU)': {
            'alphas': [round(x, 2) for x in np.arange(0.0, 1.1, 0.1)],
            'val_sets': ['valid1', 'valid2'],
            'parallel': True,
            'num_gpus': 8
        },
        '좁은 범위 (병렬 8 GPU)': {
            'alphas': [round(x, 2) for x in np.arange(0.0, 0.6, 0.1)],
            'val_sets': ['valid1', 'valid2'],
            'parallel': True,
            'num_gpus': 8
        },
    }

    # Realistic evaluation time
    EVAL_TIME = 120.0  # 2분 per evaluation (realistic for YOLO)

    results = {}

    for name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")

        alphas = config['alphas']
        val_sets = config['val_sets']

        if config.get('parallel', False):
            result = simulate_parallel(alphas, val_sets, config['num_gpus'], EVAL_TIME)
        else:
            result = simulate_sequential(alphas, val_sets, EVAL_TIME)

        results[name] = {
            'time_seconds': result['time'],
            'time_minutes': result['time'] / 60,
            'num_evals': len(alphas) * len(val_sets),
            'alphas': alphas
        }

    # Print comparison table
    print("\n" + "="*80)
    print("📊 COMPARISON TABLE")
    print("="*80)
    print(f"\n{'Strategy':<30} {'Alphas':<10} {'Evals':<10} {'Time':<15} {'Speedup':<10}")
    print("─" * 80)

    baseline_time = None
    for name, data in results.items():
        if baseline_time is None:
            baseline_time = data['time_seconds']
            speedup = "1.0x"
        else:
            speedup = f"{baseline_time / data['time_seconds']:.1f}x"

        print(f"{name:<30} {len(data['alphas']):<10} {data['num_evals']:<10} "
              f"{data['time_minutes']:<15.1f} {speedup:<10}")

    print("─" * 80)

    # Recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    fastest = min(results.items(), key=lambda x: x[1]['time_seconds'])
    print(f"\n⚡ Fastest: {fastest[0]}")
    print(f"   Time: {fastest[1]['time_minutes']:.1f} minutes")

    print(f"\n🎯 Best for A6000 x8:")
    print(f"   Strategy: 좁은 범위 (병렬 8 GPU)")
    print(f"   Expected time: {results['좁은 범위 (병렬 8 GPU)']['time_minutes']:.1f} minutes")
    print(f"   Speedup: {results['전체 범위 (순차)']['time_minutes'] / results['좁은 범위 (병렬 8 GPU)']['time_minutes']:.1f}x faster")


def estimate_real_time():
    """실제 시간 추정"""
    print("\n" + "="*80)
    print("⏱️  REAL-TIME ESTIMATION (Based on your setup)")
    print("="*80)

    # Assumptions
    REAL_EVAL_TIME = 120  # seconds per evaluation (A6000 기준)

    scenarios = {
        '현재 (순차, 전체 범위)': {
            'alphas': 11,
            'valsets': 2,
            'parallel': False
        },
        '최적화 (순차, 좁은 범위)': {
            'alphas': 4,  # 0.0, 0.1, 0.2, 0.3
            'valsets': 2,
            'parallel': False
        },
        '병렬 (8 GPU, 전체 범위)': {
            'alphas': 11,
            'valsets': 2,
            'parallel': True,
            'num_gpus': 8
        },
        '병렬 + 최적화 (8 GPU, 좁은 범위)': {
            'alphas': 4,
            'valsets': 2,
            'parallel': True,
            'num_gpus': 8
        },
    }

    print(f"\n{'Scenario':<40} {'Evals':<10} {'Time (min)':<15} {'Time (hour)':<15}")
    print("─" * 80)

    for name, config in scenarios.items():
        total_evals = config['alphas'] * config['valsets']

        if config.get('parallel', False):
            # Parallel: batch by num_gpus
            batches = int(np.ceil(total_evals / config['num_gpus']))
            time_seconds = batches * REAL_EVAL_TIME
        else:
            # Sequential
            time_seconds = total_evals * REAL_EVAL_TIME

        time_minutes = time_seconds / 60
        time_hours = time_minutes / 60

        print(f"{name:<40} {total_evals:<10} {time_minutes:<15.1f} {time_hours:<15.2f}")

    print("─" * 80)

    print("\n⚠️  현재 5시간+ 소요 → 병렬화 안 되고 있거나 다른 문제!")
    print("   Expected (순차): ~44 min")
    print("   Expected (병렬 8 GPU): ~6 min")
    print("   Actual: 5+ hours ← 비정상!")

    print("\n🔍 가능한 원인:")
    print("   1. test.py가 매우 느림 (데이터 로딩 문제?)")
    print("   2. GPU를 제대로 사용 안 함")
    print("   3. 검증 세트가 매우 큼")
    print("   4. I/O 병목")


if __name__ == '__main__':
    print("="*80)
    print("WiSE-FT SIMULATION & VERIFICATION TOOL")
    print("="*80)

    # Run simulations
    compare_strategies()

    # Estimate real time
    estimate_real_time()

    print("\n" + "="*80)
    print("✅ Simulation complete!")
    print("="*80)
