#!/usr/bin/env python3
"""
WiSE-FT Parallel Evaluation - 병렬 평가로 속도 대폭 개선

A6000 8개 GPU를 모두 활용하여 여러 알파를 동시에 평가
"""

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict
import json


def evaluate_single_alpha_valset(alpha: float, val_set: str, model_path: str,
                                  data_yaml: str, gpu_id: int, args) -> Dict:
    """
    단일 알파/검증세트 평가 (병렬 실행용)

    Args:
        alpha: Alpha value
        val_set: Validation set name
        model_path: Merged model path
        data_yaml: Temporary data yaml for this valset
        gpu_id: GPU ID to use
        args: Arguments

    Returns:
        {'alpha': float, 'valset': str, 'metrics': {...}}
    """
    print(f"[GPU {gpu_id}] Evaluating α={alpha:.3f} on {val_set}...")

    # Run test.py on specific GPU
    cmd = [
        sys.executable, 'test.py',
        '--data', data_yaml,
        '--weights', model_path,
        '--img-size', str(args.img_size),
        '--batch-size', str(args.batch_size),
        '--conf-thres', str(args.conf_thres),
        '--iou-thres', str(args.iou_thres),
        '--task', 'val',
        '--device', str(gpu_id),  # GPU 할당!
        '--save-txt',
        '--save-json',
        '--exist-ok'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=1800, check=True)  # 30분 (대용량 검증 세트 지원)

        # Parse results
        output = result.stdout
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'map50': 0.0,
            'map': 0.0,
            'fitness': 0.0
        }

        for line in output.split('\n'):
            if 'all' in line.lower() and any(x in line for x in ['0.', '1.0']):
                parts = line.split()
                try:
                    numbers = [float(x) for x in parts if x.replace('.', '').replace('-', '').isdigit()]
                    if len(numbers) >= 4:
                        metrics['precision'] = numbers[-4]
                        metrics['recall'] = numbers[-3]
                        metrics['map50'] = numbers[-2]
                        metrics['map'] = numbers[-1]
                        metrics['fitness'] = 0.1 * metrics['map50'] + 0.9 * metrics['map']
                        break
                except (ValueError, IndexError):
                    continue

        print(f"[GPU {gpu_id}] ✓ α={alpha:.3f}, {val_set}: fitness={metrics['fitness']:.4f}")

        return {
            'alpha': alpha,
            'valset': val_set,
            'metrics': metrics
        }

    except subprocess.TimeoutExpired:
        print(f"[GPU {gpu_id}] ✗ Timeout α={alpha:.3f}, {val_set}")
        return None
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ Error α={alpha:.3f}, {val_set}: {e}")
        return None


def parallel_evaluate_wiseft(scratch_path: str, finetuned_path: str,
                             data_yaml: str, val_sets: List[str],
                             alphas: List[float], num_gpus: int, args) -> List[Dict]:
    """
    병렬로 여러 알파/검증세트 평가

    Args:
        scratch_path: Scratch model
        finetuned_path: Finetuned model
        data_yaml: Base data yaml
        val_sets: List of validation set names
        alphas: List of alpha values to test
        num_gpus: Number of GPUs to use
        args: Arguments

    Returns:
        List of results with per_valset metrics
    """
    import torch
    from copy import deepcopy
    import yaml

    print("\n" + "="*80)
    print("🚀 PARALLEL WISEFT EVALUATION")
    print("="*80)
    print(f"GPUs: {num_gpus}")
    print(f"Alphas: {len(alphas)}")
    print(f"Validation sets: {len(val_sets)}")
    print(f"Total evaluations: {len(alphas) * len(val_sets)}")
    print(f"Parallel jobs: {num_gpus}")
    print("="*80)

    # Load models
    print("\n📦 Loading models...")
    scratch_ckpt = torch.load(scratch_path, map_location='cpu')
    finetuned_ckpt = torch.load(finetuned_path, map_location='cpu')
    scratch_sd = scratch_ckpt['model'].float().state_dict()
    finetuned_sd = finetuned_ckpt['model'].float().state_dict()

    # Prepare output directory
    output_dir = Path(args.output_dir) / 'parallel_eval'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create all merged models first
    print("\n⚙️  Creating merged models...")
    merged_models = {}
    for alpha in alphas:
        merged_path = output_dir / f'alpha_{alpha:.3f}.pt'

        # Merge weights
        merged_sd = {}
        for key in scratch_sd.keys():
            if key in finetuned_sd:
                merged_sd[key] = (1 - alpha) * scratch_sd[key] + alpha * finetuned_sd[key]
            else:
                merged_sd[key] = scratch_sd[key]

        output_ckpt = deepcopy(finetuned_ckpt)
        output_ckpt['model'].load_state_dict(merged_sd)
        torch.save(output_ckpt, merged_path)

        merged_models[alpha] = str(merged_path)
        print(f"  ✓ α={alpha:.3f}")

    # Create temporary data yamls for each valset
    print("\n📝 Creating temporary data configs...")
    temp_yamls = {}
    with open(data_yaml, 'r') as f:
        base_config = yaml.safe_load(f)

    for val_set in val_sets:
        temp_config = base_config.copy()

        # Update val path
        original_val = base_config.get('val', '')

        # Handle both string and list formats
        if isinstance(original_val, list):
            # If val is a list, use the first one as base
            original_val = original_val[0]
            print(f"  ℹ️  val is a list, using first entry: {original_val}")

        original_val = Path(original_val)

        if original_val.suffix == '.txt':
            new_val_path = original_val.parent / f'{val_set}.txt'
        else:
            new_val_path = original_val.parent / val_set

        temp_config['val'] = str(new_val_path)

        temp_yaml_path = output_dir / f'data_{val_set}.yaml'
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(temp_config, f)

        temp_yamls[val_set] = str(temp_yaml_path)
        print(f"  ✓ {val_set}: {new_val_path}")

    # Create task list
    tasks = []
    for alpha in alphas:
        for val_set in val_sets:
            tasks.append({
                'alpha': alpha,
                'val_set': val_set,
                'model_path': merged_models[alpha],
                'data_yaml': temp_yamls[val_set]
            })

    print(f"\n🔄 Running {len(tasks)} evaluations in parallel on {num_gpus} GPUs...")
    print("="*80)

    # Parallel execution
    results = []
    gpu_queue = list(range(num_gpus))  # Available GPUs

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        future_to_task = {}

        # Submit initial batch
        for task in tasks[:num_gpus]:
            gpu_id = gpu_queue.pop(0)
            future = executor.submit(
                evaluate_single_alpha_valset,
                task['alpha'], task['val_set'], task['model_path'],
                task['data_yaml'], gpu_id, args
            )
            future_to_task[future] = (task, gpu_id)

        # Process remaining tasks as GPUs become available
        task_idx = num_gpus
        completed = 0

        for future in as_completed(future_to_task):
            task, gpu_id = future_to_task[future]
            result = future.result()

            if result:
                results.append(result)

            completed += 1
            print(f"\n[Progress] {completed}/{len(tasks)} completed ({completed/len(tasks)*100:.1f}%)")

            # Submit next task
            if task_idx < len(tasks):
                next_task = tasks[task_idx]
                future = executor.submit(
                    evaluate_single_alpha_valset,
                    next_task['alpha'], next_task['val_set'],
                    next_task['model_path'], next_task['data_yaml'],
                    gpu_id, args
                )
                future_to_task[future] = (next_task, gpu_id)
                task_idx += 1
            else:
                gpu_queue.append(gpu_id)  # Return GPU to pool

    print("\n" + "="*80)
    print("✅ Parallel evaluation complete!")
    print("="*80)

    # Organize results by alpha
    results_by_alpha = {}
    for r in results:
        alpha = r['alpha']
        if alpha not in results_by_alpha:
            results_by_alpha[alpha] = {
                'alpha': alpha,
                'merged_model_path': merged_models[alpha],
                'metrics': {
                    'per_valset': {},
                    'overall': None
                }
            }

        results_by_alpha[alpha]['metrics']['per_valset'][r['valset']] = r['metrics']

    # Calculate overall average for each alpha
    for alpha, data in results_by_alpha.items():
        valset_metrics = list(data['metrics']['per_valset'].values())

        avg_metrics = {}
        for key in ['precision', 'recall', 'map50', 'map', 'fitness']:
            values = [m[key] for m in valset_metrics]
            avg_metrics[key] = sum(values) / len(values)

        data['metrics']['overall'] = avg_metrics

    return list(results_by_alpha.values())


if __name__ == '__main__':
    """
    사용 예시:

    python wiseft_sweep_parallel.py \
        --scratch models/600.pt \
        --finetuned models/620.pt \
        --data data.yaml \
        --val-sets valid1 valid2 \
        --num-gpus 8
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=str, required=True)
    parser.add_argument('--finetuned', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--val-sets', nargs='+', default=['valid1', 'valid2'])
    parser.add_argument('--alpha-min', type=float, default=0.0)
    parser.add_argument('--alpha-max', type=float, default=0.5)
    parser.add_argument('--focus-range', type=float, default=0.1)
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--conf-thres', type=float, default=0.001)
    parser.add_argument('--iou-thres', type=float, default=0.6)
    parser.add_argument('--output-dir', type=str, default='runs/wiseft_parallel')

    args = parser.parse_args()

    # Generate alpha list
    alphas = []
    current = args.alpha_min
    while current <= args.alpha_max + 1e-6:
        alphas.append(round(current, 3))
        current += args.focus_range

    print(f"\n🎯 Alphas to test: {alphas}")

    # Run parallel evaluation
    start_time = time.time()

    results = parallel_evaluate_wiseft(
        args.scratch, args.finetuned, args.data,
        args.val_sets, alphas, args.num_gpus, args
    )

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)

    for r in sorted(results, key=lambda x: x['alpha']):
        alpha = r['alpha']
        overall = r['metrics']['overall']['fitness']

        print(f"\nα={alpha:.3f}  Overall: {overall:.4f}")
        for valset, metrics in r['metrics']['per_valset'].items():
            print(f"  {valset}: {metrics['fitness']:.4f}")

    # Save results
    output_file = Path(args.output_dir) / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"⚡ Speedup: ~{len(alphas)*len(args.val_sets)/args.num_gpus:.1f}x faster than sequential")
    print("="*80)
