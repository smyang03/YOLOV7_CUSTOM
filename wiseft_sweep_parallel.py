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
from datetime import datetime


def log_message(msg, log_file=None):
    """타임스탬프와 함께 로그 출력"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {msg}"
    print(log_line, flush=True)

    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_line + '\n')


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
        print(f"\n⚠️  Performance degradation detected on: {', '.join(negative_changes)}")
        print(f"   💡 WiSE-FT may help recover some performance!")
        print("="*80)
        return True
    else:
        print(f"\n✅ No trade-off detected - fine-tuned model performs well on all validation sets")
        print("="*80)
        return False


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
    # 로그 파일 설정
    log_dir = Path(args.output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'gpu{gpu_id}_alpha{alpha:.3f}_{val_set}.log'

    start_time = time.time()
    # 간결한 시작 로그 (화면)
    print(f"[GPU {gpu_id}] 🚀 α={alpha:.3f}, {val_set} - 평가 시작", flush=True)

    # 상세 로그는 파일에만 기록
    with open(log_file, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting: α={alpha:.3f}, {val_set}\n")

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
        '--device', str(gpu_id),
        '--save-txt',
        '--save-json',
        '--exist-ok'
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []
        last_progress_time = time.time()

        # 출력 읽기 (화면 출력 최소화)
        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line)

                # 로그 파일에만 저장
                with open(log_file, 'a') as f:
                    f.write(line)

                # 2분마다만 간단한 진행 상황 표시
                current_time = time.time()
                if current_time - last_progress_time > 120:  # 30초 → 2분
                    elapsed = current_time - start_time
                    print(f"[GPU {gpu_id}] ⏳ α={alpha:.3f}, {val_set} - 진행 중 ({elapsed/60:.0f}분 경과)", flush=True)
                    last_progress_time = current_time

        # 프로세스 종료 대기
        try:
            return_code = process.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            process.kill()
            elapsed = time.time() - start_time
            print(f"[GPU {gpu_id}] ⏱️  α={alpha:.3f}, {val_set} - 타임아웃 ({elapsed/60:.0f}분)", flush=True)
            return None

        if return_code != 0:
            print(f"[GPU {gpu_id}] ❌ α={alpha:.3f}, {val_set} - 에러 발생", flush=True)
            return None

        # 결과 파싱
        output = ''.join(output_lines)
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

        elapsed = time.time() - start_time
        print(f"[GPU {gpu_id}] ✅ α={alpha:.3f}, {val_set} - 완료! fitness={metrics['fitness']:.4f} ({elapsed/60:.0f}분)", flush=True)

        return {
            'alpha': alpha,
            'valset': val_set,
            'metrics': metrics,
            'elapsed_time': elapsed
        }

    except Exception as e:
        elapsed = time.time() - start_time
        log_message(f"[GPU {gpu_id}] ❌ Error: {str(e)} ({elapsed/60:.1f} min)", log_file)
        return None


def evaluate_baseline_parallel(model_path: str, model_name: str, data_yaml: str,
                               val_sets: List[str], num_gpus: int, args) -> Dict:
    """
    Evaluate baseline model (scratch or finetuned) on all validation sets in parallel

    Returns:
        {'per_valset': {...}, 'overall': {...}}
    """
    print(f"\n🔹 Evaluating {model_name} baseline...")

    # Create temporary data yamls
    output_dir = Path(args.output_dir) / 'baseline_eval'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_yaml, 'r') as f:
        base_config = yaml.safe_load(f)

    temp_yamls = {}
    for val_set in val_sets:
        temp_config = base_config.copy()
        original_val = base_config.get('val', '')

        if isinstance(original_val, list):
            original_val = original_val[0]

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

    # Parallel evaluation
    results = {}
    gpu_queue = list(range(num_gpus))

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        future_to_valset = {}

        for val_set in val_sets:
            gpu_id = gpu_queue.pop(0) if gpu_queue else 0
            future = executor.submit(
                evaluate_single_alpha_valset,
                -1.0,  # Dummy alpha for baseline
                val_set,
                model_path,
                temp_yamls[val_set],
                gpu_id,
                args
            )
            future_to_valset[future] = (val_set, gpu_id)

        for future in as_completed(future_to_valset):
            val_set, gpu_id = future_to_valset[future]
            result = future.result()

            if result:
                results[val_set] = result['metrics']

            gpu_queue.append(gpu_id)

    # Calculate overall average
    avg_metrics = {}
    for key in ['precision', 'recall', 'map50', 'map', 'fitness']:
        values = [metrics[key] for metrics in results.values()]
        avg_metrics[key] = sum(values) / len(values)

    return {
        'per_valset': results,
        'overall': avg_metrics
    }


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

    # 실험 계획 명확히 표시
    print("\n" + "="*80)
    print("📋 WiSE-FT 실험 계획")
    print("="*80)
    print(f"\n평가할 모델:")
    print(f"  - Scratch 모델: {args.scratch}")
    print(f"  - Fine-tuned 모델: {args.finetuned}")
    print(f"\n평가할 Validation Sets:")
    for vs in args.val_sets:
        print(f"  - {vs}")
    print(f"\n평가할 Alpha 값: {alphas}")
    print(f"  (총 {len(alphas)}개 alpha × {len(args.val_sets)}개 validation set = {len(alphas)*len(args.val_sets)}개 평가)")
    print(f"\n병렬 처리: GPU {args.num_gpus}개 사용")
    print(f"예상 시간: ~{len(alphas)*len(args.val_sets)/args.num_gpus*10:.0f}분")
    print("\n💡 Overall = Valid1과 Valid2의 평균 fitness")
    print("="*80)

    # Evaluate baselines first
    print("\n" + "="*80)
    print("📏 BASELINE 모델 평가")
    print("="*80)

    start_time = time.time()

    # Scratch baseline (α=0.0)
    scratch_metrics = evaluate_baseline_parallel(
        args.scratch, "Scratch (α=0.0)", args.data,
        args.val_sets, args.num_gpus, args
    )

    # Fine-tuned baseline (α=1.0)
    finetuned_metrics = evaluate_baseline_parallel(
        args.finetuned, "Fine-tuned (α=1.0)", args.data,
        args.val_sets, args.num_gpus, args
    )

    # Trade-off analysis
    has_tradeoff = analyze_baseline_tradeoff(
        scratch_metrics, finetuned_metrics, args.val_sets
    )

    # Run WiSE-FT sweep
    print("\n" + "="*80)
    print("🔍 WISE-FT ALPHA SWEEP")
    print("="*80)

    results = parallel_evaluate_wiseft(
        args.scratch, args.finetuned, args.data,
        args.val_sets, alphas, args.num_gpus, args
    )

    elapsed = time.time() - start_time

    # 결과 요약 (간결하게)
    print("\n" + "="*80)
    print("📊 평가 결과 요약")
    print("="*80)
    print(f"\n💡 Overall = {' + '.join(args.val_sets)} 평균")
    print()

    # 헤더
    print(f"{'Alpha':<8} │ {'Overall':<10} │", end='')
    for val_set in args.val_sets:
        print(f" {val_set:<10} │", end='')
    print(" 비고")
    print("─" * 80)

    # Scratch baseline
    print(f"{'0.0*':<8} │ {scratch_metrics['overall']['fitness']:<10.4f} │", end='')
    for val_set in args.val_sets:
        print(f" {scratch_metrics['per_valset'][val_set]['fitness']:<10.4f} │", end='')
    print(" Scratch 모델")

    # WiSE-FT results
    for r in sorted(results, key=lambda x: x['alpha']):
        alpha = r['alpha']
        overall = r['metrics']['overall']['fitness']
        print(f"{alpha:<8.3f} │ {overall:<10.4f} │", end='')
        for val_set in args.val_sets:
            fitness = r['metrics']['per_valset'][val_set]['fitness']
            print(f" {fitness:<10.4f} │", end='')
        print()

    # Fine-tuned baseline
    print(f"{'1.0*':<8} │ {finetuned_metrics['overall']['fitness']:<10.4f} │", end='')
    for val_set in args.val_sets:
        print(f" {finetuned_metrics['per_valset'][val_set]['fitness']:<10.4f} │", end='')
    print(" Fine-tuned 모델")

    print("─" * 80)
    print("* Baseline 모델 (WiSE-FT 아님)")

    # Find best alpha
    best_result = max(results, key=lambda x: x['metrics']['overall']['fitness'])
    print(f"\n🏆 최고 성능: α={best_result['alpha']:.3f}, Overall={best_result['metrics']['overall']['fitness']:.4f}")

    # Save complete results including baselines
    all_results = {
        'baselines': {
            'scratch': {
                'alpha': 0.0,
                'model_path': args.scratch,
                'metrics': scratch_metrics
            },
            'finetuned': {
                'alpha': 1.0,
                'model_path': args.finetuned,
                'metrics': finetuned_metrics
            }
        },
        'wiseft_results': results,
        'best_wiseft': {
            'alpha': best_result['alpha'],
            'fitness': best_result['metrics']['overall']['fitness'],
            'metrics': best_result['metrics']
        },
        'has_tradeoff': has_tradeoff
    }

    output_file = Path(args.output_dir) / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"⚡ Speedup: ~{len(alphas)*len(args.val_sets)/args.num_gpus:.1f}x faster than sequential")
    print("="*80)
