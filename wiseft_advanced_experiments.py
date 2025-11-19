#!/usr/bin/env python3
"""
WiSE-FT 고급 실험 스크립트
다양한 개선 전략을 테스트하기 위한 도구
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

# ============================================================================
# 1. Fine-grained Alpha Search
# ============================================================================

def generate_fine_grained_alphas(start=0.0, end=0.15, step=0.02):
    """세밀한 alpha 값 생성"""
    alphas = []
    current = start
    while current <= end + 1e-6:
        alphas.append(round(current, 3))
        current += step
    return alphas

def run_fine_grained_search():
    """Fine-grained alpha search 실행"""
    print("="*80)
    print("Fine-grained Alpha Search (0.0~0.15, step=0.02)")
    print("="*80)

    alphas = generate_fine_grained_alphas(0.0, 0.15, 0.02)
    print(f"\nAlpha values to test: {alphas}")
    print(f"Total evaluations: {len(alphas)} alphas × 2 valsets = {len(alphas)*2}")

    print("\n실행 명령:")
    print("""
python wiseft_sweep_parallel.py \\
    --scratch new_list/600.pt \\
    --finetuned new_list/620.pt \\
    --data new_list/data.yaml \\
    --val-sets valid1 valid2 \\
    --alpha-min 0.0 \\
    --alpha-max 0.15 \\
    --focus-range 0.02 \\
    --num-gpus 8 \\
    --batch-size 64
    """)

    # 예상 결과 시뮬레이션
    print("\n예상 결과:")
    print("Alpha=0.04~0.08 사이에서:")
    print("  - Valid2 개선: +0.5~1.0%")
    print("  - Valid1 손실: -1~2%")
    print("  - Overall 손실 최소화")

# ============================================================================
# 2. Weighted Optimization
# ============================================================================

def calculate_weighted_fitness(results: List[Dict], w_valid1: float, w_valid2: float) -> List[Dict]:
    """가중치 적용된 fitness 계산"""
    weighted_results = []

    for r in results:
        v1_fitness = r['metrics']['per_valset']['valid1']['fitness']
        v2_fitness = r['metrics']['per_valset']['valid2']['fitness']

        weighted_fitness = w_valid1 * v1_fitness + w_valid2 * v2_fitness

        weighted_results.append({
            'alpha': r['alpha'],
            'weighted_fitness': weighted_fitness,
            'valid1_fitness': v1_fitness,
            'valid2_fitness': v2_fitness,
            'weights': (w_valid1, w_valid2)
        })

    return weighted_results

def find_optimal_alpha_weighted(results: List[Dict], weight_scenarios: List[Tuple[float, float]]):
    """다양한 가중치 시나리오별 최적 alpha 찾기"""
    print("="*80)
    print("Weighted Optimization Analysis")
    print("="*80)

    for w1, w2 in weight_scenarios:
        print(f"\n가중치: Valid1={w1:.1f}, Valid2={w2:.1f}")
        print("─" * 60)

        weighted = calculate_weighted_fitness(results, w1, w2)
        best = max(weighted, key=lambda x: x['weighted_fitness'])

        print(f"  최적 Alpha: {best['alpha']:.2f}")
        print(f"  Weighted Fitness: {best['weighted_fitness']:.4f}")
        print(f"  Valid1: {best['valid1_fitness']:.4f}")
        print(f"  Valid2: {best['valid2_fitness']:.4f}")

        # 상위 3개
        top3 = sorted(weighted, key=lambda x: x['weighted_fitness'], reverse=True)[:3]
        print(f"\n  상위 3개 Alpha:")
        for i, r in enumerate(top3, 1):
            print(f"    {i}. α={r['alpha']:.2f}: {r['weighted_fitness']:.4f}")

def analyze_pareto_frontier(results: List[Dict]):
    """Pareto frontier 분석 (trade-off curve)"""
    print("\n" + "="*80)
    print("Pareto Frontier Analysis")
    print("="*80)

    # (Valid1, Valid2) 점들 추출
    points = []
    for r in results:
        v1 = r['metrics']['per_valset']['valid1']['fitness']
        v2 = r['metrics']['per_valset']['valid2']['fitness']
        points.append({'alpha': r['alpha'], 'v1': v1, 'v2': v2})

    # Pareto dominant 찾기
    pareto = []
    for p in points:
        is_dominated = False
        for q in points:
            if q['v1'] > p['v1'] and q['v2'] > p['v2']:
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(p)

    print(f"\nPareto-optimal solutions: {len(pareto)}")
    for p in sorted(pareto, key=lambda x: x['alpha']):
        print(f"  α={p['alpha']:.2f}: Valid1={p['v1']:.4f}, Valid2={p['v2']:.4f}")

# ============================================================================
# 3. Precision/Recall 분석
# ============================================================================

def analyze_precision_recall_tradeoff(results: List[Dict]):
    """Precision vs Recall 경향 분석"""
    print("="*80)
    print("Precision/Recall Trade-off Analysis")
    print("="*80)

    print(f"\n{'Alpha':<8} {'V1 P':<8} {'V1 R':<8} {'V1 P/R':<10} {'V2 P':<8} {'V2 R':<8} {'V2 P/R':<10}")
    print("─" * 80)

    for r in results:
        alpha = r['alpha']
        v1_p = r['metrics']['per_valset']['valid1']['precision']
        v1_r = r['metrics']['per_valset']['valid1']['recall']
        v2_p = r['metrics']['per_valset']['valid2']['precision']
        v2_r = r['metrics']['per_valset']['valid2']['recall']

        v1_ratio = v1_p / v1_r if v1_r > 0 else 0
        v2_ratio = v2_p / v2_r if v2_r > 0 else 0

        print(f"{alpha:<8.2f} {v1_p:<8.3f} {v1_r:<8.3f} {v1_ratio:<10.2f} {v2_p:<8.3f} {v2_r:<8.3f} {v2_ratio:<10.2f}")

    # Alpha=0.1 강조
    print("\n💡 Alpha=0.1에서 Valid2의 P/R 균형 (1.02)에 주목!")
    print("   → Recall 상승이 fitness 개선의 핵심")

def analyze_localization_quality(results: List[Dict]):
    """Localization 품질 분석 (mAP50/mAP 비율)"""
    print("\n" + "="*80)
    print("Localization Quality Analysis")
    print("="*80)

    print(f"\n{'Alpha':<8} {'V1 mAP50/mAP':<15} {'V2 mAP50/mAP':<15} {'경향':<20}")
    print("─" * 80)

    for i, r in enumerate(results):
        alpha = r['alpha']
        v1_map50 = r['metrics']['per_valset']['valid1']['map50']
        v1_map = r['metrics']['per_valset']['valid1']['map']
        v2_map50 = r['metrics']['per_valset']['valid2']['map50']
        v2_map = r['metrics']['per_valset']['valid2']['map']

        v1_ratio = v1_map50 / v1_map if v1_map > 0 else 0
        v2_ratio = v2_map50 / v2_map if v2_map > 0 else 0

        trend = ""
        if i > 0:
            prev = results[i-1]
            prev_v1 = prev['metrics']['per_valset']['valid1']['map50'] / prev['metrics']['per_valset']['valid1']['map']
            prev_v2 = prev['metrics']['per_valset']['valid2']['map50'] / prev['metrics']['per_valset']['valid2']['map']

            if v1_ratio > prev_v1:
                trend += "V1↑ "
            else:
                trend += "V1↓ "

            if v2_ratio > prev_v2:
                trend += "V2↑"
            else:
                trend += "V2↓"

        print(f"{alpha:<8.2f} {v1_ratio:<15.3f} {v2_ratio:<15.3f} {trend:<20}")

    print("\n💡 비율 증가 = Localization 품질 하락")
    print("   → Fine-tuned 모델이 bounding box를 부정확하게 예측")

# ============================================================================
# 4. Layer-wise 전략 시뮬레이션
# ============================================================================

def simulate_layerwise_strategy():
    """Layer-wise alpha 전략 시뮬레이션"""
    print("="*80)
    print("Layer-wise Alpha Strategy Simulation")
    print("="*80)

    strategies = [
        {
            "name": "Conservative (대부분 Scratch)",
            "backbone": 0.02,
            "neck": 0.05,
            "head": 0.08
        },
        {
            "name": "Balanced",
            "backbone": 0.05,
            "neck": 0.10,
            "head": 0.15
        },
        {
            "name": "Aggressive (더 많은 Fine-tuned)",
            "backbone": 0.10,
            "neck": 0.15,
            "head": 0.20
        }
    ]

    for strategy in strategies:
        print(f"\n전략: {strategy['name']}")
        print(f"  Backbone alpha: {strategy['backbone']:.2f}")
        print(f"  Neck alpha: {strategy['neck']:.2f}")
        print(f"  Head alpha: {strategy['head']:.2f}")

        # 가중 평균 (대략적)
        avg_alpha = (strategy['backbone'] * 0.5 +
                     strategy['neck'] * 0.3 +
                     strategy['head'] * 0.2)
        print(f"  → 대략적 평균 alpha: {avg_alpha:.3f}")

    print("\n구현 예시 코드:")
    print("""
def layer_wise_merge(scratch_sd, finetuned_sd, strategy):
    merged_sd = {}
    for key in scratch_sd.keys():
        if 'model.0' in key or 'model.1' in key:  # Backbone
            alpha = strategy['backbone']
        elif 'model.24' in key:  # Neck
            alpha = strategy['neck']
        elif 'model.105' in key:  # Head
            alpha = strategy['head']
        else:
            alpha = strategy['backbone']  # Default

        merged_sd[key] = (1 - alpha) * scratch_sd[key] + alpha * finetuned_sd[key]
    return merged_sd
    """)

# ============================================================================
# 5. 실험 결과 시각화 (ASCII)
# ============================================================================

def visualize_results_ascii(results: List[Dict]):
    """ASCII 기반 결과 시각화"""
    print("="*80)
    print("Results Visualization")
    print("="*80)

    alphas = [r['alpha'] for r in results]
    v1_fitness = [r['metrics']['per_valset']['valid1']['fitness'] for r in results]
    v2_fitness = [r['metrics']['per_valset']['valid2']['fitness'] for r in results]

    # 정규화
    max_fitness = max(max(v1_fitness), max(v2_fitness))
    min_fitness = min(min(v1_fitness), min(v2_fitness))

    def scale(val):
        return int((val - min_fitness) / (max_fitness - min_fitness) * 40)

    print("\nFitness Trend:")
    print("  1.0 |")

    # 각 높이별로 출력
    for height in range(40, -1, -2):
        line = f"  {height/40*max_fitness:.2f} |"
        for i, alpha in enumerate(alphas):
            v1_h = scale(v1_fitness[i])
            v2_h = scale(v2_fitness[i])

            if abs(v1_h - height) < 1:
                line += " ●"
            elif abs(v2_h - height) < 1:
                line += " ○"
            else:
                line += "  "

        print(line)

    print("      +" + "─" * (len(alphas) * 2))
    alpha_labels = "       " + "  ".join([f"{a:.1f}" for a in alphas])
    print(alpha_labels)
    print("\n  ● Valid1    ○ Valid2")

# ============================================================================
# 6. 실험 제안 생성기
# ============================================================================

def generate_experiment_plan():
    """실험 계획 생성"""
    print("="*80)
    print("📋 Experiment Plan Generator")
    print("="*80)

    experiments = [
        {
            "phase": 1,
            "name": "Fine-grained Search",
            "priority": "🔴 High",
            "effort": "Low",
            "command": """
python wiseft_sweep_parallel.py \\
    --scratch new_list/600.pt \\
    --finetuned new_list/620.pt \\
    --data new_list/data.yaml \\
    --val-sets valid1 valid2 \\
    --alpha-min 0.0 \\
    --alpha-max 0.15 \\
    --focus-range 0.02 \\
    --num-gpus 8 \\
    --batch-size 64
            """,
            "expected_time": "30-60 min",
            "expected_insight": "Sweet spot between 0.04-0.08"
        },
        {
            "phase": 2,
            "name": "Full Range Sweep",
            "priority": "🟡 Medium",
            "effort": "Medium",
            "command": """
python wiseft_sweep_parallel.py \\
    --alpha-min 0.0 \\
    --alpha-max 1.0 \\
    --focus-range 0.1 \\
    --num-gpus 8
            """,
            "expected_time": "60-90 min",
            "expected_insight": "Complete curve, Alpha=1.0 performance"
        },
        {
            "phase": 3,
            "name": "Weighted Optimization",
            "priority": "🟢 Low",
            "effort": "Low",
            "command": "python wiseft_advanced_experiments.py --mode weighted",
            "expected_time": "5 min (post-processing)",
            "expected_insight": "Optimal alpha for different priorities"
        }
    ]

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Phase {exp['phase']}: {exp['name']}")
        print(f"{'='*60}")
        print(f"Priority: {exp['priority']}")
        print(f"Effort: {exp['effort']}")
        print(f"Expected Time: {exp['expected_time']}")
        print(f"\nCommand:")
        print(exp['command'])
        print(f"\n예상 결과:")
        print(f"  {exp['expected_insight']}")

# ============================================================================
# Main
# ============================================================================

def main():
    """메인 함수"""
    import sys

    # 현재 결과 로드 (예시)
    current_results = [
        {
            "alpha": 0.0,
            "metrics": {
                "per_valset": {
                    "valid1": {"precision": 0.871, "recall": 0.798, "map50": 0.828, "map": 0.649, "fitness": 0.6669},
                    "valid2": {"precision": 0.707, "recall": 0.458, "map50": 0.480, "map": 0.377, "fitness": 0.3873}
                }
            }
        },
        {
            "alpha": 0.1,
            "metrics": {
                "per_valset": {
                    "valid1": {"precision": 0.889, "recall": 0.771, "map50": 0.810, "map": 0.625, "fitness": 0.6435},
                    "valid2": {"precision": 0.553, "recall": 0.541, "map50": 0.519, "map": 0.379, "fitness": 0.3930}
                }
            }
        },
        {
            "alpha": 0.2,
            "metrics": {
                "per_valset": {
                    "valid1": {"precision": 0.876, "recall": 0.678, "map50": 0.735, "map": 0.542, "fitness": 0.5613},
                    "valid2": {"precision": 0.599, "recall": 0.385, "map50": 0.398, "map": 0.273, "fitness": 0.2855}
                }
            }
        },
        {
            "alpha": 0.3,
            "metrics": {
                "per_valset": {
                    "valid1": {"precision": 0.633, "recall": 0.528, "map50": 0.521, "map": 0.366, "fitness": 0.3815},
                    "valid2": {"precision": 0.423, "recall": 0.341, "map50": 0.280, "map": 0.190, "fitness": 0.1990}
                }
            }
        }
    ]

    print("\n" + "="*80)
    print("WiSE-FT Advanced Experiments")
    print("="*80)

    # 모든 분석 실행
    run_fine_grained_search()
    print("\n")

    weight_scenarios = [
        (0.5, 0.5),  # Equal
        (0.7, 0.3),  # Valid1 priority
        (0.3, 0.7),  # Valid2 priority
        (0.1, 0.9),  # Valid2 강력 우선
    ]
    find_optimal_alpha_weighted(current_results, weight_scenarios)

    analyze_pareto_frontier(current_results)
    analyze_precision_recall_tradeoff(current_results)
    analyze_localization_quality(current_results)
    simulate_layerwise_strategy()
    visualize_results_ascii(current_results)
    generate_experiment_plan()

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\n다음 단계:")
    print("1. Phase 1 실험 실행 (Fine-grained search)")
    print("2. 결과 분석 및 최적 alpha 확인")
    print("3. Phase 2-3 실험 진행")

if __name__ == '__main__':
    main()
