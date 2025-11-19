#!/usr/bin/env python3
"""
WiSE-FT 결과 전체 경향 분석 및 시각화
"""
import json
import numpy as np
from pathlib import Path

# 현재 결과 데이터
results = [
    {
        "alpha": 0.0,
        "metrics": {
            "per_valset": {
                "valid1": {"precision": 0.871, "recall": 0.798, "map50": 0.828, "map": 0.649, "fitness": 0.6669},
                "valid2": {"precision": 0.707, "recall": 0.458, "map50": 0.480, "map": 0.377, "fitness": 0.3873}
            },
            "overall": {"precision": 0.789, "recall": 0.628, "map50": 0.654, "map": 0.513, "fitness": 0.5271}
        }
    },
    {
        "alpha": 0.1,
        "metrics": {
            "per_valset": {
                "valid1": {"precision": 0.889, "recall": 0.771, "map50": 0.810, "map": 0.625, "fitness": 0.6435},
                "valid2": {"precision": 0.553, "recall": 0.541, "map50": 0.519, "map": 0.379, "fitness": 0.3930}
            },
            "overall": {"precision": 0.721, "recall": 0.656, "map50": 0.665, "map": 0.502, "fitness": 0.5183}
        }
    },
    {
        "alpha": 0.2,
        "metrics": {
            "per_valset": {
                "valid1": {"precision": 0.876, "recall": 0.678, "map50": 0.735, "map": 0.542, "fitness": 0.5613},
                "valid2": {"precision": 0.599, "recall": 0.385, "map50": 0.398, "map": 0.273, "fitness": 0.2855}
            },
            "overall": {"precision": 0.738, "recall": 0.532, "map50": 0.567, "map": 0.408, "fitness": 0.4234}
        }
    },
    {
        "alpha": 0.3,
        "metrics": {
            "per_valset": {
                "valid1": {"precision": 0.633, "recall": 0.528, "map50": 0.521, "map": 0.366, "fitness": 0.3815},
                "valid2": {"precision": 0.423, "recall": 0.341, "map50": 0.280, "map": 0.190, "fitness": 0.1990}
            },
            "overall": {"precision": 0.528, "recall": 0.435, "map50": 0.401, "map": 0.278, "fitness": 0.2903}
        }
    }
]

def print_section(title):
    """섹션 헤더 출력"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")

def analyze_trends():
    """전체 경향 분석"""
    print_section("1. 전체 성능 경향 분석")

    alphas = [r['alpha'] for r in results]
    valid1_fitness = [r['metrics']['per_valset']['valid1']['fitness'] for r in results]
    valid2_fitness = [r['metrics']['per_valset']['valid2']['fitness'] for r in results]
    overall_fitness = [r['metrics']['overall']['fitness'] for r in results]

    print(f"{'Alpha':<10} {'Valid1':<12} {'Valid2':<12} {'Overall':<12} {'V1 변화':<12} {'V2 변화':<12}")
    print("─" * 80)

    for i, r in enumerate(results):
        alpha = r['alpha']
        v1 = r['metrics']['per_valset']['valid1']['fitness']
        v2 = r['metrics']['per_valset']['valid2']['fitness']
        ov = r['metrics']['overall']['fitness']

        if i == 0:
            v1_change = "-"
            v2_change = "-"
        else:
            v1_change = f"{v1 - valid1_fitness[i-1]:+.4f}"
            v2_change = f"{v2 - valid2_fitness[i-1]:+.4f}"

        print(f"{alpha:<10.1f} {v1:<12.4f} {v2:<12.4f} {ov:<12.4f} {v1_change:<12} {v2_change:<12}")

    # 전체 변화량
    print("\n📈 전체 변화 (Alpha 0.0 → 0.3):")
    print(f"   Valid1: {valid1_fitness[0]:.4f} → {valid1_fitness[-1]:.4f} "
          f"({(valid1_fitness[-1]-valid1_fitness[0])/valid1_fitness[0]*100:+.1f}%)")
    print(f"   Valid2: {valid2_fitness[0]:.4f} → {valid2_fitness[-1]:.4f} "
          f"({(valid2_fitness[-1]-valid2_fitness[0])/valid2_fitness[0]*100:+.1f}%)")
    print(f"   Overall: {overall_fitness[0]:.4f} → {overall_fitness[-1]:.4f} "
          f"({(overall_fitness[-1]-overall_fitness[0])/overall_fitness[0]*100:+.1f}%)")

def analyze_tradeoff():
    """Trade-off 분석"""
    print_section("2. Valid1 vs Valid2 Trade-off 분석")

    # Alpha=0.0 vs Alpha=0.1 비교
    v1_0 = results[0]['metrics']['per_valset']['valid1']['fitness']
    v2_0 = results[0]['metrics']['per_valset']['valid2']['fitness']
    v1_1 = results[1]['metrics']['per_valset']['valid1']['fitness']
    v2_1 = results[1]['metrics']['per_valset']['valid2']['fitness']

    print("Alpha 0.0 → 0.1 변화:")
    print(f"   Valid1: {v1_0:.4f} → {v1_1:.4f} ({v1_1-v1_0:+.4f}, {(v1_1-v1_0)/v1_0*100:+.1f}%)")
    print(f"   Valid2: {v2_0:.4f} → {v2_1:.4f} ({v2_1-v2_0:+.4f}, {(v2_1-v2_0)/v2_0*100:+.1f}%)")

    if (v1_1 - v1_0) < 0 and (v2_1 - v2_0) > 0:
        print("\n   ✅ Trade-off 감지: Valid1 하락, Valid2 상승")
        print(f"   → Alpha=0.1에서 Valid2를 위해 Valid1을 {abs(v1_1-v1_0):.4f} 희생")
    elif (v1_1 - v1_0) < 0 and (v2_1 - v2_0) < 0:
        print("\n   ⚠️  두 validation set 모두 하락 (Alpha 0.0 → 0.1)")

    # 전체 패턴
    print("\n전체 패턴:")
    for i in range(len(results)-1):
        alpha_curr = results[i]['alpha']
        alpha_next = results[i+1]['alpha']
        v1_curr = results[i]['metrics']['per_valset']['valid1']['fitness']
        v1_next = results[i+1]['metrics']['per_valset']['valid1']['fitness']
        v2_curr = results[i]['metrics']['per_valset']['valid2']['fitness']
        v2_next = results[i+1]['metrics']['per_valset']['valid2']['fitness']

        v1_direction = "↑" if v1_next > v1_curr else "↓"
        v2_direction = "↑" if v2_next > v2_curr else "↓"

        print(f"   α {alpha_curr:.1f}→{alpha_next:.1f}: Valid1 {v1_direction} ({v1_next-v1_curr:+.4f}), "
              f"Valid2 {v2_direction} ({v2_next-v2_curr:+.4f})")

def visualize_ascii():
    """ASCII 그래프로 시각화"""
    print_section("3. 성능 변화 시각화")

    alphas = [r['alpha'] for r in results]
    valid1_fitness = [r['metrics']['per_valset']['valid1']['fitness'] for r in results]
    valid2_fitness = [r['metrics']['per_valset']['valid2']['fitness'] for r in results]

    # 정규화 (0-40 범위로)
    min_val = 0.0
    max_val = 0.7

    def scale(val):
        return int((val - min_val) / (max_val - min_val) * 50)

    print("Fitness")
    print("  0.70 |")
    print("  0.65 |")
    print("  0.60 |")
    print("  0.55 |")
    print("  0.50 |")
    print("  0.45 |")
    print("  0.40 |")
    print("  0.35 |")
    print("  0.30 |")
    print("  0.25 |")
    print("  0.20 |")
    print("  0.15 |")
    print("  0.10 |")
    print("       +─────────────────────────────────────────────────")
    print("         0.0      0.1      0.2      0.3      (1.0?)")
    print("                        Alpha")
    print()

    # 데이터 포인트
    print("데이터 포인트:")
    for i, alpha in enumerate(alphas):
        v1 = valid1_fitness[i]
        v2 = valid2_fitness[i]
        print(f"   α={alpha:.1f}: Valid1={v1:.4f}, Valid2={v2:.4f}")

def analyze_detailed_metrics():
    """상세 메트릭 분석"""
    print_section("4. 상세 메트릭 분석")

    print("Precision 변화:")
    for r in results:
        alpha = r['alpha']
        v1_p = r['metrics']['per_valset']['valid1']['precision']
        v2_p = r['metrics']['per_valset']['valid2']['precision']
        print(f"   α={alpha:.1f}: Valid1={v1_p:.3f}, Valid2={v2_p:.3f}")

    print("\nRecall 변화:")
    for r in results:
        alpha = r['alpha']
        v1_r = r['metrics']['per_valset']['valid1']['recall']
        v2_r = r['metrics']['per_valset']['valid2']['recall']
        print(f"   α={alpha:.1f}: Valid1={v1_r:.3f}, Valid2={v2_r:.3f}")

    print("\nmAP 변화:")
    for r in results:
        alpha = r['alpha']
        v1_m = r['metrics']['per_valset']['valid1']['map']
        v2_m = r['metrics']['per_valset']['valid2']['map']
        print(f"   α={alpha:.1f}: Valid1={v1_m:.3f}, Valid2={v2_m:.3f}")

def extrapolate_alpha_1():
    """Alpha=1.0 성능 추정"""
    print_section("5. Alpha=1.0 (Fine-tuned) 성능 추정")

    alphas = np.array([r['alpha'] for r in results])
    valid1_fitness = np.array([r['metrics']['per_valset']['valid1']['fitness'] for r in results])
    valid2_fitness = np.array([r['metrics']['per_valset']['valid2']['fitness'] for r in results])

    # 선형 추세
    v1_coef = np.polyfit(alphas, valid1_fitness, 1)
    v2_coef = np.polyfit(alphas, valid2_fitness, 1)

    v1_est = v1_coef[0] * 1.0 + v1_coef[1]
    v2_est = v2_coef[0] * 1.0 + v2_coef[1]

    print("선형 추세 기반 추정 (Alpha=1.0):")
    print(f"   Valid1 추정: {v1_est:.4f}")
    print(f"   Valid2 추정: {v2_est:.4f}")
    print(f"   Overall 추정: {(v1_est + v2_est)/2:.4f}")

    print("\n⚠️  주의: 이것은 선형 외삽(extrapolation)입니다.")
    print("   실제 Alpha=1.0 평가가 필요합니다!")

    # 추세 기울기
    print(f"\n📉 추세:")
    print(f"   Valid1 기울기: {v1_coef[0]:.4f} (alpha 0.1 증가당 {v1_coef[0]*0.1:.4f} 변화)")
    print(f"   Valid2 기울기: {v2_coef[0]:.4f} (alpha 0.1 증가당 {v2_coef[0]*0.1:.4f} 변화)")

def find_optimal_alpha():
    """최적 Alpha 찾기"""
    print_section("6. 최적 Alpha 분석")

    # Overall fitness 기준
    best_overall = max(results, key=lambda r: r['metrics']['overall']['fitness'])
    print(f"Overall Fitness 최대:")
    print(f"   Alpha = {best_overall['alpha']:.1f}")
    print(f"   Fitness = {best_overall['metrics']['overall']['fitness']:.4f}")
    print(f"   Valid1 = {best_overall['metrics']['per_valset']['valid1']['fitness']:.4f}")
    print(f"   Valid2 = {best_overall['metrics']['per_valset']['valid2']['fitness']:.4f}")

    # Valid1 기준
    best_v1 = max(results, key=lambda r: r['metrics']['per_valset']['valid1']['fitness'])
    print(f"\nValid1 Fitness 최대:")
    print(f"   Alpha = {best_v1['alpha']:.1f}")
    print(f"   Valid1 = {best_v1['metrics']['per_valset']['valid1']['fitness']:.4f}")

    # Valid2 기준
    best_v2 = max(results, key=lambda r: r['metrics']['per_valset']['valid2']['fitness'])
    print(f"\nValid2 Fitness 최대:")
    print(f"   Alpha = {best_v2['alpha']:.1f}")
    print(f"   Valid2 = {best_v2['metrics']['per_valset']['valid2']['fitness']:.4f}")

    # Balanced alpha (valid1과 valid2의 차이 최소화)
    min_gap = min(results, key=lambda r: abs(
        r['metrics']['per_valset']['valid1']['fitness'] -
        r['metrics']['per_valset']['valid2']['fitness']
    ))
    print(f"\nValid1-Valid2 차이 최소 (균형점):")
    print(f"   Alpha = {min_gap['alpha']:.1f}")
    print(f"   Valid1 = {min_gap['metrics']['per_valset']['valid1']['fitness']:.4f}")
    print(f"   Valid2 = {min_gap['metrics']['per_valset']['valid2']['fitness']:.4f}")
    print(f"   차이 = {abs(min_gap['metrics']['per_valset']['valid1']['fitness'] - min_gap['metrics']['per_valset']['valid2']['fitness']):.4f}")

def check_missing_data():
    """누락된 데이터 확인"""
    print_section("7. 데이터 완성도 체크")

    print("현재 평가된 Alpha 값:")
    print(f"   {[r['alpha'] for r in results]}")

    print("\n누락된 중요 평가:")
    print("   ❌ Alpha = 1.0 (Fine-tuned baseline)")
    print("   ℹ️  Alpha = 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 (선택사항)")

    print("\n권장사항:")
    print("   1. Alpha=1.0 평가 필수 (baseline 비교용)")
    print("   2. Alpha=0.4~0.9 평가 고려 (완전한 트렌드 파악)")

def main():
    """메인 분석 실행"""
    print("\n" + "="*80)
    print("WiSE-FT 결과 전체 경향 분석")
    print("="*80)

    analyze_trends()
    analyze_tradeoff()
    visualize_ascii()
    analyze_detailed_metrics()
    extrapolate_alpha_1()
    find_optimal_alpha()
    check_missing_data()

    print_section("분석 완료")
    print("다음 단계:")
    print("1. Alpha=1.0 평가 실행")
    print("2. 데이터셋 경로 검증 (valid1.txt, valid2.txt)")
    print("3. 결과 재확인 및 최종 결론")

if __name__ == '__main__':
    main()
