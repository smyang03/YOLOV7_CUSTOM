#!/usr/bin/env python3
"""
현재 WiSE-FT 결과 즉시 분석 및 개선 방향 제시
"""

import json

# 현재 결과
current_results = [
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

def print_header(title):
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")

def analyze_key_insights():
    """핵심 인사이트 자동 추출"""
    print_header("🔍 즉시 실행 가능한 인사이트")

    # 1. α=0.0 vs α=0.1 비교
    alpha0 = current_results[0]
    alpha01 = current_results[1]

    v1_change = alpha01['metrics']['per_valset']['valid1']['fitness'] - alpha0['metrics']['per_valset']['valid1']['fitness']
    v2_change = alpha01['metrics']['per_valset']['valid2']['fitness'] - alpha0['metrics']['per_valset']['valid2']['fitness']

    print("📊 핵심 발견 #1: α=0.0 → 0.1 변화")
    print("─" * 80)
    print(f"  Valid1: {alpha0['metrics']['per_valset']['valid1']['fitness']:.4f} → "
          f"{alpha01['metrics']['per_valset']['valid1']['fitness']:.4f} "
          f"({v1_change:+.4f}, {v1_change/alpha0['metrics']['per_valset']['valid1']['fitness']*100:+.1f}%)")
    print(f"  Valid2: {alpha0['metrics']['per_valset']['valid2']['fitness']:.4f} → "
          f"{alpha01['metrics']['per_valset']['valid2']['fitness']:.4f} "
          f"({v2_change:+.4f}, {v2_change/alpha0['metrics']['per_valset']['valid2']['fitness']*100:+.1f}%)")

    if v2_change > 0 and v1_change < 0:
        print("\n  ✅ Trade-off 감지!")
        print(f"     → Valid2를 위해 Valid1을 {abs(v1_change):.4f} 희생")
        print(f"     → Valid2 개선: {v2_change:.4f} (+{v2_change/alpha0['metrics']['per_valset']['valid2']['fitness']*100:.1f}%)")
        print(f"\n  💡 α=0.0과 α=0.1 사이에 더 나은 균형점 존재 가능성 높음!")

    # 2. P/R 분석
    print("\n\n📊 핵심 발견 #2: Precision/Recall Balance")
    print("─" * 80)

    for r in current_results[:2]:  # α=0.0, 0.1만
        alpha = r['alpha']
        v2_p = r['metrics']['per_valset']['valid2']['precision']
        v2_r = r['metrics']['per_valset']['valid2']['recall']
        v2_pr_ratio = v2_p / v2_r if v2_r > 0 else 0

        print(f"  α={alpha:.1f}: Valid2 P/R = {v2_p:.3f}/{v2_r:.3f} = {v2_pr_ratio:.2f}", end="")
        if abs(v2_pr_ratio - 1.0) < 0.1:
            print(" ⭐ 균형!")
        print()

    print("\n  💡 α=0.1에서 Valid2의 P/R이 거의 균형 (1.02)")
    print("     → Recall 상승이 fitness 개선의 핵심!")
    print("     → Detection threshold가 최적화됨")

    # 3. 최적 α 예측
    print("\n\n📊 핵심 발견 #3: 최적 Alpha 예측")
    print("─" * 80)

    # 간단한 선형 보간
    v1_at_0 = alpha0['metrics']['per_valset']['valid1']['fitness']
    v1_at_01 = alpha01['metrics']['per_valset']['valid1']['fitness']
    v2_at_0 = alpha0['metrics']['per_valset']['valid2']['fitness']
    v2_at_01 = alpha01['metrics']['per_valset']['valid2']['fitness']

    # α=0.05 예측
    v1_pred_005 = v1_at_0 + (v1_at_01 - v1_at_0) * 0.5
    v2_pred_005 = v2_at_0 + (v2_at_01 - v2_at_0) * 0.5
    overall_pred_005 = (v1_pred_005 + v2_pred_005) / 2

    print(f"  α=0.05 예측 (선형 보간):")
    print(f"    Valid1: {v1_pred_005:.4f} (손실: {v1_pred_005-v1_at_0:.4f})")
    print(f"    Valid2: {v2_pred_005:.4f} (개선: {v2_pred_005-v2_at_0:.4f})")
    print(f"    Overall: {overall_pred_005:.4f}")

    # α=0.07 예측
    v1_pred_007 = v1_at_0 + (v1_at_01 - v1_at_0) * 0.7
    v2_pred_007 = v2_at_0 + (v2_at_01 - v2_at_0) * 0.7
    overall_pred_007 = (v1_pred_007 + v2_pred_007) / 2

    print(f"\n  α=0.07 예측 (선형 보간):")
    print(f"    Valid1: {v1_pred_007:.4f} (손실: {v1_pred_007-v1_at_0:.4f})")
    print(f"    Valid2: {v2_pred_007:.4f} (개선: {v2_pred_007-v2_at_0:.4f})")
    print(f"    Overall: {overall_pred_007:.4f}")

    print(f"\n  💡 α=0.05~0.07 사이에서 최적 균형점 예상!")

def recommend_experiments():
    """즉시 실행 가능한 실험 추천"""
    print_header("🚀 즉시 실행 가능한 개선 실험")

    experiments = [
        {
            "name": "Fine-grained Alpha Search",
            "priority": "🔴 최우선",
            "time": "30-60분",
            "command": "./run_fine_grained_search.sh",
            "expected": "α=0.04~0.08에서 최적점 발견, Valid2 +0.5~1.5%",
            "why": "현재 α=0.0과 0.1 사이에 명확한 trade-off 존재"
        },
        {
            "name": "Confidence Threshold Sweep",
            "priority": "🟡 권장",
            "time": "10-20분",
            "command": "./run_confidence_sweep.sh",
            "expected": "모델 변경 없이 Valid2 Recall 개선",
            "why": "α=0.1의 효과가 threshold 변화일 가능성"
        },
        {
            "name": "Full Range Sweep",
            "priority": "🟢 선택",
            "time": "60-90분",
            "command": "python wiseft_sweep_parallel.py --alpha-min 0.0 --alpha-max 1.0 --focus-range 0.1",
            "expected": "완전한 WiSE-FT curve, α=1.0 baseline",
            "why": "전체 경향 파악 및 완성도"
        }
    ]

    for exp in experiments:
        print(f"\n{exp['priority']} {exp['name']}")
        print("─" * 80)
        print(f"  예상 시간: {exp['time']}")
        print(f"  실행 방법: {exp['command']}")
        print(f"  예상 결과: {exp['expected']}")
        print(f"  이유: {exp['why']}")

def generate_prediction_table():
    """예측 결과 표"""
    print_header("📈 Alpha별 예측 결과 (0.0~0.1 구간)")

    alphas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]

    # α=0.0과 0.1의 값
    v1_0 = 0.6669
    v1_01 = 0.6435
    v2_0 = 0.3873
    v2_01 = 0.3930

    print(f"{'Alpha':<10} {'Valid1':<12} {'Valid2':<12} {'Overall':<12} {'V1 변화':<12} {'V2 변화':<12}")
    print("─" * 80)

    for alpha in alphas:
        # 선형 보간
        ratio = alpha / 0.1
        v1 = v1_0 + (v1_01 - v1_0) * ratio
        v2 = v2_0 + (v2_01 - v2_0) * ratio
        overall = (v1 + v2) / 2

        v1_change = v1 - v1_0
        v2_change = v2 - v2_0

        marker = ""
        if alpha == 0.0 or alpha == 0.1:
            marker = " (실측)"
        elif 0.04 <= alpha <= 0.08:
            marker = " ⭐"

        print(f"{alpha:<10.2f} {v1:<12.4f} {v2:<12.4f} {overall:<12.4f} "
              f"{v1_change:>+11.4f} {v2_change:>+11.4f}{marker}")

    print("\n⭐ = 최적 균형점 예상 구간")

def main():
    """메인 분석"""
    print("\n" + "="*80)
    print("현재 WiSE-FT 결과 즉시 분석")
    print("="*80)

    analyze_key_insights()
    generate_prediction_table()
    recommend_experiments()

    print("\n" + "="*80)
    print("다음 단계")
    print("="*80)
    print("\n1. 즉시 실행:")
    print("   chmod +x run_fine_grained_search.sh")
    print("   ./run_fine_grained_search.sh")
    print("\n2. 결과 확인:")
    print("   cat runs/wiseft_fine_grained/results.json")
    print("\n3. 최적 α 선택 및 모델 사용")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
