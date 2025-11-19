#!/usr/bin/env python3
"""
WiSE-FT 결과를 읽기 쉬운 표로 변환
"""

import json

# 현재 결과
results = [
    {"alpha": 0.0, "metrics": {"per_valset": {"valid1": {"precision": 0.871, "recall": 0.798, "map50": 0.828, "map": 0.649, "fitness": 0.6669}, "valid2": {"precision": 0.707, "recall": 0.458, "map50": 0.480, "map": 0.377, "fitness": 0.3873}}, "overall": {"precision": 0.789, "recall": 0.628, "map50": 0.654, "map": 0.513, "fitness": 0.5271}}},
    {"alpha": 0.1, "metrics": {"per_valset": {"valid1": {"precision": 0.889, "recall": 0.771, "map50": 0.810, "map": 0.625, "fitness": 0.6435}, "valid2": {"precision": 0.553, "recall": 0.541, "map50": 0.519, "map": 0.379, "fitness": 0.3930}}, "overall": {"precision": 0.721, "recall": 0.656, "map50": 0.665, "map": 0.502, "fitness": 0.5183}}},
    {"alpha": 0.2, "metrics": {"per_valset": {"valid1": {"precision": 0.876, "recall": 0.678, "map50": 0.735, "map": 0.542, "fitness": 0.5613}, "valid2": {"precision": 0.599, "recall": 0.385, "map50": 0.398, "map": 0.273, "fitness": 0.2855}}, "overall": {"precision": 0.738, "recall": 0.532, "map50": 0.567, "map": 0.408, "fitness": 0.4234}}},
    {"alpha": 0.3, "metrics": {"per_valset": {"valid1": {"precision": 0.633, "recall": 0.528, "map50": 0.521, "map": 0.366, "fitness": 0.3815}, "valid2": {"precision": 0.423, "recall": 0.341, "map50": 0.280, "map": 0.190, "fitness": 0.1990}}, "overall": {"precision": 0.528, "recall": 0.435, "map50": 0.401, "map": 0.278, "fitness": 0.2903}}}
]

# Alpha 정렬
results = sorted(results, key=lambda x: x['alpha'])

def print_section(title):
    print(f"\n{'='*100}")
    print(f"{title:^100}")
    print(f"{'='*100}\n")

# 1. Fitness 요약 (가장 중요)
print_section("📊 Fitness 요약 (Overall 성능)")

print(f"{'Alpha':^8} │ {'Valid1':^10} │ {'Valid2':^10} │ {'Overall':^10} │ {'Valid1 변화':^12} │ {'Valid2 변화':^12} │ {'추천':^15}")
print("─" * 100)

baseline_v1 = results[0]['metrics']['per_valset']['valid1']['fitness']
baseline_v2 = results[0]['metrics']['per_valset']['valid2']['fitness']

for r in results:
    alpha = r['alpha']
    v1_fit = r['metrics']['per_valset']['valid1']['fitness']
    v2_fit = r['metrics']['per_valset']['valid2']['fitness']
    overall_fit = r['metrics']['overall']['fitness']

    v1_change = v1_fit - baseline_v1
    v2_change = v2_fit - baseline_v2

    # 추천
    recommend = ""
    if alpha == 0.0:
        recommend = "✅ 최고 성능"
    elif alpha == 0.1:
        recommend = "⭐ Valid2↑"
    else:
        recommend = "❌"

    print(f"{alpha:^8.1f} │ {v1_fit:^10.4f} │ {v2_fit:^10.4f} │ {overall_fit:^10.4f} │ {v1_change:>+11.4f} │ {v2_change:>+11.4f} │ {recommend:^15}")

# 2. Valid1 상세
print_section("📌 Valid1 상세 성능")

print(f"{'Alpha':^8} │ {'Precision':^10} │ {'Recall':^10} │ {'mAP50':^10} │ {'mAP':^10} │ {'Fitness':^10} │ {'P/R 비율':^10}")
print("─" * 100)

for r in results:
    alpha = r['alpha']
    m = r['metrics']['per_valset']['valid1']
    pr_ratio = m['precision'] / m['recall'] if m['recall'] > 0 else 0

    print(f"{alpha:^8.1f} │ {m['precision']:^10.3f} │ {m['recall']:^10.3f} │ {m['map50']:^10.3f} │ {m['map']:^10.3f} │ {m['fitness']:^10.4f} │ {pr_ratio:^10.2f}")

# 3. Valid2 상세
print_section("📌 Valid2 상세 성능")

print(f"{'Alpha':^8} │ {'Precision':^10} │ {'Recall':^10} │ {'mAP50':^10} │ {'mAP':^10} │ {'Fitness':^10} │ {'P/R 비율':^10}")
print("─" * 100)

for r in results:
    alpha = r['alpha']
    m = r['metrics']['per_valset']['valid2']
    pr_ratio = m['precision'] / m['recall'] if m['recall'] > 0 else 0

    marker = ""
    if abs(pr_ratio - 1.0) < 0.1:
        marker = " ⭐균형"

    print(f"{alpha:^8.1f} │ {m['precision']:^10.3f} │ {m['recall']:^10.3f} │ {m['map50']:^10.3f} │ {m['map']:^10.3f} │ {m['fitness']:^10.4f} │ {pr_ratio:^10.2f}{marker}")

# 4. Precision/Recall 비교
print_section("🔍 Precision vs Recall 비교")

print(f"{'Alpha':^8} │ {'Valid1 P':^10} │ {'Valid1 R':^10} │ {'V1 P/R':^10} │ {'Valid2 P':^10} │ {'Valid2 R':^10} │ {'V2 P/R':^10}")
print("─" * 100)

for r in results:
    alpha = r['alpha']
    v1 = r['metrics']['per_valset']['valid1']
    v2 = r['metrics']['per_valset']['valid2']

    v1_pr = v1['precision'] / v1['recall'] if v1['recall'] > 0 else 0
    v2_pr = v2['precision'] / v2['recall'] if v2['recall'] > 0 else 0

    print(f"{alpha:^8.1f} │ {v1['precision']:^10.3f} │ {v1['recall']:^10.3f} │ {v1_pr:^10.2f} │ {v2['precision']:^10.3f} │ {v2['recall']:^10.3f} │ {v2_pr:^10.2f}")

# 5. 변화량 분석
print_section("📈 Alpha 변화에 따른 변화량")

print(f"{'구간':^15} │ {'Valid1 Δ':^12} │ {'Valid2 Δ':^12} │ {'Overall Δ':^12} │ {'패턴':^20}")
print("─" * 100)

for i in range(len(results) - 1):
    curr = results[i]
    next_r = results[i + 1]

    alpha_range = f"{curr['alpha']:.1f} → {next_r['alpha']:.1f}"

    v1_delta = next_r['metrics']['per_valset']['valid1']['fitness'] - curr['metrics']['per_valset']['valid1']['fitness']
    v2_delta = next_r['metrics']['per_valset']['valid2']['fitness'] - curr['metrics']['per_valset']['valid2']['fitness']
    overall_delta = next_r['metrics']['overall']['fitness'] - curr['metrics']['overall']['fitness']

    # 패턴 판단
    pattern = ""
    if v1_delta < 0 and v2_delta > 0:
        pattern = "Trade-off ⚖️"
    elif v1_delta < 0 and v2_delta < 0:
        pattern = "둘 다 하락 ⬇️"
    elif v1_delta > 0 and v2_delta > 0:
        pattern = "둘 다 상승 ⬆️"

    print(f"{alpha_range:^15} │ {v1_delta:>+11.4f} │ {v2_delta:>+11.4f} │ {overall_delta:>+11.4f} │ {pattern:^20}")

# 6. 최고 성능 하이라이트
print_section("🏆 최고 성능")

best_overall = max(results, key=lambda x: x['metrics']['overall']['fitness'])
best_v1 = max(results, key=lambda x: x['metrics']['per_valset']['valid1']['fitness'])
best_v2 = max(results, key=lambda x: x['metrics']['per_valset']['valid2']['fitness'])

print(f"Overall 최고:  α={best_overall['alpha']:.1f}, fitness={best_overall['metrics']['overall']['fitness']:.4f}")
print(f"Valid1 최고:   α={best_v1['alpha']:.1f}, fitness={best_v1['metrics']['per_valset']['valid1']['fitness']:.4f}")
print(f"Valid2 최고:   α={best_v2['alpha']:.1f}, fitness={best_v2['metrics']['per_valset']['valid2']['fitness']:.4f}")

# 7. 빠른 의사결정 가이드
print_section("🎯 빠른 의사결정 가이드")

print("상황별 추천:")
print(f"  1. Overall 성능 최우선:       α={best_overall['alpha']:.1f} (fitness {best_overall['metrics']['overall']['fitness']:.4f})")
print(f"  2. Valid1 성능 최우선:        α={best_v1['alpha']:.1f} (fitness {best_v1['metrics']['per_valset']['valid1']['fitness']:.4f})")
print(f"  3. Valid2 성능 최우선:        α={best_v2['alpha']:.1f} (fitness {best_v2['metrics']['per_valset']['valid2']['fitness']:.4f})")
print(f"  4. Valid2 약간 개선 원함:     α=0.1 (Valid2 +1.5%, Valid1 -3.5%)")
print(f"  5. 더 나은 균형점 찾기:       Fine-grained search 실행 (α=0.0~0.15, step=0.02)")

print("\n" + "="*100)
print("💡 핵심 인사이트:")
print("  - α=0.0이 Overall 최고 성능")
print("  - α=0.1에서 Valid2가 +1.5% 개선되지만 Valid1은 -3.5% 손실")
print("  - α≥0.2는 급격히 성능 하락 → 의미 없음")
print("  - α=0.0~0.1 사이를 더 세밀하게 탐색하면 최적 균형점 찾을 가능성 높음")
print("="*100 + "\n")
