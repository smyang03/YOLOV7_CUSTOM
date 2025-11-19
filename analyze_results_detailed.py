#!/usr/bin/env python3
"""
WiSE-FT 결과 상세 분석 스크립트
- 모든 Alpha 값 비교
- Trade-off 시각화
- 최적점 추천
"""

import json
import sys

# 제공된 결과 데이터
data = {
  "baselines": {
    "scratch": {
      "alpha": 0.0,
      "model_path": "new_list/600.pt",
      "metrics": {
        "per_valset": {
          "valid2": {"precision": 0.707, "recall": 0.458, "map50": 0.48, "map": 0.377, "fitness": 0.3873},
          "valid1": {"precision": 0.871, "recall": 0.798, "map50": 0.828, "map": 0.649, "fitness": 0.6669}
        },
        "overall": {"precision": 0.789, "recall": 0.628, "map50": 0.654, "map": 0.513, "fitness": 0.5271}
      }
    },
    "finetuned": {
      "alpha": 1.0,
      "model_path": "new_list/620.pt",
      "metrics": {
        "per_valset": {
          "valid2": {"precision": 0.965, "recall": 0.964, "map50": 0.985, "map": 0.823, "fitness": 0.8392},
          "valid1": {"precision": 0.883, "recall": 0.755, "map50": 0.804, "map": 0.619, "fitness": 0.6375}
        },
        "overall": {"precision": 0.924, "recall": 0.859, "map50": 0.895, "map": 0.721, "fitness": 0.7384}
      }
    }
  },
  "wiseft_results": [
    {"alpha": 0.0, "metrics": {"per_valset": {"valid1": {"fitness": 0.6669}, "valid2": {"fitness": 0.3873}}, "overall": {"fitness": 0.5271}}},
    {"alpha": 0.1, "metrics": {"per_valset": {"valid1": {"fitness": 0.6435}, "valid2": {"fitness": 0.3930}}, "overall": {"fitness": 0.5183}}},
    {"alpha": 0.2, "metrics": {"per_valset": {"valid1": {"fitness": 0.5613}, "valid2": {"fitness": 0.2855}}, "overall": {"fitness": 0.4234}}},
    {"alpha": 0.3, "metrics": {"per_valset": {"valid1": {"fitness": 0.3815}, "valid2": {"fitness": 0.1990}}, "overall": {"fitness": 0.2903}}}
  ]
}

def print_header(title, width=120):
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")

def print_section(title, width=120):
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width + "\n")

# 전체 결과 수집
all_results = []

# Baseline 추가
scratch = data['baselines']['scratch']
finetuned = data['baselines']['finetuned']

all_results.append({
    'alpha': 0.0,
    'name': 'Scratch',
    'v1': scratch['metrics']['per_valset']['valid1']['fitness'],
    'v2': scratch['metrics']['per_valset']['valid2']['fitness'],
    'overall': scratch['metrics']['overall']['fitness'],
    'v1_p': scratch['metrics']['per_valset']['valid1']['precision'],
    'v1_r': scratch['metrics']['per_valset']['valid1']['recall'],
    'v2_p': scratch['metrics']['per_valset']['valid2']['precision'],
    'v2_r': scratch['metrics']['per_valset']['valid2']['recall']
})

all_results.append({
    'alpha': 1.0,
    'name': 'Finetuned',
    'v1': finetuned['metrics']['per_valset']['valid1']['fitness'],
    'v2': finetuned['metrics']['per_valset']['valid2']['fitness'],
    'overall': finetuned['metrics']['overall']['fitness'],
    'v1_p': finetuned['metrics']['per_valset']['valid1']['precision'],
    'v1_r': finetuned['metrics']['per_valset']['valid1']['recall'],
    'v2_p': finetuned['metrics']['per_valset']['valid2']['precision'],
    'v2_r': finetuned['metrics']['per_valset']['valid2']['recall']
})

# WiSE-FT 결과 추가 (α=0.0 제외, 중복이므로)
for r in data['wiseft_results']:
    if r['alpha'] == 0.0:
        continue  # Scratch와 동일하므로 skip

    all_results.append({
        'alpha': r['alpha'],
        'name': f'WiSE-FT α={r["alpha"]:.1f}',
        'v1': r['metrics']['per_valset']['valid1']['fitness'],
        'v2': r['metrics']['per_valset']['valid2']['fitness'],
        'overall': r['metrics']['overall']['fitness']
    })

# Alpha 순 정렬
all_results = sorted(all_results, key=lambda x: x['alpha'])

print_header("WiSE-FT 전체 결과 상세 분석")

# 1. 전체 성능 비교
print_section("📊 1. 전체 성능 비교 (Overall Fitness)")

print(f"{'Alpha':<10} │ {'Model':<20} │ {'Overall':<12} │ {'Valid1':<12} │ {'Valid2':<12} │ {'V1 vs V2':<15}")
print("─" * 120)

for r in all_results:
    v1_v2_ratio = f"{r['v1']/r['v2']:.2f}" if r['v2'] > 0 else "N/A"
    marker = " ⭐" if r['overall'] == max([x['overall'] for x in all_results]) else ""
    print(f"{r['alpha']:<10.1f} │ {r['name']:<20} │ {r['overall']:<12.4f}{marker:<3} │ {r['v1']:<12.4f} │ {r['v2']:<12.4f} │ {v1_v2_ratio:<15}")

# 2. 순위 분석
print_section("🏆 2. 성능 순위")

print("Overall 순위:")
overall_sorted = sorted(all_results, key=lambda x: x['overall'], reverse=True)
for i, r in enumerate(overall_sorted[:5], 1):
    print(f"  {i}. α={r['alpha']:.1f} ({r['name']:<20}) : {r['overall']:.4f}")

print("\nValid1 순위:")
v1_sorted = sorted(all_results, key=lambda x: x['v1'], reverse=True)
for i, r in enumerate(v1_sorted[:5], 1):
    print(f"  {i}. α={r['alpha']:.1f} ({r['name']:<20}) : {r['v1']:.4f}")

print("\nValid2 순위:")
v2_sorted = sorted(all_results, key=lambda x: x['v2'], reverse=True)
for i, r in enumerate(v2_sorted[:5], 1):
    print(f"  {i}. α={r['alpha']:.1f} ({r['name']:<20}) : {r['v2']:.4f}")

# 3. Trade-off 분석
print_section("⚖️ 3. Trade-off 분석 (Scratch 대비)")

scratch_v1 = all_results[0]['v1']
scratch_v2 = all_results[0]['v2']
scratch_overall = all_results[0]['overall']

print(f"{'Alpha':<10} │ {'Model':<20} │ {'V1 변화':<15} │ {'V2 변화':<15} │ {'Overall 변화':<15} │ {'평가':<20}")
print("─" * 120)

for r in all_results:
    v1_delta = r['v1'] - scratch_v1
    v2_delta = r['v2'] - scratch_v2
    overall_delta = r['overall'] - scratch_overall

    v1_pct = (v1_delta / scratch_v1 * 100) if scratch_v1 > 0 else 0
    v2_pct = (v2_delta / scratch_v2 * 100) if scratch_v2 > 0 else 0
    overall_pct = (overall_delta / scratch_overall * 100) if scratch_overall > 0 else 0

    # 평가
    if r['alpha'] == 0.0:
        assessment = "Baseline"
    elif overall_delta > 0 and v2_delta > 0:
        assessment = "✅ 성공적 Trade-off"
    elif v1_delta < 0 and v2_delta > 0 and overall_delta < 0:
        assessment = "⚠️ 미흡한 Trade-off"
    elif v1_delta < 0 and v2_delta < 0:
        assessment = "❌ 양쪽 모두 하락"
    else:
        assessment = "Unknown"

    print(f"{r['alpha']:<10.1f} │ {r['name']:<20} │ {v1_pct:>+6.1f}% ({v1_delta:>+.4f}) │ {v2_pct:>+6.1f}% ({v2_delta:>+.4f}) │ {overall_pct:>+6.1f}% ({overall_delta:>+.4f}) │ {assessment:<20}")

# 4. P/R 균형 분석
print_section("🎯 4. Precision/Recall 균형 분석")

print(f"{'Alpha':<10} │ {'Model':<20} │ {'Valid1 P/R':<15} │ {'Valid2 P/R':<15} │ {'비고':<30}")
print("─" * 120)

for r in all_results:
    if 'v1_p' in r and 'v1_r' in r:
        v1_pr = r['v1_p'] / r['v1_r'] if r['v1_r'] > 0 else 0
        v2_pr = r['v2_p'] / r['v2_r'] if r['v2_r'] > 0 else 0

        note = ""
        if abs(v1_pr - 1.0) < 0.15:
            note += "V1 균형 "
        if abs(v2_pr - 1.0) < 0.15:
            note += "V2 균형 ⭐"

        print(f"{r['alpha']:<10.1f} │ {r['name']:<20} │ {v1_pr:<15.2f} │ {v2_pr:<15.2f} │ {note:<30}")

# 5. 핵심 발견
print_section("💡 5. 핵심 발견사항")

best_overall = max(all_results, key=lambda x: x['overall'])
best_v1 = max(all_results, key=lambda x: x['v1'])
best_v2 = max(all_results, key=lambda x: x['v2'])

print(f"1. 🏆 Overall 최고 성능: α={best_overall['alpha']:.1f} ({best_overall['name']})")
print(f"   - Overall: {best_overall['overall']:.4f}")
print(f"   - Valid1:  {best_overall['v1']:.4f}")
print(f"   - Valid2:  {best_overall['v2']:.4f}")
print()

print(f"2. 📊 Finetuned vs Scratch 비교:")
finetuned_data = [r for r in all_results if r['alpha'] == 1.0][0]
v1_change = finetuned_data['v1'] - scratch_v1
v2_change = finetuned_data['v2'] - scratch_v2
overall_change = finetuned_data['overall'] - scratch_overall

print(f"   - Valid1:  {v1_change:+.4f} ({v1_change/scratch_v1*100:+.1f}%)")
print(f"   - Valid2:  {v2_change:+.4f} ({v2_change/scratch_v2*100:+.1f}%) ← 116.7% 향상!")
print(f"   - Overall: {overall_change:+.4f} ({overall_change/scratch_overall*100:+.1f}%)")
print()

print("3. ⚠️ α=0.1~0.3 구간 분석:")
print("   - 모두 Scratch보다 Overall 성능 하락")
print("   - α가 증가할수록 급격히 성능 저하")
print("   - 이 구간은 비효율적")
print()

print("4. ❓ 미탐색 구간: α=0.4~0.9")
print("   - 현재 평가: α=0.0, 0.1, 0.2, 0.3, 1.0")
print("   - 미평가: α=0.4, 0.5, 0.6, 0.7, 0.8, 0.9")
print("   - 이 구간에 최적 균형점이 있을 가능성 높음")
print()

# 6. 시각화
print_section("📈 6. 성능 트렌드 시각화")

print("Overall Fitness 추이:")
print()
max_fitness = max([r['overall'] for r in all_results])
for r in all_results:
    bar_len = int(r['overall'] / max_fitness * 50)
    bar = "█" * bar_len
    print(f"  α={r['alpha']:<4.1f} │ {bar} {r['overall']:.4f}")

print()
print("Valid2 Fitness 추이 (핵심 개선 지표):")
print()
max_v2 = max([r['v2'] for r in all_results])
for r in all_results:
    bar_len = int(r['v2'] / max_v2 * 50)
    bar = "█" * bar_len
    print(f"  α={r['alpha']:<4.1f} │ {bar} {r['v2']:.4f}")

# 7. 권장사항
print_section("🚀 7. 다음 단계 권장사항")

print("🔴 최우선: α=0.4~0.9 범위 탐색")
print("─" * 120)
print("  현재 상황:")
print(f"    - α=0.0 (Scratch):   Overall {scratch_overall:.4f}")
print("    - α=0.1~0.3:         성능 하락 (비효율적)")
print(f"    - α=1.0 (Finetuned): Overall {finetuned_data['overall']:.4f} ⭐ 최고")
print()
print("  필요성:")
print("    - α=0.0~0.3과 α=1.0 사이에 큰 성능 갭 존재")
print("    - α=0.4~0.9 구간에 더 나은 균형점이 있을 가능성")
print("    - 전체 Trade-off 곡선을 보기 위해 필수")
print()
print("  실행 명령:")
print("    ./run_wiseft_expanded.sh")
print("    또는")
print("    ./run_wiseft_full_range.sh")
print()

print("🟡 2단계: Fine-grained search")
print("─" * 120)
print("  - α=0.4~0.9 결과를 보고 최적 구간을 0.02 step으로 정밀 탐색")
print("  - 예: α=0.7~0.9가 좋으면, 0.70, 0.72, 0.74, ..., 0.90 탐색")
print()

print("🟢 3단계: 최종 모델 선정 및 추가 최적화")
print("─" * 120)
print("  - Confidence threshold 조정")
print("  - NMS threshold 조정")
print("  - Layer-wise WiSE-FT (선택)")
print()

print("=" * 120)
print("분석 완료".center(120))
print("=" * 120)
