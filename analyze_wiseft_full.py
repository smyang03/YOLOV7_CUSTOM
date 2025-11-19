#!/usr/bin/env python3
"""
WiSE-FT 전체 결과 분석 (Baseline 포함)
"""

import json
import sys

def analyze_full_results(json_file):
    """전체 결과 분석"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    print("=" * 120)
    print("WiSE-FT 전체 결과 분석".center(120))
    print("=" * 120)
    print()

    # Baseline 분석
    print("📊 BASELINE 모델 성능")
    print("-" * 120)

    scratch = data['baselines']['scratch']
    finetuned = data['baselines']['finetuned']

    print(f"\n{'Model':<15} │ {'Overall Fit':<12} │ {'Valid1 Fit':<12} │ {'Valid2 Fit':<12} │ {'V1 P/R':<10} │ {'V2 P/R':<10}")
    print("─" * 120)

    # Scratch
    s_overall = scratch['metrics']['overall']['fitness']
    s_v1 = scratch['metrics']['per_valset']['valid1']['fitness']
    s_v2 = scratch['metrics']['per_valset']['valid2']['fitness']
    s_v1_pr = scratch['metrics']['per_valset']['valid1']['precision'] / scratch['metrics']['per_valset']['valid1']['recall']
    s_v2_pr = scratch['metrics']['per_valset']['valid2']['precision'] / scratch['metrics']['per_valset']['valid2']['recall']

    print(f"{'Scratch (α=0.0)':<15} │ {s_overall:<12.4f} │ {s_v1:<12.4f} │ {s_v2:<12.4f} │ {s_v1_pr:<10.2f} │ {s_v2_pr:<10.2f}")

    # Finetuned
    f_overall = finetuned['metrics']['overall']['fitness']
    f_v1 = finetuned['metrics']['per_valset']['valid1']['fitness']
    f_v2 = finetuned['metrics']['per_valset']['valid2']['fitness']
    f_v1_pr = finetuned['metrics']['per_valset']['valid1']['precision'] / finetuned['metrics']['per_valset']['valid1']['recall']
    f_v2_pr = finetuned['metrics']['per_valset']['valid2']['precision'] / finetuned['metrics']['per_valset']['valid2']['recall']

    print(f"{'Finetuned (α=1.0)':<15} │ {f_overall:<12.4f} │ {f_v1:<12.4f} │ {f_v2:<12.4f} │ {f_v1_pr:<10.2f} │ {f_v2_pr:<10.2f}")

    # Delta 계산
    print("\n📈 Baseline 비교 (Finetuned - Scratch)")
    print("-" * 120)
    overall_delta = f_overall - s_overall
    v1_delta = f_v1 - s_v1
    v2_delta = f_v2 - s_v2

    print(f"Overall: {overall_delta:+.4f} ({overall_delta/s_overall*100:+.1f}%)")
    print(f"Valid1:  {v1_delta:+.4f} ({v1_delta/s_v1*100:+.1f}%)")
    print(f"Valid2:  {v2_delta:+.4f} ({v2_delta/s_v2*100:+.1f}%)")

    print()
    print("💡 핵심 발견:")
    if f_overall > s_overall:
        print(f"  ✅ Finetuned 모델이 Overall에서 {overall_delta/s_overall*100:.1f}% 더 좋음")
    if v2_delta > 0:
        print(f"  ✅ Valid2 성능이 {v2_delta/s_v2*100:.1f}% 개선 (0.3873 → 0.8392)")
    if v1_delta < 0:
        print(f"  ⚠️  Valid1 성능이 {abs(v1_delta)/s_v1*100:.1f}% 하락 (0.6669 → 0.6375)")
    else:
        print(f"  ✅ Valid1 성능도 유지 또는 개선")

    # WiSE-FT 결과
    print()
    print("=" * 120)
    print("🔀 WiSE-FT 결과 (α=0.0~0.3)")
    print("=" * 120)
    print()

    results = sorted(data['wiseft_results'], key=lambda x: x['alpha'])

    print(f"{'Alpha':<10} │ {'Overall Fit':<12} │ {'Valid1 Fit':<12} │ {'Valid2 Fit':<12} │ {'Overall Δ':<12} │ {'V1 Δ':<12} │ {'V2 Δ':<12}")
    print("─" * 120)

    for r in results:
        alpha = r['alpha']
        overall = r['metrics']['overall']['fitness']
        v1 = r['metrics']['per_valset']['valid1']['fitness']
        v2 = r['metrics']['per_valset']['valid2']['fitness']

        # Baseline과 비교
        overall_d = overall - s_overall
        v1_d = v1 - s_v1
        v2_d = v2 - s_v2

        print(f"{alpha:<10.3f} │ {overall:<12.4f} │ {v1:<12.4f} │ {v2:<12.4f} │ {overall_d:>+11.4f} │ {v1_d:>+11.4f} │ {v2_d:>+11.4f}")

    print()
    print("💡 WiSE-FT 인사이트:")
    print("-" * 120)
    print("  ❌ α=0.0~0.3 범위는 모두 Scratch보다 못하거나 비슷함")
    print("  ❌ α=0.1~0.3은 오히려 성능을 크게 떨어뜨림")
    print("  ⚠️  현재까지 최고 성능: Finetuned (α=1.0) - Overall 0.7384")
    print()

    # 다음 단계 제안
    print("=" * 120)
    print("🚀 권장 다음 단계")
    print("=" * 120)
    print()

    print("🔴 최우선: α=0.4~0.9 범위 탐색")
    print("-" * 120)
    print("  현재 상황:")
    print(f"    - α=0.0 (Scratch): Overall {s_overall:.4f}")
    print(f"    - α=0.1~0.3: 성능 하락")
    print(f"    - α=1.0 (Finetuned): Overall {f_overall:.4f} ⭐ 최고")
    print()
    print("  예상:")
    print("    - α=0.7~0.9에서 Valid1과 Valid2의 최적 균형점 발견 가능")
    print("    - Valid1 성능을 약간 희생하면서 Valid2를 크게 개선하는 구간")
    print()
    print("  실행 명령:")
    print("    python wiseft_sweep_parallel.py \\")
    print("      --scratch new_list/600.pt \\")
    print("      --finetuned new_list/620.pt \\")
    print("      --data new_list/data.yaml \\")
    print("      --val-sets valid1 valid2 \\")
    print("      --alpha-min 0.4 --alpha-max 1.0 --alpha-step 0.1 \\")
    print("      --num-gpus 8 --batch-size 128")
    print()

    print("🟡 추가 권장: Fine-grained search in 0.7~0.9")
    print("-" * 120)
    print("  α=0.7~0.9 사이를 0.02 step으로 정밀 탐색")
    print("  예상 최적점: α≈0.8 (Valid1 약간 하락, Valid2 크게 개선)")
    print()

    print("🟢 선택: Confidence threshold 최적화")
    print("-" * 120)
    print("  Finetuned 모델에서 confidence threshold를 조정하여")
    print("  Valid1 성능 손실 없이 Valid2 성능 유지 가능성 탐색")
    print()

    print("=" * 120)
    print("📊 요약")
    print("=" * 120)
    print()
    print(f"  현재 최고 성능: Finetuned (α=1.0)")
    print(f"    - Overall: {f_overall:.4f}")
    print(f"    - Valid1:  {f_v1:.4f} (Scratch 대비 {v1_delta:+.4f})")
    print(f"    - Valid2:  {f_v2:.4f} (Scratch 대비 {v2_delta:+.4f})")
    print()
    print(f"  WiSE-FT 탐색 필요 범위: α=0.4~0.9")
    print(f"  예상 최적 균형점: α≈0.7~0.9")
    print()
    print("=" * 120)

if __name__ == '__main__':
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'wiseft_full_results.json'
    analyze_full_results(json_file)
