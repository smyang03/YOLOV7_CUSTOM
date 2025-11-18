#!/usr/bin/env python3
"""
WiSE-FT 시간 추정 도구 (numpy 불필요)
"""

def estimate_time():
    """실제 시간 추정"""
    print("="*80)
    print("⏱️  WISEFT TIME ESTIMATION")
    print("="*80)

    # 가정
    EVAL_TIME_PER_SET = 120  # 초 (A6000 기준, 2분)

    scenarios = {
        '현재 방식 (순차, α=0.0~1.0, step=0.1)': {
            'alphas': 11,  # 0.0, 0.1, ..., 1.0
            'valsets': 2,  # valid1, valid2
            'gpus': 1,     # 순차 실행
            'description': '기존 wiseft_sweep.py'
        },
        '최적화 (순차, α=0.0~0.3, step=0.1)': {
            'alphas': 4,   # 0.0, 0.1, 0.2, 0.3
            'valsets': 2,
            'gpus': 1,
            'description': '좁은 알파 범위'
        },
        '병렬 (8 GPU, α=0.0~1.0, step=0.1)': {
            'alphas': 11,
            'valsets': 2,
            'gpus': 8,
            'description': 'wiseft_sweep_parallel.py'
        },
        '병렬 + 최적화 (8 GPU, α=0.0~0.3, step=0.1)': {
            'alphas': 4,
            'valsets': 2,
            'gpus': 8,
            'description': '병렬 + 좁은 범위'
        },
        '병렬 + 최적화 (8 GPU, α=0.0~0.5, step=0.1)': {
            'alphas': 6,
            'valsets': 2,
            'gpus': 8,
            'description': '병렬 + 중간 범위'
        },
    }

    print(f"\n{'전략':<50} {'평가 횟수':<12} {'시간(분)':<12} {'시간(시)':<12}")
    print("─" * 90)

    results = []

    for name, config in scenarios.items():
        total_evals = config['alphas'] * config['valsets']

        if config['gpus'] > 1:
            # 병렬: ceil(total_evals / gpus) batches
            batches = (total_evals + config['gpus'] - 1) // config['gpus']
            time_seconds = batches * EVAL_TIME_PER_SET
        else:
            # 순차
            time_seconds = total_evals * EVAL_TIME_PER_SET

        time_minutes = time_seconds / 60
        time_hours = time_minutes / 60

        results.append({
            'name': name,
            'evals': total_evals,
            'time_minutes': time_minutes,
            'time_hours': time_hours
        })

        print(f"{name:<50} {total_evals:<12} {time_minutes:<12.1f} {time_hours:<12.2f}")

    print("─" * 90)

    # 속도 비교
    baseline = results[0]['time_minutes']
    print(f"\n{'전략':<50} {'속도 향상':<15}")
    print("─" * 90)

    for r in results:
        speedup = baseline / r['time_minutes']
        print(f"{r['name']:<50} {speedup:<15.1f}x")

    print("─" * 90)

    # 현재 문제 진단
    print("\n" + "="*80)
    print("🚨 현재 문제 진단")
    print("="*80)

    print("\n예상 시간:")
    print(f"  순차 (전체 범위):  {results[0]['time_hours']:.2f} 시간 = {results[0]['time_minutes']:.0f} 분")
    print(f"  병렬 (전체 범위):  {results[2]['time_hours']:.2f} 시간 = {results[2]['time_minutes']:.0f} 분")

    print("\n실제 시간:")
    print(f"  현재: 5+ 시간 = 300+ 분 ⚠️")

    print("\n비교:")
    print(f"  예상 대비 실제: {300 / results[0]['time_minutes']:.1f}x 느림!")

    print("\n가능한 원인:")
    print("  1. ✅ 검증 세트가 매우 큼 (이미지 수 × 평가시간)")
    print("  2. ✅ test.py 자체가 느림 (데이터 로딩, 전처리 등)")
    print("  3. ✅ I/O 병목 (디스크 속도)")
    print("  4. ❌ GPU 병렬화 안 됨 (현재 순차 실행)")

    # 추천
    print("\n" + "="*80)
    print("💡 추천 해결책")
    print("="*80)

    print("\n🚀 즉시 적용 가능:")
    print("  1. 알파 범위 좁히기:")
    print("     --alpha-max 0.3")
    print(f"     → {results[1]['time_minutes']:.0f}분 (현재 대비 ~{300/results[1]['time_minutes']:.0f}% 절약)")

    print("\n⚡ 병렬화 (가장 효과적!):")
    print("  2. wiseft_sweep_parallel.py 사용:")
    print("     python wiseft_sweep_parallel.py \\")
    print("       --scratch models/600.pt \\")
    print("       --finetuned models/620.pt \\")
    print("       --data data.yaml \\")
    print("       --val-sets valid1 valid2 \\")
    print("       --alpha-min 0.0 --alpha-max 0.3 \\")
    print("       --num-gpus 8")
    print(f"     → {results[3]['time_minutes']:.0f}분!")

    print("\n📊 검증 세트 크기 확인:")
    print("  3. 이미지 개수 확인:")
    print("     wc -l valid1.txt")
    print("     wc -l valid2.txt")

    print("\n🔍 병목 지점 찾기:")
    print("  4. test.py 프로파일링:")
    print("     time python test.py --weights model.pt --data data.yaml")

    return results


if __name__ == '__main__':
    estimate_time()

    print("\n" + "="*80)
    print("✅ 시간 추정 완료")
    print("="*80)
