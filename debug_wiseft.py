#!/usr/bin/env python3
"""
WiSE-FT 디버깅 스크립트
"""
import torch
from pathlib import Path
import sys

def analyze_model(model_path):
    """모델 파일 분석"""
    print(f"\n{'='*80}")
    print(f"분석: {model_path}")
    print('='*80)

    if not Path(model_path).exists():
        print(f"❌ 파일이 존재하지 않음!")
        return None

    try:
        ckpt = torch.load(model_path, map_location='cpu')
        print(f"✅ 로딩 성공")

        # 키 확인
        print(f"\n체크포인트 키: {list(ckpt.keys())[:10]}")

        if 'model' in ckpt:
            state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
            print(f"모델 파라미터 개수: {len(state_dict)}")

            # 몇 개 샘플 출력
            print(f"\n첫 3개 레이어:")
            for i, (key, val) in enumerate(list(state_dict.items())[:3]):
                print(f"  {key}: shape={val.shape}, mean={val.float().mean():.6f}, std={val.float().std():.6f}")

            # Epoch 정보
            if 'epoch' in ckpt:
                print(f"\nEpoch: {ckpt['epoch']}")

            # Best fitness
            if 'best_fitness' in ckpt:
                print(f"Best Fitness: {ckpt['best_fitness']}")

            return state_dict
        else:
            print(f"⚠️  'model' 키가 없음")
            return None

    except Exception as e:
        print(f"❌ 에러: {e}")
        return None

def compare_models(sd1, sd2, name1, name2):
    """두 모델의 state_dict 비교"""
    print(f"\n{'='*80}")
    print(f"모델 비교: {name1} vs {name2}")
    print('='*80)

    if sd1 is None or sd2 is None:
        print("❌ 비교 불가 (모델 로딩 실패)")
        return

    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    print(f"\n{name1} 파라미터 개수: {len(keys1)}")
    print(f"{name2} 파라미터 개수: {len(keys2)}")

    if keys1 != keys2:
        print(f"\n⚠️  파라미터 키가 다름!")
        print(f"  {name1}에만 있음: {keys1 - keys2}")
        print(f"  {name2}에만 있음: {keys2 - keys1}")
    else:
        print(f"✅ 파라미터 키 동일")

        # 값 비교
        identical = 0
        different = 0

        for key in keys1:
            if torch.equal(sd1[key], sd2[key]):
                identical += 1
            else:
                different += 1

        print(f"\n동일한 파라미터: {identical}")
        print(f"다른 파라미터: {different}")

        if different > 0:
            print(f"\n차이나는 파라미터 샘플 (처음 3개):")
            count = 0
            for key in keys1:
                if not torch.equal(sd1[key], sd2[key]):
                    diff = (sd1[key] - sd2[key]).abs().max().item()
                    print(f"  {key}: max_diff={diff:.6f}")
                    count += 1
                    if count >= 3:
                        break

def verify_merge(scratch_sd, finetuned_sd, merged_sd, alpha, name):
    """가중치 병합이 올바른지 검증"""
    print(f"\n{'='*80}")
    print(f"병합 검증: {name} (alpha={alpha})")
    print('='*80)

    if scratch_sd is None or finetuned_sd is None or merged_sd is None:
        print("❌ 검증 불가")
        return

    errors = 0
    correct = 0

    for key in scratch_sd.keys():
        if key not in finetuned_sd or key not in merged_sd:
            continue

        expected = (1 - alpha) * scratch_sd[key] + alpha * finetuned_sd[key]
        actual = merged_sd[key]

        if torch.allclose(expected, actual, rtol=1e-5, atol=1e-6):
            correct += 1
        else:
            errors += 1
            if errors <= 3:  # 처음 3개만 출력
                max_diff = (expected - actual).abs().max().item()
                print(f"  ❌ {key}: max_diff={max_diff:.6e}")

    print(f"\n올바른 병합: {correct}/{correct+errors}")
    print(f"에러: {errors}/{correct+errors}")

    if errors == 0:
        print(f"✅ 병합 완벽!")
    else:
        print(f"⚠️  병합에 문제가 있음!")

if __name__ == '__main__':
    # 모델 분석
    scratch_sd = analyze_model('new_list/600.pt')
    finetuned_sd = analyze_model('new_list/620.pt')

    # 모델 비교
    compare_models(scratch_sd, finetuned_sd, '600.pt', '620.pt')

    # 병합된 모델들 검증
    alpha_0_sd = analyze_model('runs/wiseft_parallel/parallel_eval/alpha_0.000.pt')
    verify_merge(scratch_sd, finetuned_sd, alpha_0_sd, 0.0, 'alpha_0.000.pt')

    alpha_1_sd = analyze_model('runs/wiseft_parallel/parallel_eval/alpha_0.100.pt')
    verify_merge(scratch_sd, finetuned_sd, alpha_1_sd, 0.1, 'alpha_0.100.pt')

    alpha_2_sd = analyze_model('runs/wiseft_parallel/parallel_eval/alpha_0.200.pt')
    verify_merge(scratch_sd, finetuned_sd, alpha_2_sd, 0.2, 'alpha_0.200.pt')

    print(f"\n{'='*80}")
    print("분석 완료")
    print('='*80)
