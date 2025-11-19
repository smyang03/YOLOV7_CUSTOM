#!/bin/bash
# Fine-grained Alpha Search: 0.0~0.15 구간 정밀 탐색
# 예상 시간: 30-60분
# 예상 효과: Valid2 +0.5~1.5%, Valid1 손실 최소화

echo "=================================="
echo "Fine-grained WiSE-FT Alpha Search"
echo "=================================="
echo ""
echo "목표: α=0.0~0.15 구간에서 최적 균형점 찾기"
echo "Alpha 값: 0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14"
echo "예상 시간: ~30-60분 (GPU 8개)"
echo ""

python wiseft_sweep_parallel.py \
    --scratch new_list/600.pt \
    --finetuned new_list/620.pt \
    --data new_list/data.yaml \
    --val-sets valid1 valid2 \
    --alpha-min 0.0 \
    --alpha-max 0.15 \
    --focus-range 0.02 \
    --num-gpus 8 \
    --batch-size 64 \
    --output-dir runs/wiseft_fine_grained

echo ""
echo "=================================="
echo "Fine-grained search 완료!"
echo "결과 확인: runs/wiseft_fine_grained/results.json"
echo "=================================="
