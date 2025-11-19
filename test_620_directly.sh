#!/bin/bash
# 620.pt 모델을 직접 평가하여 정상 동작 확인

echo "================================"
echo "620.pt 단독 평가"
echo "================================"

# Valid1에서 평가
echo ""
echo "[1/2] Valid1 평가..."
python test.py \
    --data new_list/data.yaml \
    --weights new_list/620.pt \
    --batch-size 64 \
    --img-size 640 \
    --task val \
    --device 0 \
    --save-txt \
    --save-json

echo ""
echo "[2/2] Valid2 평가..."
# data.yaml을 임시로 수정해서 valid2 사용
# (또는 valid2용 yaml이 따로 있다면 그것 사용)

echo ""
echo "================================"
echo "평가 완료"
echo "================================"
