#!/bin/bash
# Confidence Threshold Optimization
# α=0.0 모델로 다양한 threshold 테스트
# 예상 시간: 10-20분
# 목표: 모델 변경 없이 Valid2 성능 개선

echo "========================================"
echo "Confidence Threshold Optimization"
echo "========================================"
echo ""
echo "목표: α=0.0 모델의 confidence threshold 최적화"
echo "Valid2에서 Recall 개선 가능성 탐색"
echo ""

MODEL="runs/wiseft_parallel/parallel_eval/alpha_0.000.pt"
DATA="new_list/data.yaml"
OUTPUT_BASE="runs/conf_threshold_sweep"

# Confidence threshold 값들
CONF_THRESHOLDS=(0.0001 0.0003 0.0005 0.001 0.003 0.005 0.01)

mkdir -p "$OUTPUT_BASE"

echo "Testing ${#CONF_THRESHOLDS[@]} confidence thresholds..."
echo ""

for CONF in "${CONF_THRESHOLDS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Confidence: $CONF"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python test.py \
        --data "$DATA" \
        --weights "$MODEL" \
        --batch-size 64 \
        --img-size 640 \
        --conf-thres "$CONF" \
        --iou-thres 0.6 \
        --task val \
        --device 0 \
        --save-txt \
        --save-json \
        --name "conf_${CONF}" \
        --project "$OUTPUT_BASE"

    echo ""
done

echo "========================================"
echo "Confidence sweep 완료!"
echo "결과: $OUTPUT_BASE/"
echo "========================================"
echo ""
echo "다음 단계: 각 confidence별 결과 비교"
echo "- Recall 변화 확인"
echo "- Precision vs Recall trade-off"
echo "- Valid2에서 최적 threshold 찾기"
