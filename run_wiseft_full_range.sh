#!/bin/bash
#
# WiSE-FT 전체 범위 탐색 스크립트
# α=0.0~1.0 전체 범위를 한 번에 탐색
#

echo "================================================================================================"
echo "WiSE-FT 전체 범위 탐색: α=0.0~1.0 (step=0.1)"
echo "================================================================================================"
echo ""
echo "목적: 전체 α 범위를 탐색하여 완전한 Trade-off 곡선 그리기"
echo "예상 시간: 25-30분 (A6000 x8 GPU)"
echo ""
echo "탐색할 Alpha 값:"
echo "  0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0"
echo "  (총 11개 α × 2개 validation set = 22개 평가)"
echo ""
echo "================================================================================================"
echo ""

# 실행 명령
python wiseft_sweep_parallel.py \
  --scratch new_list/600.pt \
  --finetuned new_list/620.pt \
  --data new_list/data.yaml \
  --val-sets valid1 valid2 \
  --alpha-min 0.0 \
  --alpha-max 1.0 \
  --alpha-step 0.1 \
  --num-gpus 8 \
  --batch-size 128

EXIT_CODE=$?

echo ""
echo "================================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 전체 범위 탐색 완료!"
else
    echo "❌ 오류 발생 (Exit code: $EXIT_CODE)"
    exit $EXIT_CODE
fi
echo "================================================================================================"
echo ""
echo "📊 결과 분석:"
echo "  1. 결과 파일 확인:"
echo "     cat runs/wiseft_parallel/parallel_eval/results.json"
echo ""
echo "  2. 자동 리포트 생성:"
echo "     python generate_wiseft_report.py --output WISEFT_COMPLETE_REPORT.md"
echo ""
echo "  3. 전체 분석 (Baseline 포함):"
echo "     python analyze_wiseft_full.py"
echo ""
echo "📈 다음 단계:"
echo "  - α=0.0~1.0 결과를 보고 최적 구간 확인"
echo "  - 최적 구간에서 Fine-grained search (step=0.02)"
echo "  - 최종 모델 선정 및 Confidence threshold 최적화"
echo ""
echo "================================================================================================"
