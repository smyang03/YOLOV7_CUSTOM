#!/bin/bash
#
# WiSE-FT 확장 탐색 스크립트
# α=0.4~1.0 범위를 탐색하여 최적 균형점 찾기
#

echo "================================================================================================"
echo "WiSE-FT 확장 탐색: α=0.4~1.0"
echo "================================================================================================"
echo ""
echo "목적: α=0.4~0.9 범위에서 최적 균형점 찾기"
echo "예상 시간: 15-20분 (A6000 x8 GPU)"
echo ""
echo "현재까지 발견:"
echo "  - α=0.0 (Scratch):   Overall 0.5271, Valid1 0.6669, Valid2 0.3873"
echo "  - α=0.1~0.3:         성능 하락 (비효율적)"
echo "  - α=1.0 (Finetuned): Overall 0.7384, Valid1 0.6375, Valid2 0.8392 ⭐ 최고"
echo ""
echo "탐색할 범위: α=0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0"
echo "================================================================================================"
echo ""

# 실행 명령
python wiseft_sweep_parallel.py \
  --scratch new_list/600.pt \
  --finetuned new_list/620.pt \
  --data new_list/data.yaml \
  --val-sets valid1 valid2 \
  --alpha-min 0.4 \
  --alpha-max 1.0 \
  --alpha-step 0.1 \
  --num-gpus 8 \
  --batch-size 128

echo ""
echo "================================================================================================"
echo "탐색 완료!"
echo "================================================================================================"
echo ""
echo "다음 단계:"
echo "  1. 결과 확인: cat runs/wiseft_parallel/parallel_eval/results.json"
echo "  2. 리포트 생성: python generate_wiseft_report.py"
echo "  3. 전체 분석: python analyze_wiseft_full.py"
echo ""
echo "예상 결과:"
echo "  - α=0.7~0.9에서 최적 균형점 발견 가능"
echo "  - Valid1: 0.63~0.65 (약간 하락)"
echo "  - Valid2: 0.65~0.75 (크게 개선)"
echo "  - Overall: 0.64~0.70 (개선)"
echo ""
echo "================================================================================================"
