#!/bin/bash
#
# WiSE-FT 결과 분석 실행 스크립트
#

echo "================================================================"
echo "WiSE-FT 결과 분석 실행"
echo "================================================================"
echo ""

# 현재 디렉토리 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "작업 디렉토리: $(pwd)"
echo ""

# 1. 상세 분석 실행
echo "1️⃣ 상세 분석 실행 중..."
echo "================================================================"
python analyze_results_detailed.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ 분석 실행 실패 (Exit code: $EXIT_CODE)"
    exit $EXIT_CODE
fi

echo ""
echo "================================================================"
echo "✅ 분석 완료!"
echo "================================================================"
echo ""

# 2. 리포트 생성 (선택)
echo "2️⃣ 리포트 생성 (wiseft_full_results.json 사용)"
echo "================================================================"
if [ -f "wiseft_full_results.json" ]; then
    python generate_wiseft_report.py \
        --results wiseft_full_results.json \
        --output WISEFT_COMPREHENSIVE_REPORT.md
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 리포트 생성 완료: WISEFT_COMPREHENSIVE_REPORT.md"
    else
        echo ""
        echo "⚠️  리포트 생성 실패 (무시하고 계속)"
    fi
else
    echo "⚠️  wiseft_full_results.json 파일이 없습니다. 리포트 생성 생략."
fi

echo ""
echo "================================================================"
echo "📄 생성된 문서 확인:"
echo "================================================================"
echo "  - COMPREHENSIVE_ANALYSIS_SUMMARY.md    (종합 분석 요약)"
echo "  - TRADEOFF_EXPLANATION.md              (116% 향상 설명)"
echo "  - WISEFT_FULL_ANALYSIS_REPORT.md       (전체 분석 리포트)"
echo ""
echo "================================================================"
echo "🚀 다음 단계:"
echo "================================================================"
echo "  # α=0.4~1.0 확장 범위 탐색"
echo "  ./run_wiseft_expanded.sh"
echo ""
echo "  # 또는 α=0.0~1.0 전체 범위 탐색"
echo "  ./run_wiseft_full_range.sh"
echo ""
echo "================================================================"
