#!/bin/bash
# 밤에 돌려놓을 명령어 모음
# 작성일: 2025-11-17

echo "=========================================="
echo "밤샘 실행 명령어 스크립트"
echo "=========================================="

# 로그 디렉토리 생성
mkdir -p logs
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "로그 저장 위치: $LOG_DIR"

# ==========================================
# 옵션 1: WiSE-FT 테스트 실행
# ==========================================
run_wiseft_test() {
    echo "=========================================="
    echo "옵션 1: WiSE-FT 테스트 실행"
    echo "=========================================="

    python test_wiseft_simple.py 2>&1 | tee "$LOG_DIR/wiseft_test.log"

    if [ $? -eq 0 ]; then
        echo "✅ WiSE-FT 테스트 완료"
    else
        echo "❌ WiSE-FT 테스트 실패"
    fi
}

# ==========================================
# 옵션 2: WiSE-FT Sweep 테스트 (시뮬레이션)
# ==========================================
run_wiseft_sweep_test() {
    echo "=========================================="
    echo "옵션 2: WiSE-FT Sweep 전체 테스트"
    echo "=========================================="

    python test_wiseft_sweep.py 2>&1 | tee "$LOG_DIR/wiseft_sweep_test.log"

    if [ $? -eq 0 ]; then
        echo "✅ WiSE-FT Sweep 테스트 완료"
    else
        echo "❌ WiSE-FT Sweep 테스트 실패"
    fi
}

# ==========================================
# 옵션 3: 실제 WiSE-FT Sweep 실행 (모델 있을 때)
# ==========================================
run_wiseft_sweep_real() {
    echo "=========================================="
    echo "옵션 3: 실제 WiSE-FT Sweep 실행"
    echo "=========================================="

    # 모델 파일 경로 확인 (실제 경로로 수정 필요)
    SCRATCH_MODEL="runs/scratch/weights/best.pt"
    FINETUNED_MODEL="runs/finetuned/weights/best.pt"
    DATA_YAML="data/custom.yaml"

    if [ ! -f "$SCRATCH_MODEL" ]; then
        echo "⚠️  스크래치 모델 없음: $SCRATCH_MODEL"
        echo "모델 경로를 확인하고 스크립트를 수정하세요"
        return 1
    fi

    if [ ! -f "$FINETUNED_MODEL" ]; then
        echo "⚠️  미세조정 모델 없음: $FINETUNED_MODEL"
        echo "모델 경로를 확인하고 스크립트를 수정하세요"
        return 1
    fi

    # 기본 실행
    echo "기본 WiSE-FT Sweep 실행..."
    python wiseft_sweep.py \
        --scratch "$SCRATCH_MODEL" \
        --finetuned "$FINETUNED_MODEL" \
        --data "$DATA_YAML" \
        --output-dir "runs/wiseft/night_run_basic" \
        2>&1 | tee "$LOG_DIR/wiseft_basic.log"

    echo "✅ 기본 실행 완료"
}

# ==========================================
# 옵션 4: 고급 기능 포함 WiSE-FT Sweep
# ==========================================
run_wiseft_advanced() {
    echo "=========================================="
    echo "옵션 4: 고급 기능 포함 WiSE-FT Sweep"
    echo "=========================================="

    SCRATCH_MODEL="runs/scratch/weights/best.pt"
    FINETUNED_MODEL="runs/finetuned/weights/best.pt"
    DATA_YAML="data/custom.yaml"

    if [ ! -f "$SCRATCH_MODEL" ] || [ ! -f "$FINETUNED_MODEL" ]; then
        echo "⚠️  모델 파일이 없습니다"
        return 1
    fi

    # Phase 2 & 3 기능 모두 활성화
    echo "고급 기능 포함 WiSE-FT Sweep 실행..."
    python wiseft_sweep.py \
        --scratch "$SCRATCH_MODEL" \
        --finetuned "$FINETUNED_MODEL" \
        --data "$DATA_YAML" \
        --target-class person \
        --focus-range 0.05 \
        --enable-tradeoff-viz \
        --enable-adaptive-stop \
        --enable-layer-detail \
        --enable-confidence-intervals \
        --confidence-runs 3 \
        --enable-layerwise-alpha \
        --enable-dynamic-alpha \
        --enable-ensemble \
        --ensemble-top-k 5 \
        --output-dir "runs/wiseft/night_run_advanced" \
        2>&1 | tee "$LOG_DIR/wiseft_advanced.log"

    echo "✅ 고급 실행 완료"
}

# ==========================================
# 옵션 5: 다양한 알파 범위로 여러 실험
# ==========================================
run_multiple_experiments() {
    echo "=========================================="
    echo "옵션 5: 다양한 설정으로 여러 실험"
    echo "=========================================="

    SCRATCH_MODEL="runs/scratch/weights/best.pt"
    FINETUNED_MODEL="runs/finetuned/weights/best.pt"
    DATA_YAML="data/custom.yaml"

    if [ ! -f "$SCRATCH_MODEL" ] || [ ! -f "$FINETUNED_MODEL" ]; then
        echo "⚠️  모델 파일이 없습니다"
        return 1
    fi

    # 실험 1: 낮은 알파 범위
    echo "실험 1: 낮은 알파 범위 (0.0-0.3)"
    python wiseft_sweep.py \
        --scratch "$SCRATCH_MODEL" \
        --finetuned "$FINETUNED_MODEL" \
        --data "$DATA_YAML" \
        --alpha-min 0.0 \
        --alpha-max 0.3 \
        --focus-range 0.05 \
        --output-dir "runs/wiseft/night_low_alpha" \
        2>&1 | tee "$LOG_DIR/exp1_low_alpha.log"

    # 실험 2: 중간 알파 범위
    echo "실험 2: 중간 알파 범위 (0.2-0.6)"
    python wiseft_sweep.py \
        --scratch "$SCRATCH_MODEL" \
        --finetuned "$FINETUNED_MODEL" \
        --data "$DATA_YAML" \
        --alpha-min 0.2 \
        --alpha-max 0.6 \
        --focus-range 0.05 \
        --output-dir "runs/wiseft/night_mid_alpha" \
        2>&1 | tee "$LOG_DIR/exp2_mid_alpha.log"

    # 실험 3: 전체 범위 + 동적 탐색
    echo "실험 3: 전체 범위 + 동적 알파"
    python wiseft_sweep.py \
        --scratch "$SCRATCH_MODEL" \
        --finetuned "$FINETUNED_MODEL" \
        --data "$DATA_YAML" \
        --alpha-min 0.0 \
        --alpha-max 1.0 \
        --enable-dynamic-alpha \
        --output-dir "runs/wiseft/night_dynamic" \
        2>&1 | tee "$LOG_DIR/exp3_dynamic.log"

    echo "✅ 모든 실험 완료"
}

# ==========================================
# 옵션 6: 테스트만 실행 (빠른 검증)
# ==========================================
run_quick_tests() {
    echo "=========================================="
    echo "옵션 6: 빠른 테스트 모음"
    echo "=========================================="

    echo "1. WiSE-FT 단위 테스트..."
    python test_wiseft_simple.py 2>&1 | tee "$LOG_DIR/unit_test.log"

    echo ""
    echo "2. WiSE-FT Sweep 테스트..."
    if [ -f "test_wiseft_sweep.py" ]; then
        python test_wiseft_sweep.py 2>&1 | tee "$LOG_DIR/sweep_test.log"
    else
        echo "⚠️  test_wiseft_sweep.py 파일 없음"
    fi

    echo "✅ 모든 테스트 완료"
}

# ==========================================
# 메인 메뉴
# ==========================================
main_menu() {
    echo ""
    echo "=========================================="
    echo "실행할 옵션을 선택하세요:"
    echo "=========================================="
    echo "1. WiSE-FT 테스트 실행 (가장 빠름, ~1분)"
    echo "2. WiSE-FT Sweep 전체 테스트 (~5-10분)"
    echo "3. 실제 WiSE-FT Sweep - 기본 (모델 필요, ~30-60분)"
    echo "4. 실제 WiSE-FT Sweep - 고급 기능 포함 (모델 필요, ~1-2시간)"
    echo "5. 다양한 설정으로 여러 실험 (모델 필요, ~2-4시간)"
    echo "6. 빠른 테스트 모음 (~5분)"
    echo "7. 전체 자동 실행 (테스트 + 가능한 모든 실험)"
    echo ""
    read -p "선택 (1-7): " choice

    case $choice in
        1)
            run_wiseft_test
            ;;
        2)
            run_wiseft_sweep_test
            ;;
        3)
            run_wiseft_sweep_real
            ;;
        4)
            run_wiseft_advanced
            ;;
        5)
            run_multiple_experiments
            ;;
        6)
            run_quick_tests
            ;;
        7)
            echo "전체 자동 실행 시작..."
            run_wiseft_test
            run_wiseft_sweep_test
            # 모델이 있으면 실제 실험도 실행
            if [ -f "runs/scratch/weights/best.pt" ]; then
                run_wiseft_sweep_real
                run_wiseft_advanced
            fi
            ;;
        *)
            echo "❌ 잘못된 선택입니다"
            exit 1
            ;;
    esac
}

# ==========================================
# 스크립트 실행
# ==========================================

# 인터랙티브 모드
if [ "$1" == "" ]; then
    main_menu
else
    # 커맨드라인 인자로 직접 실행
    case $1 in
        test)
            run_wiseft_test
            ;;
        sweep_test)
            run_wiseft_sweep_test
            ;;
        basic)
            run_wiseft_sweep_real
            ;;
        advanced)
            run_wiseft_advanced
            ;;
        multi)
            run_multiple_experiments
            ;;
        quick)
            run_quick_tests
            ;;
        all)
            run_wiseft_test
            run_wiseft_sweep_test
            if [ -f "runs/scratch/weights/best.pt" ]; then
                run_wiseft_sweep_real
            fi
            ;;
        *)
            echo "사용법: $0 [test|sweep_test|basic|advanced|multi|quick|all]"
            exit 1
            ;;
    esac
fi

echo ""
echo "=========================================="
echo "실행 완료!"
echo "로그 위치: $LOG_DIR"
echo "=========================================="
