#!/bin/bash
# WiSE-FT 진행 상황 확인 스크립트

echo "=========================================="
echo "WiSE-FT 진행 상황 확인"
echo "=========================================="

# 1. 프로세스 확인
echo ""
echo "1. 실행 중인 프로세스:"
ps aux | grep -E "wiseft|test.py|python" | grep -v grep

# 2. GPU 사용 확인
echo ""
echo "2. GPU 사용 상황:"
nvidia-smi

# 3. 최근 로그 확인
echo ""
echo "3. 최근 로그 (temp 디렉토리):"
if [ -d "runs/wiseft/exp2/temp" ]; then
    ls -lht runs/wiseft/exp2/temp/*.log 2>/dev/null | head -5
    echo ""
    echo "최신 로그 내용:"
    tail -20 runs/wiseft/exp2/temp/*.log 2>/dev/null | head -50
fi

# 4. 생성된 파일 확인
echo ""
echo "4. 생성된 임시 모델 파일:"
ls -lht runs/wiseft/exp2/temp/*.pt 2>/dev/null | head -10

# 5. test.py 프로세스 확인
echo ""
echo "5. test.py 실행 중인지 확인:"
ps aux | grep test.py | grep -v grep
