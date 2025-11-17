# 밤샘 실행 가이드

밤에 돌려놓을 수 있는 WiSE-FT 명령어 모음

작성일: 2025-11-17

---

## 🚀 빠른 시작

### 스크립트 실행 권한 부여
```bash
chmod +x night_run_commands.sh
```

### 인터랙티브 실행
```bash
./night_run_commands.sh
```

### 직접 명령어 실행
```bash
# 빠른 테스트만
./night_run_commands.sh test

# 전체 테스트
./night_run_commands.sh sweep_test

# 실제 WiSE-FT (모델 필요)
./night_run_commands.sh basic

# 고급 기능 포함
./night_run_commands.sh advanced

# 다양한 실험
./night_run_commands.sh multi

# 빠른 테스트 모음
./night_run_commands.sh quick

# 전부 실행
./night_run_commands.sh all
```

---

## 📋 추천 명령어 (소요 시간별)

### 1. 빠른 테스트 (~1-5분)

```bash
# WiSE-FT 단위 테스트
python test_wiseft_simple.py

# 또는 스크립트로
./night_run_commands.sh test
```

**특징:**
- 모델 파일 불필요
- 핵심 로직 검증
- 빠른 실행
- 개발 중 자주 실행

---

### 2. 전체 테스트 (~5-10분)

```bash
# WiSE-FT Sweep 전체 테스트
python test_wiseft_sweep.py

# 또는 스크립트로
./night_run_commands.sh sweep_test
```

**특징:**
- 모델 파일 불필요
- 전체 워크플로우 시뮬레이션
- Phase 2, 3 기능 포함
- 배포 전 최종 검증

---

### 3. 기본 WiSE-FT Sweep (~30-60분)

**모델 파일 필요!**

```bash
python wiseft_sweep.py \
    --scratch runs/scratch/weights/best.pt \
    --finetuned runs/finetuned/weights/best.pt \
    --data data/custom.yaml \
    --output-dir runs/wiseft/night_basic
```

**특징:**
- 실제 모델로 실행
- 기본 2단계 탐색
- 최적 알파 찾기
- 실용적인 기본 설정

---

### 4. 고급 기능 포함 (~1-2시간)

**권장: 밤에 돌리기 좋음**

```bash
python wiseft_sweep.py \
    --scratch runs/scratch/weights/best.pt \
    --finetuned runs/finetuned/weights/best.pt \
    --data data/custom.yaml \
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
    --output-dir runs/wiseft/night_advanced
```

**특징:**
- Phase 2 모든 기능
- Phase 3 모든 기능
- 상세한 분석 보고서
- 최고 성능 추구

---

### 5. 다양한 실험 (~2-4시간)

**최고 추천: 밤새 돌리기**

```bash
# 실험 1: 낮은 알파 (과적합 의심 시)
python wiseft_sweep.py \
    --scratch runs/scratch/weights/best.pt \
    --finetuned runs/finetuned/weights/best.pt \
    --data data/custom.yaml \
    --alpha-min 0.0 \
    --alpha-max 0.3 \
    --focus-range 0.05 \
    --output-dir runs/wiseft/exp1_low_alpha

# 실험 2: 중간 알파 (일반적)
python wiseft_sweep.py \
    --scratch runs/scratch/weights/best.pt \
    --finetuned runs/finetuned/weights/best.pt \
    --data data/custom.yaml \
    --alpha-min 0.2 \
    --alpha-max 0.6 \
    --focus-range 0.05 \
    --output-dir runs/wiseft/exp2_mid_alpha

# 실험 3: 동적 탐색 (효율적)
python wiseft_sweep.py \
    --scratch runs/scratch/weights/best.pt \
    --finetuned runs/finetuned/weights/best.pt \
    --data data/custom.yaml \
    --enable-dynamic-alpha \
    --enable-layerwise-alpha \
    --output-dir runs/wiseft/exp3_dynamic

# 또는 스크립트로 한 번에
./night_run_commands.sh multi
```

**특징:**
- 여러 설정 비교
- 최적 설정 찾기
- 종합적인 분석
- 논문/보고서용 데이터

---

## 🎯 상황별 추천

### 상황 1: 처음 사용하는 경우
```bash
# 1단계: 테스트 먼저
python test_wiseft_simple.py

# 2단계: 기본 실행
python wiseft_sweep.py \
    --scratch <your_model>.pt \
    --finetuned <your_model>.pt \
    --data <your_data>.yaml
```

### 상황 2: 과적합 의심 (타겟 성능 ↑, 다른 클래스 ↓)
```bash
python wiseft_sweep.py \
    --scratch runs/scratch/weights/best.pt \
    --finetuned runs/finetuned/weights/best.pt \
    --data data/custom.yaml \
    --alpha-min 0.0 \
    --alpha-max 0.3 \
    --target-class person \
    --enable-tradeoff-viz
```

### 상황 3: 최고 성능 필요 (연구/논문)
```bash
python wiseft_sweep.py \
    --scratch runs/scratch/weights/best.pt \
    --finetuned runs/finetuned/weights/best.pt \
    --data data/custom.yaml \
    --focus-range 0.025 \
    --enable-confidence-intervals \
    --confidence-runs 5 \
    --enable-layerwise-alpha \
    --enable-ensemble \
    --ensemble-top-k 5
```

### 상황 4: 시간 절약 (빠른 결과)
```bash
python wiseft_sweep.py \
    --scratch runs/scratch/weights/best.pt \
    --finetuned runs/finetuned/weights/best.pt \
    --data data/custom.yaml \
    --enable-dynamic-alpha \
    --enable-adaptive-stop
```

---

## 📊 백그라운드 실행

### nohup 사용 (추천)
```bash
nohup ./night_run_commands.sh advanced > night_run.log 2>&1 &

# 로그 확인
tail -f night_run.log

# 프로세스 확인
ps aux | grep wiseft
```

### screen 사용
```bash
# screen 세션 시작
screen -S wiseft

# 명령 실행
./night_run_commands.sh advanced

# Ctrl+A, D로 detach
# 나중에 재연결: screen -r wiseft
```

### tmux 사용
```bash
# tmux 세션 시작
tmux new -s wiseft

# 명령 실행
./night_run_commands.sh advanced

# Ctrl+B, D로 detach
# 나중에 재연결: tmux attach -t wiseft
```

---

## 📁 결과 확인

### 로그 파일
```bash
# 생성된 로그 확인
ls -lh logs/

# 최신 로그 보기
cat logs/$(ls -t logs/ | head -1)/wiseft_advanced.log
```

### 결과 파일
```bash
# WiSE-FT 결과
ls -lh runs/wiseft/

# 보고서 보기
cat runs/wiseft/night_run_advanced/wiseft_report.md

# JSON 결과
cat runs/wiseft/night_run_advanced/results.json | python -m json.tool
```

### 최고 모델
```bash
# 최고 모델 확인
ls -lh runs/wiseft/*/best_merged*.pt
```

---

## ⚠️ 주의사항

### 1. 디스크 공간 확인
```bash
df -h .
```

### 2. GPU 메모리 확인
```bash
nvidia-smi
```

### 3. 배치 크기 조정
메모리 부족 시:
```bash
python wiseft_sweep.py \
    ... \
    --batch-size 16  # 기본 32에서 줄임
```

### 4. 조기 종료 설정
시간 절약:
```bash
python wiseft_sweep.py \
    ... \
    --enable-adaptive-stop \
    --early-stop \
    --stop-patience 3
```

---

## 🔍 문제 해결

### CUDA out of memory
```bash
# 배치 크기 줄이기
--batch-size 16

# 또는 이미지 크기 줄이기
--img-size 512
```

### 너무 오래 걸림
```bash
# 동적 탐색 사용
--enable-dynamic-alpha

# 조기 종료 활성화
--enable-adaptive-stop

# 알파 범위 제한
--alpha-min 0.1 --alpha-max 0.4
```

### 디스크 공간 부족
```bash
# 최고 모델만 저장
--save-best-only

# 병합 모델 저장 안 함
# (기본적으로 비활성화됨)
```

---

## 📈 예상 소요 시간

| 작업 | 소요 시간 | 권장 시간대 |
|------|-----------|-------------|
| 단위 테스트 | 1분 | 언제나 |
| 전체 테스트 | 5-10분 | 언제나 |
| 기본 Sweep | 30-60분 | 점심/저녁 |
| 고급 Sweep | 1-2시간 | 밤 |
| 다중 실험 | 2-4시간 | 밤새 |

---

## ✅ 체크리스트

밤에 실행하기 전:

- [ ] 모델 파일 경로 확인
- [ ] data.yaml 파일 확인
- [ ] 디스크 공간 충분한지 확인
- [ ] GPU 메모리 확인
- [ ] 로그 디렉토리 생성 확인
- [ ] 백그라운드 실행 설정 (nohup/screen/tmux)
- [ ] 예상 소요 시간 확인
- [ ] 알람 설정 (선택)

---

## 🎉 아침에 확인할 것

```bash
# 1. 프로세스 완료 확인
ps aux | grep wiseft

# 2. 로그 확인
tail -100 night_run.log

# 3. 결과 확인
cat runs/wiseft/*/wiseft_report.md

# 4. 최고 알파 확인
grep "RECOMMENDED ALPHA" runs/wiseft/*/wiseft_report.md

# 5. 최고 모델 확인
ls -lh runs/wiseft/*/best_merged*.pt
```

---

**Happy Night Running! 🌙**

---

## 추가 자료

- [WISEFT_README_KR.md](WISEFT_README_KR.md) - 상세 사용 가이드
- [PHASE2_PHASE3_FEATURES_KR.md](PHASE2_PHASE3_FEATURES_KR.md) - 고급 기능
- [DATA_YAML_GUIDE_KR.md](DATA_YAML_GUIDE_KR.md) - 데이터 설정
