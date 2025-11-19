# WiSE-FT 분석 및 실행 가이드

## 📁 현재 위치 확인

먼저 올바른 디렉토리에 있는지 확인하세요:

```bash
cd /workspace/datasets/YOLOV7_CUSTOM
# 또는 로컬에서
cd /home/user/YOLOV7_CUSTOM
```

## ✅ 파일 존재 확인

```bash
ls -lh analyze_results_detailed.py run_wiseft_expanded.sh
```

출력 예상:
```
-rwxr-xr-x 1 root root  11K Nov 19 09:58 analyze_results_detailed.py
-rwxr-xr-x 1 root root 2.0K Nov 19 08:48 run_wiseft_expanded.sh
```

---

## 🔍 1. 현재 결과 분석

### 방법 1: 상세 분석 스크립트 실행

```bash
python analyze_results_detailed.py
```

또는 직접 실행:

```bash
./analyze_results_detailed.py
```

### 방법 2: 통합 분석 스크립트 실행

```bash
./run_analysis.sh
```

이 스크립트는 자동으로:
- 상세 분석 실행
- 리포트 생성
- 다음 단계 안내

---

## 🚀 2. 확장 범위 탐색 (α=0.4~1.0)

### 실행 방법:

```bash
./run_wiseft_expanded.sh
```

또는:

```bash
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
```

**예상 시간:** 15-20분
**탐색 Alpha:** 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

---

## 🔄 3. 전체 범위 재탐색 (α=0.0~1.0)

### 실행 방법:

```bash
./run_wiseft_full_range.sh
```

또는:

```bash
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
```

**예상 시간:** 25-30분
**탐색 Alpha:** 0.0, 0.1, 0.2, ..., 1.0 (총 11개)

---

## 📊 4. 결과 리포트 생성

### 자동 검색 (권장):

```bash
python generate_wiseft_report.py --output MY_REPORT.md
```

자동으로 runs/ 디렉토리에서 최신 results.json 찾음

### 특정 파일 지정:

```bash
python generate_wiseft_report.py \
  --results wiseft_full_results.json \
  --output COMPLETE_REPORT.md
```

### 콘솔 출력만:

```bash
python generate_wiseft_report.py --print-only
```

---

## ❌ 문제 해결

### 1. "파일이 없다" 오류

**증상:**
```
bash: ./run_wiseft_expanded.sh: No such file or directory
```

**해결:**
```bash
# 1. 올바른 디렉토리 확인
pwd
# 출력: /workspace/datasets/YOLOV7_CUSTOM

# 2. 디렉토리 이동
cd /workspace/datasets/YOLOV7_CUSTOM

# 3. 파일 확인
ls -la run_wiseft_expanded.sh

# 4. 실행 권한 확인
chmod +x run_wiseft_expanded.sh

# 5. 다시 실행
./run_wiseft_expanded.sh
```

### 2. "권한 없음" 오류

**증상:**
```
bash: ./run_wiseft_expanded.sh: Permission denied
```

**해결:**
```bash
chmod +x run_wiseft_expanded.sh
chmod +x run_wiseft_full_range.sh
chmod +x analyze_results_detailed.py
chmod +x run_analysis.sh
```

### 3. Python 모듈 없음

**증상:**
```
ModuleNotFoundError: No module named 'yaml'
```

**해결:**
```bash
pip install pyyaml
```

---

## 📄 생성된 문서 목록

현재 디렉토리에서 확인 가능:

```bash
ls -lh *.md
```

주요 문서:
1. **COMPREHENSIVE_ANALYSIS_SUMMARY.md** - 종합 분석 요약
2. **TRADEOFF_EXPLANATION.md** - 116% 향상 상세 설명
3. **WISEFT_FULL_ANALYSIS_REPORT.md** - 전체 분석 리포트
4. **WISEFT_ANALYSIS_REPORT.md** - WiSE-FT 분석 리포트

---

## 🎯 빠른 시작 (Quick Start)

```bash
# 1. 디렉토리 이동
cd /workspace/datasets/YOLOV7_CUSTOM

# 2. 현재 결과 분석
./run_analysis.sh

# 3. 확장 범위 탐색 (권장)
./run_wiseft_expanded.sh

# 4. 결과 확인
cat runs/wiseft_parallel/parallel_eval/results.json

# 5. 리포트 생성
python generate_wiseft_report.py --output FINAL_REPORT.md
```

---

## 💡 현재 상황 요약

### 이미 평가 완료:
- ✅ α=0.0 (Scratch): Overall 0.5271
- ✅ α=0.1: Overall 0.5183
- ✅ α=0.2: Overall 0.4234
- ✅ α=0.3: Overall 0.2903
- ✅ α=1.0 (Finetuned): Overall 0.7384 ⭐ 최고

### 미탐색 구간:
- ❓ α=0.4
- ❓ α=0.5
- ❓ α=0.6
- ❓ α=0.7
- ❓ α=0.8
- ❓ α=0.9

### 다음 단계:
```bash
./run_wiseft_expanded.sh  # 이것을 실행하세요!
```

---

## 📞 추가 도움말

### 모든 스크립트 확인:

```bash
ls -lh *.sh
```

### 모든 Python 분석 도구 확인:

```bash
ls -lh analyze*.py
```

### 결과 파일 위치:

```bash
find runs -name "results.json" -type f
```

---

**실행 순서 요약:**
1. `cd /workspace/datasets/YOLOV7_CUSTOM`
2. `./run_wiseft_expanded.sh` (α=0.4~1.0 탐색)
3. `python generate_wiseft_report.py --output FINAL_REPORT.md`
