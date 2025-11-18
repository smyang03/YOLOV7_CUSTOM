# WiSE-FT 성능 최적화 가이드

## 🐌 속도 문제

### 문제: 다중 검증 세트 평가가 느림

**증상:**
```
기존 방식 (valid1+valid2 합쳐서):
  - 1회 평가 × 12 alphas = 12회 평가
  - 예상 시간: ~20분

새 방식 (valid1, valid2 개별):
  - 2회 평가 × 12 alphas = 24회 평가
  - 예상 시간: ~40분 (2배!)
```

**왜 느려지나?**
- 각 검증 세트를 **개별적으로** 평가해야 트레이드오프 분석 가능
- valid1과 valid2 성능을 따로 알아야 균형점 찾기 가능
- 이것은 **필요한 비용**입니다!

---

## ⚡ 해결 방법

### 1️⃣ 알파 범위 좁히기 (가장 효과적!)

**기본값:**
```bash
python wiseft_sweep.py \
    --alpha-min 0.0 \
    --alpha-max 1.0 \
    --focus-range 0.1
# → 11개 알파 (0.0, 0.1, ..., 1.0)
```

**최적화:**
```bash
python wiseft_sweep.py \
    --alpha-min 0.0 \
    --alpha-max 0.5 \  # ← 범위 절반으로!
    --focus-range 0.1
# → 6개 알파 (0.0, 0.1, ..., 0.5)
# 시간 50% 절약!
```

**근거:**
- 보고서에서 α=0.3 이상부터 성능 급락
- α=0.5 이상은 볼 필요 없음
- **추천: --alpha-max 0.3 또는 0.5**

### 2️⃣ Focus Range 증가

**기본값:**
```bash
--focus-range 0.1
# → [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
```

**최적화:**
```bash
--focus-range 0.2
# → [0.0, 0.2, 0.4]
# 시간 50% 절약!
```

**주의:**
- 너무 넓으면 최적 알파를 놓칠 수 있음
- Fine search가 보완해줌

### 3️⃣ Fine Search 비활성화 (빠른 탐색)

```bash
python wiseft_sweep.py \
    --enable-fine-search false \  # ← Fine search 끄기
    --focus-range 0.1
# 시간 ~30% 절약
```

**언제 사용?**
- 빠른 예비 탐색
- 대략적인 트레이드오프만 보고 싶을 때

### 4️⃣ 배치 사이즈 증가 (GPU 메모리 충분 시)

```bash
python wiseft_sweep.py \
    --batch-size 64  # 기본 32 → 64
# 평가 속도 ~30% 증가
```

**필요 조건:**
- 16GB+ VRAM
- 메모리 부족 시 OOM 에러

### 5️⃣ 이미지 크기 감소 (정확도 약간 희생)

```bash
python wiseft_sweep.py \
    --img-size 512  # 기본 640 → 512
# 평가 속도 ~40% 증가
```

**주의:**
- 정확도가 약간 떨어질 수 있음
- 최종 평가는 640으로 다시 해야 함

---

## 🎯 추천 전략

### 🚀 빠른 탐색 (5~10분)

```bash
python wiseft_sweep.py \
    --scratch new_list/600.pt \
    --finetuned new_list/620.pt \
    --data new_list/data.yaml \
    --val-sets valid1 valid2 \
    --alpha-min 0.0 \
    --alpha-max 0.3 \  # 좁은 범위
    --focus-range 0.15 \  # 넓은 간격
    --enable-fine-search false \  # Fine search 끄기
    --batch-size 64 \  # 큰 배치
    --enable-tradeoff-viz
```

**결과:**
- 알파: [0.0, 0.15, 0.30] = **3개**
- 평가 횟수: 3 × 2 = **6회**
- 예상 시간: **~5분**
- 트레이드오프 파악 가능!

### ⚖️ 균형잡힌 탐색 (15~20분)

```bash
python wiseft_sweep.py \
    --scratch new_list/600.pt \
    --finetuned new_list/620.pt \
    --data new_list/data.yaml \
    --val-sets valid1 valid2 \
    --alpha-min 0.0 \
    --alpha-max 0.5 \  # 적당한 범위
    --focus-range 0.1 \  # 기본값
    --batch-size 48
```

**결과:**
- 알파: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] = **6개**
- 평가 횟수: 6 × 2 + fine search = **~15회**
- 예상 시간: **~15분**
- 정확한 최적 알파 찾기 가능

### 🔬 정밀 탐색 (30~40분)

```bash
python wiseft_sweep.py \
    --scratch new_list/600.pt \
    --finetuned new_list/620.pt \
    --data new_list/data.yaml \
    --val-sets valid1 valid2 \
    --alpha-min 0.0 \
    --alpha-max 1.0 \  # 전체 범위
    --focus-range 0.1 \
    --enable-fine-search true
```

**결과:**
- 알파: [0.0, 0.1, ..., 1.0] = **11개**
- 평가 횟수: 11 × 2 + fine search = **~25회**
- 예상 시간: **~40분**
- 최대한 정확한 결과

---

## 💡 2단계 전략 (추천!)

### Step 1: 빠른 탐색으로 범위 좁히기

```bash
# 1단계: 빠른 스캔 (5분)
python wiseft_sweep.py \
    --scratch new_list/600.pt \
    --finetuned new_list/620.pt \
    --data new_list/data.yaml \
    --val-sets valid1 valid2 \
    --alpha-min 0.0 \
    --alpha-max 0.5 \
    --focus-range 0.25 \  # 넓게
    --enable-fine-search false \
    --batch-size 64

# 결과: α=0.25 근처가 좋음!
```

### Step 2: 해당 범위만 정밀 탐색

```bash
# 2단계: 정밀 탐색 (10분)
python wiseft_sweep.py \
    --scratch new_list/600.pt \
    --finetuned new_list/620.pt \
    --data new_list/data.yaml \
    --val-sets valid1 valid2 \
    --alpha-min 0.1 \  # 좁힌 범위
    --alpha-max 0.4 \
    --focus-range 0.05 \  # 촘촘하게
    --enable-fine-search true

# 결과: 최적 α=0.23
```

**총 시간: 15분** (40분 → 15분, 62% 절약!)

---

## 📊 시간 비교표

| 전략 | 알파 개수 | 평가 횟수 | 예상 시간 | 정확도 |
|------|----------|----------|----------|--------|
| 전체 범위 | 11 | 24 | 40분 | ★★★★★ |
| 좁은 범위 (0~0.5) | 6 | 14 | 20분 | ★★★★☆ |
| 넓은 간격 (0.15) | 4 | 10 | 15분 | ★★★☆☆ |
| 빠른 스캔 | 3 | 6 | 5분 | ★★☆☆☆ |
| **2단계 전략** | 3+7 | 6+16 | **15분** | **★★★★☆** |

---

## 🔍 당신의 경우

**보고서 분석 결과:**
- α=0.3 이상부터 성능 급락
- α=0.5에서 최악
- α=1.0에서 회복

**추천 설정:**
```bash
python wiseft_sweep.py \
    --scratch new_list/600.pt \
    --finetuned new_list/620.pt \
    --data new_list/data.yaml \
    --val-sets valid1 valid2 \
    --alpha-min 0.0 \
    --alpha-max 0.3 \  # ← 0.3까지만!
    --focus-range 0.1 \
    --batch-size 48 \
    --enable-tradeoff-viz
```

**예상 결과:**
- 알파: [0.0, 0.1, 0.2, 0.3] = 4개
- 평가 횟수: 4 × 2 + fine search = ~12회
- **예상 시간: ~12분** (40분 → 12분, 70% 절약!)
- 트레이드오프 분석 완벽히 가능!

---

## ⚠️ 주의사항

### 절대 하지 말아야 할 것

❌ **검증 세트 크기 줄이기**
```bash
# 이렇게 하지 마세요!
head -100 valid1.txt > valid1_small.txt  # ← 정확도 파괴!
```

❌ **Batch Size를 너무 크게**
```bash
--batch-size 256  # ← OOM 에러!
```

❌ **이미지 크기를 너무 작게**
```bash
--img-size 320  # ← 정확도 크게 하락
```

### 권장 조합

✅ **알파 범위 좁히기 + 배치 증가**
```bash
--alpha-max 0.3 --batch-size 64
# 안전하고 효과적!
```

✅ **2단계 전략**
```bash
# 1단계: 빠른 스캔
# 2단계: 정밀 탐색
# 최고의 효율!
```

---

## 🎯 결론

**속도가 중요하다면:**
1. **알파 범위 좁히기** (--alpha-max 0.3~0.5)
2. **배치 사이즈 증가** (--batch-size 64)
3. **2단계 전략** 사용

**정확도가 중요하다면:**
- 시간을 투자하세요 (40분은 합리적)
- 트레이드오프 분석은 그만한 가치가 있습니다!

**현재 실행 중이라면:**
- 일단 끝까지 기다리세요
- 다음번부터 위 설정 사용

---

**마지막 팁:**
처음 실행할 때는 빠른 탐색으로 트레이드오프 확인,
나중에 필요하면 정밀 탐색으로 최적 알파 찾기!
