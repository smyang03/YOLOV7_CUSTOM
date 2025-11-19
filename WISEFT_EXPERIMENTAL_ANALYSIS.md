# WiSE-FT 실험적 경향 분석 및 개선 방향 탐색

## 🔬 실험적 관점: 현상 관찰 및 이해

### 관찰 1: Alpha=0.1에서의 미묘한 Trade-off

#### 📊 현상
```
Alpha 0.0 → 0.1:
├─ Valid1: 0.6669 → 0.6435 (-0.0234, -3.5%)
├─ Valid2: 0.3873 → 0.3930 (+0.0057, +1.5%)
└─ Overall: 0.5271 → 0.5183 (-1.7%)
```

#### 🤔 왜 이런 현상이 나타나는가?

**가설 1: Fine-tuned 모델이 Valid2에 특화된 일부 특성을 가지고 있음**
- 10% Fine-tuned (Alpha=0.1) 혼합 시 Valid2 성능이 미미하게 상승
- 하지만 90% Scratch 비율로 인해 Overall 성능은 여전히 높음
- → **Sweet spot이 0.0~0.1 사이에 존재할 가능성**

**가설 2: Precision/Recall Balance 변화**

| Alpha | Valid1 P/R | Valid2 P/R |
|-------|-----------|-----------|
| 0.0   | 0.871/0.798 | 0.707/0.458 |
| 0.1   | 0.889/0.771 | 0.553/0.541 |

Valid2에서:
- Precision 하락 (0.707 → 0.553, -21.8%)
- Recall 대폭 상승 (0.458 → 0.541, +18.1%)
- → **Detection threshold가 변경되는 효과!**

#### 💡 인사이트
```
Alpha=0.1에서 Valid2는:
- Recall을 희생하던 모델 → 더 많은 객체 탐지
- False Positive 증가 가능성
- 하지만 전체 fitness는 상승 (+1.5%)
→ Valid2에 더 적합한 detection strategy
```

---

## 📈 경향의 다차원 분석

### 1. Precision vs Recall 경향

#### Valid1 패턴
```
Alpha   Precision   Recall      P/R Ratio
0.0     0.871       0.798       1.09  (균형)
0.1     0.889       0.771       1.15  (Precision 우위)
0.2     0.876       0.678       1.29  (Precision 우위)
0.3     0.633       0.528       1.20  (균형 붕괴)
```

**경향:**
- Alpha 증가 시 Recall이 더 빠르게 하락
- Alpha=0.0에서 가장 균형잡힌 P/R
- Alpha ≥ 0.3에서 전체적 성능 붕괴

#### Valid2 패턴
```
Alpha   Precision   Recall      P/R Ratio
0.0     0.707       0.458       1.54  (Precision 과다)
0.1     0.553       0.541       1.02  (균형!)
0.2     0.599       0.385       1.56  (Precision 과다)
0.3     0.423       0.341       1.24  (붕괴)
```

**흥미로운 발견!**
- **Alpha=0.1에서만 Valid2의 P/R이 균형 (1.02)**
- Alpha=0.0에서는 Precision 과다 (많이 놓치고 있음)
- Alpha=0.1이 Valid2에 더 적합한 detection strategy 제공

### 2. mAP50 vs mAP 경향

#### mAP 비율 분석
```
Valid1:
Alpha   mAP50/mAP   의미
0.0     1.276       IoU threshold에 덜 민감
0.1     1.296
0.2     1.356       IoU threshold에 더 민감
0.3     1.423       Localization 정확도 하락

Valid2:
Alpha   mAP50/mAP   의미
0.0     1.273       유사한 패턴
0.1     1.369
0.2     1.458       Localization 급격히 악화
0.3     1.474
```

**경향:**
- Alpha 증가 시 mAP50/mAP 비율 증가 → Localization 품질 하락
- Valid2가 Valid1보다 더 빠르게 악화
- → Fine-tuned 모델이 localization이 부정확함

---

## 🎯 개선 가능성 탐색

### 개선 방향 1: Fine-grained Alpha Search (0.0~0.1 구간)

#### 동기
```
현재: Alpha=0.0 (최고) vs Alpha=0.1 (Valid2 개선)
가설: 0.0과 0.1 사이에 더 나은 균형점 존재

제안 실험:
Alpha = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15]
```

#### 예상 결과
```
Alpha=0.05 근처에서:
- Valid2 개선: +0.5~1.0%
- Valid1 손실: -1~2%
- Overall 손실 최소화
```

#### 실험 코드
```bash
python wiseft_sweep_parallel.py \
    --scratch new_list/600.pt \
    --finetuned new_list/620.pt \
    --data new_list/data.yaml \
    --val-sets valid1 valid2 \
    --alpha-min 0.0 \
    --alpha-max 0.15 \
    --focus-range 0.02 \
    --num-gpus 8 \
    --batch-size 64
```

---

### 개선 방향 2: Weighted WiSE-FT

#### 동기
```
현재: Overall = (Valid1 + Valid2) / 2
문제: Valid2 성능이 중요하다면 가중치 부여 필요

제안: Overall = w1 * Valid1 + w2 * Valid2
     where w1 + w2 = 1
```

#### 시나리오별 최적 Alpha

**시나리오 A: Valid1 중시 (w1=0.7, w2=0.3)**
```
Alpha   Weighted Score
0.0     0.7*0.6669 + 0.3*0.3873 = 0.583
0.1     0.7*0.6435 + 0.3*0.3930 = 0.568
→ Alpha=0.0 여전히 최적
```

**시나리오 B: Valid2 중시 (w1=0.3, w2=0.7)**
```
Alpha   Weighted Score
0.0     0.3*0.6669 + 0.7*0.3873 = 0.471
0.1     0.3*0.6435 + 0.7*0.3930 = 0.468
→ Alpha=0.0 여전히 근소하게 우세

하지만 fine-grained search에서:
Alpha=0.05 예상: 0.3*0.655 + 0.7*0.390 = 0.470
→ 가능성 있음
```

**시나리오 C: Valid2 최우선 (w1=0.1, w2=0.9)**
```
Alpha   Weighted Score
0.0     0.1*0.6669 + 0.9*0.3873 = 0.415
0.1     0.1*0.6435 + 0.9*0.3930 = 0.418 ← 최적!
→ Alpha=0.1이 최적
```

#### 구현
```python
def find_best_alpha_weighted(results, w_valid1, w_valid2):
    best = None
    best_score = -1

    for r in results:
        v1 = r['metrics']['per_valset']['valid1']['fitness']
        v2 = r['metrics']['per_valset']['valid2']['fitness']
        score = w_valid1 * v1 + w_valid2 * v2

        if score > best_score:
            best_score = score
            best = r

    return best, best_score
```

---

### 개선 방향 3: Layer-wise WiSE-FT

#### 동기
```
가설: 다른 레이어는 다른 alpha가 필요할 수 있음
- Early layers: 일반적 특징 (낮은 alpha - scratch 유지)
- Late layers: Task-specific 특징 (높은 alpha - finetuned 활용)
```

#### 제안 전략
```python
# Backbone: Alpha=0.05 (거의 scratch)
# Neck: Alpha=0.10
# Head: Alpha=0.15 (더 많은 finetuned)
```

#### 예상 효과
- Valid1: 일반적 특징 보존으로 성능 유지
- Valid2: Task-specific 특징 활용으로 개선
- 더 나은 균형 달성 가능

#### 구현 예시
```python
def layer_wise_merge(scratch_sd, finetuned_sd):
    merged_sd = {}

    for key in scratch_sd.keys():
        # 레이어 타입별 alpha 결정
        if 'model.0' in key or 'model.1' in key:  # Early layers
            alpha = 0.05
        elif 'model.24' in key:  # Neck
            alpha = 0.10
        elif 'model.105' in key:  # Head
            alpha = 0.15
        else:
            alpha = 0.10  # Default

        merged_sd[key] = (1 - alpha) * scratch_sd[key] + alpha * finetuned_sd[key]

    return merged_sd
```

---

### 개선 방향 4: Non-linear Interpolation

#### 동기
```
현재: Linear interpolation
문제: 선형이 최적이 아닐 수 있음

대안:
1. Sigmoid-based
2. Cosine annealing
3. Exponential
```

#### Sigmoid-based Merging
```python
import math

def sigmoid_merge(scratch, finetuned, alpha, temperature=5.0):
    # Sigmoid를 통한 부드러운 전환
    # alpha=0.5 근처에서 급격한 변화, 양 끝에서 완만

    s = 1 / (1 + math.exp(-temperature * (alpha - 0.5)))
    return (1 - s) * scratch + s * finetuned

# 예시:
# alpha=0.1: s≈0.007 (거의 scratch)
# alpha=0.5: s=0.5 (정확히 중간)
# alpha=0.9: s≈0.993 (거의 finetuned)
```

#### 장점
- Alpha=0.1 근처에서 더 scratch에 가까움
- 미세 조정 효과
- Valid1 손실 최소화하면서 Valid2 개선 가능

---

### 개선 방향 5: Metric-specific Optimization

#### 관찰
```
Valid2에서 Alpha=0.1:
- Precision: 0.707 → 0.553 (하락)
- Recall: 0.458 → 0.541 (상승)
- 최종 fitness: 상승

→ Recall 개선이 핵심!
```

#### 제안: Recall-focused WiSE-FT
```python
# Confidence threshold 조정을 통한 P/R balance
# WiSE-FT + Post-processing

def optimize_for_recall(model, target_recall=0.55):
    # Lower confidence threshold for Valid2
    # Higher for Valid1

    conf_thresholds = {
        'valid1': 0.001,  # Original
        'valid2': 0.0005  # Lower for more recall
    }
```

#### 또는 Fitness 함수 재정의
```python
# 현재: fitness = 0.1 * mAP50 + 0.9 * mAP

# Valid2용: Recall 가중
fitness_v2 = 0.3 * recall + 0.1 * mAP50 + 0.6 * mAP

# Valid1용: 균형
fitness_v1 = 0.1 * mAP50 + 0.9 * mAP
```

---

### 개선 방향 6: Ensemble Approach

#### 동기
```
WiSE-FT는 단일 merged model
대안: 여러 alpha의 앙상블
```

#### 제안
```python
# Test-time ensemble
predictions = []
weights = []

# Alpha=0.0 (Valid1 강점)
predictions.append(model_alpha_0.predict())
weights.append(0.7)

# Alpha=0.1 (Valid2 강점)
predictions.append(model_alpha_01.predict())
weights.append(0.3)

# Weighted ensemble
final_pred = weighted_average(predictions, weights)
```

#### 예상 효과
- Valid1: 0.0 모델 기여도 높음
- Valid2: 0.1 모델 기여도 높음
- 각 domain에 적합한 모델 활용

---

## 🧪 추천 실험 순서

### Phase 1: Fine-grained Search (즉시 실행 가능)
```bash
# Alpha 0.0~0.15 범위를 0.02 간격으로
python wiseft_sweep_parallel.py \
    --alpha-min 0.0 \
    --alpha-max 0.15 \
    --focus-range 0.02
```

**목표:** 최적 균형점 찾기
**예상 시간:** ~30-60분 (GPU 8개)
**예상 발견:** Alpha=0.04~0.08 사이 sweet spot

### Phase 2: Full Range (완성도)
```bash
# Alpha 0.0~1.0 전체 평가
python wiseft_sweep_parallel.py \
    --alpha-min 0.0 \
    --alpha-max 1.0 \
    --focus-range 0.1
```

**목표:** 전체 경향 파악
**예상 발견:** 0.3 이후 급락 확인, 1.0 성능 측정

### Phase 3: Weighted Optimization
```python
# 다양한 가중치로 최적 alpha 재계산
weights = [
    (0.5, 0.5),  # Equal
    (0.7, 0.3),  # Valid1 중시
    (0.3, 0.7),  # Valid2 중시
]

for w1, w2 in weights:
    best_alpha = find_optimal(results, w1, w2)
    print(f"Weights ({w1}, {w2}): Best alpha = {best_alpha}")
```

### Phase 4: Advanced Techniques (선택)
- Layer-wise merging
- Non-linear interpolation
- Ensemble approaches

---

## 📊 경향 기반 가설

### 가설 1: "Sweet Spot Hypothesis"
```
관찰: Alpha=0.0 (최고), Alpha=0.1 (Valid2 개선)
가설: Alpha=0.04~0.08 사이에 진정한 최적점 존재
검증: Fine-grained search
```

### 가설 2: "P/R Balance Hypothesis"
```
관찰: Alpha=0.1에서 Valid2의 P/R이 균형(1.02)
가설: 각 validation set은 다른 P/R balance를 선호
검증: Confidence threshold sweep
```

### 가설 3: "Localization Quality Hypothesis"
```
관찰: Alpha 증가 시 mAP50/mAP 비율 증가
가설: Fine-tuned 모델이 localization 부정확
검증: IoU threshold별 성능 분석
```

### 가설 4: "Domain Gap Hypothesis"
```
관찰: Valid1 >> Valid2 (1.7배)
가설: Valid2가 본질적으로 더 어려운 domain
검증: 데이터 분포 분석, class 별 성능
```

---

## 🎨 시각화 제안

### 1. 3D Surface Plot
```
X축: Alpha (0.0~1.0)
Y축: Weight(Valid2) (0.0~1.0)
Z축: Weighted Fitness

→ 최적 조합 찾기
```

### 2. Pareto Frontier
```
X축: Valid1 Fitness
Y축: Valid2 Fitness

각 alpha를 점으로 표시
→ Trade-off curve 시각화
→ Dominated/Non-dominated solutions
```

### 3. Metric Heatmap
```
       Alpha=0.0  0.1   0.2   0.3
V1 P   [0.871]  0.889 0.876 0.633
V1 R   [0.798]  0.771 0.678 0.528
V2 P   0.707    0.553 0.599 0.423
V2 R   0.458   [0.541] 0.385 0.341

→ 어떤 메트릭이 어디서 최대인지 한눈에
```

---

## 💡 핵심 인사이트 요약

### 1. Alpha=0.1의 특별함
- Valid2에서 **유일하게** Precision/Recall 균형
- Recall 대폭 상승 (+18%)
- 전체 fitness도 미미하게 상승

### 2. 0.0~0.1 구간이 중요
- 이 구간에 최적해 존재 가능성 높음
- Fine-grained search 필수

### 3. Valid2의 특성
- Recall이 본질적으로 낮음 (0.458)
- Precision 과다 (많은 객체 놓침)
- Alpha=0.1이 detection strategy 개선

### 4. Localization vs Detection
- Alpha 증가 → Detection 악화
- 하지만 Localization도 악화
- Fine-tuned 모델의 근본적 문제

---

## 🚀 다음 단계 액션 플랜

**즉시 실행 (High Impact, Low Effort):**
1. Fine-grained search (Alpha 0.0~0.15, step=0.02)
2. Weighted fitness 계산 (다양한 w1/w2)

**중기 실험 (High Impact, Medium Effort):**
3. Full range sweep (Alpha 0.0~1.0)
4. Confidence threshold optimization
5. P/R curve 분석

**장기 탐구 (High Impact, High Effort):**
6. Layer-wise merging
7. Non-linear interpolation
8. Ensemble approaches

---

**핵심 메시지:**
현재 결과는 "실패"가 아니라 **"Alpha=0.0~0.1 구간에 흥미로운 현상이 있다"**는 발견입니다.
이 구간을 더 세밀하게 탐색하면 Valid2 성능을 개선하면서 Valid1 손실을 최소화하는
**최적 균형점**을 찾을 수 있을 것입니다! 🎯
