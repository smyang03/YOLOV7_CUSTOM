# WiSE-FT 핵심 인사이트 및 개선 전략 요약

## 🎯 핵심 발견 (Key Insights)

### 1. **Pareto Frontier: 오직 2개의 최적해**

```
Pareto-optimal solutions:
├─ α=0.00: Valid1=0.6669, Valid2=0.3873 (전체 최고)
└─ α=0.10: Valid1=0.6435, Valid2=0.3930 (Valid2 최고)

→ 나머지 α=0.2, 0.3은 dominated (비효율적)
```

**의미:**
- **실용적으로는 α=0.0 또는 α=0.1만 고려하면 됨**
- α=0.2 이상은 두 validation set 모두에서 손해

---

### 2. **Alpha=0.1의 특별한 의미: P/R Balance**

#### 현상
```
Valid2에서:
- α=0.0: P/R ratio = 1.54 (Precision 과다, Recall 부족)
- α=0.1: P/R ratio = 1.02 (완벽한 균형!)
- α=0.2: P/R ratio = 1.56 (다시 불균형)
```

#### 메커니즘
```
α=0.1에서 Recall 대폭 상승:
- Recall: 0.458 → 0.541 (+18.1%)
- Precision: 0.707 → 0.553 (-21.8%)
- Fitness: 0.3873 → 0.3930 (+1.5%) ✓

→ Fine-tuned 모델의 10% 혼합이 detection threshold를 최적화
```

**인사이트:**
- Alpha=0.1은 Valid2에 대해 **더 aggressive한 detection strategy** 제공
- 더 많은 객체를 탐지하면서도 전체 정확도 유지

---

### 3. **Localization Quality 하락 패턴**

```
mAP50/mAP 비율 (높을수록 localization 나쁨):

Valid1:  1.276 → 1.296 → 1.356 → 1.423
Valid2:  1.273 → 1.369 → 1.458 → 1.474

→ Alpha 증가 시 localization 품질 지속 하락
```

**의미:**
- Fine-tuned 모델이 **bounding box를 부정확하게 예측**
- Detection은 하지만 위치가 정확하지 않음
- → Fine-tuning 시 localization loss가 제대로 학습 안됨

---

### 4. **Weighted Optimization 분석**

```
Valid2 우선순위별 최적 Alpha:

Equal (0.5/0.5):     α=0.0 (0.5271)
Valid1 중시 (0.7/0.3): α=0.0 (0.5830)
Valid2 중시 (0.3/0.7): α=0.0 (0.4712)
Valid2 강력 (0.1/0.9): α=0.1 (0.4181) ✓

→ Valid2가 90% 이상 중요해야 α=0.1 선택
```

**실용적 가이드:**
- 대부분의 경우: **α=0.0 사용**
- Valid2 성능이 압도적으로 중요한 경우만: **α=0.1 고려**

---

## 🔬 개선 가능성 탐색

### 전략 1: Fine-grained Search (0.0~0.15) ⭐⭐⭐⭐⭐

**가설:**
```
α=0.0 (최고)과 α=0.1 (Valid2 개선) 사이에
진정한 sweet spot이 존재할 가능성
```

**실험:**
```bash
python wiseft_sweep_parallel.py \
    --alpha-min 0.0 \
    --alpha-max 0.15 \
    --focus-range 0.02 \
    --num-gpus 8
```

**예상 결과:**
- α=0.04~0.08에서 최적 균형점 발견
- Valid2 개선: +0.5~1.0%
- Valid1 손실: -1~2%
- Overall 손실 최소화

**Impact:** 🔴 High | **Effort:** 🟢 Low | **Priority:** 1순위

---

### 전략 2: Layer-wise Merging ⭐⭐⭐⭐

**아이디어:**
```
레이어별로 다른 alpha 적용:
- Backbone (일반적 특징): α=0.02~0.05 (scratch 유지)
- Neck: α=0.05~0.10
- Head (task-specific): α=0.08~0.15 (finetuned 활용)
```

**예상 효과:**
```
Conservative 전략 (평균 α≈0.041):
- Valid1 손실 최소화 (backbone 보존)
- Valid2 약간 개선 (head 활용)
- Localization 품질 개선 가능
```

**구현 난이도:** Medium
**Impact:** 🟡 Medium-High | **Priority:** 2순위

---

### 전략 3: Confidence Threshold Optimization ⭐⭐⭐

**발견:**
```
α=0.1에서 Valid2의 Recall 개선 = 더 낮은 threshold
→ Post-processing으로도 동일 효과 가능?
```

**실험:**
```python
# α=0.0 모델로 다양한 confidence threshold 테스트
conf_thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01]

for conf in conf_thresholds:
    evaluate(model_alpha0, conf_thres=conf)
```

**예상:**
- Valid2에서 conf=0.0005일 때 α=0.1과 유사한 결과
- 모델 변경 없이 threshold만 조정으로 개선

**Impact:** 🟢 Low-Medium | **Effort:** 🟢 Very Low | **Priority:** 3순위

---

### 전략 4: Non-linear Interpolation ⭐⭐

**아이디어:**
```python
# Sigmoid-based merging
s = 1 / (1 + exp(-temp * (α - 0.5)))
merged = (1-s) * scratch + s * finetuned

# α=0.1 → s≈0.007 (선형보다 scratch에 가까움)
→ Valid1 손실 최소화하면서 Valid2 개선
```

**Impact:** 🟡 Medium | **Effort:** 🟡 Medium | **Priority:** 4순위

---

### 전략 5: Ensemble ⭐⭐

**방법:**
```python
# Test-time ensemble
predictions = [
    (model_alpha0.predict(), 0.7),  # Valid1 강점
    (model_alpha01.predict(), 0.3)  # Valid2 강점
]
final = weighted_average(predictions)
```

**장점:**
- 각 domain에 최적화된 모델 활용
- 추론 시 유연성

**단점:**
- 2배 추론 시간
- 복잡도 증가

**Impact:** 🟢 Low-Medium | **Effort:** 🟡 Medium | **Priority:** 5순위

---

## 📊 실험 우선순위 및 예상 ROI

| 전략 | Priority | Impact | Effort | 예상 개선 | 시간 |
|------|----------|--------|--------|-----------|------|
| Fine-grained Search | 🥇 1 | High | Low | +0.5~1% | 30-60분 |
| Layer-wise Merging | 🥈 2 | Med-High | Medium | +1~2% | 2-4시간 |
| Conf Threshold Opt | 🥉 3 | Low-Med | Very Low | +0.5% | 10-20분 |
| Non-linear Interp | 4 | Medium | Medium | +0.3~0.8% | 1-2시간 |
| Ensemble | 5 | Low-Med | Medium | +0.3~0.5% | 1-2시간 |

---

## 🎯 즉시 실행 가능한 액션 플랜

### Phase 1: Quick Wins (오늘)

**1. Fine-grained Search** (30-60분)
```bash
python wiseft_sweep_parallel.py \
    --alpha-min 0.0 --alpha-max 0.15 --focus-range 0.02 \
    --num-gpus 8 --batch-size 64
```

**2. Confidence Threshold Test** (20분)
```bash
# α=0.0 모델로 다양한 threshold 테스트
for conf in 0.0001 0.0005 0.001 0.005; do
    python test.py --conf-thres $conf --weights alpha_0.000.pt
done
```

**예상 성과:**
- 최적 α 발견 (0.04~0.08 예상)
- Valid2 성능 +0.5~1.5% 개선
- 총 소요 시간: ~1시간

---

### Phase 2: Advanced Techniques (이번 주)

**3. Layer-wise Merging 구현**
- Conservative, Balanced, Aggressive 전략 테스트
- 각 전략별 성능 비교

**4. Full Range Sweep (α=0.0~1.0)**
- 완전한 경향 파악
- α=1.0 baseline 확인

**예상 성과:**
- 더 정교한 모델 선택
- 전체 WiSE-FT curve 완성

---

## 💡 핵심 메시지

### 현재 상태 해석
```
❌ "실패" 아님
✅ "α=0.0~0.1 구간에 흥미로운 현상 발견"

- α=0.0: Overall 최고
- α=0.1: Valid2 P/R balance 최적
- α=0.0~0.1 사이에 더 나은 균형점 존재 가능성 높음
```

### 실용적 권장사항

**지금 바로 사용한다면:**
- **추천: α=0.0 (600.pt)**
- Valid2가 매우 중요하다면: α=0.1 고려

**더 나은 성능을 원한다면:**
1. Fine-grained search 실행 (30분)
2. 최적 α 찾기 (예상: 0.04~0.08)
3. 1~2% 개선 기대

**연구/실험 목적이라면:**
- 모든 전략 테스트
- Layer-wise, non-linear 등 탐구
- 메커니즘 이해 심화

---

## 🔍 향후 탐구 질문

1. **왜 α=0.1에서만 Valid2의 P/R이 균형을 이루는가?**
   - Fine-tuned 모델의 어떤 특성이 이를 가능하게 하는가?
   - Layer별 기여도 분석 필요

2. **Localization 품질 하락의 근본 원인은?**
   - Fine-tuning 시 localization loss weight 문제?
   - Data augmentation 부족?
   - Anchor 설정 문제?

3. **Valid1과 Valid2의 본질적 차이는?**
   - 객체 크기 분포?
   - 배경 복잡도?
   - Class 분포?
   - → 데이터 분석 필요

4. **Layer-wise merging의 최적 전략은?**
   - 어느 레이어가 domain-specific한가?
   - 어느 레이어가 general한가?

---

## 📚 참고 문서

- `WISEFT_EXPERIMENTAL_ANALYSIS.md` - 전체 실험적 분석
- `wiseft_advanced_experiments.py` - 고급 실험 도구
- `WISEFT_ANALYSIS_REPORT.md` - 기본 분석 보고서

---

**마지막 업데이트:** 2025-11-19
**상태:** 실험 설계 완료, Phase 1 실행 대기
**다음 단계:** Fine-grained search 실행 → 최적 α 발견
