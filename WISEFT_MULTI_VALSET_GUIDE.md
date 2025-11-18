# WiSE-FT 다중 검증 세트 분석 가이드

## 🎯 개요

이 업데이트는 WiSE-FT가 **여러 검증 세트에서 개별 성능을 추적**하고, **검증 세트 간 트레이드오프를 시각화**할 수 있도록 개선되었습니다.

## 🔄 핵심 변경사항

### 1️⃣ 다중 검증 세트 평가

**이전 방식:**
- 모든 검증 데이터를 합쳐서 한 번에 평가
- 전체 평균 성능만 확인 가능
- valid1과 valid2 개별 성능을 알 수 없음

**개선된 방식:**
- 각 검증 세트별로 개별 평가
- 검증 세트별 성능 + 전체 평균 모두 제공
- 트레이드오프 명확히 확인 가능

### 2️⃣ 베이스라인 트레이드오프 분석

스크래치 모델과 파인튜닝 모델을 **각 검증 세트에서 먼저 평가**하여 트레이드오프가 존재하는지 확인합니다.

```
예시 출력:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 BASELINE TRADE-OFF ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model           Metric     valid1        valid2        Overall
─────────────────────────────────────────────────────────────────
Scratch         fitness    0.7500        0.4500        0.6000
Fine-tuned      fitness    0.5000        0.8000        0.6500
─────────────────────────────────────────────────────────────────

Validation Set  Change         % Change       Direction
─────────────────────────────────────────────────────────────────
valid1          -0.2500        -33.3%         ↓ Degraded
valid2          +0.3500        +77.8%         ↑ Improved
─────────────────────────────────────────────────────────────────

⚠️  TRADE-OFF DETECTED!
   ✅ Improved on: valid2
   ❌ Degraded on: valid1

   💡 WiSE-FT is RECOMMENDED to find balance!
```

### 3️⃣ 알파 스윕 시 개별 평가

각 알파 값마다 모든 검증 세트에서 개별 평가를 수행합니다.

```python
# 각 α마다 실행:
α=0.1:
  valid1에서 평가 → fitness=0.72
  valid2에서 평가 → fitness=0.52
  Overall: fitness=0.62

α=0.2:
  valid1에서 평가 → fitness=0.68
  valid2에서 평가 → fitness=0.65
  Overall: fitness=0.665  ← 균형점!
```

### 4️⃣ 트레이드오프 시각화

valid1 vs valid2 성능을 ASCII 산점도로 시각화합니다.

```
valid2 fitness ↑
│
0.800│                          F (α=1.0)
0.700│                    ●
0.650│              ● (α=0.2) ← 균형점!
0.550│         ●
0.450│    S (α=0.0)
0.400│
└────────────────────────→ valid1 fitness
  0.500   0.600   0.700

Legend:
  S : Scratch model (α=0.0)
  F : Fine-tuned model (α=1.0)
  ● : WiSE-FT merged models

Goal: 우상단 (둘 다 높음)
```

### 5️⃣ 다중 기준 알파 선택

단일 평균만이 아닌 다양한 기준으로 최적 알파를 추천합니다.

```
🎯 OPTIMAL ALPHA RECOMMENDATIONS (Multi-Criteria)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  Best Overall Average FITNESS:
   α = 0.200
   Overall fitness: 0.6650
   valid1: 0.6800
   valid2: 0.6500

2️⃣  Most Balanced (minimum difference):
   α = 0.200
   valid1: 0.6800
   valid2: 0.6500
   Difference: 0.0300

3️⃣  Best Worst-Case (maximize minimum):
   α = 0.200
   Minimum fitness: 0.6500
   valid1: 0.6800
   valid2: 0.6500 ⬅️ MIN

4️⃣  Best Total Sum:
   α = 0.200
   Total fitness: 1.3300
   valid1: 0.6800
   valid2: 0.6500

🏆 RECOMMENDED ALPHA: 0.200
   (Selected by 4/4 criteria)
```

## 📦 새로운 함수들

### `evaluate_model_multi_valset()`
여러 검증 세트에서 개별 평가 후 결과 통합

**반환 형식:**
```python
{
    'overall': {
        'precision': 0.85,
        'recall': 0.80,
        'map50': 0.82,
        'fitness': 0.665
    },
    'per_valset': {
        'valid1': {
            'precision': 0.90,
            'fitness': 0.68,
            ...
        },
        'valid2': {
            'precision': 0.80,
            'fitness': 0.65,
            ...
        }
    }
}
```

### `analyze_baseline_tradeoff()`
베이스라인 모델들의 검증 세트별 성능 비교 및 트레이드오프 분석

### `visualize_valset_tradeoff()`
검증 세트 간 성능 트레이드오프를 ASCII 산점도로 시각화

### `find_best_alpha_multi_criteria()`
다양한 기준(평균, 균형, worst-case, 합계)으로 최적 알파 추천

## 🚀 사용 방법

### 기본 사용

```bash
python wiseft_sweep.py \
    --scratch models/600.pt \
    --finetuned models/620.pt \
    --data data/custom.yaml \
    --val-sets valid1 valid2 \
    --focus-range 0.1 \
    --enable-tradeoff-viz
```

### 데이터 준비

**data.yaml 구조:**
```yaml
train: ./data/train.txt
val: ./data/valid_all.txt  # 이 경로가 base
nc: 3
names: ['person', 'car', 'dog']
```

**검증 세트 파일:**
```
data/
├── train.txt
├── valid_all.txt  (선택사항)
├── valid1.txt     ← 필수!
└── valid2.txt     ← 필수!
```

스크립트는 `valid_all.txt`의 경로를 기준으로 `valid1.txt`, `valid2.txt`를 자동으로 찾습니다.

### 고급 옵션

```bash
python wiseft_sweep.py \
    --scratch models/600.pt \
    --finetuned models/620.pt \
    --data data/custom.yaml \
    --val-sets valid1 valid2 \
    --focus-range 0.1 \
    --alpha-min 0.0 \
    --alpha-max 0.5 \
    --enable-tradeoff-viz \
    --enable-fine-search \
    --metric fitness
```

## 📊 동작 흐름

```
1. Weight Analysis (기존과 동일)
   ↓
2. 베이스라인 평가 (각 검증 세트별!)
   - Scratch: valid1=0.75, valid2=0.45
   - Fine-tuned: valid1=0.50, valid2=0.80
   ↓
3. 트레이드오프 분석
   → 트레이드오프 발견! WiSE-FT 추천
   ↓
4. Coarse Search (각 알파마다 valid1/valid2 평가)
   α=0.1: valid1=0.72, valid2=0.52
   α=0.2: valid1=0.68, valid2=0.65 ← 균형!
   α=0.3: valid1=0.62, valid2=0.72
   ↓
5. Fine Search (최적 알파 주변 세밀 탐색)
   ↓
6. 트레이드오프 시각화
   - valid1 vs valid2 산점도
   ↓
7. 다중 기준 알파 선택
   - 평균, 균형, worst-case, 합계
   ↓
8. 최종 추천
   → α=0.2 (4/4 criteria)
```

## 📝 results.json 형식

```json
{
  "best_alpha": 0.2,
  "all_results": [
    {
      "alpha": 0.0,
      "metrics": {
        "overall": {
          "precision": 0.80,
          "recall": 0.725,
          "map50": 0.76,
          "fitness": 0.60
        },
        "per_valset": {
          "valid1": {
            "precision": 0.90,
            "recall": 0.85,
            "map50": 0.87,
            "fitness": 0.75
          },
          "valid2": {
            "precision": 0.70,
            "recall": 0.60,
            "map50": 0.65,
            "fitness": 0.45
          }
        }
      }
    },
    {
      "alpha": 0.2,
      "metrics": {
        "overall": {"fitness": 0.665},
        "per_valset": {
          "valid1": {"fitness": 0.68},
          "valid2": {"fitness": 0.65}
        }
      }
    }
  ]
}
```

## 🎯 언제 사용해야 하나요?

### ✅ 사용을 권장하는 경우

1. **치명적 망각(Catastrophic Forgetting)이 의심될 때**
   - 파인튜닝 후 기존 데이터셋 성능이 크게 하락

2. **여러 도메인의 데이터가 있을 때**
   - valid1: 실내 데이터
   - valid2: 실외 데이터

3. **성능 균형이 중요할 때**
   - 한쪽 성능만 높은 것보다 둘 다 적당히 유지하고 싶을 때

### ❌ 사용이 불필요한 경우

1. 파인튜닝이 모든 검증 세트에서 성능 향상
2. 검증 세트가 하나뿐인 경우
3. 트레이드오프가 없는 경우

## 🔍 트러블슈팅

### 문제: "File not found: valid1.txt"

**원인:** 검증 세트 파일이 올바른 위치에 없음

**해결:**
```bash
# data.yaml의 val 경로 확인
cat data/custom.yaml | grep "val:"

# 같은 디렉토리에 valid1.txt, valid2.txt 생성
ls data/valid*.txt
```

### 문제: "No trade-off detected"

**상황:** 파인튜닝이 모든 검증 세트에서 성능 향상

**대응:**
- WiSE-FT가 불필요할 수 있음
- 파인튜닝 모델을 그대로 사용하는 것이 나을 수 있음
- 계속 진행하려면 'y' 입력

## 📚 관련 문서

- [WiSE-FT 논문](https://arxiv.org/abs/2109.01903)
- [기존 WiSE-FT 가이드](WISEFT_README_KR.md)
- [Phase 2/3 고급 기능](PHASE2_PHASE3_FEATURES_KR.md)

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 남겨주세요!

---

**마지막 업데이트:** 2025-11-18
**버전:** 2.0 (Multi-Validation Set Support)
