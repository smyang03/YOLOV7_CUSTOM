# 📊 Enhanced analyze_results.py Output Examples

## 개선사항 요약

1. ✅ **Best epoch에서 모든 validation set의 성능 표시**
2. ✅ **각 validation set의 overall + per-class 결과 표시**
3. ✅ **선택한 validation set의 통계 정보 (Min, Max, Mean, Best Epoch)**

---

## Example 1: Combined validation set, person class, fitness metric

```bash
$ python analyze_results.py --results results.txt --val-set Combined --class person --metric fitness
```

### Output:

```
================================================================================
Analysis Results
================================================================================
Results file: results.txt
Validation set: Combined
Class: person
Metric: fitness

🏆 Best Epoch: 17
   Best fitness: 0.570810

   Detailed Metrics:
   - Precision: 0.804900
   - Recall: 0.710100
   - mAP@.5: 0.763500
   - mAP@.5:.95: 0.549400
   - Images: 236
   - Fitness: 0.570810

📊 Top 5 Epochs:
   1. Epoch  17: fitness = 0.570810
   2. Epoch  15: fitness = 0.523456
   3. Epoch  16: fitness = 0.518234
   4. Epoch  14: fitness = 0.512890
   5. Epoch  13: fitness = 0.507234

================================================================================
📈 Best Model Performance (Epoch 17) Across All Validation Sets
================================================================================

🔹 Combined:
   Overall: P=0.6795, R=0.5679, mAP@.5=0.6123, mAP@.5:.95=0.4523, fitness=0.4528
   Per-class:
     • Drum           : P=0.5234, R=0.4567, mAP@.5=0.5012, mAP@.5:.95=0.3456, fitness=0.3614 (images=78)
     • Helmet         : P=0.6234, R=0.4890, mAP@.5=0.5678, mAP@.5:.95=0.4123, fitness=0.4278 (images=69)
     • car            : P=0.7456, R=0.6234, mAP@.5=0.6890, mAP@.5:.95=0.5234, fitness=0.5402 (images=68)
     • person         : P=0.8049, R=0.7101, mAP@.5=0.7635, mAP@.5:.95=0.5494, fitness=0.5708 (images=236)

🔹 test1:
   Overall: P=0.6123, R=0.5234, mAP@.5=0.5678, mAP@.5:.95=0.4012, fitness=0.4179
   Per-class:
     • Drum           : P=0.5234, R=0.4567, mAP@.5=0.5012, mAP@.5:.95=0.3456, fitness=0.3614 (images=78)
     • Helmet         : P=0.6234, R=0.4890, mAP@.5=0.5678, mAP@.5:.95=0.4123, fitness=0.4278 (images=69)
     • person         : P=0.7234, R=0.6345, mAP@.5=0.7012, mAP@.5:.95=0.4890, fitness=0.5102 (images=67)

🔹 test2:
   Overall: P=0.7467, R=0.6124, mAP@.5=0.6568, mAP@.5:.95=0.5034, fitness=0.5187
   Per-class:
     • car            : P=0.7456, R=0.6234, mAP@.5=0.6890, mAP@.5:.95=0.5234, fitness=0.5402 (images=68)
     • person         : P=0.8864, R=0.7857, mAP@.5=0.8258, mAP@.5:.95=0.6098, fitness=0.6314 (images=169)

================================================================================
📊 Selected Validation Set Statistics: Combined
================================================================================

Total epochs analyzed: 18
Class: person

Metric Statistics:
Metric                 Min        Max       Mean   Best Epoch
------------------------------------------------------------
P                   0.4523     0.8049     0.6234           17
R                   0.3456     0.7101     0.5123           17
mAP@.5              0.3890     0.7635     0.5678           17
mAP@.5:.95          0.2456     0.5494     0.4012           17
fitness             0.2567     0.5708     0.3945           17

================================================================================
```

---

## Example 2: test1 validation set, all classes, mAP@.5:.95 metric

```bash
$ python analyze_results.py --results results.txt --val-set test1 --class all --metric map
```

### Output:

```
================================================================================
Analysis Results
================================================================================
Results file: results.txt
Validation set: test1
Class: all
Metric: map

🏆 Best Epoch: 16
   Best map: 0.421200

   Detailed Metrics:
   - Precision: 0.623400
   - Recall: 0.534500
   - mAP@.5: 0.578900
   - mAP@.5:.95: 0.421200
   - Fitness: 0.436970

📊 Top 5 Epochs:
   1. Epoch  16: map = 0.421200
   2. Epoch  17: map = 0.401200
   3. Epoch  15: map = 0.398700
   4. Epoch  14: map = 0.385600
   5. Epoch  18: map = 0.378900

================================================================================
📈 Best Model Performance (Epoch 16) Across All Validation Sets
================================================================================

🔹 Combined:
   Overall: P=0.6795, R=0.5679, mAP@.5=0.6123, mAP@.5:.95=0.4523, fitness=0.4528
   Per-class:
     • Drum           : P=0.5234, R=0.4567, mAP@.5=0.5012, mAP@.5:.95=0.3456, fitness=0.3614 (images=78)
     • Helmet         : P=0.6234, R=0.4890, mAP@.5=0.5678, mAP@.5:.95=0.4123, fitness=0.4278 (images=69)
     • car            : P=0.7456, R=0.6234, mAP@.5=0.6890, mAP@.5:.95=0.5234, fitness=0.5402 (images=68)
     • person         : P=0.7845, R=0.6890, mAP@.5=0.7456, mAP@.5:.95=0.5234, fitness=0.5456 (images=236)

🔹 test1:
   Overall: P=0.6234, R=0.5345, mAP@.5=0.5789, mAP@.5:.95=0.4212, fitness=0.4370
   Per-class:
     • Drum           : P=0.5234, R=0.4567, mAP@.5=0.5012, mAP@.5:.95=0.3456, fitness=0.3614 (images=78)
     • Helmet         : P=0.6234, R=0.4890, mAP@.5=0.5678, mAP@.5:.95=0.4123, fitness=0.4278 (images=69)
     • person         : P=0.7234, R=0.6345, mAP@.5=0.7012, mAP@.5:.95=0.4890, fitness=0.5102 (images=67)

🔹 test2:
   Overall: P=0.7356, R=0.6013, mAP@.5=0.6457, mAP@.5:.95=0.4834, fitness=0.4994
   Per-class:
     • car            : P=0.7123, R=0.5890, mAP@.5=0.6456, mAP@.5:.95=0.4789, fitness=0.4956 (images=68)
     • person         : P=0.7589, R=0.6136, mAP@.5=0.6458, mAP@.5:.95=0.4879, fitness=0.5032 (images=169)

================================================================================
📊 Selected Validation Set Statistics: test1
================================================================================

Total epochs analyzed: 20
Class: all

Metric Statistics:
Metric                 Min        Max       Mean   Best Epoch
------------------------------------------------------------
P                   0.4234     0.6234     0.5234           16
R                   0.3456     0.5345     0.4456           16
mAP@.5              0.3678     0.5789     0.4789           16
mAP@.5:.95          0.2345     0.4212     0.3456           16
fitness             0.2456     0.4370     0.3456           16

================================================================================
```

---

## Example 3: test2 validation set, car class, mAP@.5 metric

```bash
$ python analyze_results.py --results results.txt --val-set test2 --class car --metric map50
```

### Output:

```
================================================================================
Analysis Results
================================================================================
Results file: results.txt
Validation set: test2
Class: car
Metric: map50

🏆 Best Epoch: 18
   Best map50: 0.723400

   Detailed Metrics:
   - Precision: 0.789500
   - Recall: 0.656700
   - mAP@.5: 0.723400
   - mAP@.5:.95: 0.567800
   - Images: 68
   - Fitness: 0.583360

📊 Top 5 Epochs:
   1. Epoch  18: map50 = 0.723400
   2. Epoch  17: map50 = 0.689000
   3. Epoch  19: map50 = 0.678900
   4. Epoch  16: map50 = 0.645600
   5. Epoch  15: map50 = 0.623400

================================================================================
📈 Best Model Performance (Epoch 18) Across All Validation Sets
================================================================================

🔹 Combined:
   Overall: P=0.7123, R=0.5989, mAP@.5=0.6456, mAP@.5:.95=0.4789, fitness=0.4955
   Per-class:
     • Drum           : P=0.5567, R=0.4890, mAP@.5=0.5234, mAP@.5:.95=0.3678, fitness=0.3833 (images=78)
     • Helmet         : P=0.6456, R=0.5123, mAP@.5=0.5890, mAP@.5:.95=0.4345, fitness=0.4500 (images=69)
     • car            : P=0.7895, R=0.6567, mAP@.5=0.7234, mAP@.5:.95=0.5678, fitness=0.5834 (images=68)
     • person         : P=0.8234, R=0.7345, mAP@.5=0.7890, mAP@.5:.95=0.5890, fitness=0.6090 (images=236)

🔹 test1:
   Overall: P=0.6345, R=0.5456, mAP@.5=0.5890, mAP@.5:.95=0.4234, fitness=0.4400
   Per-class:
     • Drum           : P=0.5567, R=0.4890, mAP@.5=0.5234, mAP@.5:.95=0.3678, fitness=0.3833 (images=78)
     • Helmet         : P=0.6456, R=0.5123, mAP@.5=0.5890, mAP@.5:.95=0.4345, fitness=0.4500 (images=69)
     • person         : P=0.7012, R=0.6345, mAP@.5=0.6890, mAP@.5:.95=0.4679, fitness=0.4900 (images=67)

🔹 test2:
   Overall: P=0.7901, R=0.6522, mAP@.5=0.7022, mAP@.5:.95=0.5344, fitness=0.5512
   Per-class:
     • car            : P=0.7895, R=0.6567, mAP@.5=0.7234, mAP@.5:.95=0.5678, fitness=0.5834 (images=68)
     • person         : P=0.9456, R=0.8101, mAP@.5=0.8890, mAP@.5:.95=0.6101, fitness=0.6380 (images=169)

================================================================================
📊 Selected Validation Set Statistics: test2
================================================================================

Total epochs analyzed: 20
Class: car

Metric Statistics:
Metric                 Min        Max       Mean   Best Epoch
------------------------------------------------------------
P                   0.5234     0.7895     0.6567           18
R                   0.4123     0.6567     0.5345           18
mAP@.5              0.4567     0.7234     0.5890           18
mAP@.5:.95          0.3234     0.5678     0.4456           18
fitness             0.3345     0.5834     0.4567           18

================================================================================
```

---

## 핵심 개선사항

### 1. 📈 Best Model Performance Across All Validation Sets

Best epoch에서 모든 validation set의 성능을 한눈에 확인:
- ✅ Overall 메트릭 (P, R, mAP@.5, mAP@.5:.95, fitness)
- ✅ Per-class 메트릭 (각 클래스별 상세 정보)
- ✅ 이미지 수 정보

**장점:**
- Best epoch 선정이 다른 validation set에 미치는 영향 파악
- Combined로 best epoch를 찾았을 때, 실제 test1, test2 성능 확인
- 클래스별 성능 차이 한눈에 비교

### 2. 📊 Selected Validation Set Statistics

선택한 validation set의 전체 학습 과정 통계:
- ✅ 분석된 총 epoch 수
- ✅ 각 메트릭별 Min, Max, Mean 값
- ✅ 각 메트릭별 best epoch 정보

**장점:**
- 학습 안정성 확인 (Min-Max 범위)
- 평균 대비 best epoch 성능 비교
- 다른 메트릭의 best epoch 확인 (예: fitness best vs mAP@.5 best)

---

## 실제 사용 시나리오

### 시나리오 1: Combined로 best epoch 찾고 개별 성능 확인

```bash
python analyze_results.py --results results.txt --val-set Combined --class all --metric fitness
```

**확인 가능한 정보:**
- Combined 기준 best epoch: 17
- 해당 epoch에서 test1 성능: fitness=0.4370
- 해당 epoch에서 test2 성능: fitness=0.5187
- 각 클래스별 성능 차이

**판단:**
- test2가 test1보다 성능이 좋음
- person 클래스가 다른 클래스보다 성능이 좋음
- Combined 평균으로 선택하는 것이 적절한지 판단 가능

### 시나리오 2: 특정 클래스 성능 최적화

```bash
python analyze_results.py --results results.txt --val-set test2 --class person --metric fitness
```

**확인 가능한 정보:**
- person 클래스 best epoch 정보
- 해당 epoch에서 다른 validation set의 person 성능
- person 클래스의 전체 학습 과정 통계

**판단:**
- person 클래스만 고려하면 다른 epoch가 더 좋을 수 있음
- Overall best와 class-specific best가 다를 수 있음

### 시나리오 3: 학습 안정성 확인

**Statistics 섹션 활용:**
```
Metric Statistics:
Metric                 Min        Max       Mean   Best Epoch
------------------------------------------------------------
fitness             0.2567     0.5708     0.3945           17
```

**분석:**
- Min-Max 차이가 크면: 학습이 불안정하거나 초반 성능이 낮음
- Mean과 Best의 차이: Improvement 정도 확인
- 다른 메트릭의 Best Epoch 비교: 메트릭별 최적점 확인

---

## 요약

**새로운 정보:**
1. ✅ Best epoch에서 모든 validation set 성능 (Overall + Per-class)
2. ✅ 선택한 validation set의 통계 (Min, Max, Mean, Best Epoch for each metric)
3. ✅ 더 포괄적인 성능 분석

**사용 효과:**
- 더 정확한 모델 선택 가능
- Validation set 간 성능 차이 파악
- 클래스별 성능 비교
- 학습 안정성 확인
