# YOLOv7 Custom Features

이 문서는 YOLOv7에 추가된 커스텀 기능들을 설명합니다.

## 1. Multiple Validation Sets Support

여러 validation set을 동시에 평가하고 결과를 확인할 수 있습니다.

### 설정 방법

`data/custom.yaml` 파일에서 validation set을 리스트로 지정:

```yaml
val:
  - path: ../datasets/test1/images
    name: test1
  - path: ../datasets/test2/images
    name: test2
```

### 결과 출력 형식

**콘솔 및 results.txt:**
```
============================================================
Evaluating on test1
============================================================
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95
                 all         583         231       0.78       0.756       0.774       0.511
              person         583          67      0.858       0.507       0.655       0.434

============================================================
Evaluating on test2
============================================================
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95
                 all         895         265       0.61        0.64       0.713       0.497
              person         895         169      0.883       0.828       0.866       0.609

============================================================
Combined Results (Average)
============================================================
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95
                 all            -           -      0.695       0.698       0.744       0.504
              person            -         236      0.871       0.668       0.761       0.522
```

**results.txt 파일:**
```
  0/29     38.1G  0.008147  0.001051  0.001665   0.01086         0       640
  [      test1]      0.5133     0.2468     0.2197      0.117    0.02265   0.007445   0.004805
    [      test1][         person] Images:    67, P:     0.263, R:    0.0299, mAP@.5:      0.013, mAP@.5:.95:    0.00246
    [      test1][         Helmet] Images:    69, P:     0.863, R:      0.101, mAP@.5:      0.118, mAP@.5:.95:     0.0657
  [      test2]      0.6458     0.4678     0.5002      0.334    0.01164    0.00349   0.002329
    [      test2][         person] Images:   169, P:     0.841, R:      0.663, mAP@.5:      0.715, mAP@.5:.95:      0.491
  [    Combined]      0.5796     0.3573     0.3599     0.2255    0.01715   0.005468   0.003567
    [    Combined][         person] Images:   236, P:     0.552, R:      0.346, mAP@.5:      0.364, mAP@.5:.95:      0.247
```

## 2. Best Model Selection (--best-val-set)

학습 시 어떤 validation set을 기준으로 best 모델을 선택할지 지정할 수 있습니다.

### 사용 방법

```bash
# 첫 번째 validation set 사용 (기본값)
python train.py --best-val-set first

# 마지막 validation set 사용
python train.py --best-val-set last

# Combined (모든 validation set의 평균) 사용
python train.py --best-val-set Combined

# 특정 validation set 사용
python train.py --best-val-set test1
python train.py --best-val-set test2
```

### Fitness 계산 공식

Best 모델은 다음 fitness 값이 가장 높은 epoch가 선택됩니다:

```
fitness = 0.0 × P + 0.0 × R + 0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95
```

- **P**: Precision
- **R**: Recall
- **mAP@0.5**: mAP at IoU threshold 0.5
- **mAP@0.5:0.95**: mAP averaged over IoU thresholds 0.5 to 0.95

### 예제

```bash
# test2 데이터셋을 기준으로 best 모델 선택
python -m torch.distributed.launch \
    --nproc_per_node 8 \
    train.py \
    --data data/custom.yaml \
    --weights yolov7.pt \
    --epochs 300 \
    --batch-size 128 \
    --best-val-set test2
```

## 3. Result Analysis Tool (analyze_results.py)

학습 완료 후 results.txt 파일을 분석하여 최적의 epoch를 찾는 도구입니다.

### 기본 사용법

```bash
# 기본 분석 (첫 번째 validation set, 전체 클래스, fitness 기준)
python analyze_results.py --results runs/train/exp/results.txt

# 특정 validation set 분석
python analyze_results.py --results runs/train/exp/results.txt --val-set test1
python analyze_results.py --results runs/train/exp/results.txt --val-set test2
python analyze_results.py --results runs/train/exp/results.txt --val-set Combined

# 특정 클래스 분석
python analyze_results.py --results runs/train/exp/results.txt --val-set test1 --class person
python analyze_results.py --results runs/train/exp/results.txt --val-set test2 --class car

# 특정 메트릭 기준으로 분석
python analyze_results.py --results runs/train/exp/results.txt --metric map50
python analyze_results.py --results runs/train/exp/results.txt --metric map
python analyze_results.py --results runs/train/exp/results.txt --metric precision
python analyze_results.py --results runs/train/exp/results.txt --metric recall
```

### 사용 가능한 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--results` | results.txt 파일 경로 | `runs/train/exp/results.txt` |
| `--val-set` | 분석할 validation set (test1, test2, Combined 등) | `test1` |
| `--class` | 분석할 클래스명 또는 'all' | `all` |
| `--metric` | 최적화할 메트릭 (fitness, map50, map, precision, recall) | `fitness` |
| `--plot` | 그래프 생성 (matplotlib 필요) | False |
| `--list` | 사용 가능한 validation set 및 클래스 목록 출력 | False |

### 출력 예제

```bash
$ python analyze_results.py --results runs/train/exp/results.txt --val-set Combined --class person

================================================================================
Analysis Results
================================================================================
Results file: runs/train/exp/results.txt
Validation set: Combined
Class: person
Metric: fitness

🏆 Best Epoch: 245
   Best fitness: 0.523456

   Detailed Metrics:
   - Precision: 0.871234
   - Recall: 0.667890
   - mAP@.5: 0.761234
   - mAP@.5:.95: 0.521890
   - Images: 236
   - Fitness: 0.523456

📊 Top 5 Epochs:
   1. Epoch 245: fitness = 0.523456
   2. Epoch 243: fitness = 0.521234
   3. Epoch 248: fitness = 0.519876
   4. Epoch 241: fitness = 0.518234
   5. Epoch 239: fitness = 0.516789
================================================================================
```

### 그래프 생성

`--plot` 옵션을 사용하면 메트릭 변화 그래프를 PNG 파일로 저장합니다 (matplotlib 필요):

```bash
python analyze_results.py \
    --results runs/train/exp/results.txt \
    --val-set Combined \
    --class person \
    --metric fitness \
    --plot
```

출력: `analysis_Combined_person_fitness.png`

### 사용 가능한 validation set 및 클래스 확인

```bash
python analyze_results.py --results runs/train/exp/results.txt --list
```

출력:
```
================================================================================
Available Validation Sets and Classes
================================================================================

📊 test1:
   - all (overall metrics)
   - Drum
   - Helmet
   - Sitting
   - person

📊 test2:
   - all (overall metrics)
   - car
   - person
   - traffic light

📊 Combined:
   - all (overall metrics)
   - Drum
   - Helmet
   - Sitting
   - car
   - person
   - traffic light
================================================================================
```

## 4. 워크플로우 예제

### 학습부터 분석까지 전체 과정

```bash
# 1. 학습 시작 (Combined 결과로 best 모델 선택)
python -m torch.distributed.launch \
    --nproc_per_node 8 \
    train.py \
    --data data/custom.yaml \
    --weights yolov7.pt \
    --epochs 300 \
    --batch-size 128 \
    --best-val-set Combined

# 2. 학습 완료 후 사용 가능한 데이터 확인
python analyze_results.py \
    --results runs/train/exp/results.txt \
    --list

# 3. test1에서 person 클래스의 best epoch 찾기
python analyze_results.py \
    --results runs/train/exp/results.txt \
    --val-set test1 \
    --class person \
    --metric map50

# 4. Combined에서 전체 클래스의 best epoch 찾기 (그래프 포함)
python analyze_results.py \
    --results runs/train/exp/results.txt \
    --val-set Combined \
    --class all \
    --metric fitness \
    --plot

# 5. 특정 epoch의 weight 사용
# Best epoch가 245라면:
python test.py \
    --data data/custom.yaml \
    --weights runs/train/exp/weights/epoch_245.pt \
    --batch-size 32
```

## 5. 팁

### Best 모델 선택 전략

- **단일 타겟 데이터셋**: `--best-val-set test1` 처럼 메인 타겟 데이터셋 지정
- **여러 데이터셋에서 균형잡힌 성능**: `--best-val-set Combined` 사용
- **특정 클래스 중심**: analyze_results.py로 해당 클래스의 best epoch 찾기

### 결과 파일 관리

- `results.txt`: 모든 epoch의 상세 결과
- `results.png`: 학습 곡선 (자동 생성)
- `weights/best.pt`: --best-val-set 기준 best 모델
- `weights/last.pt`: 마지막 epoch 모델
- `weights/epoch_*.pt`: 각 epoch별 체크포인트 (--save_period 사용 시)

### 디버깅

특정 validation set을 찾을 수 없다는 경고가 나오면:
```bash
# 사용 가능한 validation set 확인
python analyze_results.py --results runs/train/exp/results.txt --list

# data.yaml의 val 설정 확인
cat data/custom.yaml
```

## 6. 문제 해결

### TypeError: isinstance() arg 2 must be a type or tuple of types

이 에러가 발생하면 `utils/general.py`의 `check_dataset` 함수 파라미터가 `dict`로 되어 있는지 확인하세요. `data_dict`로 변경되어야 합니다.

### Combined 결과가 표시되지 않음

Multiple validation sets를 사용할 때만 Combined 결과가 표시됩니다. 단일 validation set 사용 시에는 표시되지 않습니다.

### Best 모델이 예상과 다름

`--best-val-set` 옵션을 확인하세요. 기본값은 `first`이므로 첫 번째 validation set을 사용합니다.
