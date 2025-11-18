# data.yaml 파일 구조 가이드

YOLOv7 및 WiSE-FT 도구에서 사용하는 데이터셋 설정 파일 구조 설명

---

## 📋 기본 구조

### 필수 항목

```yaml
# 학습 데이터 경로
train: <경로>

# 검증 데이터 경로
val: <경로>

# 클래스 수
nc: <숫자>

# 클래스 이름 리스트
names: [<클래스1>, <클래스2>, ...]
```

### 선택 항목

```yaml
# 데이터 다운로드 명령
download: <명령어>

# 테스트 데이터 경로
test: <경로>
```

---

## 📂 경로 지정 방법

데이터 경로는 3가지 방법으로 지정할 수 있습니다:

### 1. 디렉토리 경로
```yaml
train: ./data/images/train/
val: ./data/images/val/
```

### 2. 텍스트 파일 (권장)
```yaml
train: ./data/train.txt
val: ./data/val.txt
```

**train.txt 예시:**
```
./data/images/train/img1.jpg
./data/images/train/img2.jpg
./data/images/train/img3.jpg
```

### 3. 리스트 (다중 경로)
```yaml
train: [./data/train1/, ./data/train2/]
val: [./data/val1/, ./data/val2/]
```

---

## 🔢 검증 세트 2개 사용 (다중 검증)

### 방법 1: YAML 리스트 형식
```yaml
train: ./data/train.txt

val:
  - ./data/val_set1.txt
  - ./data/val_set2.txt

nc: 3
names: ['person', 'car', 'dog']
```

### 방법 2: 한 줄 리스트 형식
```yaml
train: ./data/train.txt
val: [./data/val_set1.txt, ./data/val_set2.txt]
nc: 3
names: ['person', 'car', 'dog']
```

---

## 📝 실제 예시

### 예시 1: 기본 커스텀 데이터셋

```yaml
# custom.yaml
train: ./datasets/my_data/train.txt
val: ./datasets/my_data/val.txt
test: ./datasets/my_data/test.txt

nc: 3
names: ['person', 'car', 'dog']
```

### 예시 2: COCO 데이터셋

```yaml
# coco.yaml
download: bash ./scripts/get_coco.sh

train: ./coco/train2017.txt
val: ./coco/val2017.txt
test: ./coco/test-dev2017.txt

nc: 80
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', ...]
```

### 예시 3: 다중 검증 세트 (실내/실외)

```yaml
# indoor_outdoor.yaml
train: ./data/train_all.txt

val:
  - ./data/val_indoor.txt   # 실내 환경
  - ./data/val_outdoor.txt  # 실외 환경

nc: 5
names: ['person', 'car', 'bicycle', 'dog', 'cat']
```

### 예시 4: COCO + 커스텀 Person 데이터셋 (WiSE-FT 예시)

```yaml
# coco_person.yaml
train: ./data/coco_person/train.txt

# 2개 검증 세트로 성능 측정
val:
  - ./data/coco_person/val_coco.txt     # 일반 객체 탐지 성능
  - ./data/coco_person/val_person.txt   # Person 타겟 성능

nc: 80
names: ['person', 'bicycle', 'car', ...]
```

**WiSE-FT 사용 시:**
```bash
python wiseft_sweep.py \
    --scratch runs/coco/weights/best.pt \
    --finetuned runs/person_ft/weights/best.pt \
    --data data/coco_person.yaml \
    --target-class person
```

---

## 🎯 WiSE-FT에서 다중 검증 세트 활용

### 사용 목적

다중 검증 세트를 사용하면 다음을 동시에 측정할 수 있습니다:

1. **일반 성능**: 원본 데이터셋에서의 성능 유지 확인
2. **타겟 성능**: 미세조정 목표 데이터에서의 성능 개선 확인

### 평가 방식

WiSE-FT는 각 검증 세트에 대해:
- 개별적으로 성능 평가
- 전체 메트릭 계산
- 각 세트별 클래스 성능 추적

### 예시 출력

```
Alpha: 0.15

검증 세트: val_coco.txt
  Precision: 0.72
  Recall: 0.68
  mAP@.5: 0.70
  Person: 0.68, Car: 0.72, Dog: 0.70

검증 세트: val_person.txt
  Precision: 0.85
  Recall: 0.82
  mAP@.5: 0.84
  Person: 0.88, Car: 0.80, Dog: 0.82

전체 평균 Fitness: 0.675
```

---

## 📌 주의사항

### 1. 경로 형식
- **절대 경로** 또는 **상대 경로** 사용 가능
- 상대 경로는 스크립트 실행 위치 기준
- 경로 구분자: Linux/Mac `/`, Windows `\` 또는 `/`

### 2. 텍스트 파일 형식
각 줄에 이미지 경로 하나씩:
```
./images/img1.jpg
./images/img2.jpg
./images/img3.jpg
```

**주의:** 라벨 파일(.txt)은 자동으로 찾습니다
- 이미지: `./images/img1.jpg`
- 라벨: `./labels/img1.txt` (자동 매핑)

### 3. 클래스 인덱스
- `names` 리스트의 **인덱스 순서**가 중요
- 라벨 파일의 클래스 번호와 일치해야 함

```yaml
names: ['person', 'car', 'dog']
# person=0, car=1, dog=2
```

라벨 파일 예시:
```
0 0.5 0.5 0.3 0.4  # person
1 0.7 0.3 0.2 0.3  # car
```

---

## 🔧 데이터셋 준비 체크리스트

- [ ] 이미지 파일 준비 (.jpg, .png 등)
- [ ] 라벨 파일 준비 (.txt, YOLO 형식)
- [ ] 디렉토리 구조 설정
  ```
  data/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/
      ├── val/
      └── test/
  ```
- [ ] 경로 리스트 파일 생성 (.txt)
- [ ] data.yaml 파일 작성
- [ ] 클래스 수 및 이름 확인
- [ ] 경로 테스트 (파일 존재 여부 확인)

---

## 🛠️ 유용한 스크립트

### 경로 리스트 파일 자동 생성

```bash
# Linux/Mac
find ./data/images/train -name "*.jpg" > ./data/train.txt
find ./data/images/val -name "*.jpg" > ./data/val.txt

# 또는 Python
python << EOF
import os
from pathlib import Path

# train.txt 생성
train_dir = Path('./data/images/train')
with open('./data/train.txt', 'w') as f:
    for img in train_dir.glob('*.jpg'):
        f.write(f'{img}\n')

# val.txt 생성
val_dir = Path('./data/images/val')
with open('./data/val.txt', 'w') as f:
    for img in val_dir.glob('*.jpg'):
        f.write(f'{img}\n')
EOF
```

---

## 📚 참고 자료

- [YOLOv7 공식 문서](https://github.com/WongKinYiu/yolov7)
- [WISEFT_README_KR.md](WISEFT_README_KR.md) - WiSE-FT 도구 가이드
- [COCO 데이터셋](http://cocodataset.org)

---

**작성일**: 2025-11-17
**버전**: 1.0
