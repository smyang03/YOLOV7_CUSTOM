# YOLOv7용 WiSE-FT Sweep 도구

**자동화된 가중치 공간 앙상블 미세 조정 최적화**

스크래치 및 미세 조정된 YOLOv7 모델 간의 최적 혼합 비율을 찾아 타겟 클래스 성능 향상과 일반 성능 유지 간의 균형을 맞춥니다.

---

## 🎯 문제 정의

특정 데이터로 YOLOv7을 미세 조정할 때:

### 시나리오 A: 타겟 클래스만으로 미세 조정
```
✅ 타겟 클래스 (예: 사람): 60% → 85% (+25% 향상)
❌ 다른 클래스 (예: 자동차, 개): 70% → 45% (-25% 치명적 망각)
```

### 시나리오 B: 혼합 데이터로 미세 조정
```
⚠️ 타겟 클래스: 60% → 68% (+8% 제한적 향상)
✅ 다른 클래스: 70% → 68% (-2% 유지됨)
```

### ✨ WiSE-FT 솔루션: 두 가지 장점 모두 활용
```
🎉 타겟 클래스: 60% → 72% (+12% 우수한 향상)
🎉 다른 클래스: 70% → 66% (-4% 허용 가능한 트레이드오프)

공식: merged = 85% scratch + 15% finetuned (α=0.15)
```

---

## 🚀 빠른 시작

### 설치
```bash
pip install torch torchvision numpy pyyaml
```

### 기본 사용법
```bash
python wiseft_sweep.py \
    --scratch runs/exp_scratch/weights/best.pt \
    --finetuned runs/exp_finetuned/weights/best.pt \
    --data data/custom.yaml
```

### 예상 출력
```
================================================================================
🎯 WISEFT SWEEP 실행 요약
================================================================================

✅ 권장 ALPHA: 0.150

성능 비교:
  스크래치 기준선:   0.650
  미세조정 기준선: 0.520
  최적 병합 (α=0.15): 0.690

  스크래치 대비 향상:   +6.15%
  미세조정 대비 향상: +32.69%

💡 해석:
  Alpha = 0.15는 다음을 의미합니다: 85% scratch + 15% finetuned

  이 최적 혼합 비율은 다음 간의 최상의 균형을 달성합니다:
  - 일반 객체 탐지 능력 보존 (스크래치 모델로부터)
  - 미세 조정 개선 사항 활용 (미세조정 모델로부터)
================================================================================
```

---

## 📂 데이터셋 준비

WiSE-FT Sweep을 사용하기 전에 YOLOv7 형식의 데이터셋 YAML 파일이 필요합니다.

### data.yaml 파일 구조

```yaml
# 학습/검증/테스트 경로
train: ./data/coco/train2017.txt  # 학습 이미지 경로 리스트 파일
val: ./data/coco/val2017.txt      # 검증 이미지 경로 리스트 파일
test: ./data/coco/test2017.txt    # 테스트 이미지 (선택사항)

# 클래스 개수
nc: 80  # number of classes

# 클래스 이름 리스트 (순서대로 0, 1, 2, ...)
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']
```

### 커스텀 데이터셋 예제

**예제 1: 단일 클래스 (사람만)**
```yaml
train: ./data/person_only/train.txt
val: ./data/person_only/val.txt

nc: 1

names: ['person']
```

**예제 2: 다중 클래스 (자동차, 트럭, 버스)**
```yaml
train: ./data/vehicles/train.txt
val: ./data/vehicles/val.txt

nc: 3

names: ['car', 'truck', 'bus']
```

### 이미지 경로 리스트 파일 (.txt)

train.txt, val.txt 파일에는 각 줄에 이미지 경로를 하나씩 작성합니다:

```
/path/to/images/image1.jpg
/path/to/images/image2.jpg
/path/to/images/image3.jpg
```

### 라벨 파일 형식 (.txt)

각 이미지에 대응하는 라벨 파일이 필요합니다:

```
이미지: /path/to/images/image1.jpg
라벨:   /path/to/labels/image1.txt
```

라벨 파일 내용 (YOLO 형식):
```
<class_id> <x_center> <y_center> <width> <height>

예제:
0 0.5 0.5 0.3 0.4
2 0.2 0.3 0.1 0.15
```

- `class_id`: 클래스 인덱스 (0부터 시작, names 리스트의 인덱스)
- `x_center, y_center`: 바운딩 박스 중심 좌표 (0~1로 정규화)
- `width, height`: 바운딩 박스 너비와 높이 (0~1로 정규화)

### 디렉토리 구조

```
data/
├── custom.yaml                 # 데이터셋 설정 파일
├── train.txt                   # 학습 이미지 경로 리스트
├── val.txt                     # 검증 이미지 경로 리스트
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt         # YOLO 형식 라벨
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        └── ...
```

**참고**: 전체 예제는 [data_example.yaml](data_example.yaml)을 확인하세요.

---

## 📖 작동 방식

### 1단계: 가중치 변화 분석
```
스크래치 및 미세조정 모델 간의 레이어별 변화를 분석합니다:

레이어 그룹          평균 변화    해석
─────────────────────────────────────────────────
백본 (0-50)      3%            최소 변화
넥 (51-74)         12%           중간 변화
헤드 (75-105)        45%           유의미한 변화 ⚠️

💡 권장사항: α = 0.05-0.30
   이유: 헤드는 크게 변경되었지만 백본은 안정적으로 유지됨.
           낮은-중간 alpha를 권장합니다.
```

### 2단계: 거친 검색 (Coarse Search)
```
큰 간격(예: 0.1)으로 alpha를 테스트합니다:

Alpha    Precision    Recall    mAP@.5    mAP@.5:.95    Fitness
──────────────────────────────────────────────────────────────────
0.05     0.680        0.650     0.670     0.625         0.630
0.15     0.720        0.690     0.710     0.665         0.675  ← 최고
0.25     0.700        0.670     0.690     0.645         0.655
```

### 3단계: 세밀한 검색 (Fine Search)
```
최적의 거친 alpha(0.15) 주변을 세밀하게 검색합니다:

Alpha    Precision    Recall    mAP@.5    mAP@.5:.95    Fitness
──────────────────────────────────────────────────────────────────
0.10     0.705        0.675     0.695     0.650         0.660
0.125    0.715        0.685     0.705     0.660         0.670
0.15     0.720        0.690     0.710     0.665         0.675  ← 최고
0.175    0.710        0.680     0.700     0.655         0.665
0.20     0.700        0.670     0.690     0.645         0.655
```

### 4단계: 최적 모델 저장
```
✅ 최적 alpha: 0.15
✅ 병합된 모델 저장 위치: runs/wiseft/exp/best_merged.pt
✅ 전체 보고서: runs/wiseft/exp/wiseft_report.md
```

---

## 🔧 고급 사용법

### Alpha 범위 사용자 정의
```bash
python wiseft_sweep.py \
    --scratch runs/exp1/weights/best.pt \
    --finetuned runs/exp2/weights/best.pt \
    --data data/custom.yaml \
    --alpha-min 0.1 \
    --alpha-max 0.5 \
    --focus-range 0.05  # 더 세밀한 간격
```

### 특정 클래스 타겟팅
```bash
python wiseft_sweep.py \
    --scratch runs/exp1/weights/best.pt \
    --finetuned runs/exp2/weights/best.pt \
    --data data/custom.yaml \
    --target-class person  # 'person' 클래스에 최적화
```

### 다중 검증 세트
```bash
python wiseft_sweep.py \
    --scratch runs/exp1/weights/best.pt \
    --finetuned runs/exp2/weights/best.pt \
    --data data/custom.yaml \
    --val-sets test1 test2 Combined  # 여러 세트에서 테스트
```

### 다른 메트릭 최적화
```bash
python wiseft_sweep.py \
    --scratch runs/exp1/weights/best.pt \
    --finetuned runs/exp2/weights/best.pt \
    --data data/custom.yaml \
    --metric map50  # fitness 대신 mAP@.5 최적화
```

### 조기 종료 활성화
```bash
python wiseft_sweep.py \
    --scratch runs/exp1/weights/best.pt \
    --finetuned runs/exp2/weights/best.pt \
    --data data/custom.yaml \
    --early-stop \
    --stop-threshold 0.05 \
    --stop-patience 3
```

---

## 📊 명령줄 인수

### 필수
| 인수 | 타입 | 설명 |
|----------|------|-------------|
| `--scratch` | str | 스크래치 학습 모델 경로 |
| `--finetuned` | str | 미세조정 모델 경로 |
| `--data` | str | 데이터셋 YAML 파일 |

### Alpha 구성
| 인수 | 타입 | 기본값 | 설명 |
|----------|------|---------|-------------|
| `--focus-range` | float | 0.1 | 거친 검색의 alpha 간격 |
| `--alpha-min` | float | auto | 최소 alpha (가중치 분석에서 자동 감지) |
| `--alpha-max` | float | 1.0 | 최대 alpha |
| `--skip-zero` | flag | True | alpha=0.0 (스크래치 모델) 건너뛰기 |

### 검색 전략
| 인수 | 타입 | 기본값 | 설명 |
|----------|------|---------|-------------|
| `--enable-fine-search` | flag | True | 2단계 검색 활성화 |
| `--fine-range` | float | focus-range/2 | 세밀한 검색 간격 |
| `--fine-window` | float | 2*focus-range | 세밀한 검색 윈도우 크기 |

### 평가
| 인수 | 타입 | 기본값 | 설명 |
|----------|------|---------|-------------|
| `--metric` | str | fitness | 최적화할 메트릭 (fitness, map50, map, precision, recall) |
| `--val-sets` | list | ['test1'] | 검증 세트 |
| `--target-class` | str | None | 타겟 클래스 이름 또는 인덱스 |
| `--img-size` | int | 640 | 이미지 크기 |
| `--batch-size` | int | 32 | 배치 크기 |
| `--conf-thres` | float | 0.001 | 신뢰도 임계값 |
| `--iou-thres` | float | 0.6 | IoU 임계값 |

### 조기 종료
| 인수 | 타입 | 기본값 | 설명 |
|----------|------|---------|-------------|
| `--early-stop` | flag | False | 조기 종료 활성화 |
| `--stop-threshold` | float | 0.05 | 성능 하락 임계값 |
| `--stop-patience` | int | 3 | 종료 전 연속 하락 횟수 |

### 출력
| 인수 | 타입 | 기본값 | 설명 |
|----------|------|---------|-------------|
| `--output-dir` | str | runs/wiseft | 출력 디렉토리 |
| `--save-best-only` | flag | True | 최적 모델만 저장 |
| `--save-merged-models` | flag | False | 모든 병합 모델 저장 |
| `--report-format` | str | markdown | 보고서 형식 (markdown, text, json) |

---

## 📁 출력 파일

```
runs/wiseft/exp/
├── best_merged.pt              # ⭐ 최적 alpha 병합 모델 (이것을 사용하세요!)
├── wiseft_report.md            # 📄 전체 상세 보고서
├── results.json                # 📊 JSON 형식의 모든 결과
└── temp/                       # 임시 파일 (삭제 가능)
    ├── alpha_0.10.pt
    ├── alpha_0.15.pt
    └── ...
```

---

## 🎓 Alpha 이해하기

### Alpha (α)란 무엇인가?

Alpha는 스크래치 및 미세조정 모델 간의 혼합 비율을 제어합니다:

```python
merged_weight = (1 - α) * scratch_weight + α * finetuned_weight
```

### Alpha 값 설명

| Alpha | 의미 | 사용 시점 |
|-------|---------|-------------|
| 0.0 | 100% 스크래치 | 절대 안됨 (그냥 스크래치 모델 사용) |
| 0.1 | 90% 스크래치 + 10% 미세조정 | 심각한 치명적 망각 감지됨 |
| 0.2 | 80% 스크래치 + 20% 미세조정 | 이상적인 미세 조정 (일반적으로 최고) |
| 0.5 | 50% 스크래치 + 50% 미세조정 | 균형 잡힌 접근 |
| 0.8 | 20% 스크래치 + 80% 미세조정 | 전체 모델이 잘 미세조정됨 |
| 1.0 | 100% 미세조정 | 절대 안됨 (그냥 미세조정 모델 사용) |

### 시나리오별 권장 범위

**시나리오 1: 헤드 중심 미세 조정 (이상적)**
```
가중치 변화: 백본 3%, 헤드 45%
권장 α: 0.05 - 0.30
이유: 일반 특징 보존, 일부 타겟 개선 사항 채택
```

**시나리오 2: 과적합 (치명적 망각)**
```
가중치 변화: 백본 18%, 헤드 85%
권장 α: 0.0 - 0.20
이유: 치명적 망각 방지
```

**시나리오 3: 전체 모델 미세 조정**
```
가중치 변화: 백본 15%, 헤드 25%
권장 α: 0.2 - 0.6
이유: 전체 미세 조정 이점 활용
```

---

## 🧪 테스트 및 검증

단위 테스트 실행:
```bash
python test_wiseft_simple.py
```

예상 출력:
```
✅ 모든 테스트 통과 (10/10)
✅ 워크플로우 시뮬레이션 성공적으로 완료!
```

자세한 테스트 결과는 [WISEFT_TEST_REPORT.md](WISEFT_TEST_REPORT.md)를 참조하세요.

---

## 💡 실제 예제

### 문제
```yaml
데이터셋: COCO (80개 클래스) + 커스텀 사람 데이터셋

스크래치 모델 (COCO로 학습):
  전체 mAP@.5: 0.65
  사람: 0.60
  자동차: 0.70
  개: 0.68

미세조정 모델 (사람만으로 학습):
  전체 mAP@.5: 0.52  ⚠️ 하락!
  사람: 0.85  ✅ 향상!
  자동차: 0.45  ❌ 치명적 망각!
  개: 0.42  ❌ 치명적 망각!
```

### WiSE-FT 솔루션
```bash
python wiseft_sweep.py \
    --scratch runs/coco/weights/best.pt \
    --finetuned runs/person_finetuned/weights/best.pt \
    --data data/coco_custom.yaml \
    --target-class person
```

### 결과
```yaml
최적 Alpha: 0.15 (85% 스크래치 + 15% 미세조정)

병합 모델:
  전체 mAP@.5: 0.69  ✅ 스크래치 대비 +6%
  사람: 0.72  ✅ 스크래치 대비 +20%
  자동차: 0.67  ✅ -4%만 (WiSE-FT 미사용 시 -36%)
  개: 0.66  ✅ -3%만 (WiSE-FT 미사용 시 -38%)

트레이드오프: 타겟 클래스 +20%, 기타 클래스 -3~4%만
```

---

## ❓ FAQ

### Q1: 얼마나 걸리나요?
**A**: 데이터셋 크기와 GPU에 따라 다릅니다:
- 소형 데이터셋 (1000개 이미지): ~30분
- 중형 데이터셋 (5000개 이미지): ~1시간
- 대형 데이터셋 (10000개 이상 이미지): ~2-3시간

### Q2: 중간에 멈추고 다시 시작할 수 있나요?
**A**: 현재는 불가능합니다. 필요한 경우 results.json을 저장하고 나중에 분석하세요.

### Q3: 모든 alpha가 비슷하게 수행된다면?
**A**: 이것은 다음을 의미합니다:
1. 미세 조정이 가중치를 많이 변경하지 않았거나,
2. 모델이 매우 유사하거나,
3. 메트릭이 충분히 민감하지 않음

시도: 보고서에서 가중치 분석 확인, 또는 다른 메트릭 사용 (--metric map50)

### Q4: YOLOv5/v8에서 사용할 수 있나요?
**A**: 개념은 동일하지만 다음을 조정해야 할 수 있습니다:
- 백본/넥/헤드에 대한 레이어 인덱스
- 체크포인트 구조
- test.py 호출

### Q5: 다중 클래스 미세 조정에도 작동하나요?
**A**: 예! 여러 --target-class 값을 지정하거나 전체 메트릭을 사용하세요.

### Q6: --skip-zero를 사용해야 하나요?
**A**: 예 (기본값). Alpha=0.0은 그냥 스크래치 모델입니다 - 테스트할 필요가 없습니다.

---

## 🔗 관련 개념

### WiSE-FT 논문
- **제목**: "Robust fine-tuning of zero-shot models"
- **저자**: Mitchell Wortsman et al.
- **학회**: CVPR 2022
- **핵심 아이디어**: 사전 학습 및 미세조정 모델 간의 선형 보간

### 관련 기법
- **Model Soup**: 여러 미세조정 모델 평균화
- **EWC (Elastic Weight Consolidation)**: 망각 방지를 위한 정규화
- **Progressive Neural Networks**: 새로운 작업을 위한 새로운 용량 추가

---

## 🛠️ 문제 해결

### 문제: "RuntimeError: CUDA out of memory"
**해결책**: --batch-size 줄이기:
```bash
python wiseft_sweep.py ... --batch-size 16
```

### 문제: "test.py not found"
**해결책**: YOLOv7 디렉토리에 있는지 확인:
```bash
cd /path/to/yolov7
python wiseft_sweep.py ...
```

### 문제: "모든 alpha가 동일한 성능"
**해결책**:
1. 가중치 분석 확인 - 모델이 실제로 다른가요?
2. 다른 메트릭 시도: --metric map50
3. 더 세밀한 검색을 위해 focus-range 증가: --focus-range 0.05

### 문제: "Alpha 권장사항이 잘못된 것 같음"
**해결책**: 수동으로 재정의:
```bash
python wiseft_sweep.py ... --alpha-min 0.1 --alpha-max 0.5
```

---

## 📚 추가 리소스

- [WISEFT_TEST_REPORT_KR.md](WISEFT_TEST_REPORT_KR.md) - 자세한 테스트 결과
- [PHASE2_PHASE3_FEATURES_KR.md](PHASE2_PHASE3_FEATURES_KR.md) - 고급 기능 가이드
- [wiseft_sweep.py](wiseft_sweep.py) - 소스 코드
- [test_wiseft_simple.py](test_wiseft_simple.py) - 단위 테스트

---

## 📝 인용

연구에서 이 도구를 사용하는 경우 다음을 인용해 주세요:

```bibtex
@misc{wiseft_yolov7,
  title={WiSE-FT Sweep Tool for YOLOv7},
  author={Your Name},
  year={2024},
  howpublished={\\url{https://github.com/yourusername/yolov7_wiseft}}
}
```

그리고 원본 WiSE-FT 논문:
```bibtex
@inproceedings{wortsman2022robust,
  title={Robust fine-tuning of zero-shot models},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Kim, Jong Wook and Li, Mike and Kornblith, Simon and Roelofs, Rebecca and Lopes, Raphael Gontijo and Hajishirzi, Hannaneh and Farhadi, Ali and Namkoong, Hongseok and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7959--7971},
  year={2022}
}
```

---

## 📜 라이선스

MIT License - 자유롭게 사용 및 수정하세요!

---

## 🤝 기여

기여를 환영합니다! 다음 사항을 준수해 주세요:
1. 저장소 포크
2. 기능 브랜치 생성
3. 새 기능에 대한 테스트 추가
4. Pull request 제출

---

**상태**: ✅ 프로덕션 준비 완료
**버전**: 1.0.0 MVP
**최종 업데이트**: 2025-11-15

---

*즐거운 WiSE-FT-ing! 🎉*
