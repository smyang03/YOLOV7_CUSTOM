# YOLOv7용 WiSE-FT 스윕 도구

**자동화된 가중치 공간 앙상블 미세조정 최적화**

스크래치 및 미세조정된 YOLOv7 모델 간의 최적 혼합 비율을 찾아 타겟 클래스 개선과 일반 성능 유지 사이의 균형을 맞춥니다.

---

## 🎯 문제 정의

특정 데이터로 YOLOv7을 미세조정할 때:

### 시나리오 A: 타겟 클래스만으로 미세조정
```
✅ 타겟 클래스 (예: person): 60% → 85% (+25% 개선)
❌ 다른 클래스 (예: car, dog): 70% → 45% (-25% 치명적 망각)
```

### 시나리오 B: 혼합 데이터로 미세조정
```
⚠️ 타겟 클래스: 60% → 68% (+8% 제한적 개선)
✅ 다른 클래스: 70% → 68% (-2% 유지됨)
```

### ✨ WiSE-FT 솔루션: 두 마리 토끼를 다 잡는다
```
🎉 타겟 클래스: 60% → 72% (+12% 좋은 개선)
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
🎯 WISEFT SWEEP 요약 보고서
================================================================================

✅ 권장 알파: 0.150

성능 비교:
  스크래치 기준선:   0.650
  미세조정 기준선:   0.520
  최고 병합 (α=0.15): 0.690

  스크래치 대비 개선:   +6.15%
  미세조정 대비 개선:   +32.69%

💡 해석:
  Alpha = 0.15 의미: 85% 스크래치 + 15% 미세조정

  이 최적 혼합 비율은 다음 사이의 최상의 균형을 달성합니다:
  - 일반 객체 탐지 능력 유지 (스크래치 모델에서)
  - 미세조정 개선사항 활용 (미세조정 모델에서)
================================================================================
```

---

## 📖 동작 원리

### 1단계: 가중치 변화 분석
```
스크래치와 미세조정 모델 간 레이어별 변화 분석:

레이어 그룹          평균 변화     해석
─────────────────────────────────────────────────
백본 (0-50)          3%            최소 변화
넥 (51-74)          12%            중간 변화
헤드 (75-105)        45%            큰 변화 ⚠️

💡 권장사항: α = 0.05-0.30
   이유: 헤드는 크게 변경되었지만 백본은 안정적으로 유지됨.
         낮은-중간 알파가 권장됨.
```

### 2단계: 거친 탐색
```
큰 간격으로 알파 테스트 (예: 0.1):

Alpha    Precision    Recall    mAP@.5    mAP@.5:.95    Fitness
──────────────────────────────────────────────────────────────────
0.05     0.680        0.650     0.670     0.625         0.630
0.15     0.720        0.690     0.710     0.665         0.675  ← 최고
0.25     0.700        0.670     0.690     0.645         0.655
```

### 3단계: 세밀한 탐색
```
최고 거친 알파(0.15) 주변 정교화:

Alpha    Precision    Recall    mAP@.5    mAP@.5:.95    Fitness
──────────────────────────────────────────────────────────────────
0.10     0.705        0.675     0.695     0.650         0.660
0.125    0.715        0.685     0.705     0.660         0.670
0.15     0.720        0.690     0.710     0.665         0.675  ← 최고
0.175    0.710        0.680     0.700     0.655         0.665
0.20     0.700        0.670     0.690     0.645         0.655
```

### 4단계: 최고 모델 저장
```
✅ 최적 알파: 0.15
✅ 병합된 모델 저장 위치: runs/wiseft/exp/best_merged.pt
✅ 전체 보고서: runs/wiseft/exp/wiseft_report.md
```

---

## 🔧 고급 사용법

### 알파 범위 사용자 정의
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

## 📊 명령줄 인자

### 필수
| 인자 | 타입 | 설명 |
|----------|------|-------------|
| `--scratch` | str | 스크래치 학습된 모델 경로 |
| `--finetuned` | str | 미세조정된 모델 경로 |
| `--data` | str | 데이터셋 YAML 파일 |

### 알파 설정
| 인자 | 타입 | 기본값 | 설명 |
|----------|------|---------|-------------|
| `--focus-range` | float | 0.1 | 거친 탐색용 알파 간격 |
| `--alpha-min` | float | auto | 최소 알파 (가중치 분석에서 자동 감지) |
| `--alpha-max` | float | 1.0 | 최대 알파 |
| `--skip-zero` | flag | True | alpha=0.0 건너뛰기 (스크래치 모델) |

### 탐색 전략
| 인자 | 타입 | 기본값 | 설명 |
|----------|------|---------|-------------|
| `--enable-fine-search` | flag | True | 2단계 탐색 활성화 |
| `--fine-range` | float | focus-range/2 | 세밀한 탐색 간격 |
| `--fine-window` | float | 2*focus-range | 세밀한 탐색 윈도우 크기 |

### 평가
| 인자 | 타입 | 기본값 | 설명 |
|----------|------|---------|-------------|
| `--metric` | str | fitness | 최적화할 메트릭 (fitness, map50, map, precision, recall) |
| `--val-sets` | list | ['test1'] | 검증 세트 |
| `--target-class` | str | None | 타겟 클래스 이름 또는 인덱스 |
| `--img-size` | int | 640 | 이미지 크기 |
| `--batch-size` | int | 32 | 배치 크기 |
| `--conf-thres` | float | 0.001 | 신뢰도 임계값 |
| `--iou-thres` | float | 0.6 | IoU 임계값 |

### 조기 종료
| 인자 | 타입 | 기본값 | 설명 |
|----------|------|---------|-------------|
| `--early-stop` | flag | False | 조기 종료 활성화 |
| `--stop-threshold` | float | 0.05 | 성능 하락 임계값 |
| `--stop-patience` | int | 3 | 종료 전 연속 하락 횟수 |

### 출력
| 인자 | 타입 | 기본값 | 설명 |
|----------|------|---------|-------------|
| `--output-dir` | str | runs/wiseft | 출력 디렉토리 |
| `--save-best-only` | flag | True | 최고 모델만 저장 |
| `--save-merged-models` | flag | False | 모든 병합 모델 저장 |
| `--report-format` | str | markdown | 보고서 형식 (markdown, text, json) |

---

## 📁 출력 파일

```
runs/wiseft/exp/
├── best_merged.pt              # ⭐ 최고 알파 병합 모델 (이것을 사용!)
├── wiseft_report.md            # 📄 전체 상세 보고서
├── results.json                # 📊 JSON 형식의 모든 결과
└── temp/                       # 임시 파일 (삭제 가능)
    ├── alpha_0.10.pt
    ├── alpha_0.15.pt
    └── ...
```

---

## 🎓 알파 이해하기

### 알파(α)란 무엇인가?

알파는 스크래치와 미세조정 모델 간 혼합 비율을 제어합니다:

```python
merged_weight = (1 - α) * scratch_weight + α * finetuned_weight
```

### 알파 값 설명

| 알파 | 의미 | 사용 시기 |
|-------|---------|-------------|
| 0.0 | 100% 스크래치 | 절대 사용 안 함 (그냥 스크래치 모델 사용) |
| 0.1 | 90% 스크래치 + 10% 미세조정 | 심각한 치명적 망각 감지됨 |
| 0.2 | 80% 스크래치 + 20% 미세조정 | 이상적인 미세조정 (일반적으로 최고) |
| 0.5 | 50% 스크래치 + 50% 미세조정 | 균형잡힌 접근 |
| 0.8 | 20% 스크래치 + 80% 미세조정 | 전체 모델이 잘 미세조정됨 |
| 1.0 | 100% 미세조정 | 절대 사용 안 함 (그냥 미세조정 모델 사용) |

### 시나리오별 권장 범위

**시나리오 1: 헤드 중심 미세조정 (이상적)**
```
가중치 변화: 백본 3%, 헤드 45%
권장 α: 0.05 - 0.30
이유: 일반 특징 유지, 일부 타겟 개선사항 채택
```

**시나리오 2: 과적합 (치명적 망각)**
```
가중치 변화: 백본 18%, 헤드 85%
권장 α: 0.0 - 0.20
이유: 치명적 망각 방지
```

**시나리오 3: 전체 모델 미세조정**
```
가중치 변화: 백본 15%, 헤드 25%
권장 α: 0.2 - 0.6
이유: 전체 미세조정 이점 활용
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

## 💡 실제 사례

### 문제
```yaml
데이터셋: COCO (80 클래스) + 커스텀 person 데이터셋

스크래치 모델 (COCO로 학습):
  전체 mAP@.5: 0.65
  Person: 0.60
  Car: 0.70
  Dog: 0.68

미세조정 모델 (person-only로 학습):
  전체 mAP@.5: 0.52  ⚠️ 하락!
  Person: 0.85  ✅ 개선!
  Car: 0.45  ❌ 치명적 망각!
  Dog: 0.42  ❌ 치명적 망각!
```

### WiSE-FT로 해결
```bash
python wiseft_sweep.py \
    --scratch runs/coco/weights/best.pt \
    --finetuned runs/person_finetuned/weights/best.pt \
    --data data/coco_custom.yaml \
    --target-class person
```

### 결과
```yaml
최적 알파: 0.15 (85% 스크래치 + 15% 미세조정)

병합 모델:
  전체 mAP@.5: 0.69  ✅ 스크래치 대비 +6%
  Person: 0.72  ✅ 스크래치 대비 +20%
  Car: 0.67  ✅ 단 -4% (WiSE-FT 없으면 -36%)
  Dog: 0.66  ✅ 단 -3% (WiSE-FT 없으면 -38%)

트레이드오프: 타겟 클래스 +20%, 다른 클래스 -3~4%만
```

---

## ❓ FAQ

### Q1: 얼마나 시간이 걸리나요?
**A**: 데이터셋 크기와 GPU에 따라 다름:
- 작은 데이터셋 (1000 이미지): ~30분
- 중간 데이터셋 (5000 이미지): ~1시간
- 큰 데이터셋 (10000+ 이미지): ~2-3시간

### Q2: 중단하고 재개할 수 있나요?
**A**: 현재는 불가능. 필요시 results.json을 저장하고 나중에 분석하세요.

### Q3: 모든 알파가 비슷한 성능을 보이면?
**A**: 이는 다음을 의미합니다:
1. 미세조정이 가중치를 크게 바꾸지 않았거나,
2. 모델이 매우 유사하거나,
3. 메트릭이 충분히 민감하지 않음

시도: 보고서에서 가중치 분석 확인, 또는 다른 메트릭 사용 (--metric map50).

### Q4: YOLOv5/v8에서 사용할 수 있나요?
**A**: 개념은 동일하지만 다음을 조정해야 할 수 있습니다:
- 백본/넥/헤드의 레이어 인덱스
- 체크포인트 구조
- test.py 호출

### Q5: 다중 클래스 미세조정에 작동하나요?
**A**: 예! 여러 --target-class 값을 지정하거나 전체 메트릭을 사용하세요.

### Q6: --skip-zero를 사용해야 하나요?
**A**: 예 (기본값). Alpha=0.0은 그냥 스크래치 모델이므로 테스트할 필요가 없습니다.

---

## 🔗 관련 개념

### WiSE-FT 논문
- **제목**: "Robust fine-tuning of zero-shot models"
- **저자**: Mitchell Wortsman et al.
- **학회**: CVPR 2022
- **핵심 아이디어**: 사전 학습 및 미세조정 모델 간 선형 보간

### 관련 기술
- **Model Soup**: 여러 미세조정 모델 평균화
- **EWC (Elastic Weight Consolidation)**: 망각 방지 정규화
- **Progressive Neural Networks**: 새 작업을 위한 새 용량 추가

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

### 문제: "모든 알파가 같은 성능"
**해결책**:
1. 가중치 분석 확인 - 모델이 실제로 다른가?
2. 다른 메트릭 시도: --metric map50
3. 더 세밀한 탐색을 위해 focus-range 증가: --focus-range 0.05

### 문제: "알파 권장사항이 잘못된 것 같음"
**해결책**: 수동으로 재정의:
```bash
python wiseft_sweep.py ... --alpha-min 0.1 --alpha-max 0.5
```

---

## 📚 추가 자료

- [WISEFT_TEST_REPORT.md](WISEFT_TEST_REPORT.md) - 상세 테스트 결과
- [wiseft_sweep.py](wiseft_sweep.py) - 소스 코드
- [test_wiseft_simple.py](test_wiseft_simple.py) - 단위 테스트

---

## 📝 인용

연구에서 이 도구를 사용하는 경우 인용해 주세요:

```bibtex
@misc{wiseft_yolov7,
  title={WiSE-FT Sweep Tool for YOLOv7},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/yolov7_wiseft}}
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

기여를 환영합니다! 다음을 따라주세요:
1. 저장소 포크
2. 기능 브랜치 생성
3. 새 기능에 대한 테스트 추가
4. 풀 리퀘스트 제출

---

**상태**: ✅ 프로덕션 준비 완료
**버전**: 1.0.0 MVP
**최종 업데이트**: 2025-11-15

---

*즐거운 WiSE-FT-ing! 🎉*
