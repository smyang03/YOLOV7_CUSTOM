#!/usr/bin/env python3
"""
GT와 추론 결과의 매칭 검증 스크립트
"""

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in xywh format (normalized)
    box format: [x_center, y_center, width, height]
    """
    # Convert xywh to xyxy
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # Calculate intersection area
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Calculate union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


# GT 데이터
gt_labels = [
    [3, 0.07318, 0.17176, 0.01927, 0.02870],
    [0, 0.12370, 0.34491, 0.06406, 0.25833],
    [0, 0.07865, 0.29491, 0.04583, 0.27685]
]

# 추론 결과
pred_labels = [
    [0, 0.07786, 0.29444, 0.04427, 0.27778, 0.96387],
    [0, 0.12422, 0.34444, 0.06302, 0.26111, 0.95264],
    [3, 0.07318, 0.17083, 0.01927, 0.03056, 0.88965]
]

print("=" * 80)
print("GT와 추론 결과 매칭 검증")
print("=" * 80)
print()

print("Ground Truth (GT):")
for i, gt in enumerate(gt_labels):
    print(f"  GT[{i}]: class={gt[0]}, center=({gt[1]:.5f}, {gt[2]:.5f}), size=({gt[3]:.5f}, {gt[4]:.5f})")
print()

print("추론 결과 (Predictions):")
for i, pred in enumerate(pred_labels):
    print(f"  Pred[{i}]: class={pred[0]}, center=({pred[1]:.5f}, {pred[2]:.5f}), size=({pred[3]:.5f}, {pred[4]:.5f}), conf={pred[5]:.5f}")
print()

# IoU 매트릭스 계산
print("IoU Matrix (GT x Prediction):")
print("-" * 80)
n = len(gt_labels)
cost_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

print(f"{'':8s}", end='')
for j in range(n):
    print(f"Pred[{j}]  ", end='')
print()

for i, gt in enumerate(gt_labels):
    print(f"GT[{i}]   ", end='')
    for j, pred in enumerate(pred_labels):
        if gt[0] != pred[0]:  # 클래스 불일치
            iou = calculate_iou(gt[1:5], pred[1:5])
            cost_matrix[i][j] = 1e9
            print(f"  X({iou:.3f})", end='')
        else:  # 클래스 일치
            iou = calculate_iou(gt[1:5], pred[1:5])
            cost_matrix[i][j] = -iou
            print(f"  {iou:.5f} ", end='')
    print()
print()

# Hungarian algorithm 시뮬레이션
try:
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print("Hungarian Algorithm을 사용한 최적 매칭:")
except ImportError:
    print("scipy를 사용할 수 없어 순차 매칭을 수행합니다:")
    row_ind = list(range(n))
    col_ind = []
    used = set()
    for i in range(n):
        best_j = -1
        best_cost = 1e9
        for j in range(n):
            if j not in used and cost_matrix[i][j] < best_cost:
                best_cost = cost_matrix[i][j]
                best_j = j
        col_ind.append(best_j)
        used.add(best_j)

print("-" * 80)
iou_threshold = 0.9
all_match = True

for i, j in zip(row_ind, col_ind):
    gt = gt_labels[i]
    pred = pred_labels[j]
    iou = calculate_iou(gt[1:5], pred[1:5])
    class_match = "O" if gt[0] == pred[0] else "X"
    iou_match = "O" if iou >= iou_threshold else "X"

    print(f"GT[{i}] (class={gt[0]}) <-> Pred[{j}] (class={pred[0]}, conf={pred[5]:.5f})")
    print(f"  IoU: {iou:.5f}, Class Match: {class_match}, IoU >= {iou_threshold}: {iou_match}")

    if gt[0] != pred[0] or iou < iou_threshold:
        all_match = False
    print()

print("=" * 80)
print(f"매칭 결과: {'성공 (Correct)' if all_match else '실패 (Mismatch)'}")
print(f"IoU Threshold: {iou_threshold}")
print("=" * 80)
