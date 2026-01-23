import argparse
import time
from pathlib import Path
from datetime import datetime
import shutil
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# Try to import scipy for Hungarian algorithm, fallback to greedy if not available
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using greedy matching instead of Hungarian algorithm")


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


def load_gt_labels(image_path, gt_root):
    """
    Load ground truth labels for an image
    Converts: JPEGImages/sub/image.jpg -> gt_root/sub/image.txt
    Returns: list of [class, x, y, w, h] (normalized)
    """
    image_path = Path(image_path)

    # Find JPEGImages in the path and replace with gt_root
    parts = list(image_path.parts)
    try:
        jpeg_idx = parts.index('JPEGImages')
        gt_parts = parts[:jpeg_idx] + [gt_root] + parts[jpeg_idx+1:]
    except ValueError:
        # If JPEGImages not found, try relative path
        gt_parts = [gt_root] + parts[parts.index(image_path.name)-1:] if image_path.name in parts else [gt_root, image_path.name]

    gt_path = Path(*gt_parts).with_suffix('.txt')

    labels = []
    if gt_path.exists():
        with open(gt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        labels.append([cls, x, y, w, h])

    return labels, gt_path.exists()


def match_predictions_to_gt(gt_labels, pred_labels, iou_threshold):
    """
    Match predictions to ground truth using Hungarian algorithm
    gt_labels: list of [class, x, y, w, h]
    pred_labels: list of [class, x, y, w, h, conf]
    Returns: True if all objects match (same count, class, and IoU >= threshold)
    """
    # If both empty, it's a match
    if len(gt_labels) == 0 and len(pred_labels) == 0:
        return True, []

    # If counts don't match, no match
    if len(gt_labels) != len(pred_labels):
        return False, []

    # If no objects, already handled above
    if len(gt_labels) == 0:
        return True, []

    # Build cost matrix (negative IoU, since we want to maximize IoU)
    n = len(gt_labels)
    cost_matrix = np.zeros((n, n))

    for i, gt in enumerate(gt_labels):
        for j, pred in enumerate(pred_labels):
            # Check class match
            if gt[0] != pred[0]:
                cost_matrix[i][j] = 1e9  # Very high cost for class mismatch
            else:
                iou = calculate_iou(gt[1:5], pred[1:5])
                cost_matrix[i][j] = -iou  # Negative because we minimize cost

    # Use Hungarian algorithm or greedy matching
    if SCIPY_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        # Greedy matching: for each GT, find best matching prediction
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
        col_ind = np.array(col_ind)
        row_ind = np.array(row_ind)

    # Check if all matches satisfy IoU threshold
    matched_pairs = []
    for i, j in zip(row_ind, col_ind):
        gt = gt_labels[i]
        pred = pred_labels[j]

        # Check class match
        if gt[0] != pred[0]:
            return False, []

        # Check IoU threshold
        iou = calculate_iou(gt[1:5], pred[1:5])
        if iou < iou_threshold:
            return False, []

        matched_pairs.append((i, j, iou))

    return True, matched_pairs


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # GT matching mode
    match_gt_mode = opt.match_gt
    match_iou = opt.match_iou
    max_save = opt.max_save
    gt_root = opt.gt_root
    recursive = opt.recursive

    # Directories
    if match_gt_mode and opt.save_dir:
        save_dir = Path(opt.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run

    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # GT matching mode: create additional directories
    if match_gt_mode:
        (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
        (save_dir / 'labels_Correct').mkdir(parents=True, exist_ok=True)
        (save_dir / 'JPEGImages').mkdir(parents=True, exist_ok=True)
        (save_dir / 'result').mkdir(parents=True, exist_ok=True)

        # Initialize counters and log
        log_file = save_dir / 'matching_log.txt'
        log_data = {
            'start_time': datetime.now(),
            'total_processed': 0,
            'matched_with_objects': 0,
            'matched_empty': 0,
            'failed_gt_not_found': 0,
            'failed_count_mismatch': 0,
            'failed_iou_low': 0,
            'saved_total': 0,
            'saved_with_objects': 0,
            'saved_empty': 0,
            'max_empty': int(max_save * 0.01) if max_save > 0 else 0
        }

        # Write initial log
        with open(log_file, 'w') as f:
            f.write("=== YOLO GT Matching Results ===\n")
            f.write(f"Start Time: {log_data['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nParameters:\n")
            f.write(f"  - Source: {source}\n")
            f.write(f"  - Weights: {weights}\n")
            f.write(f"  - GT Root: {gt_root}\n")
            f.write(f"  - Save Dir: {save_dir}\n")
            f.write(f"  - Recursive Search: {recursive}\n")
            f.write(f"  - Match IoU: {match_iou:.2f}\n")
            f.write(f"  - Max Save: {max_save}\n")
            f.write(f"  - Max Empty (1%): {log_data['max_empty']}\n\n")

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, recursive=recursive)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    stop_processing = False
    for path, img, im0s, vid_cap in dataset:
        if stop_processing:
            break
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # GT matching mode
            if match_gt_mode:
                log_data['total_processed'] += 1

                # Load GT labels
                gt_labels, gt_exists = load_gt_labels(p, gt_root)

                if not gt_exists:
                    log_data['failed_gt_not_found'] += 1
                    print(f"Warning: GT label not found for {p.name}")
                    continue

                # Convert predictions to normalized format
                pred_labels = []
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in det:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        pred_labels.append([int(cls.item()), xywh[0], xywh[1], xywh[2], xywh[3], conf.item()])

                # Match predictions to GT
                is_match, matched_pairs = match_predictions_to_gt(gt_labels, pred_labels, match_iou)

                # Check if we should save (considering empty limit)
                is_empty = len(gt_labels) == 0 and len(pred_labels) == 0
                should_save = False

                if is_match:
                    if is_empty:
                        log_data['matched_empty'] += 1
                        if log_data['saved_empty'] < log_data['max_empty']:
                            should_save = True
                    else:
                        log_data['matched_with_objects'] += 1
                        should_save = True
                else:
                    # Categorize failure reason
                    if len(gt_labels) != len(pred_labels):
                        log_data['failed_count_mismatch'] += 1
                    else:
                        log_data['failed_iou_low'] += 1

                # Save if matched and within limits
                if should_save and log_data['saved_total'] < max_save:
                    # Maintain directory structure
                    # Find relative path from source
                    source_path = Path(source)
                    try:
                        rel_path = p.relative_to(source_path)
                    except ValueError:
                        rel_path = p.name

                    # Remove leading 'JPEGImages' from relative path if present
                    if isinstance(rel_path, Path) and len(rel_path.parts) > 0 and rel_path.parts[0] == 'JPEGImages':
                        rel_path = Path(*rel_path.parts[1:]) if len(rel_path.parts) > 1 else Path(rel_path.name)

                    # Save image
                    img_save_path = save_dir / 'JPEGImages' / rel_path
                    img_save_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, img_save_path)

                    # Save GT labels
                    gt_save_path = save_dir / 'labels' / rel_path.with_suffix('.txt')
                    gt_save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(gt_save_path, 'w') as f:
                        for label in gt_labels:
                            f.write(f"{label[0]} {label[1]:.5f} {label[2]:.5f} {label[3]:.5f} {label[4]:.5f}\n")

                    # Save predictions with confidence (5 decimal places)
                    pred_save_path = save_dir / 'labels_Correct' / rel_path.with_suffix('.txt')
                    pred_save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(pred_save_path, 'w') as f:
                        for pred in pred_labels:
                            f.write(f"{pred[0]} {pred[1]:.5f} {pred[2]:.5f} {pred[3]:.5f} {pred[4]:.5f} {pred[5]:.5f}\n")

                    # Update save_path for result image (with directory structure)
                    if save_img:
                        result_img_path = save_dir / 'result' / rel_path
                        result_img_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path = str(result_img_path)

                    # Update counters
                    log_data['saved_total'] += 1
                    if is_empty:
                        log_data['saved_empty'] += 1
                    else:
                        log_data['saved_with_objects'] += 1

                # Print progress
                print(f"Processing: {log_data['total_processed']} | "
                      f"Matched: {log_data['matched_with_objects'] + log_data['matched_empty']} | "
                      f"Saved: {log_data['saved_total']}/{max_save} | "
                      f"Empty: {log_data['saved_empty']}/{log_data['max_empty']}")

                # Stop if we reached max save
                if log_data['saved_total'] >= max_save:
                    print(f"\nReached max save limit ({max_save}). Stopping.")
                    stop_processing = True
                    break

            else:
                # Original detection mode
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    # Write final log for GT matching mode
    if match_gt_mode:
        log_data['end_time'] = datetime.now()
        duration = (log_data['end_time'] - log_data['start_time']).total_seconds()

        with open(log_file, 'a') as f:
            f.write("\nProgress:\n")
            f.write(f"Total Images Processed: {log_data['total_processed']}\n")
            f.write(f"  - Matched (with objects): {log_data['matched_with_objects']}\n")
            f.write(f"  - Matched (empty): {log_data['matched_empty']}\n")
            f.write(f"  - Failed (GT not found): {log_data['failed_gt_not_found']}\n")
            f.write(f"  - Failed (object count mismatch): {log_data['failed_count_mismatch']}\n")
            f.write(f"  - Failed (IoU too low): {log_data['failed_iou_low']}\n\n")

            f.write("Saved:\n")
            f.write(f"  - Total: {log_data['saved_total']} / {max_save}\n")
            f.write(f"  - With Objects: {log_data['saved_with_objects']}\n")
            f.write(f"  - Empty (0 objects): {log_data['saved_empty']} / {log_data['max_empty']} (1%)\n\n")

            stop_reason = "Max save limit reached" if log_data['saved_total'] >= max_save else "All images processed"
            f.write(f"Stopped: {stop_reason}\n")
            f.write(f"End Time: {log_data['end_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration:.1f}s\n")

        print(f"\n=== GT Matching Summary ===")
        print(f"Total Processed: {log_data['total_processed']}")
        print(f"Saved: {log_data['saved_total']} / {max_save}")
        print(f"  - With Objects: {log_data['saved_with_objects']}")
        print(f"  - Empty: {log_data['saved_empty']} / {log_data['max_empty']}")
        print(f"Log saved to: {log_file}")

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    # GT matching mode parameters
    parser.add_argument('--match-gt', action='store_true', help='enable GT matching mode')
    parser.add_argument('--match-iou', type=float, default=0.9, help='IoU threshold for GT matching')
    parser.add_argument('--max-save', type=int, default=10000, help='maximum number of matched samples to save')
    parser.add_argument('--save-dir', type=str, default='', help='directory to save matched results')
    parser.add_argument('--gt-root', type=str, default='labels', help='GT labels root directory (replaces JPEGImages)')
    parser.add_argument('--recursive', action='store_true', help='search for images in subdirectories recursively')

    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
