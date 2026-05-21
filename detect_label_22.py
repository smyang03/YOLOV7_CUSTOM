import argparse
import time
import platform
from pathlib import Path
import logging
import sys
import hashlib
import cv2
import torch
import numpy as np
from tqdm import tqdm
import os
import shutil

YOLOV7_ROOT = Path(os.environ.get("YOLOV7_ROOT", r"D:\codes\YOLOV7"))
if not getattr(sys, "frozen", False) and YOLOV7_ROOT.exists():
    sys.path.insert(0, str(YOLOV7_ROOT))

from models.experimental import attempt_load
from utils.datasets import LoadImagestxt, read_label_file, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, 
    xyxy2xywh, xywhn2xyxy, strip_optimizer, 
    set_logging, increment_path, calculate_iou
)
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


RESULT_CATEGORIES = ['good_detect', 'miss_detect', 'false_detect', 'background', 'low_conf']
RESULT_GROUPS = {
    'good_detect': 'GOOD',
    'background': 'GOOD',
    'miss_detect': 'MISS',
    'false_detect': 'FAIL',
    'low_conf': 'FAIL',
}
IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
DEBUG_STEM_MAX_LENGTH = 80
DEBUG_PANEL_WIDTH = 640
DEBUG_PANEL_HEIGHT = 360
DEBUG_HEADING_HEIGHT = 32
DEBUG_BOX_THICKNESS = 2
DEBUG_LABEL_SCALE = 0.38
DEBUG_LABEL_THICKNESS = 1
DEBUG_TITLE_SCALE = 0.72
DEBUG_TITLE_THICKNESS = 2
DEBUG_INFO_SCALE = 0.40
DEBUG_INFO_THICKNESS = 1


def create_grouped_result_dirs(base_path):
    """GOOD/MISS/FAIL 기준 결과 폴더를 생성합니다."""
    base_path = Path(base_path)
    dirs = {}
    for group_name in sorted(set(RESULT_GROUPS.values())):
        group_dir = base_path / group_name
        (group_dir / 'JPEGImages').mkdir(parents=True, exist_ok=True)
        (group_dir / 'labels').mkdir(parents=True, exist_ok=True)
        dirs[group_name] = group_dir
    return dirs


def prepare_source_txt(source, save_dir, logger):
    """source가 폴더이면 이미지 목록 txt를 생성하고, txt이면 그대로 사용합니다."""
    source_path = Path(source)
    if source_path.is_file():
        return str(source_path)
    if not source_path.is_dir():
        raise FileNotFoundError(f"source path not found: {source}")

    image_paths = sorted(
        path for path in source_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"source directory has no supported images: {source}")

    generated_txt = Path(save_dir) / 'source_images.txt'
    generated_txt.write_text(
        "\n".join(str(path) for path in image_paths) + "\n",
        encoding='utf-8',
    )
    logger.info(f"Generated source image list: {generated_txt} ({len(image_paths)} images)")
    return str(generated_txt)


def imread_unicode(path):
    """Windows 한글/Unicode path를 안전하게 읽기 위한 OpenCV wrapper입니다."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, image):
    """Windows 한글/Unicode path를 안전하게 저장하기 위한 OpenCV wrapper입니다."""
    path = Path(path)
    ext = path.suffix if path.suffix else '.jpg'
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def make_debug_image_path(debug_dir, image_path):
    """긴 원본 파일명으로 인한 Windows path length 문제를 줄인 debug image 경로를 만듭니다."""
    source_path = Path(image_path)
    safe_stem = ''.join(
        char if char not in '<>:"/\\|?*' and ord(char) >= 32 else '_'
        for char in source_path.stem
    ).strip(' .')
    if not safe_stem:
        safe_stem = 'image'
    safe_stem = safe_stem[:DEBUG_STEM_MAX_LENGTH]
    digest = hashlib.sha1(str(source_path).encode('utf-8')).hexdigest()[:10]
    return Path(debug_dir) / f"{safe_stem}_{digest}_debug.jpg"


def get_class_name(names, cls_id):
    """model.names 범위를 벗어난 class id도 debug label에서 안전하게 표시합니다."""
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    if isinstance(names, dict) and cls_id in names:
        return str(names[cls_id])
    return f"class_{cls_id}"


def get_class_color(colors, cls_id):
    """class id 기준 debug 색상을 반환합니다."""
    if colors:
        return colors[int(cls_id) % len(colors)]
    return [0, 255, 0]


def fit_text_scale(text, max_width, initial_scale, thickness, min_scale=0.8):
    """긴 text가 패널 폭을 넘지 않도록 font scale을 줄입니다."""
    scale = initial_scale
    while scale > min_scale:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if text_width <= max_width:
            return scale
        scale *= 0.9
    return min_scale


def draw_debug_box(image, xyxy, label, color):
    """640x360 debug panel에 맞춘 얇은 bbox와 작은 class label을 그립니다."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness=DEBUG_BOX_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    if not label:
        return

    text_size, baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        DEBUG_LABEL_SCALE,
        DEBUG_LABEL_THICKNESS,
    )
    text_w, text_h = text_size
    pad_x, pad_y = 3, 2
    label_w = text_w + pad_x * 2
    label_h = text_h + baseline + pad_y * 2
    label_x1 = min(x1, max(0, width - label_w - 1))
    label_y1 = y1 - label_h if y1 - label_h >= 0 else y1 + DEBUG_BOX_THICKNESS
    label_y1 = min(max(0, label_y1), max(0, height - label_h - 1))
    label_x2 = label_x1 + label_w
    label_y2 = label_y1 + label_h

    cv2.rectangle(
        image,
        (label_x1, label_y1),
        (label_x2, label_y2),
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (label_x1 + pad_x, label_y2 - baseline - pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        DEBUG_LABEL_SCALE,
        (255, 255, 255),
        DEBUG_LABEL_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def scale_xyxy_to_panel(xyxy, src_shape):
    """원본 좌표계를 640x360 debug panel 좌표계로 변환합니다."""
    src_h, src_w = src_shape[:2]
    x_scale = DEBUG_PANEL_WIDTH / max(src_w, 1)
    y_scale = DEBUG_PANEL_HEIGHT / max(src_h, 1)
    values = [float(v) for v in xyxy]
    return [
        values[0] * x_scale,
        values[1] * y_scale,
        values[2] * x_scale,
        values[3] * y_scale,
    ]


def create_fixed_debug_panel(base_image, box_entries, title, subtitle, bg_color):
    """bbox를 640x360 이미지 기준으로 다시 그리고 heading을 붙입니다."""
    if base_image.ndim == 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)

    panel_image = cv2.resize(
        base_image,
        (DEBUG_PANEL_WIDTH, DEBUG_PANEL_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    for xyxy, label, color in box_entries:
        scaled_xyxy = scale_xyxy_to_panel(xyxy, base_image.shape)
        draw_debug_box(panel_image, scaled_xyxy, label, color)

    panel = np.zeros(
        (DEBUG_HEADING_HEIGHT + DEBUG_PANEL_HEIGHT, DEBUG_PANEL_WIDTH, 3),
        dtype=np.uint8,
    )
    panel[:DEBUG_HEADING_HEIGHT, :] = np.array(bg_color, dtype=np.uint8)
    panel[DEBUG_HEADING_HEIGHT:, :, :] = panel_image

    title_scale = DEBUG_TITLE_SCALE
    title_thickness = DEBUG_TITLE_THICKNESS
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness)[0]
    info_x = min(12 + title_size[0] + 18, DEBUG_PANEL_WIDTH - 1)
    subtitle_scale = fit_text_scale(
        subtitle,
        max_width=max(80, DEBUG_PANEL_WIDTH - info_x - 12),
        initial_scale=DEBUG_INFO_SCALE,
        thickness=DEBUG_INFO_THICKNESS,
        min_scale=0.28,
    )

    cv2.putText(
        panel,
        title,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (255, 255, 255),
        title_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        subtitle,
        (info_x, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        subtitle_scale,
        (245, 245, 245),
        DEBUG_INFO_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    return panel


class UnicodeSafeLoadImagestxt(LoadImagestxt):
    """LoadImagestxt의 cv2.imread 경로를 Unicode-safe 방식으로 교체합니다."""

    def __next__(self):
        if self.count == self.nf:
            print(f'All {self.nf} files processed in {time.time() - self.start_time:.2f} seconds')
            raise StopIteration

        path = self.files[self.count]
        percent_complete = (self.count / self.nf) * 100
        elapsed_time = time.time() - self.start_time
        if self.count > 0:
            files_per_second = self.count / elapsed_time
            remaining_files = self.nf - self.count
            estimated_remaining_time = remaining_files / files_per_second if files_per_second > 0 else 0
            time_info = f' | ETA: {estimated_remaining_time:.1f}s'
        else:
            time_info = ''

        bar_length = 20
        filled_length = int(bar_length * self.count / self.nf)
        progress_bar = '█' * filled_length + '░' * (bar_length - filled_length)

        if self.video_flag[self.count]:
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:
                    print(f'All {self.nf} files processed in {time.time() - self.start_time:.2f} seconds')
                    raise StopIteration
                path = self.files[self.count]
                self.new_video(path)
                ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'[{progress_bar}] {percent_complete:.1f}% | Video {self.count + 1}/{self.nf} (frame {self.frame}/{self.nframes}){time_info}: {path}')
        else:
            self.count += 1
            img0 = imread_unicode(path)
            assert img0 is not None, f'Failed to load image with Unicode-safe reader: {path}'
            print(f'[{progress_bar}] {percent_complete:.1f}% | Image {self.count}/{self.nf}{time_info}: {path}')

        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap


def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def is_cuda_runtime_error(exc):
    """CUDA runtime/device 호환성 문제인지 확인합니다."""
    message = str(exc).lower()
    keywords = [
        'cuda error',
        'no kernel image is available',
        'cuda-capable device',
        'invalid device function',
        'no nvidia driver',
        'cuda unavailable',
    ]
    return isinstance(exc, (RuntimeError, AssertionError)) and any(keyword in message for keyword in keywords)


def cuda_smoke_test(device_index=0, logger=None):
    """실제 CUDA tensor 연산과 torchvision NMS가 가능한지 짧게 확인합니다."""
    try:
        if not torch.cuda.is_available():
            return False
        device = torch.device(f'cuda:{device_index}')
        x = torch.ones((8,), device=device)
        y = (x * 2).sum()
        torch.cuda.synchronize(device)

        from torchvision.ops import nms

        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
            device=device,
        )
        scores = torch.tensor([0.9, 0.8], device=device)
        _ = nms(boxes, scores, 0.5)
        torch.cuda.synchronize(device)
        _ = y.item()
        return True
    except Exception as exc:
        if logger:
            logger.warning(f"CUDA smoke test failed on cuda:{device_index}: {exc}")
        return False


def resolve_runtime_device(device_arg, logger):
    """자동 GPU 우선 선택을 실제 CUDA smoke test 기준으로 결정합니다."""
    requested = str(device_arg or '').strip().lower()
    if requested == 'cpu':
        return torch.device('cpu')

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return torch.device('cpu')

    first_device = 0
    if requested:
        try:
            first_device = int(requested.split(',')[0])
        except ValueError:
            logger.warning(f"Invalid device option '{device_arg}', using CPU")
            return torch.device('cpu')

    if first_device < 0 or first_device >= torch.cuda.device_count():
        logger.warning(f"CUDA device index out of range: {first_device}, using CPU")
        return torch.device('cpu')

    if not cuda_smoke_test(first_device, logger):
        logger.warning("CUDA runtime is not compatible with this GPU/build, falling back to CPU")
        return torch.device('cpu')

    device = torch.device(f'cuda:{first_device}')
    props = torch.cuda.get_device_properties(first_device)
    logger.info(
        f"Using CUDA:{first_device} ({props.name}, compute capability {props.major}.{props.minor})"
    )
    return device


def load_model(weights, device, img_size, trace=True):
    """Load the detection model."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model from {weights}")
    model = attempt_load(weights, map_location=torch.device('cpu'))
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    model.eval()

    if device.type != 'cpu':
        model = model.to(device)
    
    # Enable half precision if on CUDA
    half = device.type != 'cpu'
    if half:
        model.half()
    
    # Trace model for better performance if requested
    if trace and device.type != 'cpu':
        try:
            logger.info("Converting model to Traced-model...")
            # 입력과 모델의 타입을 일치시키기 위해 half precision으로 통일
            rand_example = torch.zeros(1, 3, img_size, img_size).to(device)
            if half:
                rand_example = rand_example.half()
            else:
                # 모델이 half가 아니라면 float32로 유지
                rand_example = rand_example.float()
                
            model = TracedModel(model, device, img_size, rand_example)
            logger.info("Successfully converted to Traced-model")
        except Exception as e:
            logger.warning(f"Failed to create traced model: {e}")
            logger.info("Using non-traced model instead")
    
    # Warmup
    if device.type != 'cpu':
        logger.info("Warming up model...")
        img = torch.zeros(1, 3, img_size, img_size).to(device)
        img = img.half() if half else img.float()
        _ = model(img)
    
    return model, stride, img_size, half


def process_batch(img, model, device, half, augment=False, conf_thres=0.25, iou_thres=0.45, 
                  classes=None, agnostic_nms=False):
    """Process a batch of images through the model."""
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        pred = model(img, augment=augment)[0]
    
    # Apply NMS
    pred = non_max_suppression(
        pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms
    )
    
    return pred


def safe_scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    안전하게 좌표를 스케일링하는 함수.
    원본 scale_coords 함수와 동일하지만 에러 처리가 강화됨.
    
    Args:
        img1_shape: 소스 이미지 형태 (높이, 너비)
        coords: 변환할 좌표 (xyxy 형식)
        img0_shape: 대상 이미지 형태 (높이, 너비)
        ratio_pad: 선택적 비율과 패딩 값
        
    Returns:
        변환된 좌표
    """
    # 입력 형태 확인 및 수정
    if isinstance(img0_shape, torch.Tensor):
        img0_shape = img0_shape.cpu().numpy()
    if isinstance(img1_shape, torch.Tensor):
        img1_shape = img1_shape.cpu().numpy()
        
    # 차원 확인
    if len(img0_shape) >= 3:  # 높이, 너비, 채널
        img0_h, img0_w = img0_shape[0], img0_shape[1]
    else:  # 높이, 너비
        img0_h, img0_w = img0_shape[0], img0_shape[1] if len(img0_shape) > 1 else img0_shape[0]
    
    if len(img1_shape) >= 3:
        img1_h, img1_w = img1_shape[0], img1_shape[1]
    else:
        img1_h, img1_w = img1_shape[0], img1_shape[1] if len(img1_shape) > 1 else img1_shape[0]
    
    # 좌표 복사하여 원본 유지
    coords_result = coords.clone() if isinstance(coords, torch.Tensor) else coords.copy()
    
    # 스케일링 비율 계산
    if ratio_pad is None:
        gain = min(img1_h / img0_h, img1_w / img0_w)
        pad = (img1_w - img0_w * gain) / 2, (img1_h - img0_h * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    # 좌표 변환
    coords_result[:, [0, 2]] -= pad[0]  # x padding
    coords_result[:, [1, 3]] -= pad[1]  # y padding
    coords_result[:, :4] /= gain
    
    # 클리핑 (이미지 범위 내로 제한)
    coords_result[:, 0].clamp_(0, img0_w)  # x1
    coords_result[:, 1].clamp_(0, img0_h)  # y1
    coords_result[:, 2].clamp_(0, img0_w)  # x2
    coords_result[:, 3].clamp_(0, img0_h)  # y2
    
    return coords_result


def evaluate_detections(pred, gt_boxes, im0s, img, gn, names, conf_thres, iou_thres, colors=None, 
                        visualize=False, min_recall=0.8, classes=None):
    """
    Evaluate detections against ground truth, filtering for specific classes.
    
    Args:
        pred: Model predictions
        gt_boxes: Ground truth boxes
        im0s: Original image
        img: Preprocessed image tensor
        gn: Normalization gain
        names: Class names
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        colors: Colors for visualization
        visualize: Whether to generate visualization
        min_recall: Minimum recall for good detection
        classes: List of specific classes to evaluate. If None, all classes are evaluated.
        
    Returns:
        dict: Results info including detection category, precision, recall, etc.
    """
    logger = logging.getLogger(__name__)
    
    if colors is None:
        colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    
    result = {
        'category': 'unknown',
        'matched_gt': set(),
        'matched_pred': set(),
        'precision': 0,
        'recall': 0,
        'pred_info': [],
        'bbox_count': 0,
        'visualization': None
    }
    
    # Process first detection batch (should be single image)
    det = pred[0] if len(pred) > 0 and pred[0] is not None else None
    
    # Create debug visualization images
    im_pred = im0s.copy()  # Detection visualization
    im_gt = im0s.copy()    # Ground truth visualization
    pred_visual_entries = []
    gt_visual_entries = []
    
    # 클래스 필터링된 ground truth boxes만 선택
    filtered_gt_boxes = []
    selected_classes = [int(cls) for cls in classes] if classes is not None else None
    if selected_classes is not None:
        for gt_idx, gt_box in enumerate(gt_boxes):
            gt_cls = int(gt_box[0])
            if gt_cls in selected_classes:
                filtered_gt_boxes.append((gt_idx, gt_box))
    else:
        filtered_gt_boxes = [(gt_idx, gt_box) for gt_idx, gt_box in enumerate(gt_boxes)]
    
    # First add ground truth boxes to GT visualization
    for gt_idx, gt_box in filtered_gt_boxes:
        try:
            xyxy = xywhn2xyxy(torch.tensor(gt_box[1:]).unsqueeze(0),  # gn을 곱하지 않음
                                w=im_gt.shape[1], h=im_gt.shape[0]).view(-1).tolist()
            if visualize:
                cls_id = int(gt_box[0])
                label = f'{get_class_name(names, cls_id)} GT'
                gt_visual_entries.append((xyxy, label, get_class_color(colors, cls_id)))
        except Exception as e:
            logger.error(f"Error processing GT box {gt_idx}: {e}")
            continue
    
    # Match detections with ground truth
    all_conf_good = True  # 모든 검출 결과의 신뢰도가 좋은지 확인
    
    if det is not None and len(det) > 0:
        # 안전한 좌표 스케일링을 위한 수정
        try:
            # 이미지 형태 확인 및 스케일링 수행
            if len(im0s.shape) == 3:  # 색상 채널이 있는 이미지 (높이, 너비, 채널)
                img0_shape = im0s.shape[:2]  # (높이, 너비)만 사용
            else:
                img0_shape = im0s.shape  # 그레이스케일 이미지나 다른 형태
                
            if len(img.shape) == 4:  # 배치, 채널, 높이, 너비
                img1_shape = img.shape[2:]  # (높이, 너비)
            else:
                img1_shape = img.shape[1:]  # 첫 번째 차원이 없는 경우
                
            # 안전한 스케일링 함수 사용
            det[:, :4] = safe_scale_coords(img1_shape, det[:, :4], img0_shape).round()
        except Exception as e:
            logger.error(f"Error scaling coordinates: {e}")
            logger.debug(f"img shape: {img.shape}, im0s shape: {im0s.shape}")
            # 스케일링 실패 시 원본 좌표 유지
        
        # 클래스 필터링된 detection 결과만 선택
        filtered_det = []
        if selected_classes is not None:
            for i, (*xyxy, conf, cls) in enumerate(det):
                cls_id = int(cls)
                if cls_id == 2 and cls_id == 5 and cls_id == 7:
                    cls_id = 1
                elif cls_id == 14:
                    cls_id = 2
                elif cls_id == 15 and cls_id == 16:
                    cls_id = 3
                    
                if cls_id in selected_classes:
                    filtered_det.append((i, (*xyxy, conf, cls)))
        else:
            filtered_det = [(i, d) for i, d in enumerate(det)]
        
        # 모든 검출 결과의 신뢰도 확인
        all_conf_good = all(conf >= conf_thres for _, (*_, conf, _) in filtered_det)
        
        # Add predictions to visualization image
        for pred_idx, (*xyxy, conf, cls) in filtered_det:
            cls_id = int(cls)
            pred_bbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            result['pred_info'].append((cls_id, pred_bbox, conf.item()))
            
            if visualize:
                label = f'{get_class_name(names, cls_id)} {conf:.2f}'
                pred_visual_entries.append((xyxy, label, get_class_color(colors, cls_id)))
        
        # For each prediction, find the best matching GT box
        for pred_idx, (*xyxy, conf, cls) in filtered_det:
            pred_cls = int(cls)
            pred_bbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            
            # Find best matching GT box
            best_iou = 0
            best_gt_idx = -1
            
            for orig_gt_idx, gt_box in filtered_gt_boxes:
                gt_cls = int(gt_box[0])
                
                # Only match if classes are the same
                if pred_cls == gt_cls:
                    iou = calculate_iou(pred_bbox, gt_box[1:])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = orig_gt_idx
            
            # If IoU is above threshold, consider it a match
            if best_iou > iou_thres and best_gt_idx >= 0:
                result['matched_gt'].add(best_gt_idx)
                result['matched_pred'].add(pred_idx)
    
    # Calculate precision and recall based on filtered ground truth and predictions
    filtered_gt_count = len(filtered_gt_boxes)
    filtered_pred_count = len(filtered_det) if det is not None and len(det) > 0 else 0
    
    if filtered_gt_count > 0:
        result['recall'] = len(result['matched_gt']) / filtered_gt_count
    else:
        result['recall'] = 1.0 if filtered_pred_count == 0 else 0.0
        
    if filtered_pred_count > 0:
        result['precision'] = len(result['matched_pred']) / filtered_pred_count
    else:
        result['precision'] = 1.0 if filtered_gt_count == 0 else 0.0
    
    # Determine result category
    if filtered_gt_count > 0:
        if result['recall'] >= min_recall:
            if all_conf_good:
                result['category'] = 'good_detect'
            else:
                result['category'] = 'low_conf'
        elif result['recall'] > 0 and result['recall'] < min_recall:
            result['category'] = 'miss_detect'
        else:
            result['category'] = 'false_detect'
    else:
        # No ground truth, but we have detections - false positive
        if filtered_pred_count > 0:
            result['category'] = 'false_detect'
        else:
            # No ground truth, no detections - correctly did nothing
            result['category'] = 'background'
    result['bbox_count'] = filtered_pred_count
    # Create combined visualization if requested
    if visualize:
        try:
            pred_subtitle = (
                f"Precision {result['precision']:.2f} | Recall {result['recall']:.2f} | "
                f"Boxes {filtered_pred_count}"
            )
            gt_subtitle = f"Ground truth boxes {filtered_gt_count}"
            im_pred_with_heading = create_fixed_debug_panel(
                im_pred,
                pred_visual_entries,
                title="PRED",
                subtitle=pred_subtitle,
                bg_color=(42, 96, 42),
            )
            im_gt_with_heading = create_fixed_debug_panel(
                im_gt,
                gt_visual_entries,
                title="GT",
                subtitle=gt_subtitle,
                bg_color=(96, 56, 36),
            )
            
            # Combine side by side
            result['visualization'] = np.hstack((im_pred_with_heading, im_gt_with_heading))
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            result['visualization'] = None
    
    return result

def save_results(result, path, target_dir, label_path=None):
    """Save results to appropriate directories."""
    # Create paths
    p = Path(path)
    target_img_path = target_dir / 'JPEGImages' / p.name
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(target_img_path), exist_ok=True)
    
    # Copy image
    shutil.copy(str(path), str(target_img_path))
    
    # Copy label if it exists
    if label_path and os.path.exists(label_path):
        target_label_path = target_dir / 'labels' / (p.stem + '.txt')
        os.makedirs(os.path.dirname(target_label_path), exist_ok=True)
        shutil.copy(label_path, str(target_label_path))
    
    return target_img_path


def detect(opt, stop_event=None):
    """Main detection function."""
    logger = setup_logging()
    logger.info(f"Starting detection with options: {opt}")
    is_windows = platform.system() == 'Windows'
    if opt.view_img and not is_windows:
        logger.warning("view_img option is only supported on Windows. Disabling.")

    # opt.view_img = True
    # opt.save_debug_images = True
    # Set up save directory
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging to file as well
    file_handler = logging.FileHandler(save_dir / 'detection_log.txt', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    # Initialize device
    device = resolve_runtime_device(opt.device, logger)

    # Load model
    try:
        model, stride, imgsz, half = load_model(
            opt.weights, device, opt.img_size, trace=not opt.no_trace
        )
    except Exception as exc:
        if device.type == 'cuda' and is_cuda_runtime_error(exc):
            logger.warning(f"CUDA model initialization failed, retrying on CPU: {exc}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            device = torch.device('cpu')
            model, stride, imgsz, half = load_model(
                opt.weights, device, opt.img_size, trace=False
            )
        else:
            raise
    
    # Get model class names and set up colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Set up dataset
    try:
        source_txt = prepare_source_txt(opt.source, save_dir, logger)
        dataset = UnicodeSafeLoadImagestxt(source_txt, img_size=imgsz, stride=stride)
        logger.info(f"Dataset loaded with {len(dataset)} images")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Create result directories
    try:
        result_dirs = create_grouped_result_dirs(opt.savedirs)
        debug_dirs = {}
        for group_name, group_dir in result_dirs.items():
            debug_dir = group_dir / 'debug_images'
            debug_dir.mkdir(exist_ok=True, parents=True)
            debug_dirs[group_name] = debug_dir
            
        logger.info(f"Result directories created at {opt.savedirs}")
    except Exception as e:
        logger.error(f"Failed to create result directories: {e}")
        return
    
    # Statistics
    stats = {cat: 0 for cat in RESULT_CATEGORIES}
    group_stats = {group_name: 0 for group_name in sorted(set(RESULT_GROUPS.values()))}
    total_bounding_boxes = 0
    debug_images_saved = 0
    debug_images_failed = 0
    total_images = 0
    successful_images = 0
    interrupted = False
    t0 = time.time()
    
    # Process images
    for idx, item in enumerate(tqdm(dataset, desc="Processing images")):
        if stop_event is not None and stop_event.is_set():
            interrupted = True
            logger.warning("Stop requested. Detection loop stopped before processing next image.")
            break

        path = None
        try:
            if item is None: continue
            path, img, im0s, vid_cap = item
            if img is None: continue
            if img.shape is None or im0s.shape is None: continue
            if not hasattr(img,'shape') or not hasattr(im0s,'shape'): continue
            if len(img.shape) == 0: continue
            logger.info(f"Processing image {idx+1}/{len(dataset)}:Results: {stats}")
            total_images += 1
            
            # Load ground truth
            image_path = Path(path)
            label_path = str(image_path.with_suffix('.txt')).replace('JPEGImages', 'labels')
            if not os.path.exists(label_path):
                # 다른 확장자도 시도
                label_path = label_path.replace('.txt', '.txt')
                if not os.path.exists(label_path):
                    logger.warning(f"Label file not found for {path}")
                    gt_boxes = []  # 빈 GT 박스 목록
                else:
                    gt_boxes = read_label_file(label_path)
            else:
                gt_boxes = read_label_file(label_path)
            
            # Process batch
            try:
                pred = process_batch(
                    img, model, device, half,
                    augment=opt.augment,
                    conf_thres=opt.conf_thres,
                    iou_thres=opt.iou_thres,
                    classes=opt.classes,
                    agnostic_nms=opt.agnostic_nms
                )
            except Exception as exc:
                if device.type == 'cuda' and is_cuda_runtime_error(exc):
                    logger.warning(f"CUDA inference failed, switching to CPU and retrying: {exc}")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    device = torch.device('cpu')
                    model, stride, imgsz, half = load_model(
                        opt.weights, device, opt.img_size, trace=False
                    )
                    pred = process_batch(
                        img, model, device, half,
                        augment=opt.augment,
                        conf_thres=opt.conf_thres,
                        iou_thres=opt.iou_thres,
                        classes=opt.classes,
                        agnostic_nms=opt.agnostic_nms
                    )
                else:
                    raise
            
            # Get normalization gain for image
            if len(im0s.shape) == 3:  # (높이, 너비, 채널)
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # [w, h, w, h]
            else:  # (높이, 너비)
                h, w = im0s.shape
                gn = torch.tensor([w, h, w, h])
            
            # Evaluate detections
            result = evaluate_detections(
                pred, gt_boxes, im0s, img, gn, names, 
                opt.conf_thres, opt.iou_thres, colors,
                visualize=opt.view_img or opt.save_debug_images,
                min_recall=opt.min_recall,
                classes=opt.classes
            )
            
            # Update statistics
            category = result['category']
            group_name = RESULT_GROUPS.get(category, 'FAIL')
            if category not in stats:
                stats[category] = 0
            stats[category] += 1
            group_stats[group_name] += 1
            total_bounding_boxes += int(result.get('bbox_count', len(result.get('pred_info', []))))

            # Save results if needed
            if not opt.nosave:
                target_dir = result_dirs[group_name]
                save_results(result, path, target_dir, label_path)
            
            # Save visualization if requested
            if opt.save_debug_images and result['visualization'] is not None:
                debug_img_path = make_debug_image_path(debug_dirs[group_name], path)
                try:
                    if imwrite_unicode(debug_img_path, result['visualization']):
                        debug_images_saved += 1
                        logger.info(f"Saved debug image: {debug_img_path}")
                    else:
                        debug_images_failed += 1
                        logger.error(f"Failed to encode debug image: {debug_img_path}")
                except Exception as e:
                    debug_images_failed += 1
                    logger.error(f"Failed to save debug image: {e}")
            
            # Show visualization if requested
            if opt.view_img and result['visualization'] is not None and is_windows:
                cv2.imshow('Detection Evaluation', cv2.resize(result['visualization'], (1600, 900)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            successful_images += 1
                
        except Exception as e:
            logger.error(f"Error processing image {path if path else idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Print final statistics
    processing_time = time.time() - t0
    seconds_per_image = processing_time / total_images if total_images else 0.0
    logger.info(f"Processed {total_images} images in {processing_time:.2f}s ({seconds_per_image:.3f}s per image)")
    logger.info(f"Successfully processed: {successful_images}/{total_images} images")
    logger.info(f"Results: {stats}")
    logger.info(f"Grouped results: {group_stats}")
    logger.info(f"Total bounding boxes: {total_bounding_boxes}")
    logger.info(f"Debug images saved: {debug_images_saved}, failed: {debug_images_failed}")
    logger.info(f"Interrupted: {interrupted}")

    def ratio(count):
        return count / total_images if total_images else 0.0
    
    # Save summary
    with open(save_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"Total images: {total_images}\n")
        f.write(f"Successfully processed: {successful_images}\n")
        f.write(f"Interrupted: {'yes' if interrupted else 'no'}\n")
        f.write(f"Processing time: {processing_time:.2f}s ({seconds_per_image:.3f}s per image)\n")
        f.write(f"GOOD: {group_stats['GOOD']}개 ({ratio(group_stats['GOOD']):.1%})\n")
        f.write(f"MISS(미감지): {group_stats['MISS']}개 ({ratio(group_stats['MISS']):.1%})\n")
        f.write(f"FAIL(오감지): {group_stats['FAIL']}개 ({ratio(group_stats['FAIL']):.1%})\n")
        f.write(f"총 bounding_box: {total_bounding_boxes}개\n")
        f.write(f"debug images saved: {debug_images_saved}개\n")
        f.write(f"debug images failed: {debug_images_failed}개\n")
        f.write("\n[세부 category]\n")
        f.write(f"good_detect: {stats['good_detect']}개 ({ratio(stats['good_detect']):.1%})\n")
        f.write(f"miss_detect: {stats['miss_detect']}개 ({ratio(stats['miss_detect']):.1%})\n")
        f.write(f"false_detect: {stats['false_detect']}개 ({ratio(stats['false_detect']):.1%})\n")
        f.write(f"low_conf: {stats['low_conf']}개 ({ratio(stats['low_conf']):.1%})\n")
        f.write(f"background: {stats['background']}개 ({ratio(stats['background']):.1%})\n")
    
    logger.info(f"Done. Results saved to {save_dir}")
    
    # Close windows
    if opt.view_img and is_windows:
        cv2.destroyAllWindows()

    return save_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='D:\\code\\SIAV2-AI-MODEL\\models\\detectnetwork\\00.안전환경\\pt\SIAV2_Detector_YOLOV7_SafeEnv_V4.0.0_FP32_240903_BASE.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='Z:\\101.etc\\core\\core\\data1\\valid.txt', help='source image list txt file')
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--min-recall', type=float, default=0.8, help='Minimum recall for good detection')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-debug-images', action='store_true', help='save debug images')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true',help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1, 3, 4],help='filter by class')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--savedirs', type=str, default='Z:\\101.etc\\core\\core\\data1', help='base directory for saving results')
    
    return parser.parse_args()


if __name__ == '__main__':
    try:
        opt = parse_args()
        print(opt)
        
        with torch.no_grad():
            detect(opt)
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()
