"""
detect_label_22.py  —  YOLOv7 Detection Evaluation Engine  v1.0.0
=================================================================
• GOOD / MISS / FAIL 분류 평가
• GPU / CPU 자동 선택 (CUDA smoke-test 포함)
• Unicode(한글) 경로 완전 지원
• HTML 리포트 자동 생성
• GUI 연동을 위한 progress_callback 지원
"""
import argparse
import datetime
import time
import platform
from pathlib import Path
import logging
import sys
import hashlib
import traceback
import cv2
import torch
import numpy as np
from tqdm import tqdm
import os
import shutil

APP_VERSION = '1.0.0'

# ── YOLOv7 경로 자동 인식 ─────────────────────────────────────────
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


# ── 결과 분류 상수 ────────────────────────────────────────────────
RESULT_CATEGORIES = ['good_detect', 'miss_detect', 'false_detect', 'background', 'low_conf']
RESULT_GROUPS = {
    'good_detect': 'GOOD',
    'background':  'GOOD',
    'miss_detect': 'MISS',
    'false_detect':'FAIL',
    'low_conf':    'FAIL',
}
IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
REPORT_SAMPLE_LIMIT = 3   # HTML 리포트에 보여줄 카테고리별 샘플 이미지 수

# ── 디버그 패널 상수 ──────────────────────────────────────────────
DEBUG_STEM_MAX_LENGTH  = 80
DEBUG_PANEL_WIDTH      = 640
DEBUG_PANEL_HEIGHT     = 360
DEBUG_HEADING_HEIGHT   = 32
DEBUG_BOX_THICKNESS    = 2
DEBUG_LABEL_SCALE      = 0.38
DEBUG_LABEL_THICKNESS  = 1
DEBUG_TITLE_SCALE      = 0.72
DEBUG_TITLE_THICKNESS  = 2
DEBUG_INFO_SCALE       = 0.40
DEBUG_INFO_THICKNESS   = 1


# ══════════════════════════════════════════════════════════════════
#  파일 / 경로 유틸리티
# ══════════════════════════════════════════════════════════════════

def create_grouped_result_dirs(base_path):
    """GOOD / MISS / FAIL 결과 폴더를 생성합니다."""
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


def _load_image_b64(path, max_width=1024):
    """이미지를 로드해 max_width px로 축소 후 JPEG base64 문자열로 반환합니다.
    HTML 리포트 인라인 샘플 이미지 생성에 사용됩니다."""
    try:
        import base64
        data = np.fromfile(str(path), dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > max_width:
            scale     = max_width / w
            new_w     = max_width
            new_h     = max(1, int(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode('utf-8')
    except Exception:
        return None


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


# ══════════════════════════════════════════════════════════════════
#  시각화 유틸리티
# ══════════════════════════════════════════════════════════════════

def get_class_name(names, cls_id):
    """model.names 범위를 벗어난 class id도 안전하게 표시합니다."""
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
    """640×360 debug panel에 맞춘 얇은 bbox와 작은 class label을 그립니다."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(image, (x1, y1), (x2, y2), color,
                  thickness=DEBUG_BOX_THICKNESS, lineType=cv2.LINE_AA)
    if not label:
        return

    text_size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, DEBUG_LABEL_SCALE, DEBUG_LABEL_THICKNESS)
    text_w, text_h = text_size
    pad_x, pad_y = 3, 2
    label_w = text_w + pad_x * 2
    label_h = text_h + baseline + pad_y * 2
    label_x1 = min(x1, max(0, width - label_w - 1))
    label_y1 = y1 - label_h if y1 - label_h >= 0 else y1 + DEBUG_BOX_THICKNESS
    label_y1 = min(max(0, label_y1), max(0, height - label_h - 1))
    label_x2 = label_x1 + label_w
    label_y2 = label_y1 + label_h

    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2),
                  color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(image, label,
                (label_x1 + pad_x, label_y2 - baseline - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX, DEBUG_LABEL_SCALE,
                (255, 255, 255), DEBUG_LABEL_THICKNESS, lineType=cv2.LINE_AA)


def scale_xyxy_to_panel(xyxy, src_shape):
    """원본 좌표계를 640×360 debug panel 좌표계로 변환합니다."""
    src_h, src_w = src_shape[:2]
    x_scale = DEBUG_PANEL_WIDTH / max(src_w, 1)
    y_scale = DEBUG_PANEL_HEIGHT / max(src_h, 1)
    values = [float(v) for v in xyxy]
    return [values[0]*x_scale, values[1]*y_scale, values[2]*x_scale, values[3]*y_scale]


def create_fixed_debug_panel(base_image, box_entries, title, subtitle, bg_color):
    """bbox를 640×360 기준으로 다시 그리고 heading을 붙입니다."""
    if base_image.ndim == 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)

    panel_image = cv2.resize(base_image, (DEBUG_PANEL_WIDTH, DEBUG_PANEL_HEIGHT),
                             interpolation=cv2.INTER_AREA)
    for xyxy, label, color in box_entries:
        scaled_xyxy = scale_xyxy_to_panel(xyxy, base_image.shape)
        draw_debug_box(panel_image, scaled_xyxy, label, color)

    panel = np.zeros((DEBUG_HEADING_HEIGHT + DEBUG_PANEL_HEIGHT, DEBUG_PANEL_WIDTH, 3),
                     dtype=np.uint8)
    panel[:DEBUG_HEADING_HEIGHT, :] = np.array(bg_color, dtype=np.uint8)
    panel[DEBUG_HEADING_HEIGHT:, :, :] = panel_image

    title_scale     = DEBUG_TITLE_SCALE
    title_thickness = DEBUG_TITLE_THICKNESS
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX,
                                  title_scale, title_thickness)[0]
    info_x = min(12 + title_size[0] + 18, DEBUG_PANEL_WIDTH - 1)
    subtitle_scale = fit_text_scale(
        subtitle, max_width=max(80, DEBUG_PANEL_WIDTH - info_x - 12),
        initial_scale=DEBUG_INFO_SCALE, thickness=DEBUG_INFO_THICKNESS, min_scale=0.28)

    cv2.putText(panel, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX,
                title_scale, (255, 255, 255), title_thickness, lineType=cv2.LINE_AA)
    cv2.putText(panel, subtitle, (info_x, 23), cv2.FONT_HERSHEY_SIMPLEX,
                subtitle_scale, (245, 245, 245), DEBUG_INFO_THICKNESS, lineType=cv2.LINE_AA)
    return panel


# ══════════════════════════════════════════════════════════════════
#  데이터셋 로더 (Unicode 안전)
# ══════════════════════════════════════════════════════════════════

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
            fps = self.count / elapsed_time
            remaining = self.nf - self.count
            time_info = f' | ETA: {remaining / fps:.1f}s' if fps > 0 else ''
        else:
            time_info = ''

        bar_length = 20
        filled = int(bar_length * self.count / self.nf)
        progress_bar = '█' * filled + '░' * (bar_length - filled)

        if self.video_flag[self.count]:
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:
                    raise StopIteration
                path = self.files[self.count]
                self.new_video(path)
                ret_val, img0 = self.cap.read()
            self.frame += 1
            print(f'[{progress_bar}] {percent_complete:.1f}% | Video {self.count+1}/{self.nf} '
                  f'(frame {self.frame}/{self.nframes}){time_info}: {path}')
        else:
            self.count += 1
            img0 = imread_unicode(path)
            # [FIX] assert → 명시적 예외 (PyInstaller -O 플래그에서도 안전하게 동작)
            if img0 is None:
                raise RuntimeError(f'Failed to load image: {path}')
            print(f'[{progress_bar}] {percent_complete:.1f}% | Image {self.count}/{self.nf}{time_info}: {path}')

        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return path, img, img0, self.cap


# ══════════════════════════════════════════════════════════════════
#  로깅 설정
# ══════════════════════════════════════════════════════════════════

def setup_logging(log_level=logging.INFO):
    """
    로깅 설정.
    호출할 때마다 핸들러를 새로 구성합니다.
    GUI에서 여러 번 detect()를 실행해도 핸들러가 누적되지 않습니다.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # 이전에 추가된 핸들러 모두 제거 (중복 누적 방지)
    for handler in logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)

    # StreamHandler 새로 추가
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

    # root logger 전파 방지 (중복 출력 억제)
    logger.propagate = False
    return logger


# ══════════════════════════════════════════════════════════════════
#  CUDA 장치 선택
# ══════════════════════════════════════════════════════════════════

def is_cuda_runtime_error(exc):
    """CUDA runtime/device 호환성 문제인지 확인합니다."""
    message = str(exc).lower()
    keywords = [
        'cuda error', 'no kernel image is available', 'cuda-capable device',
        'invalid device function', 'no nvidia driver', 'cuda unavailable',
    ]
    return isinstance(exc, (RuntimeError, AssertionError)) and any(k in message for k in keywords)


def cuda_smoke_test(device_index=0, logger=None):
    """실제 CUDA tensor 연산과 torchvision NMS가 가능한지 확인합니다."""
    try:
        if not torch.cuda.is_available():
            return False
        device = torch.device(f'cuda:{device_index}')
        x = torch.ones((8,), device=device)
        y = (x * 2).sum()
        torch.cuda.synchronize(device)
        from torchvision.ops import nms
        boxes  = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]], device=device)
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
    """GPU 우선, CUDA smoke test 실패 시 CPU로 fallback합니다."""
    requested = str(device_arg or '').strip().lower()
    if requested == 'cpu':
        return torch.device('cpu')

    if not torch.cuda.is_available():
        logger.warning("CUDA not available — using CPU")
        return torch.device('cpu')

    first_device = 0
    if requested:
        try:
            first_device = int(requested.split(',')[0])
        except ValueError:
            logger.warning(f"Invalid device option '{device_arg}' — using CPU")
            return torch.device('cpu')

    if first_device < 0 or first_device >= torch.cuda.device_count():
        logger.warning(f"CUDA device index out of range: {first_device} — using CPU")
        return torch.device('cpu')

    if not cuda_smoke_test(first_device, logger):
        logger.warning("CUDA runtime incompatible with this GPU/build — falling back to CPU")
        return torch.device('cpu')

    device = torch.device(f'cuda:{first_device}')
    props = torch.cuda.get_device_properties(first_device)
    logger.info(f"Using CUDA:{first_device} ({props.name}, compute {props.major}.{props.minor})")
    return device


# ══════════════════════════════════════════════════════════════════
#  모델 로드 / 추론
# ══════════════════════════════════════════════════════════════════

def load_model(weights, device, img_size, trace=True):
    """YOLOv7 모델을 로드합니다."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model: {weights}")
    model = attempt_load(weights, map_location=torch.device('cpu'))
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    model.eval()

    if device.type != 'cpu':
        model = model.to(device)

    half = device.type != 'cpu'
    if half:
        model.half()

    if trace and device.type != 'cpu':
        try:
            logger.info("Converting to TracedModel...")
            rand_example = torch.zeros(1, 3, img_size, img_size).to(device)
            rand_example = rand_example.half() if half else rand_example.float()
            model = TracedModel(model, device, img_size, rand_example)
            logger.info("TracedModel conversion succeeded")
        except Exception as e:
            logger.warning(f"TracedModel failed ({e}) — using standard model")

    if device.type != 'cpu':
        logger.info("Warming up model...")
        img = torch.zeros(1, 3, img_size, img_size).to(device)
        img = img.half() if half else img.float()
        _ = model(img)

    return model, stride, img_size, half


def process_batch(img, model, device, half, augment=False,
                  conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False):
    """단일 이미지를 추론합니다."""
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img, augment=augment)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres,
                               classes=classes, agnostic=agnostic_nms)
    return pred


def safe_scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """letterbox 좌표를 원본 이미지 좌표로 안전하게 역변환합니다."""
    if isinstance(img0_shape, torch.Tensor):
        img0_shape = img0_shape.cpu().numpy()
    if isinstance(img1_shape, torch.Tensor):
        img1_shape = img1_shape.cpu().numpy()

    if len(img0_shape) >= 3:
        img0_h, img0_w = img0_shape[0], img0_shape[1]
    else:
        img0_h, img0_w = img0_shape[0], (img0_shape[1] if len(img0_shape) > 1 else img0_shape[0])

    if len(img1_shape) >= 3:
        img1_h, img1_w = img1_shape[0], img1_shape[1]
    else:
        img1_h, img1_w = img1_shape[0], (img1_shape[1] if len(img1_shape) > 1 else img1_shape[0])

    coords_result = coords.clone() if isinstance(coords, torch.Tensor) else coords.copy()

    if ratio_pad is None:
        gain = min(img1_h / img0_h, img1_w / img0_w)
        pad = (img1_w - img0_w * gain) / 2, (img1_h - img0_h * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords_result[:, [0, 2]] -= pad[0]
    coords_result[:, [1, 3]] -= pad[1]
    coords_result[:, :4] /= gain
    coords_result[:, 0].clamp_(0, img0_w)
    coords_result[:, 1].clamp_(0, img0_h)
    coords_result[:, 2].clamp_(0, img0_w)
    coords_result[:, 3].clamp_(0, img0_h)
    return coords_result


# ══════════════════════════════════════════════════════════════════
#  평가 로직
# ══════════════════════════════════════════════════════════════════

def evaluate_detections(pred, gt_boxes, im0s, img, gn, names, conf_thres, iou_thres,
                        colors=None, visualize=False, min_recall=0.8, classes=None):
    """
    예측 결과를 GT와 비교하여 good_detect / miss_detect / false_detect /
    background / low_conf 중 하나로 분류합니다.
    """
    logger = logging.getLogger(__name__)

    if colors is None:
        colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    result = {
        'category':     'unknown',
        'matched_gt':   set(),
        'matched_pred': set(),
        'precision':    0.0,
        'recall':       0.0,
        'pred_info':    [],
        'bbox_count':   0,
        'visualization': None,
    }

    det = pred[0] if len(pred) > 0 and pred[0] is not None else None

    im_pred = im0s.copy()
    im_gt   = im0s.copy()
    pred_visual_entries = []
    gt_visual_entries   = []

    # ── GT 필터링 ────────────────────────────────────────────────
    selected_classes = [int(c) for c in classes] if classes is not None else None
    if selected_classes is not None:
        filtered_gt_boxes = [(i, b) for i, b in enumerate(gt_boxes)
                             if int(b[0]) in selected_classes]
    else:
        filtered_gt_boxes = list(enumerate(gt_boxes))

    for gt_idx, gt_box in filtered_gt_boxes:
        try:
            xyxy = xywhn2xyxy(torch.tensor(gt_box[1:]).unsqueeze(0),
                              w=im_gt.shape[1], h=im_gt.shape[0]).view(-1).tolist()
            if visualize:
                cls_id = int(gt_box[0])
                gt_visual_entries.append((xyxy, f'{get_class_name(names, cls_id)} GT',
                                          get_class_color(colors, cls_id)))
        except Exception as e:
            logger.error(f"Error processing GT box {gt_idx}: {e}")

    # ── 예측 처리 ────────────────────────────────────────────────
    filtered_det = []          # ← 반드시 if 블록 외부에서 초기화
    all_conf_good = True

    if det is not None and len(det) > 0:
        try:
            img0_shape = im0s.shape[:2] if len(im0s.shape) == 3 else im0s.shape
            img1_shape = img.shape[2:]  if len(img.shape) == 4  else img.shape[1:]
            det[:, :4] = safe_scale_coords(img1_shape, det[:, :4], img0_shape).round()
        except Exception as e:
            logger.error(f"Coordinate scaling failed: {e}")

        # 클래스 필터링 + 클래스 ID 리매핑 (버그 수정: and → in)
        for i, (*xyxy, conf, cls) in enumerate(det):
            cls_id = int(cls)
            # [FIX] 원래 'and' 조건은 수학적으로 불가능 → 'in (...)' 으로 수정
            if cls_id in (2, 5, 7):
                cls_id = 1
            elif cls_id == 14:
                cls_id = 2
            elif cls_id in (15, 16):
                cls_id = 3

            if selected_classes is None or cls_id in selected_classes:
                filtered_det.append((i, (*xyxy, conf, torch.tensor(float(cls_id)))))

        all_conf_good = all(conf >= conf_thres for _, (*_, conf, _) in filtered_det)

        for pred_idx, (*xyxy, conf, cls) in filtered_det:
            cls_id = int(cls)
            pred_bbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            result['pred_info'].append((cls_id, pred_bbox, conf.item()))
            if visualize:
                label = f'{get_class_name(names, cls_id)} {conf:.2f}'
                pred_visual_entries.append((xyxy, label, get_class_color(colors, cls_id)))

        # IoU 매칭
        for pred_idx, (*xyxy, conf, cls) in filtered_det:
            pred_cls  = int(cls)
            pred_bbox = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            best_iou, best_gt_idx = 0.0, -1
            for orig_gt_idx, gt_box in filtered_gt_boxes:
                if pred_cls == int(gt_box[0]):
                    iou = calculate_iou(pred_bbox, gt_box[1:])
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, orig_gt_idx
            if best_iou > iou_thres and best_gt_idx >= 0:
                result['matched_gt'].add(best_gt_idx)
                result['matched_pred'].add(pred_idx)

        # ── 클래스별 어노테이션 레벨 통계 ────────────────────────
        per_class_stats = {}

        for orig_gt_idx, gt_box in filtered_gt_boxes:
            cls_id = int(gt_box[0])
            s = per_class_stats.setdefault(
                cls_id, {'gt': 0, 'matched_gt': 0, 'miss': 0, 'false': 0})
            s['gt'] += 1
            if orig_gt_idx in result['matched_gt']:
                s['matched_gt'] += 1
            else:
                s['miss'] += 1

        for pred_idx, (*xyxy, conf, cls) in filtered_det:
            if pred_idx not in result['matched_pred']:
                cls_id = int(cls)
                s = per_class_stats.setdefault(
                    cls_id, {'gt': 0, 'matched_gt': 0, 'miss': 0, 'false': 0})
                s['false'] += 1

        result['per_class_stats'] = per_class_stats

    # ── Precision / Recall 계산 ──────────────────────────────────
    filtered_gt_count   = len(filtered_gt_boxes)
    filtered_pred_count = len(filtered_det)

    if filtered_gt_count > 0:
        result['recall'] = len(result['matched_gt']) / filtered_gt_count
    else:
        result['recall'] = 1.0 if filtered_pred_count == 0 else 0.0

    if filtered_pred_count > 0:
        result['precision'] = len(result['matched_pred']) / filtered_pred_count
    else:
        result['precision'] = 1.0 if filtered_gt_count == 0 else 0.0

    # ── 카테고리 결정 ────────────────────────────────────────────
    if filtered_gt_count > 0:
        if result['recall'] >= min_recall:
            result['category'] = 'good_detect' if all_conf_good else 'low_conf'
        elif result['recall'] > 0:
            result['category'] = 'miss_detect'
        else:
            result['category'] = 'false_detect'
    else:
        result['category'] = 'false_detect' if filtered_pred_count > 0 else 'background'

    result['bbox_count'] = filtered_pred_count

    # ── 시각화 생성 ──────────────────────────────────────────────
    if visualize:
        try:
            pred_sub = (f"Precision {result['precision']:.2f} | "
                        f"Recall {result['recall']:.2f} | Boxes {filtered_pred_count}")
            gt_sub = f"Ground truth boxes {filtered_gt_count}"
            im_pred_panel = create_fixed_debug_panel(
                im_pred, pred_visual_entries, "PRED", pred_sub, (42, 96, 42))
            im_gt_panel = create_fixed_debug_panel(
                im_gt, gt_visual_entries, "GT", gt_sub, (96, 56, 36))
            result['visualization'] = np.hstack((im_pred_panel, im_gt_panel))
        except Exception as e:
            logger.error(f"Visualization error: {e}")

    return result


# ══════════════════════════════════════════════════════════════════
#  라벨 경로 탐색
# ══════════════════════════════════════════════════════════════════

def _find_label_path(image_path: Path):
    """
    이미지 경로에 대응하는 GT 라벨 파일(.txt) 경로를 탐색합니다.

    탐색 전략 (순서대로 시도):
    1. 이미지와 같은 디렉터리의 .txt
    2. 이미지 경로의 'JPEGImages' 부분을 'labels'로 교체 (대소문자 무시)
    3. 부모 폴더에 labels/ 서브폴더가 있으면 그 안의 .txt
    """
    stem = image_path.stem

    # 1) 같은 디렉터리
    candidate = image_path.with_suffix('.txt')
    if candidate.exists():
        return candidate

    # 2) 경로 문자열에서 JPEGImages → labels 교체 (대소문자 무시)
    parts = image_path.parts
    for i, part in enumerate(parts):
        if part.lower() == 'jpegimages':
            new_parts = parts[:i] + ('labels',) + parts[i+1:]
            candidate = Path(*new_parts).with_suffix('.txt')
            if candidate.exists():
                return candidate
            break

    # 3) 부모 디렉터리의 labels/ 서브폴더
    labels_dir = image_path.parent.parent / 'labels'
    candidate = labels_dir / (stem + '.txt')
    if candidate.exists():
        return candidate

    return None


# ══════════════════════════════════════════════════════════════════
#  결과 저장
# ══════════════════════════════════════════════════════════════════

def save_results(result, path, target_dir, label_path=None):
    """이미지와 라벨을 GOOD / MISS / FAIL 폴더에 복사합니다."""
    logger = logging.getLogger(__name__)
    p = Path(path)
    target_img_path = target_dir / 'JPEGImages' / p.name
    os.makedirs(target_img_path.parent, exist_ok=True)

    # Unicode 경로 안전 복사
    try:
        img_data = np.fromfile(str(path), dtype=np.uint8)
        img_data.tofile(str(target_img_path))
    except Exception:
        try:
            shutil.copy(str(path), str(target_img_path))
        except Exception as e:
            logger.error(f"Failed to copy image {path}: {e}")

    if label_path and os.path.exists(str(label_path)):
        target_label_path = target_dir / 'labels' / (p.stem + '.txt')
        os.makedirs(target_label_path.parent, exist_ok=True)
        try:
            shutil.copy(str(label_path), str(target_label_path))
        except Exception as e:
            logger.error(f"Failed to copy label {label_path}: {e}")

    return target_img_path
def write_category_list_files(result_dirs, category_items, logger):
    """
    그룹 폴더(GOOD/MISS/FAIL) 내부에 카테고리별 이미지 목록 txt를 저장합니다.
    예) .../GOOD/good_detect_list.txt
    """
    for category in RESULT_CATEGORIES:
        group_name = RESULT_GROUPS.get(category, 'FAIL')
        group_dir = result_dirs.get(group_name)
        if group_dir is None:
            continue

        list_path = Path(group_dir) / f"{category}_list.txt"
        items = category_items.get(category, [])
        try:
            with open(list_path, 'w', encoding='utf-8') as f:
                for name in items:
                    f.write(f"{name}\n")
            logger.info(f"Saved category list: {list_path} ({len(items)} items)")
        except Exception as e:
            logger.error(f"Failed to save category list {list_path}: {e}")


# ══════════════════════════════════════════════════════════════════
#  HTML 리포트 생성
# ══════════════════════════════════════════════════════════════════

def generate_html_report(save_dir, stats, group_stats, total_images,
                          successful_images, processing_time, opt,
                          interrupted=False, category_items=None,
                          sample_limit=5, sample_vis_paths=None):
    """검출 결과를 담은 HTML 리포트를 생성합니다."""
    save_dir = Path(save_dir)

    def pct(n):
        return f"{n / total_images * 100:.1f}" if total_images else "0.0"

    def fmt_time(sec):
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    good = group_stats.get('GOOD', 0)
    miss = group_stats.get('MISS', 0)
    fail = group_stats.get('FAIL', 0)
    spi  = processing_time / total_images if total_images else 0.0
    now  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    interrupted_banner = (
        '<div class="banner-warn">⚠ 검출이 중단되었습니다. 결과는 불완전할 수 있습니다.</div>'
        if interrupted else ''
    )

    import html as _html  # HTML 이스케이프 (파일명 특수문자 대응)

    classes_str = ', '.join(str(c) for c in opt.classes) if opt.classes else '전체'
    category_labels = {
        'good_detect': '정상 검출',
        'background': '배경(검출 없음)',
        'miss_detect': '미검출',
        'false_detect': '오검출',
        'low_conf': '저신뢰 검출',
    }
    category_items = category_items or {}
    diff_categories = ['miss_detect', 'false_detect', 'low_conf']

    category_rows = ""
    for cat in RESULT_CATEGORIES:
        items = category_items.get(cat, [])
        display_name = category_labels.get(cat, cat)
        # [FIX] 파일명에 <, >, & 등 특수문자가 있어도 HTML이 깨지지 않도록 이스케이프
        escaped_items = [_html.escape(name) for name in items[:20]]
        items_html = '<br>'.join(escaped_items) if escaped_items else '-'
        category_rows += (
            f"<tr><td>{display_name}</td><td>{len(items)}개</td>"
            f"<td>{items_html}</td></tr>"
        )

    diff_preview_html = ""
    for cat in diff_categories:
        items = category_items.get(cat, [])
        if not items:
            continue
        display_name = category_labels.get(cat, cat)
        diff_preview_html += f"<h3>{display_name} 샘플 ({min(len(items), sample_limit)} / {len(items)}개)</h3><ul>"
        for item in items[:sample_limit]:
            # [FIX] 파일명 HTML 이스케이프
            diff_preview_html += f"<li>{_html.escape(item)}</li>"
        diff_preview_html += "</ul>"

    # 샘플 이미지 HTML 생성 (base64 임베드)
    _svp = sample_vis_paths or {}
    sample_images_html = ""
    for cat in diff_categories:
        paths      = _svp.get(cat, [])
        items_list = category_items.get(cat, [])
        total_cnt  = len(items_list)
        display_name = category_labels.get(cat, cat)

        sample_images_html += f'<h3>{display_name} &nbsp;<span class="cnt">({len(paths)}/{total_cnt}개 표시)</span></h3>'
        if paths:
            sample_images_html += '<div class="img-sample">'
            for p in paths:
                b64 = _load_image_b64(p)
                fn  = _html.escape(Path(p).name) if b64 else ''
                if b64:
                    sample_images_html += (
                        f'<img src="data:image/jpeg;base64,{b64}" alt="{fn}" '
                        f'title="{fn}">'
                    )
            sample_images_html += '</div>'
        elif total_cnt:
            sample_images_html += (
                '<p class="no-sample">샘플 이미지 없음'
                ' — <code>debug 이미지 저장</code> 옵션 활성화 시 자동으로 수집됩니다.</p>'
            )
        else:
            sample_images_html += '<p class="no-sample">해당 카테고리 없음</p>'

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detection Report — {save_dir.name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#0d1117;color:#e6edf3;padding:24px 16px;line-height:1.5}}
.wrap{{max-width:960px;margin:0 auto}}
h1{{font-size:22px;color:#58a6ff;padding-bottom:12px;
    border-bottom:1px solid #30363d;margin-bottom:8px}}
.meta{{font-size:12px;color:#8b949e;margin-bottom:24px}}
.banner-warn{{background:#2a1515;border:1px solid #da3633;color:#f85149;
              padding:10px 16px;border-radius:6px;margin-bottom:16px;font-size:13px}}
.cards{{display:flex;gap:12px;margin-bottom:24px;flex-wrap:wrap}}
.card{{flex:1;min-width:180px;padding:20px;border-radius:8px;
       border:1px solid #30363d;background:#161b22}}
.card.good{{background:#0d2218;border-color:#2ea043}}
.card.miss{{background:#2a1f0a;border-color:#9e6a03}}
.card.fail{{background:#1a0c0c;border-color:#da3633}}
.card-title{{font-size:11px;font-weight:700;letter-spacing:1px;
             text-transform:uppercase;margin-bottom:6px}}
.card.good .card-title{{color:#3fb950}}
.card.miss .card-title{{color:#d29922}}
.card.fail .card-title{{color:#f85149}}
.card.total .card-title{{color:#8b949e}}
.card-num{{font-size:36px;font-weight:700;line-height:1}}
.card-pct{{font-size:13px;color:#8b949e;margin-top:4px}}
.bar{{height:5px;background:#21262d;border-radius:3px;
      margin-top:12px;overflow:hidden}}
.bar-g{{height:100%;background:#3fb950}}
.bar-m{{height:100%;background:#d29922}}
.bar-f{{height:100%;background:#f85149}}
.bar-t{{height:100%;background:#388bfd}}
h2{{font-size:16px;margin:28px 0 12px;color:#cdd6f4;
    border-left:3px solid #388bfd;padding-left:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:24px}}
th,td{{padding:9px 12px;border:1px solid #21262d;text-align:left}}
th{{background:#161b22;color:#8b949e;font-weight:600;font-size:11px}}
tr:nth-child(odd){{background:#0d1117}}
tr:nth-child(even){{background:#161b22}}
h3{{font-size:14px;margin:14px 0 8px;color:#8ab4f8}}
ul{{margin:0 0 14px 20px}}
li{{margin-bottom:4px;font-size:13px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px}}
.info-box{{padding:14px 16px;background:#161b22;border-radius:8px;
           border:1px solid #30363d}}
.info-lbl{{font-size:11px;color:#8b949e;margin-bottom:4px}}
.info-val{{font-size:14px;font-weight:600;word-break:break-all}}
.img-sample{{margin-bottom:16px}}
.img-sample img{{width:100%;border-radius:4px;border:1px solid #30363d;
                 margin-bottom:8px;display:block}}
.no-sample{{color:#8b949e;font-size:13px;font-style:italic;margin:4px 0 12px}}
.cnt{{font-size:12px;color:#8b949e;font-weight:normal}}
.footer{{text-align:center;font-size:11px;color:#8b949e;
         margin-top:32px;padding-top:16px;border-top:1px solid #21262d}}
</style>
</head>
<body>
<div class="wrap">
  <h1>🔍 검출 리포트</h1>
  <div class="meta">생성: {now} &nbsp;|&nbsp; 저장 위치: {save_dir} &nbsp;|&nbsp; v{APP_VERSION}</div>
  {interrupted_banner}

  <div class="cards">
    <div class="card good">
      <div class="card-title">GOOD</div>
      <div class="card-num">{good}</div>
      <div class="card-pct">{pct(good)}%</div>
      <div class="bar"><div class="bar-g" style="width:{pct(good)}%"></div></div>
    </div>
    <div class="card miss">
      <div class="card-title">미검출</div>
      <div class="card-num">{miss}</div>
      <div class="card-pct">{pct(miss)}%</div>
      <div class="bar"><div class="bar-m" style="width:{pct(miss)}%"></div></div>
    </div>
    <div class="card fail">
      <div class="card-title">오검출</div>
      <div class="card-num">{fail}</div>
      <div class="card-pct">{pct(fail)}%</div>
      <div class="bar"><div class="bar-f" style="width:{pct(fail)}%"></div></div>
    </div>
    <div class="card total">
      <div class="card-title">총 처리</div>
      <div class="card-num">{total_images}</div>
      <div class="card-pct">성공 {successful_images}장 ({pct(successful_images)}%)</div>
      <div class="bar"><div class="bar-t" style="width:{pct(successful_images)}%"></div></div>
    </div>
  </div>

  <h2>세부 카테고리</h2>
  <table>
    <thead>
      <tr><th>카테고리</th><th>그룹</th><th>건수</th><th>비율</th></tr>
    </thead>
    <tbody>
      <tr><td>good_detect</td><td style="color:#3fb950">정상</td>
          <td>{stats.get('good_detect',0)}</td><td>{pct(stats.get('good_detect',0))}%</td></tr>
      <tr><td>background</td><td style="color:#3fb950">정상</td>
          <td>{stats.get('background',0)}</td><td>{pct(stats.get('background',0))}%</td></tr>
      <tr><td>miss_detect</td><td style="color:#d29922">미검출</td>
          <td>{stats.get('miss_detect',0)}</td><td>{pct(stats.get('miss_detect',0))}%</td></tr>
      <tr><td>false_detect</td><td style="color:#f85149">오검출</td>
          <td>{stats.get('false_detect',0)}</td><td>{pct(stats.get('false_detect',0))}%</td></tr>
      <tr><td>low_conf</td><td style="color:#f85149">오검출</td>
          <td>{stats.get('low_conf',0)}</td><td>{pct(stats.get('low_conf',0))}%</td></tr>
    </tbody>
  </table>

  <h2>카테고리별 데이터 목록</h2>
  <table>
    <thead>
      <tr><th>카테고리</th><th>건수</th><th>이미지 목록(최대 20개)</th></tr>
    </thead>
    <tbody>
      {category_rows}
    </tbody>
  </table>

  <h2>차이점 샘플 이미지</h2>
  <div class="info-box" style="margin-bottom:16px">
    <div class="info-lbl">확인 가이드</div>
    <div class="info-val">각 카테고리에서 최대 {sample_limit}장을 샘플로 보여줍니다.
      왼쪽 패널: 모델 예측(PRED) &nbsp;|&nbsp; 오른쪽 패널: 실제 정답(GT)</div>
  </div>
  {sample_images_html}

  <h2>처리 정보</h2>
  <div class="grid2">
    <div class="info-box">
      <div class="info-lbl">총 처리 시간</div>
      <div class="info-val">{fmt_time(processing_time)}</div>
    </div>
    <div class="info-box">
      <div class="info-lbl">이미지당 처리 시간</div>
      <div class="info-val">{spi:.3f} s/img</div>
    </div>
    <div class="info-box">
      <div class="info-lbl">모델 가중치</div>
      <div class="info-val">{opt.weights}</div>
    </div>
    <div class="info-box">
      <div class="info-lbl">추론 설정</div>
      <div class="info-val">
        conf {opt.conf_thres} &nbsp;|&nbsp; iou {opt.iou_thres} &nbsp;|&nbsp;
        recall {opt.min_recall}<br>img-size {opt.img_size} &nbsp;|&nbsp; classes [{classes_str}]
      </div>
    </div>
  </div>

  <div class="footer">Detect Label 22 v{APP_VERSION} — 객체검출 모델 평가 도구</div>
</div>
</body>
</html>"""

    report_path = save_dir / 'report.html'
    report_path.write_text(html, encoding='utf-8')
    return report_path


# ══════════════════════════════════════════════════════════════════
#  메인 검출 루프
# ══════════════════════════════════════════════════════════════════

def detect(opt, stop_event=None, progress_callback=None):
    """
    메인 검출 함수.
    progress_callback(current, total, category, stats, group_stats, elapsed, eta)
    형태의 콜백을 지원합니다 (GUI 연동용).
    """
    logger = setup_logging()
    logger.info(f"detect_label_22 v{APP_VERSION}  |  options: {opt}")
    is_windows = platform.system() == 'Windows'
    if opt.view_img and not is_windows:
        logger.warning("view_img is only supported on Windows — disabling.")

    # ── 저장 디렉터리 ────────────────────────────────────────────
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(save_dir / 'detection_log.txt', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # ── 장치 선택 ────────────────────────────────────────────────
    # [NOTE] fh는 함수 종료 시 반드시 닫힘 (finally 블록에서 처리)
    device = resolve_runtime_device(opt.device, logger)

    # ── 모델 로드 (CUDA 실패 시 CPU 재시도) ─────────────────────
    try:
        model, stride, imgsz, half = load_model(
            opt.weights, device, opt.img_size, trace=not opt.no_trace)
    except Exception as exc:
        if device.type == 'cuda' and is_cuda_runtime_error(exc):
            logger.warning(f"CUDA model init failed — retrying on CPU: {exc}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            device = torch.device('cpu')
            model, stride, imgsz, half = load_model(
                opt.weights, device, opt.img_size, trace=False)
        else:
            raise

    names  = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # ── 데이터셋 ─────────────────────────────────────────────────
    try:
        source_txt = prepare_source_txt(opt.source, save_dir, logger)
        dataset    = UnicodeSafeLoadImagestxt(source_txt, img_size=imgsz, stride=stride)
        logger.info(f"Dataset: {len(dataset)} images")
    except Exception as e:
        logger.error(f"Dataset load failed: {e}")
        return None

    # ── 결과 폴더 ────────────────────────────────────────────────
    try:
        result_dirs = create_grouped_result_dirs(opt.savedirs)
        debug_dirs  = {}
        for gname, gdir in result_dirs.items():
            dd = gdir / 'debug_images'
            dd.mkdir(exist_ok=True, parents=True)
            debug_dirs[gname] = dd
        logger.info(f"Result dirs created: {opt.savedirs}")
    except Exception as e:
        logger.error(f"Result dir creation failed: {e}")
        return None

    # ── 통계 초기화 ──────────────────────────────────────────────
    stats         = {cat: 0 for cat in RESULT_CATEGORIES}
    category_items = {cat: [] for cat in RESULT_CATEGORIES}
    group_stats   = {gname: 0 for gname in sorted(set(RESULT_GROUPS.values()))}
    # 리포트 샘플 이미지 수집 (카테고리별 최대 REPORT_SAMPLE_LIMIT장)
    sample_vis_paths    = {cat: [] for cat in ['miss_detect', 'false_detect', 'low_conf']}
    report_samples_dir  = save_dir / 'report_samples'
    # 클래스별 GT 어노테이션 레벨 통계
    global_class_stats: dict = {}
    total_bb      = 0
    debug_saved   = 0
    debug_failed  = 0
    total_images  = 0
    success_images= 0
    interrupted   = False
    t0            = time.time()

    # ── progress_callback 있으면 tqdm 생략 ───────────────────────
    iterable = dataset if progress_callback else tqdm(dataset, desc="Processing")

    # ── 이미지 루프 ──────────────────────────────────────────────
    for idx, item in enumerate(iterable):
        if stop_event is not None and stop_event.is_set():
            interrupted = True
            logger.warning("Stop requested — breaking.")
            break

        path     = None
        category = 'skip'
        try:
            if item is None:
                continue
            path, img, im0s, vid_cap = item
            if img is None or im0s is None:
                continue
            if not (hasattr(img, 'shape') and hasattr(im0s, 'shape')):
                continue
            if len(img.shape) == 0:
                continue

            total_images += 1
            logger.info(f"[{total_images}/{len(dataset)}] {path}")

            # ── GT 라벨 로드 ──────────────────────────────────
            image_path = Path(path)
            label_path = _find_label_path(image_path)
            if label_path is None:
                logger.warning(f"Label not found: {image_path.with_suffix('.txt')}")
                gt_boxes = []
            else:
                gt_boxes = read_label_file(str(label_path))

            # ── 추론 (CUDA 실패 시 CPU 재시도) ────────────────
            try:
                pred = process_batch(
                    img, model, device, half,
                    augment=opt.augment,
                    conf_thres=opt.conf_thres,
                    iou_thres=opt.iou_thres,
                    classes=opt.classes,
                    agnostic_nms=opt.agnostic_nms,
                )
            except Exception as exc:
                if device.type == 'cuda' and is_cuda_runtime_error(exc):
                    logger.warning(f"CUDA inference failed — switching to CPU: {exc}")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    device = torch.device('cpu')
                    model, stride, imgsz, half = load_model(
                        opt.weights, device, opt.img_size, trace=False)
                    pred = process_batch(
                        img, model, device, half,
                        augment=opt.augment,
                        conf_thres=opt.conf_thres,
                        iou_thres=opt.iou_thres,
                        classes=opt.classes,
                        agnostic_nms=opt.agnostic_nms,
                    )
                else:
                    raise

            # gn: 정규화 게인 [w, h, w, h]
            if len(im0s.shape) == 3:
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
            else:
                h, w = im0s.shape
                gn = torch.tensor([w, h, w, h])

            # ── 평가 ──────────────────────────────────────────
            # 샘플이 아직 부족하면 visualize 강제 활성화 (최초 처리 이미지 한정)
            _still_need = (
                total_images <= REPORT_SAMPLE_LIMIT * 30 and
                any(len(sample_vis_paths[c]) < REPORT_SAMPLE_LIMIT
                    for c in sample_vis_paths)
            )
            result = evaluate_detections(
                pred, gt_boxes, im0s, img, gn, names,
                opt.conf_thres, opt.iou_thres, colors,
                visualize=(opt.view_img or opt.save_debug_images or _still_need),
                min_recall=opt.min_recall,
                classes=opt.classes,
            )

            # ── 통계 업데이트 ──────────────────────────────────
            category   = result['category']
            group_name = RESULT_GROUPS.get(category, 'FAIL')
            stats[category]        += 1
            group_stats[group_name]+= 1
            category_items[category].append(str(Path(path).name))
            total_bb += int(result.get('bbox_count', len(result.get('pred_info', []))))

            # 클래스별 통계 누적
            for _cid, _cs in result.get('per_class_stats', {}).items():
                _gs = global_class_stats.setdefault(
                    _cid, {'gt': 0, 'matched_gt': 0, 'miss': 0, 'false': 0})
                for _k in ('gt', 'matched_gt', 'miss', 'false'):
                    _gs[_k] += _cs[_k]

            # 리포트용 샘플 이미지 저장 (MISS/FAIL 카테고리만, 슬롯이 남은 경우)
            if (
                category in sample_vis_paths
                and len(sample_vis_paths[category]) < REPORT_SAMPLE_LIMIT
                and result['visualization'] is not None
            ):
                try:
                    report_samples_dir.mkdir(parents=True, exist_ok=True)
                    sn  = len(sample_vis_paths[category])
                    sp  = report_samples_dir / f"{category}_{sn}.jpg"
                    if imwrite_unicode(sp, result['visualization']):
                        sample_vis_paths[category].append(sp)
                except Exception as _se:
                    logger.warning(f"샘플 이미지 저장 실패: {_se}")

            # ── 결과 저장 ──────────────────────────────────────
            if not opt.nosave:
                save_results(result, path, result_dirs[group_name],
                             str(label_path) if label_path else None)

            # ── 디버그 이미지 ──────────────────────────────────
            if opt.save_debug_images and result['visualization'] is not None:
                dbg_path = make_debug_image_path(debug_dirs[group_name], path)
                try:
                    if imwrite_unicode(dbg_path, result['visualization']):
                        debug_saved += 1
                    else:
                        debug_failed += 1
                        logger.error(f"Failed to encode debug image: {dbg_path}")
                except Exception as e:
                    debug_failed += 1
                    logger.error(f"Debug image save error: {e}")

            # ── 화면 표시 ──────────────────────────────────────
            if opt.view_img and result['visualization'] is not None and is_windows:
                cv2.imshow('Detection Evaluation',
                           cv2.resize(result['visualization'], (1600, 900)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            success_images += 1

        except Exception as e:
            logger.error(f"Error processing {path or idx}: {e}")
            logger.error(traceback.format_exc())

        # ── progress callback (항상 호출) ────────────────────────
        if progress_callback:
            elapsed = time.time() - t0
            per_img = elapsed / (idx + 1) if idx >= 0 else 0
            eta     = per_img * (len(dataset) - idx - 1)
            try:
                progress_callback(
                    idx + 1, len(dataset), category,
                    dict(stats), dict(group_stats), elapsed, eta,
                    {k: dict(v) for k, v in global_class_stats.items()},
                )
            except Exception:
                pass

    # ── 최종 통계 ────────────────────────────────────────────────
    proc_time = time.time() - t0
    spi = proc_time / total_images if total_images else 0.0

    logger.info(f"Finished — {total_images} images in {proc_time:.2f}s ({spi:.3f}s/img)")
    logger.info(f"Success: {success_images}/{total_images}  |  stats: {stats}")
    logger.info(f"Groups: {group_stats}  |  total bbox: {total_bb}")
    logger.info(f"Debug: saved={debug_saved} failed={debug_failed}  |  interrupted={interrupted}")

    def r(n): return n / total_images if total_images else 0.0

    # ── summary.txt ──────────────────────────────────────────────
    with open(save_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"Detect Label 22  v{APP_VERSION}\n")
        f.write(f"생성: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"총 이미지: {total_images}\n")
        f.write(f"성공: {success_images}\n")
        f.write(f"중단: {'예' if interrupted else '아니오'}\n")
        f.write(f"처리 시간: {proc_time:.2f}s  ({spi:.3f}s/img)\n\n")
        f.write(f"GOOD      : {group_stats['GOOD']}개  ({r(group_stats['GOOD']):.1%})\n")
        f.write(f"MISS(미감지): {group_stats['MISS']}개  ({r(group_stats['MISS']):.1%})\n")
        f.write(f"FAIL(오감지): {group_stats['FAIL']}개  ({r(group_stats['FAIL']):.1%})\n")
        f.write(f"총 bbox: {total_bb}개\n\n")
        f.write("[세부 카테고리]\n")
        for cat in RESULT_CATEGORIES:
            f.write(f"  {cat}: {stats[cat]}개  ({r(stats[cat]):.1%})\n")

    # ── 그룹 폴더별 카테고리 리스트 txt 저장 ───────────────────────
    write_category_list_files(result_dirs, category_items, logger)

    # ── HTML 리포트 ───────────────────────────────────────────────
    try:
        report_path = generate_html_report(
            save_dir, stats, group_stats, total_images,
            success_images, proc_time, opt, interrupted,
            category_items=category_items, sample_limit=REPORT_SAMPLE_LIMIT,
            sample_vis_paths=sample_vis_paths,
        )
        logger.info(f"Report: {report_path}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")

    if opt.view_img and is_windows:
        cv2.destroyAllWindows()

    # [FIX] FileHandler 명시적 닫기 — Windows에서 재실행 시 파일 잠금 방지
    try:
        fh.close()
        logger.removeHandler(fh)
    except Exception:
        pass

    return save_dir


# ══════════════════════════════════════════════════════════════════
#  CLI 진입점
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv7 Detection Evaluation')
    parser.add_argument('--weights',     type=str, default='best.pt')
    parser.add_argument('--source',      type=str, default='data/images')
    parser.add_argument('--img-size',    type=int, default=1280)
    parser.add_argument('--conf-thres',  type=float, default=0.25)
    parser.add_argument('--iou-thres',   type=float, default=0.45)
    parser.add_argument('--min-recall',  type=float, default=0.8)
    parser.add_argument('--device',      default='')
    parser.add_argument('--view-img',    action='store_true')
    parser.add_argument('--save-debug-images', action='store_true')
    parser.add_argument('--save-txt',    action='store_true')
    parser.add_argument('--nosave',      action='store_true')
    parser.add_argument('--classes',     nargs='+', type=int, default=[0, 1, 3, 4])
    parser.add_argument('--augment',     action='store_true')
    parser.add_argument('--project',     default='runs/detect')
    parser.add_argument('--name',        default='exp')
    parser.add_argument('--exist-ok',    action='store_true')
    parser.add_argument('--no-trace',    action='store_true')
    parser.add_argument('--agnostic-nms',action='store_true')
    parser.add_argument('--savedirs',    type=str, default='outputs/sorted')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        opt = parse_args()
        print(opt)
        with torch.no_grad():
            detect(opt)
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
