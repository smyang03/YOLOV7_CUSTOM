"""
test_valid.py - 다중 검증 세트 테스트
test.py의 test() 함수를 그대로 사용하되,
data yaml의 val이 리스트이면 각 val별로 테스트 후 요약 출력

Usage:
    python test_valid.py --data data/custom.yaml --weights best.pt

data yaml 예시:
    val:
      - ./data/images/val_indoor/
      - ./data/images/val_outdoor/
"""

import argparse
import yaml
from pathlib import Path

from utils.general import check_file
from test import test  # test.py의 test() 함수 재사용


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test_valid.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)
    print(opt)

    # yaml 로드
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    task_key = opt.task if opt.task in ('train', 'val', 'test') else 'val'
    val_data = data_dict.get(task_key, data_dict.get('val'))

    if isinstance(val_data, list) and len(val_data) > 1:
        # ── 다중 검증 세트 ──────────────────────────────────────
        print(f'\n다중 검증 세트 감지: {len(val_data)}개')
        original_name = opt.name
        all_results = []

        for i, val_path in enumerate(val_data):
            val_name = Path(val_path).stem
            print(f'\n{"="*60}')
            print(f'[{i+1}/{len(val_data)}] 검증 세트: {val_name}')
            print(f'경로: {val_path}')
            print(f'{"="*60}')

            # 해당 val 경로만 담은 data dict 생성
            data_single = data_dict.copy()
            data_single[task_key] = val_path

            # 결과 저장 폴더명에 val 이름 포함
            opt.name = f'{original_name}_{val_name}'

            results = test(data_single,
                           opt.weights,
                           opt.batch_size,
                           opt.img_size,
                           opt.conf_thres,
                           opt.iou_thres,
                           opt.save_json,
                           opt.single_cls,
                           opt.augment,
                           opt.verbose,
                           save_txt=opt.save_txt | opt.save_hybrid,
                           save_hybrid=opt.save_hybrid,
                           save_conf=opt.save_conf,
                           trace=not opt.no_trace,
                           v5_metric=opt.v5_metric
                           )
            all_results.append((val_name, results))

        opt.name = original_name

        # 전체 요약 출력
        print(f'\n{"="*60}')
        print('전체 검증 세트 요약')
        print(f'{"="*60}')
        print(('%25s' + '%12s' * 4) % ('Val Set', 'P', 'R', 'mAP@.5', 'mAP@.5:.95'))
        pf = '%25s' + '%12.3g' * 4
        for val_name, (res, maps, t, _) in all_results:
            mp, mr, map50, map_ = res[:4]
            print(pf % (val_name, mp, mr, map50, map_))

    else:
        # ── 단일 val - test.py와 동일하게 동작 ─────────────────
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric
             )
