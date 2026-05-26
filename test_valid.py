"""
test_valid.py - 다중 weight × 다중 val 세트 테스트
test.py의 test() 함수를 재사용하여
weight 여러 개 × val 여러 개 조합을 모두 테스트 후 요약 표 출력

Usage:
    # weight 1개, val 여러 개
    python test_valid.py --data data/custom.yaml --weights best.pt

    # weight 여러 개, val 여러 개
    python test_valid.py --data data/custom.yaml --weights model_A.pt model_B.pt model_C.pt

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

    task_key  = opt.task if opt.task in ('train', 'val', 'test') else 'val'
    val_data  = data_dict.get(task_key, data_dict.get('val'))
    val_list  = val_data if isinstance(val_data, list) else [val_data]
    wt_list   = opt.weights if isinstance(opt.weights, list) else [opt.weights]

    original_name = opt.name

    # results_table[weight_name][val_name] = (mp, mr, map50, map_)
    results_table = {}

    for wi, weight in enumerate(wt_list):
        w_name = Path(weight).stem
        results_table[w_name] = {}

        print(f'\n{"#"*60}')
        print(f'[Weight {wi+1}/{len(wt_list)}] {weight}')
        print(f'{"#"*60}')

        for vi, val_path in enumerate(val_list):
            val_name = Path(val_path).stem

            print(f'\n{"="*60}')
            print(f'  [Val {vi+1}/{len(val_list)}] {val_name}')
            print(f'  경로: {val_path}')
            print(f'{"="*60}')

            # 해당 val 경로만 담은 data dict
            data_single = data_dict.copy()
            data_single[task_key] = val_path

            # 결과 저장 폴더: exp_modelA_val_indoor
            opt.name    = f'{original_name}_{w_name}_{val_name}'
            opt.weights = [weight]

            res, maps, t, _ = test(data_single,
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
            mp, mr, map50, map_ = res[:4]
            results_table[w_name][val_name] = (mp, mr, map50, map_)

    opt.name    = original_name
    opt.weights = wt_list

    # ── 전체 요약 표 출력 ───────────────────────────────────────
    val_names = [Path(v).stem for v in val_list]
    col_w     = 18  # 컬럼 너비

    print(f'\n{"="*60}')
    print('전체 요약 (mAP@.5 / mAP@.5:.95)')
    print(f'{"="*60}')

    # 헤더
    header = f'{"Weight":<25}' + ''.join(f'{v:>{col_w}}' for v in val_names)
    print(header)
    print('-' * (25 + col_w * len(val_names)))

    # 각 weight별 행
    for w_name, val_results in results_table.items():
        row = f'{w_name:<25}'
        for val_name in val_names:
            if val_name in val_results:
                mp50, map_ = val_results[val_name][2], val_results[val_name][3]
                cell = f'{mp50:.3f}/{map_:.3f}'
            else:
                cell = '-'
            row += f'{cell:>{col_w}}'
        print(row)
