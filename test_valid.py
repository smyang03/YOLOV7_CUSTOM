"""
test_valid.py - 다중 weight × 다중 val 세트 테스트 (GPU 큐 병렬 실행)

GPU가 여러 개면 각 GPU가 워커로 동작 → 끝나는 즉시 다음 weight 처리.
GPU가 1개거나 CPU면 순차 실행으로 자동 전환.
완료 후 {project}/valid_results_{timestamp}.json 자동 저장.

Usage:
    python test_valid.py --data data/custom.yaml \\
        --weights w0.pt w1.pt w2.pt --verbose --img-size 1280
"""

import argparse
import json
import queue
import yaml
import torch
import multiprocessing as mp
from datetime import datetime
from multiprocessing import Process, Queue
from argparse import Namespace
from pathlib import Path

from utils.general import check_file


# ── 결과 직렬화 (numpy → python native) ────────────────────────────────────
def _serialize(per_class):
    """per_class_results(numpy 포함)를 JSON 저장 가능한 dict로 변환"""
    if per_class is None:
        return None
    names = per_class['names']
    rows  = []
    for i, c in enumerate(per_class['ap_class']):
        rows.append({
            'class':  names[int(c)],
            'labels': int(per_class['nt'][int(c)]),
            'P':      round(float(per_class['p'][i]),  4),
            'R':      round(float(per_class['r'][i]),  4),
            'mAP50':  round(float(per_class['ap50'][i]), 4),
            'mAP':    round(float(per_class['ap'][i]),  4),
        })
    return rows


# ── GPU 워커 (자식 프로세스) ─────────────────────────────────────────────────
def _gpu_worker(gpu_id, work_queue, result_queue, val_list, data_dict, task_key, base_args):
    import test as test_module
    from test import test as test_fn

    opt = Namespace(**base_args)
    opt.device = str(gpu_id)

    while True:
        try:
            wi, weight = work_queue.get_nowait()
        except queue.Empty:
            break

        w_name         = Path(weight).stem
        weight_results = {}

        print(f'\n  [GPU {gpu_id}] 시작 → {w_name}', flush=True)

        for val_path in val_list:
            val_name = Path(val_path).stem
            print(f'  [GPU {gpu_id}] [{w_name}] → {val_name}', flush=True)

            data_single     = data_dict.copy()
            data_single[task_key] = val_path
            opt.weights     = [weight]
            opt.name        = f'{base_args["name"]}_{w_name}_{val_name}'
            test_module.opt = opt

            try:
                res, _, _, per_class = test_fn(
                    data_single, [weight],
                    opt.batch_size, opt.img_size,
                    opt.conf_thres, opt.iou_thres,
                    opt.save_json, opt.single_cls,
                    opt.augment, opt.verbose,
                    save_txt=opt.save_txt | opt.save_hybrid,
                    save_hybrid=opt.save_hybrid,
                    save_conf=opt.save_conf,
                    trace=not opt.no_trace,
                    v5_metric=opt.v5_metric,
                )
                weight_results[val_name] = {
                    'P':        round(float(res[0]), 4),
                    'R':        round(float(res[1]), 4),
                    'mAP50':    round(float(res[2]), 4),
                    'mAP':      round(float(res[3]), 4),
                    'per_class': _serialize(per_class),
                }
            except Exception as e:
                print(f'  [ERROR] GPU {gpu_id} / {w_name} / {val_name}: {e}', flush=True)
                weight_results[val_name] = None

        print(f'  [GPU {gpu_id}] 완료 ← {w_name}', flush=True)
        result_queue.put((w_name, weight_results))


# ── 병렬 실행 (큐 방식) ─────────────────────────────────────────────────────
def _run_parallel(wt_list, val_list, data_dict, task_key, opt, num_gpus):
    base_args    = vars(opt)
    work_queue   = mp.Queue()
    result_queue = mp.Queue()

    for wi, weight in enumerate(wt_list):
        work_queue.put((wi, weight))

    actual_workers = min(num_gpus, len(wt_list))
    print(f'\n병렬 모드: GPU 워커 {actual_workers}개 / Weight {len(wt_list)}개')
    print(f'  → GPU가 끝나는 즉시 다음 weight 자동 할당\n')

    processes = []
    for gpu_id in range(actual_workers):
        p = Process(
            target=_gpu_worker,
            args=(gpu_id, work_queue, result_queue, val_list, data_dict, task_key, base_args),
            daemon=False,
        )
        processes.append(p)
        p.start()

    results_table = {}
    for _ in wt_list:
        w_name, weight_results = result_queue.get()
        results_table[w_name]  = weight_results

    for p in processes:
        p.join()

    return results_table


# ── 순차 실행 ────────────────────────────────────────────────────────────────
def _run_sequential(wt_list, val_list, data_dict, task_key, opt):
    import test as test_module
    from test import test as test_fn

    results_table = {}
    original_name = opt.name

    for wi, weight in enumerate(wt_list):
        w_name = Path(weight).stem
        results_table[w_name] = {}

        print(f'\n{"#"*60}')
        print(f'[Weight {wi+1}/{len(wt_list)}] {weight}')
        print(f'{"#"*60}')

        opt.weights     = [weight]
        test_module.opt = opt

        for vi, val_path in enumerate(val_list):
            val_name = Path(val_path).stem

            print(f'\n{"="*60}')
            print(f'  [Val {vi+1}/{len(val_list)}] {val_name}')
            print(f'  경로: {val_path}')
            print(f'{"="*60}')

            data_single     = data_dict.copy()
            data_single[task_key] = val_path
            opt.name        = f'{original_name}_{w_name}_{val_name}'
            test_module.opt = opt

            try:
                res, _, _, per_class = test_fn(
                    data_single, [weight],
                    opt.batch_size, opt.img_size,
                    opt.conf_thres, opt.iou_thres,
                    opt.save_json, opt.single_cls,
                    opt.augment, opt.verbose,
                    save_txt=opt.save_txt | opt.save_hybrid,
                    save_hybrid=opt.save_hybrid,
                    save_conf=opt.save_conf,
                    trace=not opt.no_trace,
                    v5_metric=opt.v5_metric,
                )
                results_table[w_name][val_name] = {
                    'P':        round(float(res[0]), 4),
                    'R':        round(float(res[1]), 4),
                    'mAP50':    round(float(res[2]), 4),
                    'mAP':      round(float(res[3]), 4),
                    'per_class': _serialize(per_class),
                }
            except Exception as e:
                print(f'  [ERROR] {w_name}/{val_name}: {e}')
                results_table[w_name][val_name] = None

    opt.name    = original_name
    opt.weights = wt_list
    return results_table


# ── JSON 저장 ────────────────────────────────────────────────────────────────
def _save_json(results_table, wt_list, val_list, opt, data_yaml_path):
    project_dir = Path(opt.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = project_dir / f'valid_results_{ts}.json'

    payload = {
        'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_yaml':   str(data_yaml_path),
        'img_size':    opt.img_size,
        'conf_thres':  opt.conf_thres,
        'iou_thres':   opt.iou_thres,
        'weights':     wt_list,
        'val_sets':    [Path(v).stem for v in val_list],
        'results':     results_table,
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'\n결과 저장 완료: {save_path}')
    return save_path


# ── 터미널 요약 표 ────────────────────────────────────────────────────────────
def _print_summary(results_table, wt_list, val_list):
    val_names = [Path(v).stem for v in val_list]
    col_w     = 20

    print(f'\n{"="*60}')
    print('전체 요약  (mAP@.5 / mAP@.5:.95)')
    print(f'{"="*60}')
    print(f'{"Weight":<25}' + ''.join(f'{v:>{col_w}}' for v in val_names))
    print('-' * (25 + col_w * len(val_names)))

    for weight in wt_list:
        w_name      = Path(weight).stem
        val_results = results_table.get(w_name, {})
        row         = f'{w_name:<25}'
        for val_name in val_names:
            entry = val_results.get(val_name)
            if entry and entry.get('mAP50') is not None:
                cell = f'{entry["mAP50"]:.3f}/{entry["mAP"]:.3f}'
            else:
                cell = 'ERROR'
            row += f'{cell:>{col_w}}'
        print(row)


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(prog='test_valid.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt')
    parser.add_argument('--data',        type=str,   default='data/coco.yaml')
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--img-size',    type=int,   default=640)
    parser.add_argument('--conf-thres',  type=float, default=0.001)
    parser.add_argument('--iou-thres',   type=float, default=0.65)
    parser.add_argument('--task',        type=str,   default='val')
    parser.add_argument('--device',      type=str,   default='')
    parser.add_argument('--single-cls',  action='store_true')
    parser.add_argument('--augment',     action='store_true')
    parser.add_argument('--verbose',     action='store_true')
    parser.add_argument('--save-txt',    action='store_true')
    parser.add_argument('--save-hybrid', action='store_true')
    parser.add_argument('--save-conf',   action='store_true')
    parser.add_argument('--save-json',   action='store_true')
    parser.add_argument('--project',     type=str,   default='runs/test')
    parser.add_argument('--name',        type=str,   default='exp')
    parser.add_argument('--exist-ok',    action='store_true')
    parser.add_argument('--no-trace',    action='store_true')
    parser.add_argument('--v5-metric',   action='store_true')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)
    print(opt)

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    task_key = opt.task if opt.task in ('train', 'val', 'test') else 'val'
    val_raw  = data_dict.get(task_key, data_dict.get('val'))
    val_list = val_raw if isinstance(val_raw, list) else [val_raw]
    wt_list  = opt.weights if isinstance(opt.weights, list) else [opt.weights]

    num_gpus = torch.cuda.device_count()
    print(f'\n감지된 GPU: {num_gpus}개 / Weight: {len(wt_list)}개 / Val 세트: {len(val_list)}개')

    if num_gpus >= 2 and len(wt_list) > 1:
        results_table = _run_parallel(wt_list, val_list, data_dict, task_key, opt, num_gpus)
    else:
        mode = 'GPU 1개' if num_gpus == 1 else 'CPU'
        print(f'→ 순차 모드 ({mode})')
        results_table = _run_sequential(wt_list, val_list, data_dict, task_key, opt)

    _print_summary(results_table, wt_list, val_list)
    _save_json(results_table, wt_list, val_list, opt, opt.data)
