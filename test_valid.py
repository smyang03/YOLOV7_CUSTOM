"""
test_valid.py - 다중 weight × 다중 val 세트 테스트 (GPU 큐 병렬 실행)

GPU가 여러 개면 각 GPU가 워커로 동작 → 끝나는 즉시 다음 weight 처리.
GPU가 1개거나 CPU면 순차 실행으로 자동 전환.

Usage:
    # weight 1개
    python test_valid.py --data data/custom.yaml --weights best.pt

    # weight 여러 개 → GPU 큐 방식 병렬 실행
    python test_valid.py --data data/custom.yaml \\
        --weights w0.pt w1.pt w2.pt w3.pt w4.pt w5.pt w6.pt w7.pt w8.pt w9.pt

data yaml 예시:
    val:
      - ./data/images/val_A/
      - ./data/images/val_B/
"""

import argparse
import queue
import yaml
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue
from argparse import Namespace
from pathlib import Path

from utils.general import check_file


# ── GPU 워커 (자식 프로세스) ─────────────────────────────────────────────────
def _gpu_worker(gpu_id, work_queue, result_queue, val_list, data_dict, task_key, base_args):
    """
    특정 GPU를 전담하는 워커.
    work_queue 가 빌 때까지 weight를 하나씩 꺼내 모든 val 세트를 처리하고
    결과를 result_queue 에 넣은 뒤 다음 weight를 가져온다.
    """
    import test as test_module
    from test import test as test_fn

    opt = Namespace(**base_args)
    opt.device = str(gpu_id)

    while True:
        # 큐에서 다음 작업 가져오기 (비었으면 종료)
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

            data_single = data_dict.copy()
            data_single[task_key] = val_path

            opt.weights     = [weight]
            opt.name        = f'{base_args["name"]}_{w_name}_{val_name}'
            test_module.opt = opt

            try:
                res, _, _, _ = test_fn(
                    data_single,
                    [weight],
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
                    v5_metric=opt.v5_metric,
                )
                weight_results[val_name] = res[:4]   # (mp, mr, map50, map)
            except Exception as e:
                print(f'  [ERROR] GPU {gpu_id} / {w_name} / {val_name}: {e}', flush=True)
                weight_results[val_name] = None

        print(f'  [GPU {gpu_id}] 완료 ← {w_name}', flush=True)
        result_queue.put((w_name, weight_results))


# ── 병렬 실행 (큐 방식) ─────────────────────────────────────────────────────
def _run_parallel(wt_list, val_list, data_dict, task_key, opt, num_gpus):
    """
    work_queue 에 모든 weight를 넣고,
    num_gpus 개의 GPU 워커가 큐에서 꺼내가며 처리.
    weight > GPU 개수여도 끝나는 GPU가 바로 다음 weight를 가져가므로
    GPU가 놀지 않는다.
    """
    base_args  = vars(opt)

    # 작업 큐: (인덱스, weight경로) 전부 넣기
    work_queue   = mp.Queue()
    result_queue = mp.Queue()
    for wi, weight in enumerate(wt_list):
        work_queue.put((wi, weight))

    # GPU당 워커 1개 생성
    actual_workers = min(num_gpus, len(wt_list))   # weight < GPU 개수면 낭비 방지
    print(f'\n병렬 모드: GPU 워커 {actual_workers}개 / Weight {len(wt_list)}개')
    print(f'  → GPU가 끝나는 즉시 다음 weight 자동 할당\n')

    processes = []
    for gpu_id in range(actual_workers):
        p = Process(
            target=_gpu_worker,
            args=(gpu_id, work_queue, result_queue, val_list, data_dict, task_key, base_args),
            daemon=False,  # DataLoader가 내부적으로 자식 프로세스를 생성하므로 daemon=False 필수
        )
        processes.append(p)
        p.start()

    # 모든 결과 수집 (weight 개수만큼)
    results_table = {}
    for _ in wt_list:
        w_name, weight_results = result_queue.get()
        results_table[w_name]  = weight_results

    # 모든 워커 정상 종료 대기
    for p in processes:
        p.join()

    return results_table


# ── 순차 실행 ────────────────────────────────────────────────────────────────
def _run_sequential(wt_list, val_list, data_dict, task_key, opt):
    """GPU 1개 또는 CPU 환경 → 순차 실행."""
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
                res, _, _, _ = test_fn(
                    data_single,
                    [weight],
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
                    v5_metric=opt.v5_metric,
                )
                results_table[w_name][val_name] = res[:4]
            except Exception as e:
                print(f'  [ERROR] {w_name}/{val_name}: {e}')
                results_table[w_name][val_name] = None

    opt.name    = original_name
    opt.weights = wt_list
    return results_table


# ── 요약 표 출력 ─────────────────────────────────────────────────────────────
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
            if entry is not None:
                cell = f'{entry[2]:.3f}/{entry[3]:.3f}'
            else:
                cell = 'ERROR'
            row += f'{cell:>{col_w}}'
        print(row)


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)   # CUDA + multiprocessing 안정성

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

    # yaml 로드
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    task_key = opt.task if opt.task in ('train', 'val', 'test') else 'val'
    val_raw  = data_dict.get(task_key, data_dict.get('val'))
    val_list = val_raw if isinstance(val_raw, list) else [val_raw]
    wt_list  = opt.weights if isinstance(opt.weights, list) else [opt.weights]

    # GPU 감지 및 모드 결정
    num_gpus = torch.cuda.device_count()
    print(f'\n감지된 GPU: {num_gpus}개 / Weight: {len(wt_list)}개 / Val 세트: {len(val_list)}개')

    if num_gpus >= 2 and len(wt_list) > 1:
        results_table = _run_parallel(wt_list, val_list, data_dict, task_key, opt, num_gpus)
    else:
        mode = 'GPU 1개' if num_gpus == 1 else 'CPU'
        print(f'→ 순차 모드 ({mode})')
        results_table = _run_sequential(wt_list, val_list, data_dict, task_key, opt)

    _print_summary(results_table, wt_list, val_list)
