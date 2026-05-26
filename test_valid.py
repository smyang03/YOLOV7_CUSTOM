"""
test_valid.py - 다중 weight × 다중 val 세트 테스트 (GPU 병렬 실행 지원)

GPU가 여러 개면 weight별로 GPU를 나눠 병렬 실행,
GPU가 1개거나 CPU면 순차 실행으로 자동 전환.

Usage:
    # weight 1개
    python test_valid.py --data data/custom.yaml --weights best.pt

    # weight 여러 개 → GPU별 병렬 실행
    python test_valid.py --data data/custom.yaml \\
        --weights model_A.pt model_B.pt model_C.pt

data yaml 예시:
    val:
      - ./data/images/val_indoor/
      - ./data/images/val_outdoor/
"""

import argparse
import yaml
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue
from argparse import Namespace
from pathlib import Path

from utils.general import check_file


# ── 워커 함수 (자식 프로세스에서 실행) ─────────────────────────────────────
def _worker(weight, val_list, data_dict, task_key, base_args, device, result_queue):
    """단일 weight를 지정 GPU에서 모든 val 세트 순서대로 테스트"""
    import test as test_module
    from test import test as test_fn

    # 자식 프로세스 전용 opt 구성
    opt = Namespace(**base_args)
    opt.device  = str(device)
    opt.weights = [weight]
    test_module.opt = opt  # test() 내부에서 전역 opt를 읽으므로 주입

    w_name         = Path(weight).stem
    weight_results = {}

    for val_path in val_list:
        val_name = Path(val_path).stem
        print(f'\n  [GPU {device}] [{w_name}] → {val_name}', flush=True)

        data_single = data_dict.copy()
        data_single[task_key] = val_path

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
            print(f'  [ERROR] {w_name}/{val_name}: {e}', flush=True)
            weight_results[val_name] = None

    result_queue.put((w_name, weight_results))


# ── 병렬 실행 헬퍼 ──────────────────────────────────────────────────────────
def _run_parallel(wt_list, val_list, data_dict, task_key, opt, num_gpus):
    """weight를 num_gpus 단위 배치로 묶어 병렬 실행. 결과 dict 반환."""

    # GPU 수보다 weight가 많으면 배치로 나눔 (같은 GPU에 두 모델 올리지 않기 위해)
    batches = [wt_list[i:i + num_gpus] for i in range(0, len(wt_list), num_gpus)]
    results_table = {}
    base_args     = vars(opt)

    for b_idx, batch in enumerate(batches):
        print(f'\n{"#"*60}')
        print(f'배치 {b_idx+1}/{len(batches)}: {[Path(w).stem for w in batch]}')
        print(f'{"#"*60}')

        q          = Queue()
        processes  = []

        for wi, weight in enumerate(batch):
            gpu_id = wi % num_gpus
            print(f'  {Path(weight).stem} → GPU {gpu_id}')
            p = Process(
                target=_worker,
                args=(weight, val_list, data_dict, task_key, base_args, gpu_id, q),
                daemon=True,
            )
            processes.append(p)
            p.start()

        # 모든 자식 프로세스 완료 대기 + 결과 수집
        for p in processes:
            p.join()

        while not q.empty():
            w_name, weight_results = q.get()
            results_table[w_name] = weight_results

    return results_table


# ── 순차 실행 헬퍼 ──────────────────────────────────────────────────────────
def _run_sequential(wt_list, val_list, data_dict, task_key, opt):
    """GPU 1개 또는 CPU 환경에서 순차 실행. 결과 dict 반환."""
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

            data_single = data_dict.copy()
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


# ── 요약 표 출력 ────────────────────────────────────────────────────────────
def _print_summary(results_table, wt_list, val_list):
    val_names = [Path(v).stem for v in val_list]
    col_w     = 20

    print(f'\n{"="*60}')
    print('전체 요약  (mAP@.5 / mAP@.5:.95)')
    print(f'{"="*60}')
    print(f'{"Weight":<25}' + ''.join(f'{v:>{col_w}}' for v in val_names))
    print('-' * (25 + col_w * len(val_names)))

    for weight in wt_list:
        w_name       = Path(weight).stem
        val_results  = results_table.get(w_name, {})
        row          = f'{w_name:<25}'
        for val_name in val_names:
            entry = val_results.get(val_name)
            if entry is not None:
                cell = f'{entry[2]:.3f}/{entry[3]:.3f}'
            else:
                cell = 'ERROR'
            row += f'{cell:>{col_w}}'
        print(row)


# ── 메인 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)   # CUDA + multiprocessing 안정성

    parser = argparse.ArgumentParser(prog='test_valid.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
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

    # GPU 감지
    num_gpus = torch.cuda.device_count()
    print(f'\n감지된 GPU: {num_gpus}개 / Weight: {len(wt_list)}개 / Val 세트: {len(val_list)}개')

    if num_gpus >= 2 and len(wt_list) > 1:
        print('→ 병렬 모드 (weight별 GPU 분산)')
        results_table = _run_parallel(wt_list, val_list, data_dict, task_key, opt, num_gpus)
    else:
        if num_gpus == 1:
            print('→ 순차 모드 (GPU 1개)')
        else:
            print('→ 순차 모드 (CPU)')
        results_table = _run_sequential(wt_list, val_list, data_dict, task_key, opt)

    _print_summary(results_table, wt_list, val_list)
