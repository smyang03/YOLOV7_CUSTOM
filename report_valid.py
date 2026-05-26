"""
report_valid.py - test_valid.py 결과 리포트 생성기

test_valid.py 실행 후 저장된 valid_results_*.json 을 읽어
모델별 × DB별 × 클래스별 전체 리포트를 출력하고 txt 파일로 저장.

Usage:
    # 가장 최근 결과 리포트
    python report_valid.py --dir runs/test

    # 특정 json 파일 지정
    python report_valid.py --dir runs/test --file valid_results_20250526_103000.json

    # txt 파일로만 저장 (터미널 출력 없음)
    python report_valid.py --dir runs/test --quiet
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


# ── JSON 파일 탐색 ────────────────────────────────────────────────────────────
def find_result_files(dir_path: Path):
    files = sorted(dir_path.glob('valid_results_*.json'))
    return files


def load_result(json_path: Path) -> dict:
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)


# ── 리포트 생성 ───────────────────────────────────────────────────────────────
def build_report(data: dict) -> str:
    lines = []

    def w(text=''):
        lines.append(text)

    sep1 = '=' * 80
    sep2 = '-' * 80
    sep3 = '─' * 80

    # ── 헤더 ──
    w(sep1)
    w('  YOLOv7 Validation Report')
    w(f'  생성 시각 : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    w(f'  원본 실행 : {data.get("timestamp", "-")}')
    w(f'  Data YAML : {data.get("data_yaml", "-")}')
    w(f'  Img Size  : {data.get("img_size", "-")}   '
      f'conf={data.get("conf_thres", "-")}   iou={data.get("iou_thres", "-")}')
    w(sep1)

    results  = data.get('results', {})
    val_sets = data.get('val_sets', [])
    weights  = data.get('weights', list(results.keys()))

    # ── 모델별 섹션 ──────────────────────────────────────────────────────────
    for wi, weight_path in enumerate(weights):
        w_name     = Path(weight_path).stem
        w_results  = results.get(w_name, {})

        w()
        w(sep1)
        w(f'  [{wi+1}/{len(weights)}] 모델 : {w_name}')
        w(f'            경로 : {weight_path}')
        w(sep1)

        for val_name in val_sets:
            entry = w_results.get(val_name)

            w()
            w(f'  ▶ DB : {val_name}')
            w(sep2)

            if entry is None:
                w('  [ERROR] 결과 없음 (실행 중 오류 발생)')
                w(sep2)
                continue

            # 헤더
            hdr = (f'  {"Class":<22} {"Labels":>8} {"P":>8} {"R":>8} '
                   f'{"mAP@.5":>10} {"mAP@.5:.95":>12}')
            w(hdr)
            w(f'  {sep3}')

            # overall
            w(f'  {"all":<22} {"":>8} '
              f'{entry["P"]:>8.3f} {entry["R"]:>8.3f} '
              f'{entry["mAP50"]:>10.3f} {entry["mAP"]:>12.3f}')

            # per class
            per_class = entry.get('per_class')
            if per_class:
                w(f'  {sep3}')
                for row in per_class:
                    w(f'  {row["class"]:<22} {row["labels"]:>8} '
                      f'{row["P"]:>8.3f} {row["R"]:>8.3f} '
                      f'{row["mAP50"]:>10.3f} {row["mAP"]:>12.3f}')

            w(sep2)

    # ── 전체 요약 매트릭스 ────────────────────────────────────────────────────
    w()
    w(sep1)
    w('  전체 요약 매트릭스  (mAP@.5 / mAP@.5:.95)')
    w(sep1)

    col_w = 22
    header = f'  {"Weight":<28}' + ''.join(f'{v:>{col_w}}' for v in val_sets)
    w(header)
    w(f'  {"-" * (28 + col_w * len(val_sets))}')

    for weight_path in weights:
        w_name    = Path(weight_path).stem
        w_results = results.get(w_name, {})
        row       = f'  {w_name:<28}'
        for val_name in val_sets:
            entry = w_results.get(val_name)
            if entry and entry.get('mAP50') is not None:
                cell = f'{entry["mAP50"]:.3f}/{entry["mAP"]:.3f}'
            else:
                cell = 'ERROR'
            row += f'{cell:>{col_w}}'
        w(row)

    # ── DB별 클래스 상세 비교 ─────────────────────────────────────────────────
    for val_name in val_sets:
        w()
        w(sep1)
        w(f'  DB별 클래스 상세 비교 : {val_name}  (mAP@.5)')
        w(sep1)

        # 모든 클래스 이름 수집
        class_names = []
        for weight_path in weights:
            w_name  = Path(weight_path).stem
            entry   = results.get(w_name, {}).get(val_name)
            if entry and entry.get('per_class'):
                for row in entry['per_class']:
                    if row['class'] not in class_names:
                        class_names.append(row['class'])

        if not class_names:
            w('  (per-class 데이터 없음 - --verbose 옵션으로 재실행 필요)')
            continue

        # 헤더
        col = 14
        w_col = 28
        w(f'  {"Class":<22}' + ''.join(f'{Path(wp).stem:>{w_col}}' for wp in weights))
        w(f'  {"-" * (22 + w_col * len(weights))}')

        for cls in class_names:
            row = f'  {cls:<22}'
            for weight_path in weights:
                w_name = Path(weight_path).stem
                entry  = results.get(w_name, {}).get(val_name)
                cell   = '-'
                if entry and entry.get('per_class'):
                    match = next((r for r in entry['per_class'] if r['class'] == cls), None)
                    if match:
                        cell = f'{match["mAP50"]:.3f}'
                row += f'{cell:>{w_col}}'
            w(row)

    w()
    w(sep1)
    w('  End of Report')
    w(sep1)

    return '\n'.join(lines)


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='report_valid.py')
    parser.add_argument('--dir',   type=str, required=True, help='runs/test 같은 결과 폴더')
    parser.add_argument('--file',  type=str, default=None,  help='특정 json 파일명 (생략 시 최신)')
    parser.add_argument('--quiet', action='store_true',     help='터미널 출력 생략')
    opt = parser.parse_args()

    dir_path = Path(opt.dir)
    if not dir_path.exists():
        print(f'[ERROR] 폴더 없음: {dir_path}')
        sys.exit(1)

    # JSON 파일 선택
    if opt.file:
        json_path = dir_path / opt.file
        if not json_path.exists():
            print(f'[ERROR] 파일 없음: {json_path}')
            sys.exit(1)
    else:
        files = find_result_files(dir_path)
        if not files:
            print(f'[ERROR] {dir_path} 에 valid_results_*.json 파일이 없습니다.')
            print('        test_valid.py 실행 후 다시 시도하세요.')
            sys.exit(1)

        if len(files) > 1:
            print(f'JSON 파일 {len(files)}개 발견:')
            for i, f in enumerate(files):
                print(f'  [{i}] {f.name}')
            print(f'→ 가장 최신 파일 사용: {files[-1].name}')
            print('  (특정 파일 지정: --file <파일명>)')
        json_path = files[-1]

    print(f'\n로드: {json_path}')
    data   = load_result(json_path)
    report = build_report(data)

    # 터미널 출력
    if not opt.quiet:
        print('\n' + report)

    # txt 저장
    save_path = json_path.with_suffix('.txt')
    save_path.write_text(report, encoding='utf-8')
    print(f'\n리포트 저장: {save_path}')
