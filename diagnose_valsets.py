#!/usr/bin/env python3
"""
WiSE-FT 검증 세트 진단 도구
"""

import sys
from pathlib import Path
import yaml


def diagnose_validation_sets(data_yaml_path):
    """검증 세트 진단"""
    print("="*80)
    print("🔍 VALIDATION SET DIAGNOSTIC TOOL")
    print("="*80)

    # Load data.yaml
    print(f"\n📄 Loading: {data_yaml_path}")
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n📋 Config keys: {list(config.keys())}")

    # Check val field
    val_field = config.get('val', None)
    print(f"\n🔍 val field type: {type(val_field)}")
    print(f"   val field value: {val_field}")

    if isinstance(val_field, list):
        print(f"   ℹ️  val is a list with {len(val_field)} entries")
        for i, v in enumerate(val_field):
            print(f"     [{i}]: {v}")
            check_file(v)
    else:
        print(f"   ℹ️  val is a string")
        check_file(val_field)

    # Check expected validation sets
    print("\n" + "="*80)
    print("📊 CHECKING VALIDATION SET FILES")
    print("="*80)

    base_dir = Path(data_yaml_path).parent
    print(f"\nBase directory: {base_dir}")

    for val_name in ['valid1', 'valid2', 'test1', 'test2']:
        val_file = base_dir / f'{val_name}.txt'
        check_file(str(val_file))


def check_file(filepath):
    """파일 상세 정보 확인"""
    path = Path(filepath)

    print(f"\n  📁 {path.name}")
    print(f"     Path: {path}")

    if not path.exists():
        print(f"     ❌ File does not exist!")
        return

    # File size
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"     ✓ Exists")
    print(f"     Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    # Line count
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            num_lines = len(lines)
            print(f"     Images: {num_lines:,}")

            # Check first few images
            print(f"     First 3 entries:")
            for i, line in enumerate(lines[:3]):
                img_path = Path(line.strip())
                if img_path.exists():
                    img_size = img_path.stat().st_size / (1024 * 1024)
                    print(f"       [{i+1}] {img_path.name} ({img_size:.2f} MB)")
                else:
                    print(f"       [{i+1}] {img_path} ❌ NOT FOUND")

            # Estimate total size
            if num_lines > 0 and lines[0].strip():
                first_img = Path(lines[0].strip())
                if first_img.exists():
                    avg_img_size = first_img.stat().st_size / (1024 * 1024)
                    total_size_gb = (avg_img_size * num_lines) / 1024
                    print(f"     Estimated total data: {total_size_gb:.2f} GB")

                    # Estimate evaluation time
                    # Assume: 640x640 image, batch_size=64, ~200 images/sec
                    est_time_sec = num_lines / 200
                    est_time_min = est_time_sec / 60
                    print(f"     Estimated eval time: {est_time_min:.1f} minutes")

                    if est_time_min > 10:
                        print(f"     ⚠️  WARNING: This will likely timeout (> 10 min)!")

    except Exception as e:
        print(f"     ❌ Error reading file: {e}")


def recommend_solution(data_yaml_path):
    """해결책 추천"""
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    base_dir = Path(data_yaml_path).parent

    for val_name in ['valid1', 'valid2']:
        val_file = base_dir / f'{val_name}.txt'

        if not val_file.exists():
            continue

        try:
            with open(val_file, 'r') as f:
                num_lines = len(f.readlines())

            if num_lines > 5000:
                print(f"\n⚠️  {val_name}.txt has {num_lines:,} images (LARGE!)")
                print(f"   Recommendations:")
                print(f"   1. Increase timeout:")
                print(f"      Edit wiseft_sweep_parallel.py line 48:")
                print(f"      timeout=1800  # 30 minutes instead of 10")
                print(f"")
                print(f"   2. Or create a sample:")
                print(f"      head -1000 {val_file} > {base_dir}/{val_name}_sample.txt")
                print(f"      # Then use --val-sets {val_name}_sample ...")
                print(f"")
                print(f"   3. Or increase batch size:")
                print(f"      --batch-size 128  # Faster evaluation")

        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_valsets.py <data.yaml>")
        sys.exit(1)

    data_yaml = sys.argv[1]
    diagnose_validation_sets(data_yaml)
    recommend_solution(data_yaml)

    print("\n" + "="*80)
    print("✅ Diagnostic complete!")
    print("="*80)
