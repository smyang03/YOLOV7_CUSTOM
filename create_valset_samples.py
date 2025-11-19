#!/usr/bin/env python3
"""
검증 세트 샘플링 도구 - 빠른 테스트를 위한 샘플 생성
"""

import sys
from pathlib import Path
import random


def create_sample(input_file, output_file, num_samples=1000, random_sample=True):
    """
    검증 세트 샘플 생성

    Args:
        input_file: 원본 검증 세트 파일
        output_file: 샘플 파일 경로
        num_samples: 샘플 개수
        random_sample: 랜덤 샘플링 여부 (False면 처음부터)
    """
    print(f"📄 Reading: {input_file}")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"   Total images: {total_lines:,}")

    if total_lines <= num_samples:
        print(f"   ℹ️  Already small enough ({total_lines} <= {num_samples})")
        print(f"   Copying as-is...")
        sampled_lines = lines
    else:
        if random_sample:
            print(f"   🎲 Random sampling {num_samples} images...")
            sampled_lines = random.sample(lines, num_samples)
        else:
            print(f"   ✂️  Taking first {num_samples} images...")
            sampled_lines = lines[:num_samples]

    # Write sample
    print(f"💾 Writing: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.writelines(sampled_lines)

    print(f"   ✓ Created with {len(sampled_lines):,} images")

    # Calculate size reduction
    if total_lines > num_samples:
        reduction = (1 - len(sampled_lines) / total_lines) * 100
        speedup = total_lines / len(sampled_lines)
        print(f"   📊 Size reduction: {reduction:.1f}%")
        print(f"   ⚡ Expected speedup: {speedup:.1f}x")

    return output_path


def main():
    """메인 함수"""
    print("="*80)
    print("✂️  VALIDATION SET SAMPLING TOOL")
    print("="*80)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python create_valset_samples.py <base_dir> [num_samples]")
        print("\nExample:")
        print("  python create_valset_samples.py new_list 1000")
        print("\nThis will create:")
        print("  new_list/valid1.txt → new_list/valid1_sample.txt (1000 images)")
        print("  new_list/valid2.txt → new_list/valid2_sample.txt (1000 images)")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print(f"\n📁 Base directory: {base_dir}")
    print(f"🎯 Sample size: {num_samples}")
    print("")

    created_files = []

    for val_name in ['valid1', 'valid2', 'test1', 'test2']:
        input_file = base_dir / f'{val_name}.txt'

        if not input_file.exists():
            print(f"⏭️  Skipping {val_name}.txt (not found)")
            continue

        output_file = base_dir / f'{val_name}_sample.txt'

        try:
            created = create_sample(input_file, output_file, num_samples, random_sample=False)
            created_files.append(created)
            print("")
        except Exception as e:
            print(f"❌ Error processing {val_name}: {e}")
            print("")

    # Summary
    print("="*80)
    print("📋 SUMMARY")
    print("="*80)

    if created_files:
        print(f"\n✅ Created {len(created_files)} sample files:")
        for f in created_files:
            print(f"   {f}")

        print(f"\n💡 Now run WiSE-FT with samples:")
        print(f"")
        print(f"python wiseft_sweep_parallel.py \\")
        print(f"    --scratch models/600.pt \\")
        print(f"    --finetuned models/620.pt \\")
        print(f"    --data {base_dir}/data.yaml \\")
        print(f"    --val-sets valid1_sample valid2_sample \\")
        print(f"    --alpha-min 0.0 --alpha-max 0.3 \\")
        print(f"    --num-gpus 8")
        print(f"")
        print(f"Expected time: ~2 minutes (instead of 10+ minutes)")
    else:
        print("\n⚠️  No sample files created")

    print("="*80)


if __name__ == '__main__':
    main()
