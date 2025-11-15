#!/usr/bin/env python3
"""Test enhanced output format of analyze_results.py"""

import sys
sys.path.insert(0, '/home/user/YOLOV7_CUSTOM')

from analyze_results import parse_results_file, print_analysis
import argparse

# Create mock args
class MockArgs:
    def __init__(self):
        self.results = 'test_simulation/sample_results.txt'

# Parse the sample results
print("="*80)
print("Testing Enhanced analyze_results.py Output")
print("="*80)

args = MockArgs()
results = parse_results_file(args.results)

print(f"\n✅ Parsed {len(results)} epochs")

# Test the enhanced print_analysis function
print("\n" + "="*80)
print("Test Case: Combined validation set, person class, fitness metric")
print("="*80)

# Simulate the analysis
print_analysis(results, 'Combined', 'person', 'fitness')

print("\n" + "="*80)
print("Test Case: test1 validation set, all classes, map metric")
print("="*80)

print_analysis(results, 'test1', 'all', 'map')

print("\n" + "="*80)
print("✅ Enhanced Output Test Completed!")
print("="*80)
