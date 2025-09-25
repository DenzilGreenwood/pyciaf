#!/usr/bin/env python3
"""
Simple test runner to debug subprocess issues
"""

import os
import sys
import subprocess

def test_single_example():
    """Test running a single example to debug issues."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    example_path = os.path.join(current_dir, 'LLM_example', 'llm_example.py')
    example_dir = os.path.dirname(example_path)
    example_filename = os.path.basename(example_path)
    
    print(f"Current dir: {current_dir}")
    print(f"Example path: {example_path}")
    print(f"Example dir: {example_dir}")
    print(f"Example filename: {example_filename}")
    print(f"Example exists: {os.path.exists(example_path)}")
    
    if os.path.exists(example_path):
        print("\nRunning example...")
        result = subprocess.run(
            [sys.executable, example_filename], 
            cwd=example_dir,
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT length: {len(result.stdout)}")
        print(f"STDERR length: {len(result.stderr)}")
        
        if result.stdout:
            print(f"STDOUT first 300 chars:\n{result.stdout[:300]}")
        if result.stderr:
            print(f"STDERR first 300 chars:\n{result.stderr[:300]}")
            
        # Check for success indicators
        if result.returncode == 0:
            success_indicators = result.stdout.count("✅")
            completion_indicators = "Implementation Complete!" in result.stdout
            print(f"Success indicators: {success_indicators}")
            print(f"Completion found: {completion_indicators}")
    else:
        print("Example file not found!")

if __name__ == "__main__":
    test_single_example()