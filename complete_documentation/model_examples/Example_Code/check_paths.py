#!/usr/bin/env python3
"""
Directory path checker for the runner script
"""

import os


def check_paths():
    """Check what paths the runner script is looking for."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")

    examples = [
        {
            "name": "LLM",
            "path": os.path.join(current_dir, "LLM_example", "llm_example.py"),
        },
        {
            "name": "Classifier",
            "path": os.path.join(
                current_dir, "Classifier_example", "classifier_example.py"
            ),
        },
        {
            "name": "CNN",
            "path": os.path.join(current_dir, "CNN_example", "cnn_example.py"),
        },
        {
            "name": "Diffusion",
            "path": os.path.join(
                current_dir, "Diffusion_example", "diffusion_example.py"
            ),
        },
        {
            "name": "Agentic System",
            "path": os.path.join(
                current_dir, "Agentic_System_example", "agentic_system_example.py"
            ),
        },
    ]

    print("\nChecking example paths:")
    for example in examples:
        exists = os.path.exists(example["path"])
        print(f"  {example['name']}: {exists}")
        print(f"    Path: {example['path']}")
        if exists:
            # Check if directory structure is correct
            example_dir = os.path.dirname(example["path"])
            example_filename = os.path.basename(example["path"])
            print(f"    Directory: {example_dir}")
            print(f"    Filename: {example_filename}")
            print(f"    Directory exists: {os.path.exists(example_dir)}")
        print()


if __name__ == "__main__":
    check_paths()
