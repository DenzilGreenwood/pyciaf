#!/usr/bin/env python3
"""
CIAF Test Runner

Convenient script to run all CIAF tests with various options.

Usage:
    python run_tests.py                 # Run all tests
    python run_tests.py --module core   # Run specific module tests
    python run_tests.py --coverage      # Run with coverage report
    python run_tests.py --fast          # Run fast tests only
    python run_tests.py --integration   # Run integration tests only

Created: 2026-03-31
Version: 1.0.0
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd):
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    # Run from the directory containing the script (pyciaf/)
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run CIAF tests")

    parser.add_argument(
        "--module",
        choices=[
            "core",
            "agents",
            "watermarks",
            "lcm",
            "compliance",
            "web",
            "vault",
            "integration",
            "all",
        ],
        default="all",
        help="Test module to run",
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast tests only (skip slow integration tests)",
    )

    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--failfast",
        "-x",
        action="store_true",
        help="Stop on first failure",
    )

    args = parser.parse_args()

    # Build pytest command
    cmd = ["pytest"]

    # Select test files based on module
    if args.integration:
        cmd.append("tests/test_integration_comprehensive.py")
    elif args.module == "all":
        # Exclude problematic hypothesis tests
        cmd.extend([
            "tests/",
            "--ignore=tests/test_properties.py",
            "--ignore=tests/test_properties_simple.py",
        ])
    elif args.module == "core":
        cmd.extend([
            "tests/test_core_comprehensive.py",
            "tests/test_merkle.py",
            "tests/test_anchors.py",
        ])
    elif args.module == "agents":
        cmd.append("tests/test_agents.py")
    elif args.module == "watermarks":
        cmd.extend([
            "tests/test_watermarks_comprehensive.py",
            "tests/test_watermarks.py",
            "tests/test_fragment_verification.py",
            "tests/test_pdf_visual_watermarking.py",
        ])
    elif args.module == "lcm":
        cmd.extend([
            "tests/test_lcm_comprehensive.py",
            "tests/test_capsule_bugfix.py",
        ])
    elif args.module == "compliance":
        cmd.extend([
            "tests/test_compliance_comprehensive.py",
            "tests/compliance/test_acceptance.py",
        ])
    elif args.module == "web":
        cmd.extend([
            "tests/test_web_comprehensive.py",
            "tests/test_web_integration.py",
        ])
    elif args.module == "vault":
        cmd.extend([
            "tests/test_vault_comprehensive.py",
            "tests/test_vault_integration.py",
        ])

    # Add options
    if args.verbose:
        cmd.append("-v")

    if args.failfast:
        cmd.append("-x")

    if args.fast:
        cmd.extend(["-m", "not slow"])

    if args.coverage:
        cmd.extend([
            "--cov=ciaf",
            "--cov-report=html",
            "--cov-report=term",
        ])

    # Run tests
    return_code = run_command(cmd)

    if return_code == 0:
        print("\n" + "=" * 80)
        print("[SUCCESS] All tests passed!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("[FAILURE] Some tests failed!")
        print("=" * 80)

    if args.coverage:
        print("\n[REPORT] Coverage report generated: htmlcov/index.html")

    return return_code


if __name__ == "__main__":
    sys.exit(main())
