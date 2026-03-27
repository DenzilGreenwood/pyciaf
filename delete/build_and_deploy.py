#!/usr/bin/env python3
"""
CIAF Package Build and Deployment Script
========================================

This script handles the complete build and deployment process for the CIAF package.
It includes cleaning, building, testing, and optional deployment to PyPI.

Usage:
    python build_and_deploy.py [options]

Options:
    --clean-only    : Only clean build artifacts
    --build-only    : Build without deploying
    --test-only     : Only run tests
    --deploy-test   : Deploy to TestPyPI
    --deploy-prod   : Deploy to production PyPI
    --full          : Clean, build, test, and deploy to TestPyPI
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path


class CIAFBuilder:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.build_dirs = ["build", "dist", "*.egg-info"]

    def run_command(self, command, description):
        """Run a command and handle errors"""
        print(f"\n{'='*60}")
        print(f"🔄 {description}")
        print(f"{'='*60}")
        print(f"Command: {command}")

        try:
            result = subprocess.run(
                command, shell=True, check=True, capture_output=True, text=True
            )
            print("✅ Success!")
            if result.stdout:
                print("Output:", result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print("❌ Failed!")
            print("Error:", e.stderr)
            return False

    def clean_build_artifacts(self):
        """Clean all build artifacts"""
        print("\n🧹 Cleaning build artifacts...")

        # Remove build directories
        for pattern in self.build_dirs:
            if "*" in pattern:
                import glob

                for path in glob.glob(pattern):
                    if os.path.exists(path):
                        shutil.rmtree(path)
                        print(f"  Removed: {path}")
            else:
                if os.path.exists(pattern):
                    shutil.rmtree(pattern)
                    print(f"  Removed: {pattern}")

        # Remove __pycache__ directories
        for root, dirs, files in os.walk(self.root_dir):
            for dir_name in dirs:
                if dir_name == "__pycache__":
                    cache_path = os.path.join(root, dir_name)
                    shutil.rmtree(cache_path)
                    print(f"  Removed: {cache_path}")

        print("✅ Clean completed!")

    def check_dependencies(self):
        """Check required build dependencies"""
        print("\n🔍 Checking build dependencies...")

        required_packages = ["build", "twine", "pytest"]
        missing = []

        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                missing.append(package)
                print(f"  ❌ {package}")

        if missing:
            print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
            install_cmd = f"pip install {' '.join(missing)}"
            print(f"Install with: {install_cmd}")

            response = input("Install missing dependencies? (y/n): ")
            if response.lower() == "y":
                return self.run_command(install_cmd, "Installing missing dependencies")
            else:
                print("❌ Cannot proceed without required dependencies")
                return False

        return True

    def run_tests(self):
        """Run the test suite"""
        if not os.path.exists("tests"):
            print("⚠️  No tests directory found, skipping tests")
            return True

        return self.run_command("python -m pytest tests/ -v", "Running test suite")

    def build_package(self):
        """Build the package"""
        return self.run_command("python -m build", "Building package distributions")

    def check_package(self):
        """Check package with twine"""
        return self.run_command(
            "python -m twine check dist/*", "Checking package distributions"
        )

    def deploy_to_testpypi(self):
        """Deploy to TestPyPI"""
        print("\n⚠️  Deploying to TestPyPI...")
        print("You'll need your TestPyPI credentials")
        return self.run_command(
            "python -m twine upload --repository testpypi dist/*",
            "Deploying to TestPyPI",
        )

    def deploy_to_pypi(self):
        """Deploy to production PyPI"""
        print("\n⚠️  DEPLOYING TO PRODUCTION PyPI!")
        print("This will make the package publicly available")
        confirm = input("Are you sure? Type 'YES' to confirm: ")

        if confirm != "YES":
            print("❌ Deployment cancelled")
            return False

        return self.run_command(
            "python -m twine upload dist/*", "Deploying to production PyPI"
        )

    def show_package_info(self):
        """Show information about the built package"""
        print("\n📦 Package Information:")
        print("=" * 60)

        dist_dir = Path("dist")
        if dist_dir.exists():
            for file in dist_dir.iterdir():
                size = file.stat().st_size / 1024  # KB
                print(f"  {file.name} ({size:.1f} KB)")

        # Show version info
        try:
            import toml

            with open("pyproject.toml", "r") as f:
                config = toml.load(f)
                version = config["project"]["version"]
                name = config["project"]["name"]
                print(f"\n  Package: {name}")
                print(f"  Version: {version}")
        except:
            print("  Could not read package info from pyproject.toml")


def main():
    parser = argparse.ArgumentParser(description="CIAF Package Builder")
    parser.add_argument(
        "--clean-only", action="store_true", help="Only clean build artifacts"
    )
    parser.add_argument(
        "--build-only", action="store_true", help="Build without deploying"
    )
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    parser.add_argument("--deploy-test", action="store_true", help="Deploy to TestPyPI")
    parser.add_argument(
        "--deploy-prod", action="store_true", help="Deploy to production PyPI"
    )
    parser.add_argument(
        "--full", action="store_true", help="Clean, build, test, and deploy to TestPyPI"
    )

    args = parser.parse_args()

    builder = CIAFBuilder()

    print("🚀 CIAF Package Builder")
    print("=" * 60)

    # Clean only
    if args.clean_only:
        builder.clean_build_artifacts()
        return

    # Test only
    if args.test_only:
        if not builder.run_tests():
            sys.exit(1)
        return

    # Check dependencies
    if not builder.check_dependencies():
        sys.exit(1)

    # Full workflow or build-only
    if args.full or args.build_only or not any(vars(args).values()):
        # Clean first
        builder.clean_build_artifacts()

        # Run tests
        if not builder.run_tests():
            print("❌ Tests failed, stopping build")
            sys.exit(1)

        # Build package
        if not builder.build_package():
            print("❌ Build failed")
            sys.exit(1)

        # Check package
        if not builder.check_package():
            print("❌ Package check failed")
            sys.exit(1)

        # Show package info
        builder.show_package_info()

        if args.build_only:
            print("\n✅ Build completed successfully!")
            return

    # Deployment
    if args.deploy_test or args.full:
        if not builder.deploy_to_testpypi():
            print("❌ TestPyPI deployment failed")
            sys.exit(1)
        print("\n✅ Successfully deployed to TestPyPI!")
        print("Test installation with:")
        print("  pip install --index-url https://test.pypi.org/simple/ ciaf")

    if args.deploy_prod:
        if not builder.deploy_to_pypi():
            print("❌ PyPI deployment failed")
            sys.exit(1)
        print("\n✅ Successfully deployed to production PyPI!")
        print("Install with: pip install ciaf")

    if not any(vars(args).values()):
        print("\n✅ Build completed successfully!")
        print("\nNext steps:")
        print("  --deploy-test  : Deploy to TestPyPI for testing")
        print("  --deploy-prod  : Deploy to production PyPI")


if __name__ == "__main__":
    main()
