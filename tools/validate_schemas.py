#!/usr/bin/env python3
"""
CIAF Schema Validation Tool

Validates all CIAF schemas and checks for common issues.

Usage:
    python tools/validate_schemas.py [--verbose]

Author: CIAF Schema Team
Date: March 30, 2026
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")


class SchemaValidator:
    """Validates CIAF JSON schemas."""

    def __init__(self, schemas_dir: Path, verbose: bool = False):
        self.schemas_dir = schemas_dir
        self.verbose = verbose
        self.validation_results = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "warnings": [],
            "errors": [],
        }

    def validate_json(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate that file is valid JSON."""
        errors = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json.load(f)
            return True, errors
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Error reading file: {e}")
            return False, errors

    def validate_schema_structure(self, schema: Dict, file_path: Path) -> List[str]:
        """Validate schema structure and conventions."""
        warnings = []

        # Check for required top-level fields
        if "$schema" not in schema:
            warnings.append("Missing $schema field")

        if "$id" not in schema:
            warnings.append("Missing $id field")

        if "title" not in schema:
            warnings.append("Missing title field")

        if "description" not in schema:
            warnings.append("Missing description field")

        # Check schema version
        if "$schema" in schema:
            if "2020-12" not in schema["$schema"]:
                warnings.append(
                    f"Schema version should be 2020-12, got: {schema['$schema']}"
                )

        # Check for common anti-patterns
        if "properties" in schema:
            for prop_name, prop_def in schema["properties"].items():
                # Check for missing descriptions
                if "description" not in prop_def and "$ref" not in prop_def:
                    warnings.append(f"Property '{prop_name}' missing description")

                # Check for SHA-256 hashes not using common schema
                if (
                    prop_def.get("type") == "string"
                    and prop_def.get("pattern") == "^[a-f0-9]{64}$"
                    and "$ref" not in prop_def
                ):
                    warnings.append(
                        f"Property '{prop_name}' uses inline SHA-256 pattern. "
                        "Consider using common/identifiers/sha256-hash.json"
                    )

                # Check for UUIDs not using common schema
                if (
                    prop_def.get("type") == "string"
                    and prop_def.get("format") == "uuid"
                    and "$ref" not in prop_def
                ):
                    warnings.append(
                        f"Property '{prop_name}' uses inline UUID format. "
                        "Consider using common/identifiers/uuid.json"
                    )

                # Check for timestamps not using common schema
                if (
                    prop_def.get("type") == "string"
                    and prop_def.get("format") == "date-time"
                    and "$ref" not in prop_def
                ):
                    warnings.append(
                        f"Property '{prop_name}' uses inline timestamp format. "
                        "Consider using common/patterns/timestamp.json"
                    )

        return warnings

    def validate_file(self, file_path: Path) -> bool:
        """Validate a single schema file."""
        relative_path = file_path.relative_to(self.schemas_dir.parent)

        # Validate JSON
        valid_json, errors = self.validate_json(file_path)

        if not valid_json:
            self.validation_results["errors"].append(
                {"file": str(relative_path), "errors": errors}
            )
            return False

        # Load and validate structure
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                schema = json.load(f)

            warnings = self.validate_schema_structure(schema, file_path)

            if warnings:
                self.validation_results["warnings"].append(
                    {"file": str(relative_path), "warnings": warnings}
                )

            return True

        except Exception as e:
            self.validation_results["errors"].append(
                {"file": str(relative_path), "errors": [f"Validation error: {e}"]}
            )
            return False

    def validate_all(self):
        """Validate all schemas."""
        # Get all JSON files
        schema_files = list(self.schemas_dir.rglob("*.json"))

        print(f"Validating {len(schema_files)} schema files...")
        print("-" * 60)

        for file_path in sorted(schema_files):
            self.validation_results["total_files"] += 1

            if self.validate_file(file_path):
                self.validation_results["valid_files"] += 1
                if self.verbose:
                    relative_path = file_path.relative_to(self.schemas_dir.parent)
                    print(f"[OK] {relative_path}")
            else:
                self.validation_results["invalid_files"] += 1
                relative_path = file_path.relative_to(self.schemas_dir.parent)
                print(f"[ERROR] {relative_path}")

    def generate_report(self) -> str:
        """Generate validation report."""
        report = []
        report.append("\n" + "=" * 60)
        report.append("SCHEMA VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        report.append("Summary:")
        report.append(f"  Total Files: {self.validation_results['total_files']}")
        report.append(f"  Valid: {self.validation_results['valid_files']}")
        report.append(f"  Invalid: {self.validation_results['invalid_files']}")
        report.append(
            f"  Files with Warnings: {len(self.validation_results['warnings'])}"
        )
        report.append("")

        if self.validation_results["errors"]:
            report.append("ERRORS:")
            report.append("-" * 60)
            for error_info in self.validation_results["errors"]:
                report.append(f"\n{error_info['file']}:")
                for error in error_info["errors"]:
                    report.append(f"  [X] {error}")

        if self.validation_results["warnings"]:
            report.append("\nWARNINGS:")
            report.append("-" * 60)
            for warning_info in self.validation_results["warnings"]:
                report.append(f"\n{warning_info['file']}:")
                for warning in warning_info["warnings"]:
                    report.append(f"  [!] {warning}")

        report.append("\n" + "=" * 60)

        if self.validation_results["invalid_files"] == 0:
            report.append("[OK] All schemas are valid!")
        else:
            report.append(
                f"[ERROR] Found {self.validation_results['invalid_files']} invalid schema(s)"
            )

        report.append("=" * 60)

        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate CIAF JSON schemas")
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed validation progress"
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=Path(__file__).parent.parent / "ciaf" / "schemas",
        help="Path to schemas directory",
    )

    args = parser.parse_args()

    # Validate schemas directory
    if not args.schemas_dir.exists():
        print(f"Error: Schemas directory not found: {args.schemas_dir}")
        return 1

    # Create validator
    validator = SchemaValidator(schemas_dir=args.schemas_dir, verbose=args.verbose)

    # Run validation
    print("\nCIAF Schema Validation Tool")
    print(f"Schemas Directory: {args.schemas_dir}\n")

    validator.validate_all()

    # Generate report
    report = validator.generate_report()
    print(report)

    return 0 if validator.validation_results["invalid_files"] == 0 else 1


if __name__ == "__main__":
    exit(main())
