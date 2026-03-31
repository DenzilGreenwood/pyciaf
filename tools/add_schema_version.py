#!/usr/bin/env python3
"""
Add version field to all CIAF JSON schema files.

This script adds a consistent "version" field to all schema files
to support schema evolution and versioning.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Configuration
SCHEMA_VERSION = "1.0.0"
SCHEMAS_DIR = Path(__file__).parent.parent / "ciaf" / "schemas"


def add_version_to_schema(schema: Dict[str, Any], version: str) -> Dict[str, Any]:
    """
    Add version field to schema, preserving field order.

    Args:
        schema: The schema dictionary
        version: Version string to add

    Returns:
        Modified schema with version field
    """
    # Check if version already exists
    if "version" in schema:
        print(f"    Version already exists: {schema['version']}")
        return schema

    # Create new ordered dict with version after description
    result = {}
    for key, value in schema.items():
        result[key] = value
        # Add version after description field
        if key == "description":
            result["version"] = version

    # If no description field found, add version after title
    if "version" not in result:
        new_result = {}
        for key, value in schema.items():
            new_result[key] = value
            if key == "title":
                new_result["version"] = version
        result = new_result

    # If still not added (no title or description), add at beginning after $schema
    if "version" not in result:
        new_result = {}
        for key, value in schema.items():
            new_result[key] = value
            if key == "$schema":
                new_result["version"] = version
        result = new_result

    return result


def process_schema_file(file_path: Path, version: str, dry_run: bool = False) -> bool:
    """
    Process a single schema file, adding version field.

    Args:
        file_path: Path to the schema file
        version: Version string to add
        dry_run: If True, don't write changes

    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Read the schema
        with open(file_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # Skip if not a dict (shouldn't happen)
        if not isinstance(schema, dict):
            print(f"[SKIP] {file_path.relative_to(SCHEMAS_DIR)}: Not a JSON object")
            return False

        # Check if version already exists
        if "version" in schema:
            print(
                f"[OK] {file_path.relative_to(SCHEMAS_DIR)}: Version exists ({schema['version']})"
            )
            return False

        # Add version
        updated_schema = add_version_to_schema(schema, version)

        if not dry_run:
            # Write back with proper formatting
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(updated_schema, f, indent=2, ensure_ascii=False)
                f.write("\n")  # Add trailing newline

            print(
                f"[MODIFIED] {file_path.relative_to(SCHEMAS_DIR)}: Added version {version}"
            )
        else:
            print(
                f"[DRY-RUN] {file_path.relative_to(SCHEMAS_DIR)}: Would add version {version}"
            )

        return True

    except json.JSONDecodeError as e:
        print(f"[ERROR] {file_path.relative_to(SCHEMAS_DIR)}: JSON decode error - {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {file_path.relative_to(SCHEMAS_DIR)}: {e}")
        return False


def main():
    """Main execution function."""
    # Parse arguments
    dry_run = "--dry-run" in sys.argv
    version = SCHEMA_VERSION

    # Check if custom version provided
    for arg in sys.argv[1:]:
        if arg.startswith("--version="):
            version = arg.split("=", 1)[1]

    print(f"\n{'='*60}")
    print(f"Adding Schema Version: {version}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
    print(f"Directory: {SCHEMAS_DIR}")
    print(f"{'='*60}\n")

    if not SCHEMAS_DIR.exists():
        print(f"Error: Schemas directory not found: {SCHEMAS_DIR}")
        return 1

    # Find all .json files (excluding .backup files)
    schema_files = sorted(SCHEMAS_DIR.rglob("*.json"))

    # Exclude backup files and example files
    schema_files = [
        f
        for f in schema_files
        if not f.name.endswith(".backup")
        and "backups" not in f.parts
        and "example" not in f.stem.lower()
    ]

    print(f"Found {len(schema_files)} schema files\n")

    # Process each file
    modified_count = 0
    unchanged_count = 0
    error_count = 0

    for file_path in schema_files:
        try:
            was_modified = process_schema_file(file_path, version, dry_run)
            if was_modified:
                modified_count += 1
            else:
                unchanged_count += 1
        except Exception as e:
            print(f"[ERROR] {file_path.relative_to(SCHEMAS_DIR)}: {e}")
            error_count += 1

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total Files:    {len(schema_files)}")
    print(f"Modified:       {modified_count}")
    print(f"Unchanged:      {unchanged_count}")
    print(f"Errors:         {error_count}")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY-RUN complete. Run without --dry-run to apply changes.\n")
    else:
        print("Schema version update complete.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
