#!/usr/bin/env python3
"""
CIAF Schema Migration Script

Automatically migrates existing schemas to use common reusable components.

Usage:
    python tools/migrate_to_common_schemas.py [--dry-run] [--backup]

Options:
    --dry-run   Show what would be changed without modifying files
    --backup    Create .backup files before modification (default: True)
    --no-backup Skip backup file creation

Author: CIAF Schema Team
Date: March 30, 2026
"""

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import argparse

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


class SchemaMigrator:
    """Migrates CIAF schemas to use common reusable components."""

    def __init__(self, schemas_dir: Path, dry_run: bool = False, backup: bool = True):
        self.schemas_dir = schemas_dir
        self.dry_run = dry_run
        self.backup = backup
        self.migrations_applied = []
        self.migration_stats = {
            "files_processed": 0,
            "files_modified": 0,
            "patterns_replaced": 0,
            "errors": []
        }

        # Define migration patterns
        self.migration_patterns = [
            # Identifiers
            {
                "name": "UUID",
                "ref": "common/identifiers/uuid.json",
                "pattern": {
                    "type": "string",
                    "format": "uuid"
                },
                "replacement": {
                    "$ref": "common/identifiers/uuid.json"
                }
            },
            {
                "name": "SHA-256 Hash",
                "ref": "common/identifiers/sha256-hash.json",
                "pattern": {
                    "type": "string",
                    "pattern": "^[a-f0-9]{64}$"
                },
                "replacement": {
                    "$ref": "common/identifiers/sha256-hash.json"
                }
            },
            {
                "name": "Semantic Version",
                "ref": "common/identifiers/semantic-version.json",
                "pattern": {
                    "type": "string",
                    "pattern": "^\\d+\\.\\d+\\.\\d+$"
                },
                "replacement": {
                    "$ref": "common/identifiers/semantic-version.json"
                }
            },
            # Enums
            {
                "name": "Principal Type",
                "ref": "common/enums/principal-type.json",
                "pattern": {
                    "type": "string",
                    "enum": ["agent", "human", "service", "system"]
                },
                "replacement": {
                    "$ref": "common/enums/principal-type.json"
                }
            },
            {
                "name": "Environment Type",
                "ref": "common/enums/environment-type.json",
                "pattern": {
                    "type": "string",
                    "enum": ["production", "staging", "development", "test"]
                },
                "replacement": {
                    "$ref": "common/enums/environment-type.json"
                }
            },
            {
                "name": "Decision Type (Full)",
                "ref": "common/enums/decision-type.json",
                "pattern": {
                    "type": "string",
                    "enum": ["allow", "deny", "require_approval", "not_applicable"]
                },
                "replacement": {
                    "$ref": "common/enums/decision-type.json"
                }
            },
            {
                "name": "Gate Type",
                "ref": "common/enums/gate-type.json",
                "pattern": {
                    "type": "string",
                    "enum": ["provenance", "validation", "approval", "runtime", "pre_action", "post_action"]
                },
                "replacement": {
                    "$ref": "common/enums/gate-type.json"
                }
            },
            {
                "name": "Evidence Strength",
                "ref": "common/enums/evidence-strength.json",
                "pattern": {
                    "type": "string",
                    "enum": ["real", "simulated", "fallback"]
                },
                "replacement": {
                    "$ref": "common/enums/evidence-strength.json"
                }
            },
            {
                "name": "Hash Algorithm",
                "ref": "common/enums/hash-algorithm.json",
                "pattern": {
                    "type": "string",
                    "enum": ["sha256", "sha3-256", "blake3"]
                },
                "replacement": {
                    "$ref": "common/enums/hash-algorithm.json"
                }
            },
            # Patterns
            {
                "name": "Timestamp",
                "ref": "common/patterns/timestamp.json",
                "pattern": {
                    "type": "string",
                    "format": "date-time"
                },
                "replacement": {
                    "$ref": "common/patterns/timestamp.json"
                }
            },
            {
                "name": "Metadata Object",
                "ref": "common/patterns/metadata.json",
                "pattern": {
                    "type": "object"
                },
                "field_name_match": "metadata",
                "replacement": {
                    "$ref": "common/patterns/metadata.json"
                }
            },
            {
                "name": "Merkle Path",
                "ref": "common/patterns/merkle-path.json",
                "pattern": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": "^[a-f0-9]{64}$"
                    }
                },
                "field_name_match": "merkle_path",
                "replacement": {
                    "$ref": "common/patterns/merkle-path.json"
                }
            },
            {
                "name": "Policy Obligations",
                "ref": "common/patterns/policy-obligations.json",
                "pattern": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "field_name_match": "policy_obligations",
                "replacement": {
                    "$ref": "common/patterns/policy-obligations.json"
                }
            }
        ]

    def matches_pattern(self, field_def: Dict, pattern: Dict, field_name: str = None) -> bool:
        """Check if a field definition matches a migration pattern."""
        # Check field name match if specified
        if "field_name_match" in pattern:
            if not field_name or field_name != pattern["field_name_match"]:
                return False

        # Get the pattern to match
        pattern_def = pattern["pattern"]

        # Check type
        if "type" in pattern_def and field_def.get("type") != pattern_def["type"]:
            return False

        # Check format
        if "format" in pattern_def:
            if field_def.get("format") != pattern_def["format"]:
                return False

        # Check pattern (regex)
        if "pattern" in pattern_def:
            if field_def.get("pattern") != pattern_def["pattern"]:
                return False

        # Check enum (exact match including order)
        if "enum" in pattern_def:
            field_enum = field_def.get("enum", [])
            pattern_enum = pattern_def["enum"]
            # Check if enums match (order-independent)
            if set(field_enum) != set(pattern_enum):
                return False

        # Check nested items
        if "items" in pattern_def:
            if "items" not in field_def:
                return False
            # Recursively check items
            if not self.matches_pattern(field_def["items"], {"pattern": pattern_def["items"]}):
                return False

        return True

    def migrate_field(self, field_def: Dict, field_name: str) -> Tuple[Dict, str]:
        """
        Migrate a single field definition to use common schema.
        
        Returns:
            Tuple of (new_definition, migration_name) or (original, None) if no match
        """
        for migration in self.migration_patterns:
            if self.matches_pattern(field_def, migration, field_name):
                # Create new definition with $ref
                new_def = migration["replacement"].copy()
                
                # Preserve description if it exists
                if "description" in field_def:
                    new_def["description"] = field_def["description"]
                
                # Preserve default if it exists and not in pattern
                if "default" in field_def and "default" not in migration["pattern"]:
                    new_def["default"] = field_def["default"]
                
                return new_def, migration["name"]
        
        return field_def, None

    def migrate_properties(self, properties: Dict) -> Tuple[Dict, List[str]]:
        """
        Migrate all properties in a schema.
        
        Returns:
            Tuple of (new_properties, list_of_migrations_applied)
        """
        new_properties = {}
        migrations_applied = []

        for field_name, field_def in properties.items():
            # Skip if already using $ref
            if "$ref" in field_def:
                new_properties[field_name] = field_def
                continue

            # Try to migrate
            new_def, migration_name = self.migrate_field(field_def, field_name)
            
            if migration_name:
                migrations_applied.append(f"{field_name} → {migration_name}")
            
            new_properties[field_name] = new_def

        return new_properties, migrations_applied

    def migrate_schema(self, schema: Dict) -> Tuple[Dict, List[str]]:
        """
        Migrate an entire schema.
        
        Returns:
            Tuple of (new_schema, list_of_migrations_applied)
        """
        new_schema = schema.copy()
        all_migrations = []

        # Migrate properties
        if "properties" in schema:
            new_properties, migrations = self.migrate_properties(schema["properties"])
            new_schema["properties"] = new_properties
            all_migrations.extend(migrations)

        # Migrate nested objects (e.g., in allOf, anyOf, oneOf)
        for key in ["allOf", "anyOf", "oneOf"]:
            if key in schema:
                new_items = []
                for item in schema[key]:
                    if "properties" in item:
                        new_props, migrations = self.migrate_properties(item["properties"])
                        new_item = item.copy()
                        new_item["properties"] = new_props
                        all_migrations.extend(migrations)
                        new_items.append(new_item)
                    else:
                        new_items.append(item)
                new_schema[key] = new_items

        return new_schema, all_migrations

    def migrate_file(self, file_path: Path) -> bool:
        """
        Migrate a single schema file.
        
        Returns:
            True if file was modified, False otherwise
        """
        try:
            # Read schema
            with open(file_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)

            # Migrate
            new_schema, migrations = self.migrate_schema(schema)

            # Check if anything changed
            if not migrations:
                return False

            # Backup original if requested
            if self.backup and not self.dry_run:
                backup_path = file_path.with_suffix('.json.backup')
                shutil.copy2(file_path, backup_path)

            # Write new schema if not dry run
            if not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(new_schema, f, indent=2, ensure_ascii=False)
                    f.write('\n')  # Add trailing newline

            # Record migration
            self.migrations_applied.append({
                "file": str(file_path.relative_to(self.schemas_dir.parent)),
                "migrations": migrations
            })

            return True

        except Exception as e:
            self.migration_stats["errors"].append({
                "file": str(file_path),
                "error": str(e)
            })
            return False

    def migrate_all(self):
        """Migrate all schemas in the directory."""
        # Get all JSON files except those in common/ directory
        schema_files = []
        for file_path in self.schemas_dir.rglob("*.json"):
            # Skip common schemas and example files
            if "common/" in str(file_path) or "example" in file_path.name.lower():
                continue
            schema_files.append(file_path)

        print(f"Found {len(schema_files)} schema files to process")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE MIGRATION'}")
        print(f"Backup: {'Enabled' if self.backup else 'Disabled'}")
        print("-" * 60)

        for file_path in sorted(schema_files):
            self.migration_stats["files_processed"] += 1
            
            modified = self.migrate_file(file_path)
            
            if modified:
                self.migration_stats["files_modified"] += 1
                relative_path = file_path.relative_to(self.schemas_dir.parent)
                print(f"[MODIFIED] {relative_path}")
            else:
                relative_path = file_path.relative_to(self.schemas_dir.parent)
                print(f"[NO CHANGE] {relative_path}")

    def generate_report(self) -> str:
        """Generate migration report."""
        report = []
        report.append("\n" + "=" * 60)
        report.append("MIGRATION REPORT")
        report.append("=" * 60)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        report.append("")
        report.append("Summary:")
        report.append(f"  Files Processed: {self.migration_stats['files_processed']}")
        report.append(f"  Files Modified: {self.migration_stats['files_modified']}")
        report.append(f"  Total Migrations: {len(self.migrations_applied)}")
        report.append("")

        if self.migrations_applied:
            report.append("Migrations Applied:")
            report.append("-" * 60)
            for migration in self.migrations_applied:
                report.append(f"\n{migration['file']}:")
                for change in migration['migrations']:
                    report.append(f"  - {change}")

        if self.migration_stats["errors"]:
            report.append("\nErrors:")
            report.append("-" * 60)
            for error in self.migration_stats["errors"]:
                report.append(f"{error['file']}: {error['error']}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate CIAF schemas to use common reusable components"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup files"
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=Path(__file__).parent.parent / "ciaf" / "schemas",
        help="Path to schemas directory"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Output path for migration report"
    )

    args = parser.parse_args()

    # Validate schemas directory
    if not args.schemas_dir.exists():
        print(f"Error: Schemas directory not found: {args.schemas_dir}")
        return 1

    # Create migrator
    migrator = SchemaMigrator(
        schemas_dir=args.schemas_dir,
        dry_run=args.dry_run,
        backup=not args.no_backup
    )

    # Run migration
    print(f"\nCIAF Schema Migration Tool")
    print(f"Schemas Directory: {args.schemas_dir}")
    print("")

    migrator.migrate_all()

    # Generate report
    report = migrator.generate_report()
    print(report)

    # Save report if requested
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {args.report}")

    return 0 if not migrator.migration_stats["errors"] else 1


if __name__ == "__main__":
    exit(main())
