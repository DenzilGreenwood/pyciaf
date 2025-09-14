"""
CIAF Metadata Storage System

This module provides comprehensive metadata storage and retrieval capabilities
for the CIAF framework, supporting multiple storage backends and formats.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import hashlib
import json
import os
import pickle
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class MetadataStorage:
    """
    Centralized metadata storage system for CIAF framework.

    Supports multiple storage backends:
    - JSON files (default)
    - SQLite database
    - Pickle files
    """

    def __init__(
        self,
        storage_path: str = "ciaf_metadata",
        backend: str = "json",
        use_compression: bool = False,
    ):
        """
        Initialize metadata storage.

        Args:
            storage_path: Base path for metadata storage
            backend: Storage backend ('json', 'sqlite', 'pickle')
            use_compression: Use compressed storage (creates CompressedMetadataStorage instance)
        """
        if use_compression:
            # Import here to avoid circular imports
            from .metadata_storage_compressed import CompressedMetadataStorage

            self._compressed_storage = CompressedMetadataStorage(
                storage_path=storage_path,
                backend="compressed_json" if backend == "json" else backend,
            )
            self._use_compressed = True
        else:
            self._use_compressed = False
            self.storage_path = Path(storage_path)
            self.backend = backend.lower()
            self.storage_path.mkdir(parents=True, exist_ok=True)

            # Initialize backend-specific storage
            if self.backend == "sqlite":
                self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite database for metadata storage."""
        self.db_path = self.storage_path / "ciaf_metadata.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create metadata table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT,
                stage TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata_hash TEXT,
                details TEXT,
                metadata_json TEXT NOT NULL
            )
        """
        )

        # Create audit trail table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_trail (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                action TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                details TEXT,
                FOREIGN KEY (parent_id) REFERENCES metadata (id)
            )
        """
        )

        # Create compliance events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS compliance_events (
                id TEXT PRIMARY KEY,
                metadata_id TEXT,
                framework TEXT NOT NULL,
                compliance_score REAL,
                validation_status TEXT,
                timestamp TEXT NOT NULL,
                details TEXT,
                FOREIGN KEY (metadata_id) REFERENCES metadata (id)
            )
        """
        )

        conn.commit()
        conn.close()

    def save_metadata(
        self,
        model_name: str,
        stage: str,
        event_type: str,
        metadata: Dict[str, Any],
        model_version: Optional[str] = None,
        details: Optional[str] = None,
    ) -> str:
        """
        Save metadata for a specific model and pipeline stage.

        Args:
            model_name: Name of the model
            stage: Pipeline stage (e.g., 'data_ingestion', 'training', 'inference')
            event_type: Type of event (e.g., 'data_processed', 'model_trained')
            metadata: Metadata dictionary to save
            model_version: Version of the model
            details: Additional details about the event

        Returns:
            Unique identifier for the saved metadata
        """
        # Use compressed storage if enabled
        if self._use_compressed:
            return self._compressed_storage.save_metadata(
                model_name, stage, event_type, metadata, model_version, details
            )

        # Generate unique ID and timestamp
        metadata_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create metadata hash for integrity verification
        metadata_str = json.dumps(metadata, sort_keys=True)
        metadata_hash = hashlib.sha256(metadata_str.encode()).hexdigest()

        # Prepare metadata record
        record = {
            "id": metadata_id,
            "model_name": model_name,
            "model_version": model_version or "1.0.0",
            "stage": stage,
            "event_type": event_type,
            "timestamp": timestamp,
            "metadata_hash": metadata_hash,
            "details": details,
            "metadata": metadata,
        }

        # Save using selected backend
        if self.backend == "json":
            self._save_json(record)
        elif self.backend == "sqlite":
            self._save_sqlite(record)
        elif self.backend == "pickle":
            self._save_pickle(record)

        return metadata_id

    def _save_json(self, record: Dict[str, Any]):
        """Save metadata as JSON file."""
        # Organize by model and date
        model_dir = self.storage_path / record["model_name"]
        model_dir.mkdir(exist_ok=True)

        date_str = datetime.fromisoformat(
            record["timestamp"].replace("Z", "+00:00")
        ).strftime("%Y-%m-%d")
        file_path = model_dir / f"{date_str}_{record['stage']}_{record['id'][:8]}.json"

        with open(file_path, "w") as f:
            json.dump(record, f, indent=2, default=str)

    def _save_sqlite(self, record: Dict[str, Any]):
        """Save metadata to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO metadata 
            (id, model_name, model_version, stage, event_type, timestamp, metadata_hash, details, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                record["id"],
                record["model_name"],
                record["model_version"],
                record["stage"],
                record["event_type"],
                record["timestamp"],
                record["metadata_hash"],
                record["details"],
                json.dumps(record["metadata"]),
            ),
        )

        conn.commit()
        conn.close()

    def _save_pickle(self, record: Dict[str, Any]):
        """Save metadata as pickle file."""
        model_dir = self.storage_path / record["model_name"]
        model_dir.mkdir(exist_ok=True)

        file_path = model_dir / f"{record['stage']}_{record['id'][:8]}.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(record, f)

    def get_metadata(self, metadata_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata by ID.

        Args:
            metadata_id: Unique identifier of the metadata

        Returns:
            Metadata record or None if not found
        """
        # Use compressed storage if enabled
        if self._use_compressed:
            return self._compressed_storage.get_metadata(metadata_id)

        if self.backend == "sqlite":
            return self._get_sqlite(metadata_id)
        else:
            return self._search_files(metadata_id)

    def _get_sqlite(self, metadata_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, model_name, model_version, stage, event_type, timestamp, 
                   metadata_hash, details, metadata_json
            FROM metadata WHERE id = ?
        """,
            (metadata_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "model_name": row[1],
                "model_version": row[2],
                "stage": row[3],
                "event_type": row[4],
                "timestamp": row[5],
                "metadata_hash": row[6],
                "details": row[7],
                "metadata": json.loads(row[8]),
            }
        return None

    def _search_files(self, metadata_id: str) -> Optional[Dict[str, Any]]:
        """Search for metadata in file system."""
        for model_dir in self.storage_path.iterdir():
            if model_dir.is_dir():
                for file_path in model_dir.iterdir():
                    if metadata_id[:8] in file_path.name:
                        if file_path.suffix == ".json":
                            with open(file_path, "r") as f:
                                record = json.load(f)
                                if record["id"] == metadata_id:
                                    return record
                        elif file_path.suffix == ".pkl":
                            with open(file_path, "rb") as f:
                                record = pickle.load(f)
                                if record["id"] == metadata_id:
                                    return record
        return None

    def get_model_metadata(
        self, model_name: str, stage: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all metadata for a specific model.

        Args:
            model_name: Name of the model
            stage: Optional stage filter
            limit: Maximum number of records to return

        Returns:
            List of metadata records
        """
        if self.backend == "sqlite":
            return self._get_model_sqlite(model_name, stage, limit)
        else:
            return self._get_model_files(model_name, stage, limit)

    def _get_model_sqlite(
        self, model_name: str, stage: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """Get model metadata from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if stage:
            cursor.execute(
                """
                SELECT id, model_name, model_version, stage, event_type, timestamp, 
                       metadata_hash, details, metadata_json
                FROM metadata 
                WHERE model_name = ? AND stage = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (model_name, stage, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, model_name, model_version, stage, event_type, timestamp, 
                       metadata_hash, details, metadata_json
                FROM metadata 
                WHERE model_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (model_name, limit),
            )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "model_name": row[1],
                "model_version": row[2],
                "stage": row[3],
                "event_type": row[4],
                "timestamp": row[5],
                "metadata_hash": row[6],
                "details": row[7],
                "metadata": json.loads(row[8]),
            }
            for row in rows
        ]

    def _get_model_files(
        self, model_name: str, stage: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """Get model metadata from files."""
        model_dir = self.storage_path / model_name
        if not model_dir.exists():
            return []

        records = []
        for file_path in model_dir.iterdir():
            if stage and stage not in file_path.name:
                continue

            try:
                if file_path.suffix == ".json":
                    with open(file_path, "r") as f:
                        record = json.load(f)
                        records.append(record)
                elif file_path.suffix == ".pkl":
                    with open(file_path, "rb") as f:
                        record = pickle.load(f)
                        records.append(record)
            except Exception:
                continue

        # Sort by timestamp and limit
        records.sort(key=lambda x: x["timestamp"], reverse=True)
        return records[:limit]

    def get_pipeline_trace(self, model_name: str) -> Dict[str, Any]:
        """
        Get complete pipeline trace for a model.

        Args:
            model_name: Name of the model

        Returns:
            Complete pipeline trace with all stages
        """
        metadata_records = self.get_model_metadata(model_name)

        # Organize by stage
        pipeline_trace = {
            "model_name": model_name,
            "trace_generated": datetime.now(timezone.utc).isoformat(),
            "stages": {},
        }

        for record in metadata_records:
            stage = record["stage"]
            if stage not in pipeline_trace["stages"]:
                pipeline_trace["stages"][stage] = []

            pipeline_trace["stages"][stage].append(
                {
                    "id": record["id"],
                    "event_type": record["event_type"],
                    "timestamp": record["timestamp"],
                    "details": record["details"],
                    "metadata": record["metadata"],
                }
            )

        return pipeline_trace

    def add_compliance_event(
        self,
        metadata_id: str,
        framework: str,
        compliance_score: float,
        validation_status: str,
        details: Optional[str] = None,
    ) -> str:
        """
        Add compliance validation event.

        Args:
            metadata_id: Associated metadata ID
            framework: Compliance framework (e.g., 'GDPR', 'FDA', 'EEOC')
            compliance_score: Compliance score (0.0 to 1.0)
            validation_status: Status ('passed', 'failed', 'warning')
            details: Additional details

        Returns:
            Compliance event ID
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        if self.backend == "sqlite":
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO compliance_events 
                (id, metadata_id, framework, compliance_score, validation_status, timestamp, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event_id,
                    metadata_id,
                    framework,
                    compliance_score,
                    validation_status,
                    timestamp,
                    details,
                ),
            )

            conn.commit()
            conn.close()
        else:
            # Save as separate file for file-based backends
            compliance_record = {
                "id": event_id,
                "metadata_id": metadata_id,
                "framework": framework,
                "compliance_score": compliance_score,
                "validation_status": validation_status,
                "timestamp": timestamp,
                "details": details,
            }

            compliance_dir = self.storage_path / "compliance_events"
            compliance_dir.mkdir(exist_ok=True)

            file_path = compliance_dir / f"{event_id}.json"
            with open(file_path, "w") as f:
                json.dump(compliance_record, f, indent=2)

        return event_id

    def export_metadata(
        self, model_name: Optional[str] = None, format: str = "json"
    ) -> str:
        """
        Export metadata to a specific format.

        Args:
            model_name: Optional model name filter
            format: Export format ('json', 'csv', 'xml')

        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if model_name:
            metadata_records = self.get_model_metadata(model_name, limit=1000)
            filename = f"{model_name}_metadata_{timestamp}.{format}"
        else:
            # Export all metadata
            metadata_records = []
            for model_dir in self.storage_path.iterdir():
                if model_dir.is_dir() and model_dir.name != "compliance_events":
                    metadata_records.extend(
                        self.get_model_metadata(model_dir.name, limit=1000)
                    )
            filename = f"all_metadata_{timestamp}.{format}"

        export_path = self.storage_path / "exports"
        export_path.mkdir(exist_ok=True)

        file_path = export_path / filename

        if format == "json":
            with open(file_path, "w") as f:
                json.dump(metadata_records, f, indent=2, default=str)
        elif format == "csv":
            self._export_csv(metadata_records, file_path)
        elif format == "xml":
            self._export_xml(metadata_records, file_path)

        return str(file_path)

    def _export_csv(self, records: List[Dict], file_path: Path):
        """Export metadata to CSV format."""
        import csv

        if not records:
            return

        # Flatten metadata for CSV
        flattened_records = []
        for record in records:
            flat_record = {k: v for k, v in record.items() if k != "metadata"}

            # Flatten metadata dict
            if "metadata" in record:
                for k, v in record["metadata"].items():
                    flat_record[f"metadata_{k}"] = v

            flattened_records.append(flat_record)

        with open(file_path, "w", newline="") as f:
            if flattened_records:
                writer = csv.DictWriter(f, fieldnames=flattened_records[0].keys())
                writer.writeheader()
                writer.writerows(flattened_records)

    def _export_xml(self, records: List[Dict], file_path: Path):
        """Export metadata to XML format."""
        import xml.etree.ElementTree as ET

        root = ET.Element("ciaf_metadata")

        for record in records:
            record_elem = ET.SubElement(root, "metadata_record")

            for key, value in record.items():
                if key == "metadata":
                    metadata_elem = ET.SubElement(record_elem, "metadata")
                    for meta_key, meta_value in value.items():
                        meta_elem = ET.SubElement(metadata_elem, meta_key)
                        meta_elem.text = str(meta_value)
                else:
                    elem = ET.SubElement(record_elem, key)
                    elem.text = str(value)

        tree = ET.ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)

    def cleanup_old_metadata(self, days_old: int = 365):
        """
        Clean up metadata older than specified days.

        Args:
            days_old: Number of days after which to clean up metadata
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
        cutoff_iso = cutoff_date.isoformat()

        if self.backend == "sqlite":
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                DELETE FROM metadata WHERE timestamp < ?
            """,
                (cutoff_iso,),
            )

            cursor.execute(
                """
                DELETE FROM compliance_events WHERE timestamp < ?
            """,
                (cutoff_iso,),
            )

            conn.commit()
            conn.close()
        else:
            # Clean up files
            for model_dir in self.storage_path.iterdir():
                if model_dir.is_dir():
                    for file_path in model_dir.iterdir():
                        if file_path.stat().st_mtime < cutoff_date.timestamp():
                            file_path.unlink()

    def _list_json_files(self) -> List[Path]:
        """List all JSON metadata files in the storage directory."""
        json_files = []
        for model_dir in self.storage_path.iterdir():
            if model_dir.is_dir():
                json_files.extend(model_dir.glob("*.json"))
        return json_files

    @property
    def config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        return {
            "storage_path": str(self.storage_path),
            "backend": self.backend,
            "db_path": str(self.db_path) if hasattr(self, "db_path") else None,
        }


# Global metadata storage instance
_global_storage = None


def get_metadata_storage(
    storage_path: str = "ciaf_metadata",
    backend: str = "json",
    use_compression: bool = False,
) -> MetadataStorage:
    """Get global metadata storage instance."""
    global _global_storage
    if _global_storage is None:
        _global_storage = MetadataStorage(storage_path, backend, use_compression)
    return _global_storage


def save_pipeline_metadata(
    model_name: str, stage: str, event_type: str, metadata: Dict[str, Any], **kwargs
) -> str:
    """Convenience function to save metadata using global storage."""
    storage = get_metadata_storage()
    return storage.save_metadata(model_name, stage, event_type, metadata, **kwargs)


def get_pipeline_trace(model_name: str) -> Dict[str, Any]:
    """Convenience function to get pipeline trace using global storage."""
    storage = get_metadata_storage()
    return storage.get_pipeline_trace(model_name)


if __name__ == "__main__":
    # Example usage
    from datetime import timedelta

    # Initialize storage
    storage = MetadataStorage("example_metadata", "sqlite")

    # Save some example metadata
    metadata_id = storage.save_metadata(
        model_name="job_classifier",
        stage="data_ingestion",
        event_type="data_loaded",
        metadata={
            "rows": 1000,
            "columns": 10,
            "null_values": 5,
            "data_quality_score": 0.95,
        },
        details="Successfully loaded training data",
    )

    print(f"Saved metadata with ID: {metadata_id}")

    # Add compliance event
    compliance_id = storage.add_compliance_event(
        metadata_id=metadata_id,
        framework="GDPR",
        compliance_score=0.98,
        validation_status="passed",
        details="Data processing complies with GDPR requirements",
    )

    print(f"Added compliance event: {compliance_id}")

    # Get pipeline trace
    trace = storage.get_pipeline_trace("job_classifier")
    print(f"Pipeline trace: {json.dumps(trace, indent=2)}")

    # Export metadata
    export_path = storage.export_metadata("job_classifier", "json")
    print(f"Exported metadata to: {export_path}")
