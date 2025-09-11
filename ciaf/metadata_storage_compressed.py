"""
CIAF Optimized Compressed Metadata Storage

Enhanced metadata storage system with compression support for improved
performance and reduced storage footprint.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import gzip
import hashlib
import json
import lzma
import os
import pickle
import sqlite3
import struct
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

# Optional import for msgpack
try:
    import msgpack

    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    msgpack = None


CompressionType = Literal["none", "gzip", "lzma", "zlib"]
SerializationType = Literal["json", "msgpack", "pickle"]


class CompressedMetadataStorage:
    """
    Optimized metadata storage with compression and multiple serialization formats.

    Features:
    - Multiple compression algorithms (gzip, lzma, zlib)
    - Multiple serialization formats (JSON, MessagePack, Pickle)
    - Automatic compression ratio optimization
    - Backward compatibility with existing JSON storage
    - Performance benchmarking
    """

    def __init__(
        self,
        storage_path: str = "ciaf_metadata_compressed",
        backend: str = "compressed_json",
        compression: CompressionType = "lzma",
        serialization: SerializationType = "json",  # Default to JSON for compatibility
        compression_level: int = 6,
    ):
        """
        Initialize optimized metadata storage.

        Args:
            storage_path: Base path for metadata storage
            backend: Storage backend ('compressed_json', 'sqlite', 'hybrid')
            compression: Compression algorithm ('none', 'gzip', 'lzma', 'zlib')
            serialization: Serialization format ('json', 'msgpack', 'pickle')
            compression_level: Compression level (0-9, higher = better compression)
        """
        self.storage_path = Path(storage_path)
        self.backend = backend.lower()
        self.compression = compression

        # Use JSON as fallback if msgpack is requested but not available
        if serialization == "msgpack" and not MSGPACK_AVAILABLE:
            print("⚠️ MessagePack not available, falling back to JSON serialization")
            self.serialization = "json"
        else:
            self.serialization = serialization

        self.compression_level = compression_level
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.compression_stats = {
            "total_files": 0,
            "total_uncompressed_size": 0,
            "total_compressed_size": 0,
            "compression_ratio": 0.0,
        }

        # Initialize backend-specific storage
        if self.backend == "sqlite":
            self._init_sqlite()
        elif self.backend == "hybrid":
            self._init_sqlite()
            self._init_hybrid_storage()

    def _init_sqlite(self):
        """Initialize SQLite database with compression support."""
        self.db_path = self.storage_path / "ciaf_metadata_compressed.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                stage TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata_hash TEXT NOT NULL,
                details TEXT,
                compression_type TEXT NOT NULL,
                serialization_type TEXT NOT NULL,
                uncompressed_size INTEGER,
                compressed_size INTEGER,
                metadata_blob BLOB NOT NULL
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_name ON metadata(model_name)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stage ON metadata(stage)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp ON metadata(timestamp)
        """
        )

        conn.commit()
        conn.close()

    def _init_hybrid_storage(self):
        """Initialize hybrid storage for large metadata objects."""
        self.blob_dir = self.storage_path / "blobs"
        self.blob_dir.mkdir(exist_ok=True)

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data using selected format."""
        if self.serialization == "json":
            return json.dumps(data, default=str, separators=(",", ":")).encode("utf-8")
        elif self.serialization == "msgpack":
            if not MSGPACK_AVAILABLE:
                raise ValueError(
                    "MessagePack is not available. Install with: pip install msgpack"
                )
            return msgpack.packb(data, default=str, use_bin_type=True)
        elif self.serialization == "pickle":
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unsupported serialization format: {self.serialization}")

    def _deserialize_data(self, data: bytes, serialization_type: str) -> Any:
        """Deserialize data from bytes."""
        if serialization_type == "json":
            return json.loads(data.decode("utf-8"))
        elif serialization_type == "msgpack":
            if not MSGPACK_AVAILABLE:
                raise ValueError(
                    "MessagePack is not available. Install with: pip install msgpack"
                )
            return msgpack.unpackb(data, raw=False, strict_map_key=False)
        elif serialization_type == "pickle":
            return pickle.loads(data)
        else:
            raise ValueError(f"Unsupported serialization format: {serialization_type}")

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using selected algorithm."""
        if self.compression == "none":
            return data
        elif self.compression == "gzip":
            return gzip.compress(data, compresslevel=self.compression_level)
        elif self.compression == "lzma":
            return lzma.compress(data, preset=self.compression_level)
        elif self.compression == "zlib":
            return zlib.compress(data, level=self.compression_level)
        else:
            raise ValueError(f"Unsupported compression: {self.compression}")

    def _decompress_data(self, data: bytes, compression_type: str) -> bytes:
        """Decompress data."""
        if compression_type == "none":
            return data
        elif compression_type == "gzip":
            return gzip.decompress(data)
        elif compression_type == "lzma":
            return lzma.decompress(data)
        elif compression_type == "zlib":
            return zlib.decompress(data)
        else:
            raise ValueError(f"Unsupported compression: {compression_type}")

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
        Save metadata with compression and optimization.

        Args:
            model_name: Name of the model
            stage: Pipeline stage
            event_type: Type of event
            metadata: Metadata dictionary to save
            model_version: Version of the model
            details: Additional details

        Returns:
            Unique identifier for the saved metadata
        """
        # Generate unique ID and timestamp
        metadata_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Serialize metadata
        serialized_data = self._serialize_data(metadata)
        uncompressed_size = len(serialized_data)

        # Compress serialized data
        compressed_data = self._compress_data(serialized_data)
        compressed_size = len(compressed_data)

        # Create metadata hash for integrity
        metadata_hash = hashlib.sha256(serialized_data).hexdigest()

        # Update compression statistics
        self._update_compression_stats(uncompressed_size, compressed_size)

        # Prepare record
        record = {
            "id": metadata_id,
            "model_name": model_name,
            "model_version": model_version or "1.0.0",
            "stage": stage,
            "event_type": event_type,
            "timestamp": timestamp,
            "metadata_hash": metadata_hash,
            "details": details,
            "compression_type": self.compression,
            "serialization_type": self.serialization,
            "uncompressed_size": uncompressed_size,
            "compressed_size": compressed_size,
            "compressed_metadata": compressed_data,
        }

        # Save using selected backend
        if self.backend == "compressed_json":
            self._save_compressed_file(record)
        elif self.backend == "sqlite":
            self._save_sqlite_compressed(record)
        elif self.backend == "hybrid":
            self._save_hybrid(record)

        return metadata_id

    def _save_compressed_file(self, record: Dict[str, Any]):
        """Save as compressed file with metadata header."""
        model_dir = self.storage_path / record["model_name"]
        model_dir.mkdir(exist_ok=True)

        date_str = datetime.fromisoformat(
            record["timestamp"].replace("Z", "+00:00")
        ).strftime("%Y-%m-%d")
        file_path = model_dir / f"{date_str}_{record['stage']}_{record['id'][:8]}.cmeta"

        # Create file header with metadata info
        header = {
            "id": record["id"],
            "model_name": record["model_name"],
            "model_version": record["model_version"],
            "stage": record["stage"],
            "event_type": record["event_type"],
            "timestamp": record["timestamp"],
            "metadata_hash": record["metadata_hash"],
            "details": record["details"],
            "compression_type": record["compression_type"],
            "serialization_type": record["serialization_type"],
            "uncompressed_size": record["uncompressed_size"],
            "compressed_size": record["compressed_size"],
        }

        header_data = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_size = len(header_data)

        with open(file_path, "wb") as f:
            # Write header size (4 bytes)
            f.write(struct.pack("I", header_size))
            # Write header
            f.write(header_data)
            # Write compressed metadata
            f.write(record["compressed_metadata"])

    def _save_sqlite_compressed(self, record: Dict[str, Any]):
        """Save to SQLite with compressed blob."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO metadata 
            (id, model_name, model_version, stage, event_type, timestamp, 
             metadata_hash, details, compression_type, serialization_type,
             uncompressed_size, compressed_size, metadata_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                record["compression_type"],
                record["serialization_type"],
                record["uncompressed_size"],
                record["compressed_size"],
                record["compressed_metadata"],
            ),
        )

        conn.commit()
        conn.close()

    def _save_hybrid(self, record: Dict[str, Any]):
        """Save using hybrid approach: small metadata in SQLite, large blobs in files."""
        # Determine if metadata should be stored as external blob
        size_threshold = 1024 * 10  # 10KB threshold

        if record["compressed_size"] > size_threshold:
            # Save blob to external file
            blob_id = f"{record['id'][:8]}_{record['stage']}.blob"
            blob_path = self.blob_dir / blob_id

            with open(blob_path, "wb") as f:
                f.write(record["compressed_metadata"])

            # Store reference in database
            metadata_blob = blob_id.encode("utf-8")
            is_external = True
        else:
            # Store directly in database
            metadata_blob = record["compressed_metadata"]
            is_external = False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO metadata 
            (id, model_name, model_version, stage, event_type, timestamp, 
             metadata_hash, details, compression_type, serialization_type,
             uncompressed_size, compressed_size, metadata_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                record["compression_type"] + ("_external" if is_external else ""),
                record["serialization_type"],
                record["uncompressed_size"],
                record["compressed_size"],
                metadata_blob,
            ),
        )

        conn.commit()
        conn.close()

    def get_metadata(self, metadata_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decompress metadata by ID."""
        if self.backend == "compressed_json":
            return self._get_compressed_file(metadata_id)
        elif self.backend in ["sqlite", "hybrid"]:
            return self._get_sqlite_compressed(metadata_id)

        return None

    def _get_compressed_file(self, metadata_id: str) -> Optional[Dict[str, Any]]:
        """Load metadata from compressed file."""
        # Search for file with matching ID
        for model_dir in self.storage_path.iterdir():
            if model_dir.is_dir():
                for file_path in model_dir.glob(f"*_{metadata_id[:8]}.cmeta"):
                    try:
                        with open(file_path, "rb") as f:
                            # Read header size
                            header_size = struct.unpack("I", f.read(4))[0]
                            # Read header
                            header_data = f.read(header_size)
                            header = json.loads(header_data.decode("utf-8"))

                            if header["id"] == metadata_id:
                                # Read compressed metadata
                                compressed_data = f.read()
                                # Decompress
                                decompressed_data = self._decompress_data(
                                    compressed_data, header["compression_type"]
                                )
                                # Deserialize
                                metadata = self._deserialize_data(
                                    decompressed_data, header["serialization_type"]
                                )

                                return {**header, "metadata": metadata}
                    except Exception:
                        continue

        return None

    def _get_sqlite_compressed(self, metadata_id: str) -> Optional[Dict[str, Any]]:
        """Load metadata from SQLite with decompression."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, model_name, model_version, stage, event_type, timestamp,
                   metadata_hash, details, compression_type, serialization_type,
                   uncompressed_size, compressed_size, metadata_blob
            FROM metadata WHERE id = ?
        """,
            (metadata_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        compression_type = row[8]
        is_external = compression_type.endswith("_external")
        if is_external:
            compression_type = compression_type.replace("_external", "")
            # Load from external blob file
            blob_id = row[12].decode("utf-8")
            blob_path = self.blob_dir / blob_id
            with open(blob_path, "rb") as f:
                compressed_data = f.read()
        else:
            compressed_data = row[12]

        # Decompress and deserialize
        decompressed_data = self._decompress_data(compressed_data, compression_type)
        metadata = self._deserialize_data(decompressed_data, row[9])

        return {
            "id": row[0],
            "model_name": row[1],
            "model_version": row[2],
            "stage": row[3],
            "event_type": row[4],
            "timestamp": row[5],
            "metadata_hash": row[6],
            "details": row[7],
            "compression_type": compression_type,
            "serialization_type": row[9],
            "uncompressed_size": row[10],
            "compressed_size": row[11],
            "metadata": metadata,
        }

    def _update_compression_stats(self, uncompressed_size: int, compressed_size: int):
        """Update compression statistics."""
        self.compression_stats["total_files"] += 1
        self.compression_stats["total_uncompressed_size"] += uncompressed_size
        self.compression_stats["total_compressed_size"] += compressed_size

        if self.compression_stats["total_uncompressed_size"] > 0:
            self.compression_stats["compression_ratio"] = 1 - (
                self.compression_stats["total_compressed_size"]
                / self.compression_stats["total_uncompressed_size"]
            )

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics."""
        return {
            **self.compression_stats,
            "space_saved_mb": (
                self.compression_stats["total_uncompressed_size"]
                - self.compression_stats["total_compressed_size"]
            )
            / (1024 * 1024),
            "compression_percentage": self.compression_stats["compression_ratio"] * 100,
        }

    def migrate_from_json(self, source_path: str) -> int:
        """
        Migrate existing JSON metadata to compressed format.

        Args:
            source_path: Path to existing JSON metadata storage

        Returns:
            Number of files migrated
        """
        source_dir = Path(source_path)
        migrated_count = 0

        if not source_dir.exists():
            return 0

        for model_dir in source_dir.iterdir():
            if model_dir.is_dir():
                for json_file in model_dir.glob("*.json"):
                    try:
                        with open(json_file, "r") as f:
                            record = json.load(f)

                        # Extract metadata and save in compressed format
                        metadata = record.get("metadata", {})
                        self.save_metadata(
                            model_name=record["model_name"],
                            stage=record["stage"],
                            event_type=record["event_type"],
                            metadata=metadata,
                            model_version=record.get("model_version"),
                            details=record.get("details"),
                        )

                        migrated_count += 1

                    except Exception as e:
                        print(f"Failed to migrate {json_file}: {e}")
                        continue

        return migrated_count


# Convenience function for backward compatibility
def get_metadata_storage(
    storage_path: str = "ciaf_metadata_compressed",
    compression: CompressionType = "lzma",
    serialization: SerializationType = "json",  # Default to JSON for compatibility
) -> CompressedMetadataStorage:
    """Get optimized metadata storage instance."""
    return CompressedMetadataStorage(
        storage_path=storage_path, compression=compression, serialization=serialization
    )
