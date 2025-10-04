"""
Write-Once-Read-Many (WORM) durable storage implementation for CIAF.

Provides persistent storage for Merkle trees and audit records with 
SQLite and LMDB adapters for production deployments.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import sqlite3
import tempfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False

from .canonicalization import AnchorRecord
from .crypto import sha256_hash
from .enums import RecordType
from .interfaces import Merkle
from .merkle import MerkleTree


@dataclass
class WORMRecord:
    """Single WORM record entry."""
    id: str
    timestamp: str
    record_type: RecordType
    data: Dict[str, Any]
    hash: str
    
    def __post_init__(self):
        if not self.hash:
            self.hash = sha256_hash(json.dumps(self.data, sort_keys=True).encode('utf-8'))


class WORMStore(ABC):
    """Abstract base class for WORM storage implementations."""
    
    @abstractmethod
    def append_record(self, record: WORMRecord) -> str:
        """Append a record and return its ID."""
        ...
    
    @abstractmethod
    def get_record(self, record_id: str) -> Optional[WORMRecord]:
        """Retrieve a record by ID."""
        ...
    
    @abstractmethod
    def list_records(self, record_type: Optional[RecordType] = None) -> List[WORMRecord]:
        """List all records, optionally filtered by type."""
        ...
    
    @abstractmethod
    def close(self):
        """Close the store and clean up resources."""
        ...


class SQLiteWORMStore(WORMStore):
    """SQLite-based WORM store for production deployments."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite WORM store.
        
        Args:
            db_path: Path to SQLite database file. If None, creates temporary file.
        """
        self.db_path = db_path or tempfile.mktemp(suffix='.db')
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
        self.conn.execute('PRAGMA synchronous=FULL')  # Durability
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema."""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS worm_records (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                record_type TEXT NOT NULL,
                data TEXT NOT NULL,
                hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_record_type ON worm_records(record_type)
        ''')
        
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON worm_records(timestamp)
        ''')
        
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_hash ON worm_records(hash)
        ''')
        
        self.conn.commit()
    
    def append_record(self, record: WORMRecord) -> str:
        """Append a record to the WORM store."""
        # Check for duplicate ID (WORM violation)
        cursor = self.conn.execute('SELECT id FROM worm_records WHERE id = ?', (record.id,))
        if cursor.fetchone():
            raise ValueError(f"Record {record.id} already exists (WORM violation)")
        
        # Insert record
        self.conn.execute('''
            INSERT INTO worm_records (id, timestamp, record_type, data, hash)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            record.id,
            record.timestamp,
            record.record_type.value,
            json.dumps(record.data, sort_keys=True),
            record.hash
        ))
        
        self.conn.commit()
        return record.id
    
    def get_record(self, record_id: str) -> Optional[WORMRecord]:
        """Retrieve a record by ID."""
        cursor = self.conn.execute('''
            SELECT id, timestamp, record_type, data, hash
            FROM worm_records WHERE id = ?
        ''', (record_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return WORMRecord(
            id=row[0],
            timestamp=row[1],
            record_type=RecordType(row[2]),
            data=json.loads(row[3]),
            hash=row[4]
        )
    
    def list_records(self, record_type: Optional[RecordType] = None) -> List[WORMRecord]:
        """List all records, optionally filtered by type."""
        if record_type:
            cursor = self.conn.execute('''
                SELECT id, timestamp, record_type, data, hash
                FROM worm_records WHERE record_type = ?
                ORDER BY created_at ASC
            ''', (record_type.value,))
        else:
            cursor = self.conn.execute('''
                SELECT id, timestamp, record_type, data, hash
                FROM worm_records ORDER BY created_at ASC
            ''')
        
        return [
            WORMRecord(
                id=row[0],
                timestamp=row[1],
                record_type=RecordType(row[2]),
                data=json.loads(row[3]),
                hash=row[4]
            )
            for row in cursor.fetchall()
        ]
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()


class LMDBWORMStore(WORMStore):
    """LMDB-based WORM store for high-performance deployments."""
    
    def __init__(self, db_path: Optional[str] = None, map_size: int = 1024 * 1024 * 1024):
        """
        Initialize LMDB WORM store.
        
        Args:
            db_path: Path to LMDB database directory. If None, creates temporary directory.
            map_size: Maximum size of LMDB map (default 1GB)
        """
        if not LMDB_AVAILABLE:
            raise ImportError("LMDB is required for LMDBWORMStore (pip install lmdb)")
        
        self.db_path = db_path or tempfile.mkdtemp()
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        
        self.env = lmdb.open(
            self.db_path,
            map_size=map_size,
            sync=True,  # Force synchronous writes for durability
            writemap=False  # Safer for concurrent access
        )
        
        # Create sub-databases
        with self.env.begin(write=True) as txn:
            # Main records database
            self.records_db = self.env.open_db(b'records', txn=txn)
            # Index by record type
            self.type_index_db = self.env.open_db(b'type_index', txn=txn)
    
    def append_record(self, record: WORMRecord) -> str:
        """Append a record to the WORM store."""
        with self.env.begin(write=True) as txn:
            # Check for duplicate ID (WORM violation)
            if txn.get(record.id.encode('utf-8'), db=self.records_db):
                raise ValueError(f"Record {record.id} already exists (WORM violation)")
            
            # Serialize record
            record_data = {
                'id': record.id,
                'timestamp': record.timestamp,
                'record_type': record.record_type.value,
                'data': record.data,
                'hash': record.hash
            }
            
            serialized = json.dumps(record_data, sort_keys=True).encode('utf-8')
            
            # Store in main records database
            txn.put(record.id.encode('utf-8'), serialized, db=self.records_db)
            
            # Update type index
            type_key = f"{record.record_type.value}:{record.id}".encode('utf-8')
            txn.put(type_key, record.id.encode('utf-8'), db=self.type_index_db)
        
        return record.id
    
    def get_record(self, record_id: str) -> Optional[WORMRecord]:
        """Retrieve a record by ID."""
        with self.env.begin() as txn:
            data = txn.get(record_id.encode('utf-8'), db=self.records_db)
            if not data:
                return None
            
            record_dict = json.loads(data.decode('utf-8'))
            return WORMRecord(
                id=record_dict['id'],
                timestamp=record_dict['timestamp'],
                record_type=RecordType(record_dict['record_type']),
                data=record_dict['data'],
                hash=record_dict['hash']
            )
    
    def list_records(self, record_type: Optional[RecordType] = None) -> List[WORMRecord]:
        """List all records, optionally filtered by type."""
        records = []
        
        with self.env.begin() as txn:
            if record_type:
                # Use type index for efficient filtering
                cursor = txn.cursor(db=self.type_index_db)
                prefix = f"{record_type.value}:".encode('utf-8')
                
                if cursor.set_range(prefix):
                    for key, value in cursor:
                        if not key.startswith(prefix):
                            break
                        
                        record_id = value.decode('utf-8')
                        record_data = txn.get(record_id.encode('utf-8'), db=self.records_db)
                        if record_data:
                            record_dict = json.loads(record_data.decode('utf-8'))
                            records.append(WORMRecord(
                                id=record_dict['id'],
                                timestamp=record_dict['timestamp'],
                                record_type=RecordType(record_dict['record_type']),
                                data=record_dict['data'],
                                hash=record_dict['hash']
                            ))
            else:
                # Get all records
                cursor = txn.cursor(db=self.records_db)
                for key, value in cursor:
                    record_dict = json.loads(value.decode('utf-8'))
                    records.append(WORMRecord(
                        id=record_dict['id'],
                        timestamp=record_dict['timestamp'],
                        record_type=RecordType(record_dict['record_type']),
                        data=record_dict['data'],
                        hash=record_dict['hash']
                    ))
        
        return records
    
    def close(self):
        """Close the LMDB environment."""
        if self.env:
            self.env.close()


class DurableWORMMerkleTree:
    """
    WORM Merkle tree with durable storage backend.
    
    Combines in-memory Merkle tree performance with persistent storage
    for production audit scenarios.
    """
    
    def __init__(self, store: WORMStore, tree_id: str = "default"):
        """
        Initialize durable WORM Merkle tree.
        
        Args:
            store: WORM storage backend
            tree_id: Identifier for this tree instance
        """
        self.store = store
        self.tree_id = tree_id
        self.merkle_tree = MerkleTree()
        self._load_from_store()
    
    def _load_from_store(self):
        """Load existing leaves from persistent store."""
        records = self.store.list_records(RecordType.DATASET)  # Or appropriate type
        leaves = []
        
        for record in records:
            if record.data.get('tree_id') == self.tree_id:
                leaves.append(record.data['leaf_hash'])
        
        if leaves:
            self.merkle_tree = MerkleTree(leaves)
    
    def append_leaf(self, leaf_hash: str, metadata: Dict[str, Any]) -> str:
        """Append leaf to both Merkle tree and persistent store."""
        # Add to Merkle tree
        new_root = self.merkle_tree.add_leaf(leaf_hash)
        
        # Create WORM record
        record_data = {
            'tree_id': self.tree_id,
            'leaf_hash': leaf_hash,
            'metadata': metadata,
            'root_after_append': new_root
        }
        
        record = WORMRecord(
            id=f"{self.tree_id}:{leaf_hash}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            record_type=RecordType.DATASET,
            data=record_data,
            hash=""  # Will be computed in __post_init__
        )
        
        # Store persistently
        self.store.append_record(record)
        
        return new_root
    
    def get_root(self) -> str:
        """Get current Merkle root."""
        return self.merkle_tree.get_root()
    
    def get_proof(self, leaf_hash: str) -> List[tuple[str, str]]:
        """Get Merkle proof for a leaf."""
        return self.merkle_tree.get_proof(leaf_hash)
    
    def verify_proof(self, leaf_hash: str, proof: List[tuple[str, str]], root: str) -> bool:
        """Verify Merkle inclusion proof."""
        return self.merkle_tree.verify_proof(leaf_hash, proof, root)
    
    def close(self):
        """Close the underlying store."""
        self.store.close()


def create_sqlite_worm_store(db_path: Optional[str] = None) -> SQLiteWORMStore:
    """Factory function for SQLite WORM store."""
    return SQLiteWORMStore(db_path)


def create_lmdb_worm_store(db_path: Optional[str] = None, map_size: int = 1024 * 1024 * 1024) -> LMDBWORMStore:
    """Factory function for LMDB WORM store."""
    return LMDBWORMStore(db_path, map_size)