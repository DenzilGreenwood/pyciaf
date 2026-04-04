"""
CIAF Vault Backend Systems

Storage backends for the CIAF vault:
- PostgreSQL: Enterprise-grade relational database
- SQLite: Embedded database (integrated in metadata_storage.py)
- JSON: File-based storage (integrated in metadata_storage.py)
- Pickle: Binary serialization (integrated in metadata_storage.py)
- Cloud: Abstract interface for Azure, GCP, AWS backends

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

# Cloud backend abstract interface
from .cloud_backend import CloudVaultBackend

try:
    from .postgresql_backend import PostgreSQLBackend, create_postgresql_vault

    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    PostgreSQLBackend = None
    create_postgresql_vault = None

__all__ = [
    "CloudVaultBackend",
    "PostgreSQLBackend",
    "create_postgresql_vault",
    "POSTGRESQL_AVAILABLE",
]
