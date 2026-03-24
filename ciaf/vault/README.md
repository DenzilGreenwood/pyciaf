# CIAF Vault - Centralized Storage Backend

## Overview

The CIAF Vault module provides a unified storage backend for all CIAF framework operations including:

- **Metadata Storage**: Training and inference metadata
- **Receipt Storage**: Cryptographic receipts and audit trails
- **Compliance Events**: Regulatory compliance tracking
- **Training Snapshots**: Model checkpoints and training history
- **Provenance Capsules**: Data lineage and provenance tracking

## Features

✅ **Multiple Backends**:
- JSON files (default, development-friendly)
- SQLite (embedded database, production-ready)
- PostgreSQL (enterprise-grade, scalable)
- Compressed storage (optimized for large datasets)

✅ **Enterprise Capabilities**:
- Connection pooling (PostgreSQL)
- Transaction management
- ACID compliance
- Concurrent access support
- Advanced indexing and querying

✅ **Schema Validation**:
- JSON schema validation
- Type safety
- Data integrity checks

## Architecture

```
ciaf/vault/
├── __init__.py                    # Main vault exports
├── metadata_storage.py            # Core storage implementation
├── metadata_storage_compressed.py # Compressed storage backend
├── metadata_storage_optimized.py  # High-performance storage
├── metadata_config.py             # Configuration management
├── metadata_integration.py        # Integration utilities
├── metadata_tags/                 # Metadata tagging system
├── backends/                      # Storage backends
│   ├── __init__.py
│   └── postgresql_backend.py      # PostgreSQL implementation
└── databases/                     # Database files (SQLite, etc.)
```

## Quick Start

### JSON Backend (Default)

```python
from ciaf.vault import MetadataStorage

# Create storage with JSON backend
storage = MetadataStorage(
    storage_path="./my_vault",
    backend="json"
)

# Store metadata
storage.store_metadata(
    model_name="my_model",
    stage="training",
    event_type="epoch_complete",
    metadata={"epoch": 1, "loss": 0.5}
)

# Retrieve metadata
records = storage.retrieve_metadata(model_name="my_model")
```

### SQLite Backend

```python
from ciaf.vault import MetadataStorage

# Create storage with SQLite backend
storage = MetadataStorage(
    storage_path="./my_vault",
    backend="sqlite"
)

# All operations same as JSON backend
storage.store_metadata(...)
```

### PostgreSQL Backend (Enterprise)

```python
from ciaf.vault import MetadataStorage

# Create storage with PostgreSQL backend
storage = MetadataStorage(
    backend="postgresql",
    postgresql_config={
        "host": "localhost",
        "port": 5432,
        "database": "ciaf_vault",
        "user": "ciaf_user",
        "password": "your_secure_password"
    }
)

# Automatic connection pooling and transaction management
storage.store_metadata(...)
```

#### PostgreSQL Setup

1. **Install PostgreSQL client library**:
   ```bash
   pip install psycopg2-binary
   ```

2. **Create PostgreSQL database**:
   ```bash
   # Using psql
   psql -U postgres
   CREATE DATABASE ciaf_vault;
   CREATE USER ciaf_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE ciaf_vault TO ciaf_user;
   ```

3. **Or use automatic database creation**:
   ```python
   from ciaf.vault.backends import create_postgresql_vault

   vault = create_postgresql_vault(
       host="localhost",
       database="ciaf_vault",
       user="ciaf_user",
       password="your_password",
       create_database=True  # Automatically create if not exists
   )
   ```

### Compressed Storage

```python
from ciaf.vault import MetadataStorage

# Use compressed storage for large datasets
storage = MetadataStorage(
    storage_path="./my_vault",
    backend="json",
    use_compression=True
)

# Automatic compression/decompression
storage.store_metadata(...)
```

### High-Performance Storage

```python
from ciaf.vault import HighPerformanceMetadataStorage

# Optimized for high-throughput scenarios
storage = HighPerformanceMetadataStorage(
    storage_path="./my_vault",
    cache_size=10000  # In-memory cache
)

# Batch operations for better performance
storage.store_metadata_batch([...])
```

## Configuration Management

### Pre-configured Templates

```python
from ciaf.vault import (
    create_high_performance_config,
    create_compliance_first_config,
    create_balanced_config
)

# High-performance configuration
config = create_high_performance_config()

# Compliance-focused configuration
config = create_compliance_first_config()

# Balanced configuration
config = create_balanced_config()
```

### Custom Configuration

```python
from ciaf.vault import MetadataConfig

config = MetadataConfig(
    backend="postgresql",
    postgresql_config={
        "host": "db.example.com",
        "database": "production_vault"
    },
    enable_caching=True,
    cache_size=5000,
    compression=True
)
```

## Receipt Storage

The vault provides specialized storage for inference receipts:

```python
from ciaf.vault import MetadataStorage

storage = MetadataStorage(backend="postgresql", postgresql_config={...})

# Store inference receipt
storage.store_receipt(
    receipt_id="receipt_123",
    model_name="fraud_detection_v2",
    receipt_data={
        "query_hash": "abc123...",
        "output_hash": "def456...",
        "timestamp": "2026-03-24T14:30:00Z",
        "model_fingerprint": "789xyz..."
    }
)

# Retrieve receipts
receipts = storage.retrieve_receipts(
    model_name="fraud_detection_v2",
    limit=100
)
```

## PostgreSQL Schema

The PostgreSQL backend automatically creates the following tables:

- `ciaf_metadata` - Training/inference metadata
- `ciaf_inference_receipts` - Cryptographic receipts
- `ciaf_training_snapshots` - Model snapshots
- `ciaf_compliance_events` - Compliance tracking
- `ciaf_provenance_capsules` - Data lineage
- `ciaf_audit_trail` - Audit events

All tables include:
- UUID primary keys
- JSONB columns for flexible data
- Timestamps with timezone support
- GIN indexes for fast JSON queries
- Foreign key constraints for referential integrity

## Schema Validation

Vault data structures are defined in `ciaf/schemas/vault.schema.json`:

```python
import json
from pathlib import Path

# Load vault schema
schema_path = Path("ciaf/schemas/vault.schema.json")
with open(schema_path) as f:
    vault_schema = json.load(f)

# Validate data against schema
import jsonschema
jsonschema.validate(instance=your_data, schema=vault_schema)
```

## Integration with CIAF Framework

### Automatic Metadata Capture

```python
from ciaf.vault import MetadataCapture, create_model_manager

# Context manager for automatic capture
with MetadataCapture(model_name="my_model") as capture:
    # Training code here
    train_model()
    # Metadata automatically captured

# Model metadata manager
manager = create_model_manager(
    storage_path="./vault",
    backend="postgresql"
)

# Capture training metadata
manager.capture_training_metadata(
    model_name="my_model",
    epoch=1,
    metrics={"loss": 0.5, "accuracy": 0.95}
)
```

### Compliance Tracking

```python
from ciaf.vault import ComplianceTracker, create_compliance_tracker

tracker = create_compliance_tracker(
    storage_path="./vault",
    backend="postgresql"
)

# Track compliance events
tracker.track_compliance_event(
    framework="EU_AI_ACT",
    requirement="Article 10 - Data Governance",
    status="COMPLIANT",
    evidence={"audit_trail": "..."}
)

# Generate compliance summary
summary = tracker.generate_compliance_summary()
```

## Performance Considerations

### Backend Selection Guide

| Backend | Use Case | Read Speed | Write Speed | Scalability | Concurrency |
|---------|----------|------------|-------------|-------------|-------------|
| JSON | Development, small datasets | Fast | Fast | Limited | Single |
| SQLite | Production, embedded apps | Fast | Medium | Good | Limited |
| PostgreSQL | Enterprise, multi-user | Very Fast | Fast | Excellent | High |
| Compressed | Large datasets, archival | Medium | Slow | Excellent | Single |

### Performance Tips

1. **Use PostgreSQL for production**: Best performance for concurrent access
2. **Enable caching**: HighPerformanceMetadataStorage with memory cache
3. **Batch operations**: Group multiple writes for better throughput
4. **Index wisely**: PostgreSQL automatically indexes key fields
5. **Use compression**: For archival or large historical datasets

## Migration Between Backends

```python
from ciaf.vault import MetadataStorage

# Read from SQLite
source = MetadataStorage(
    storage_path="./old_vault",
    backend="sqlite"
)

# Write to PostgreSQL
target = MetadataStorage(
    backend="postgresql",
    postgresql_config={...}
)

# Migrate data
for record in source.retrieve_metadata(limit=10000):
    target.store_metadata(**record)
```

## Security Considerations

1. **PostgreSQL**:
   - Use strong passwords
   - Enable SSL/TLS connections
   - Restrict network access
   - Regular backups

2. **File-based backends**:
   - Set proper file permissions
   - Encrypt sensitive data
   - Regular backups
   - Access control via OS

3. **All backends**:
   - Validate inputs
   - Sanitize queries
   - Use parameterized queries
   - Audit access logs

## Troubleshooting

### PostgreSQL Connection Issues

```python
# Test PostgreSQL connection
from ciaf.vault.backends import PostgreSQLBackend

try:
    backend = PostgreSQLBackend(
        host="localhost",
        database="ciaf_vault",
        user="ciaf_user",
        password="your_password"
    )
    print("✅ PostgreSQL connection successful")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### Import Errors

If you see `psycopg2` import errors:
```bash
pip install psycopg2-binary
```

If on Windows and build errors occur:
```bash
pip install psycopg2-binary --no-cache-dir
```

### Performance Issues

1. Check database connections:
   ```python
   # Increase connection pool size
   storage = MetadataStorage(
       backend="postgresql",
       postgresql_config={
           ...,
           "min_connections": 5,
           "max_connections": 50
       }
   )
   ```

2. Enable caching:
   ```python
   from ciaf.vault import HighPerformanceMetadataStorage

   storage = HighPerformanceMetadataStorage(
       cache_size=10000
   )
   ```

## Examples

See the `examples/` directory for complete working examples:

- `examples/vault_basic_usage.py` - Basic vault operations
- `examples/vault_postgresql_demo.py` - PostgreSQL backend
- `examples/vault_migration.py` - Backend migration
- `complete_documentation/model_examples/` - Integration with models

## API Reference

See the docstrings in each module for detailed API documentation:

- `metadata_storage.py` - Core storage API
- `metadata_config.py` - Configuration options
- `metadata_integration.py` - Integration utilities
- `backends/postgresql_backend.py` - PostgreSQL-specific API

## Contributing

When adding new features to the vault:

1. Update the vault schema (`schemas/vault.schema.json`)
2. Add PostgreSQL table definitions if needed
3. Update this README
4. Add tests in `tests/test_vault.py`
5. Update examples

## License

See the main CIAF LICENSE file for licensing information.

---

**Version**: 1.0.0
**Last Updated**: 2026-03-24
**Author**: Denzil James Greenwood
