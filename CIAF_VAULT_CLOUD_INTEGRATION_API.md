# CIAF Vault Cloud Integration API Design

**Created:** April 4, 2026  
**Author:** Denzil James Greenwood  
**Purpose:** Define API requirements for integrating CIAF Vault concepts with Azure and GCP implementations

---

## Executive Summary

This document defines the functions and APIs needed to enable CIAF's vault concepts (centralized metadata storage, cryptographic receipts, audit trails) to work seamlessly across:

1. **Core CIAF (pyciaf)** - Vault storage backend with SQLite/PostgreSQL/JSON
2. **ciaf.azure** - Azure Key Vault, Blob Storage, Azure ML
3. **ciaf.gcp** - Cloud KMS, Cloud Storage, Vertex AI

---

## 1. Core Vault Concepts (from pyciaf/ciaf/vault/)

### 1.1 Metadata Storage (`MetadataStorage`)

**Purpose:** Centralized storage for ML lifecycle metadata

**Key Capabilities:**
- Multi-backend support (JSON, SQLite, PostgreSQL, compressed)
- Metadata versioning and integrity hashing
- Query interface for pipeline traces
- Audit trail tracking
- Compliance event logging

**Core Methods:**
```python
class MetadataStorage:
    def save_metadata(model_name, stage, event_type, metadata, model_version, details) -> str
    def get_metadata(metadata_id) -> Dict[str, Any]
    def query_metadata(model_name, stage, event_type, start_date, end_date) -> List[Dict]
    def get_pipeline_trace(model_name, model_version) -> Dict
    def verify_metadata_integrity(metadata_id) -> bool
    def save_audit_event(parent_id, action, user_id, details) -> str
    def get_audit_trail(metadata_id) -> List[Dict]
```

### 1.2 Evidence Vault (`EvidenceVault`)

**Purpose:** Tamper-evident cryptographic receipts for agent actions

**Key Capabilities:**
- HMAC-SHA256 signatures
- Hash-chaining (blockchain-like)
- Receipt verification
- Principal-based indexing

**Core Methods:**
```python
class EvidenceVault(EvidenceRecorder):
    def record_action(result: ExecutionResult) -> ActionReceipt
    def verify_receipt(receipt: ActionReceipt) -> bool
    def get_receipts_by_principal(principal_id) -> List[ActionReceipt]
    def verify_chain() -> bool
```

### 1.3 Vault Client (`VaultClient`)

**Purpose:** REST API client for centralized vault web application

**Key Capabilities:**
- Send events to centralized vault
- Generate receipts remotely
- Query vault data
- Real-time governance monitoring

**Core Methods:**
```python
class VaultClient:
    def send_core_event(model_name, event_type, stage, metadata, ...) -> Dict
    def send_inference_event(model_name, input_data, prediction, ...) -> Dict
    def send_training_event(model_name, epoch, metrics, ...) -> Dict
    def send_governance_event(...) -> Dict
    def get_receipts(model_name, start_date, end_date) -> List[Dict]
```

---

## 2. Current Cloud Implementations

### 2.1 Azure Implementation (ciaf.azure)

**Current Modules:**
- `AzureKeyVaultSigner` - HSM-backed signing (RSA-PSS 2048)
- `AzureBlobReceiptStorage` - Blob storage with lifecycle management
- `AzureMLCIAFIntegration` - Azure ML workspace integration
- `AzureMonitorCIAFPublisher` - Metrics and compliance monitoring

**Current Methods:**
```python
# Signing
AzureKeyVaultSigner:
    - sign(data) -> bytes
    - verify(data, signature) -> bool
    - get_key_metadata() -> Dict

# Storage
AzureBlobReceiptStorage:
    - store_receipt(receipt_id, receipt_data, metadata) -> str
    - get_receipt(receipt_id) -> Dict
    - query_receipts(model_name, date_range) -> List[Dict]
    - bulk_store_receipts(receipts) -> int
    
# ML Integration
AzureMLCIAFIntegration:
    - register_dataset(name, uri, description, labels) -> Tuple[Dataset, Dict]
    - register_model(name, path, description, labels) -> Tuple[Model, Dict]
    - create_deployment(model_name, endpoint_name, instance_type) -> Dict
```

### 2.2 GCP Implementation (ciaf.gcp)

**Current Modules:**
- `CloudKMSSigner` - HSM-backed signing (RSA-PSS 2048)
- `CloudStorageReceiptStorage` - GCS with lifecycle management
- `VertexAICIAFIntegration` - Vertex AI integration
- `CloudMonitoringCIAFMetrics` - Cloud Monitoring integration

**Current Methods:**
```python
# Signing
CloudKMSSigner:
    - sign(data) -> bytes
    - verify(data, signature) -> bool
    - get_key_metadata() -> Dict

# Storage
CloudStorageReceiptStorage:
    - store_receipt(receipt_id, receipt_data, metadata) -> str
    - get_receipt(receipt_id) -> Dict
    - query_receipts(prefix, max_results) -> List[str]
    - bulk_upload(receipts) -> List[str]
    
# ML Integration
VertexAICIAFIntegration:
    - register_dataset(display_name, gcs_uri, description, labels) -> Tuple[Dataset, Dict]
    - register_model(artifact_uri, display_name, container_image, labels) -> Tuple[Model, Dict]
    - deploy_model(model_id, endpoint_name, machine_type) -> Dict
```

---

## 3. Required API for Vault Integration

### 3.1 Missing Shared Interface

**Problem:** Each cloud provider has slightly different APIs, making it hard for vault to store data uniformly.

**Solution:** Create abstract base class with standard interface

```python
# File: ciaf/vault/backends/cloud_backend.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

class CloudVaultBackend(ABC):
    """
    Abstract interface for cloud-based vault storage.
    
    Implementations: AzureVaultBackend, GCPVaultBackend
    """
    
    @abstractmethod
    def store_metadata(
        self,
        metadata_id: str,
        model_name: str,
        stage: str,
        event_type: str,
        metadata: Dict[str, Any],
        timestamp: datetime,
        signature: Optional[bytes] = None
    ) -> str:
        """Store metadata with cryptographic signature."""
        pass
    
    @abstractmethod
    def get_metadata(self, metadata_id: str) -> Dict[str, Any]:
        """Retrieve metadata by ID."""
        pass
    
    @abstractmethod
    def query_metadata(
        self,
        model_name: Optional[str] = None,
        stage: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Query metadata with filters."""
        pass
    
    @abstractmethod
    def store_receipt(
        self,
        receipt_id: str,
        receipt_data: Dict[str, Any],
        signature: bytes
    ) -> str:
        """Store cryptographic receipt."""
        pass
    
    @abstractmethod
    def get_receipt(self, receipt_id: str) -> Dict[str, Any]:
        """Retrieve receipt by ID."""
        pass
    
    @abstractmethod
    def store_audit_event(
        self,
        event_id: str,
        parent_id: str,
        action: str,
        user_id: str,
        timestamp: datetime,
        details: Dict[str, Any]
    ) -> str:
        """Store audit trail event."""
        pass
    
    @abstractmethod
    def get_audit_trail(self, parent_id: str) -> List[Dict[str, Any]]:
        """Get audit trail for a resource."""
        pass
    
    @abstractmethod
    def verify_integrity(self, metadata_id: str) -> bool:
        """Verify cryptographic integrity of stored metadata."""
        pass
```

### 3.2 Azure Vault Backend Implementation

**New Module:** `ciaf.azure.vault_backend`

```python
# File: ciaf/azure/vault_backend.py

from ciaf.vault.backends.cloud_backend import CloudVaultBackend
from .signers import AzureKeyVaultSigner
from .storage import AzureBlobReceiptStorage
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class AzureVaultBackend(CloudVaultBackend):
    """
    Azure implementation of CIAF Vault backend.
    
    Uses:
    - Azure Key Vault for cryptographic signing
    - Azure Blob Storage for receipt/metadata storage
    - Azure Table Storage for indexing/querying (optional)
    """
    
    def __init__(
        self,
        vault_url: str,
        key_name: str,
        storage_account_name: str,
        container_name: str = "ciaf-vault",
        use_managed_identity: bool = True
    ):
        self.signer = AzureKeyVaultSigner(
            vault_url=vault_url,
            key_name=key_name,
            use_managed_identity=use_managed_identity
        )
        
        self.storage = AzureBlobReceiptStorage(
            account_name=storage_account_name,
            container_name=container_name,
            use_managed_identity=use_managed_identity
        )
    
    def store_metadata(
        self,
        metadata_id: str,
        model_name: str,
        stage: str,
        event_type: str,
        metadata: Dict[str, Any],
        timestamp: datetime,
        signature: Optional[bytes] = None
    ) -> str:
        """Store metadata with HSM signature in Azure Blob Storage."""
        
        # Create metadata record
        record = {
            "id": metadata_id,
            "model_name": model_name,
            "stage": stage,
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "metadata": metadata
        }
        
        # Generate signature if not provided
        if signature is None:
            data_to_sign = json.dumps(record, sort_keys=True).encode()
            signature = self.signer.sign(data_to_sign)
        
        record["signature"] = signature.hex()
        
        # Store in blob storage under metadata/ prefix
        blob_name = f"metadata/{model_name}/{stage}/{metadata_id}.json"
        return self.storage.store_receipt(
            blob_name,
            record,
            metadata={"type": "metadata", "stage": stage}
        )
    
    def query_metadata(
        self,
        model_name: Optional[str] = None,
        stage: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Query metadata using blob prefix search."""
        
        # Build prefix for efficient querying
        prefix = "metadata/"
        if model_name:
            prefix += f"{model_name}/"
            if stage:
                prefix += f"{stage}/"
        
        # Use blob storage query with filters
        results = self.storage.query_receipts(
            prefix=prefix,
            date_range=(start_date, end_date) if start_date else None
        )
        
        # Filter by event_type if specified
        if event_type:
            results = [r for r in results if r.get("event_type") == event_type]
        
        return results
    
    # ... implement other methods
```

### 3.3 GCP Vault Backend Implementation

**New Module:** `ciaf.gcp.vault_backend`

```python
# File: ciaf/gcp/vault_backend.py

from ciaf.vault.backends.cloud_backend import CloudVaultBackend
from .signers import CloudKMSSigner
from .storage import CloudStorageReceiptStorage
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class GCPVaultBackend(CloudVaultBackend):
    """
    GCP implementation of CIAF Vault backend.
    
    Uses:
    - Cloud KMS for cryptographic signing
    - Cloud Storage for receipt/metadata storage
    - Firestore for indexing/querying (optional)
    """
    
    def __init__(
        self,
        project_id: str,
        location: str,
        keyring: str,
        key_name: str,
        bucket_name: str
    ):
        self.signer = CloudKMSSigner(
            project_id=project_id,
            location=location,
            keyring=keyring,
            key_name=key_name
        )
        
        self.storage = CloudStorageReceiptStorage(
            project_id=project_id,
            bucket_name=bucket_name,
            location=location
        )
    
    def store_metadata(
        self,
        metadata_id: str,
        model_name: str,
        stage: str,
        event_type: str,
        metadata: Dict[str, Any],
        timestamp: datetime,
        signature: Optional[bytes] = None
    ) -> str:
        """Store metadata with HSM signature in Cloud Storage."""
        
        # Create metadata record
        record = {
            "id": metadata_id,
            "model_name": model_name,
            "stage": stage,
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "metadata": metadata
        }
        
        # Generate signature if not provided
        if signature is None:
            data_to_sign = json.dumps(record, sort_keys=True).encode()
            signature = self.signer.sign(data_to_sign)
        
        record["signature"] = signature.hex()
        
        # Store in GCS under metadata/ prefix
        object_path = f"metadata/{model_name}/{stage}/{metadata_id}.json"
        return self.storage.store_receipt(
            object_path,
            record,
            metadata={"type": "metadata", "stage": stage}
        )
    
    def query_metadata(
        self,
        model_name: Optional[str] = None,
        stage: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Query metadata using GCS prefix search."""
        
        # Build prefix for efficient querying
        prefix = "metadata/"
        if model_name:
            prefix += f"{model_name}/"
            if stage:
                prefix += f"{stage}/"
        
        # Query Cloud Storage
        results = self.storage.query_receipts(prefix=prefix)
        
        # Load and filter results
        metadata_records = []
        for result_path in results:
            record = self.storage.get_receipt(result_path)
            
            # Apply filters
            if event_type and record.get("event_type") != event_type:
                continue
            if start_date or end_date:
                record_time = datetime.fromisoformat(record["timestamp"])
                if start_date and record_time < start_date:
                    continue
                if end_date and record_time > end_date:
                    continue
            
            metadata_records.append(record)
        
        return metadata_records
    
    # ... implement other methods
```

---

## 4. Integration with Existing Vault

### 4.1 Extend MetadataStorage to Support Cloud Backends

```python
# File: ciaf/vault/metadata_storage.py

class MetadataStorage:
    def __init__(
        self,
        storage_path: str = "ciaf_metadata",
        backend: str = "json",
        use_compression: bool = False,
        postgresql_config: Optional[Dict[str, Any]] = None,
        cloud_backend: Optional[CloudVaultBackend] = None,  # NEW
    ):
        # ... existing code ...
        
        # NEW: Support cloud backends
        if cloud_backend:
            self.backend = "cloud"
            self._cloud_backend = cloud_backend
    
    def save_metadata(self, model_name, stage, event_type, metadata, ...) -> str:
        # ... existing code ...
        
        # NEW: Route to cloud backend if configured
        if self.backend == "cloud":
            return self._cloud_backend.store_metadata(
                metadata_id=metadata_id,
                model_name=model_name,
                stage=stage,
                event_type=event_type,
                metadata=metadata,
                timestamp=datetime.now(timezone.utc)
            )
        
        # ... existing backend routing ...
```

### 4.2 Factory Functions for Easy Setup

```python
# File: ciaf/vault/__init__.py

def get_azure_vault_backend(
    vault_url: str,
    key_name: str,
    storage_account_name: str,
    use_managed_identity: bool = True
) -> MetadataStorage:
    """
    Create MetadataStorage with Azure cloud backend.
    
    Example:
        >>> from ciaf.vault import get_azure_vault_backend
        >>> storage = get_azure_vault_backend(
        ...     vault_url="https://my-vault.vault.azure.net/",
        ...     key_name="ciaf-key",
        ...     storage_account_name="ciafreceipts"
        ... )
        >>> storage.save_metadata(...)
    """
    from ciaf.azure.vault_backend import AzureVaultBackend
    
    backend = AzureVaultBackend(
        vault_url=vault_url,
        key_name=key_name,
        storage_account_name=storage_account_name,
        use_managed_identity=use_managed_identity
    )
    
    return MetadataStorage(cloud_backend=backend)


def get_gcp_vault_backend(
    project_id: str,
    location: str,
    keyring: str,
    key_name: str,
    bucket_name: str
) -> MetadataStorage:
    """
    Create MetadataStorage with GCP cloud backend.
    
    Example:
        >>> from ciaf.vault import get_gcp_vault_backend
        >>> storage = get_gcp_vault_backend(
        ...     project_id="my-project",
        ...     location="us-central1",
        ...     keyring="ciaf-keyring",
        ...     key_name="ciaf-key",
        ...     bucket_name="ciaf-vault"
        ... )
        >>> storage.save_metadata(...)
    """
    from ciaf.gcp.vault_backend import GCPVaultBackend
    
    backend = GCPVaultBackend(
        project_id=project_id,
        location=location,
        keyring=keyring,
        key_name=key_name,
        bucket_name=bucket_name
    )
    
    return MetadataStorage(cloud_backend=backend)
```

---

## 5. API Surface Summary

### 5.1 New Functions Needed in ciaf.azure

```python
# ciaf/azure/vault_backend.py
class AzureVaultBackend(CloudVaultBackend):
    def __init__(vault_url, key_name, storage_account_name, ...)
    def store_metadata(metadata_id, model_name, stage, ...) -> str
    def get_metadata(metadata_id) -> Dict
    def query_metadata(model_name, stage, event_type, ...) -> List[Dict]
    def store_receipt(receipt_id, receipt_data, signature) -> str
    def get_receipt(receipt_id) -> Dict
    def store_audit_event(event_id, parent_id, action, ...) -> str
    def get_audit_trail(parent_id) -> List[Dict]
    def verify_integrity(metadata_id) -> bool
```

### 5.2 New Functions Needed in ciaf.gcp

```python
# ciaf/gcp/vault_backend.py
class GCPVaultBackend(CloudVaultBackend):
    def __init__(project_id, location, keyring, key_name, bucket_name)
    def store_metadata(metadata_id, model_name, stage, ...) -> str
    def get_metadata(metadata_id) -> Dict
    def query_metadata(model_name, stage, event_type, ...) -> List[Dict]
    def store_receipt(receipt_id, receipt_data, signature) -> str
    def get_receipt(receipt_id) -> Dict
    def store_audit_event(event_id, parent_id, action, ...) -> str
    def get_audit_trail(parent_id) -> List[Dict]
    def verify_integrity(metadata_id) -> bool
```

### 5.3 Enhanced Functions in Existing Modules

#### Azure Storage Enhancements
```python
# ciaf/azure/storage.py
class AzureBlobReceiptStorage:
    # NEW: Add query by prefix and date range
    def query_receipts(
        prefix: str,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        metadata_filters: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]
    
    # NEW: Bulk operations for performance
    def bulk_store_receipts(receipts: List[Dict]) -> List[str]
    
    # NEW: Verify blob integrity with signatures
    def verify_receipt_integrity(receipt_id: str) -> bool
```

#### GCP Storage Enhancements
```python
# ciaf/gcp/storage.py
class CloudStorageReceiptStorage:
    # ENHANCE: Improve query_receipts with filtering
    def query_receipts(
        prefix: str,
        max_results: int = 100,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None
    ) -> List[str]
    
    # NEW: Batch retrieve for efficiency
    def batch_get_receipts(receipt_ids: List[str]) -> List[Dict[str, Any]]
    
    # NEW: Verify object integrity
    def verify_receipt_integrity(receipt_id: str) -> bool
```

---

## 6. Usage Examples

### 6.1 Using Azure Vault Backend

```python
from ciaf.vault import get_azure_vault_backend

# Initialize vault with Azure backend
vault = get_azure_vault_backend(
    vault_url="https://ciaf-vault.vault.azure.net/",
    key_name="ciaf-signing-key",
    storage_account_name="ciafvault"
)

# Save metadata (stored in Azure Blob with Key Vault signature)
metadata_id = vault.save_metadata(
    model_name="credit-model-v1",
    stage="training",
    event_type="epoch_complete",
    metadata={"epoch": 10, "loss": 0.05, "accuracy": 0.95}
)

# Query metadata
results = vault.query_metadata(
    model_name="credit-model-v1",
    stage="training"
)

# Get pipeline trace
trace = vault.get_pipeline_trace(
    model_name="credit-model-v1",
    model_version="1.0.0"
)
```

### 6.2 Using GCP Vault Backend

```python
from ciaf.vault import get_gcp_vault_backend

# Initialize vault with GCP backend
vault = get_gcp_vault_backend(
    project_id="ciaftest",
    location="us-central1",
    keyring="ciaf-keyring",
    key_name="ciaf-signing-key",
    bucket_name="ciaf-vault"
)

# Save metadata (stored in GCS with Cloud KMS signature)
metadata_id = vault.save_metadata(
    model_name="fraud-model-v2",
    stage="inference",
    event_type="prediction_made",
    metadata={"input_hash": "abc123", "prediction": "fraud", "confidence": 0.89}
)

# Query metadata
results = vault.query_metadata(
    model_name="fraud-model-v2",
    start_date=datetime(2026, 1, 1)
)

# Verify integrity
is_valid = vault.verify_metadata_integrity(metadata_id)
```

### 6.3 Hybrid Setup (Local + Cloud)

```python
from ciaf.vault import MetadataStorage, get_gcp_vault_backend

# Local development with SQLite
local_vault = MetadataStorage(
    storage_path="./dev_vault",
    backend="sqlite"
)

# Production with GCP
prod_vault = get_gcp_vault_backend(
    project_id="ciaf-prod",
    location="us-central1",
    keyring="ciaf-prod-keyring",
    key_name="ciaf-prod-key",
    bucket_name="ciaf-prod-vault"
)

# Same API works for both!
for vault in [local_vault, prod_vault]:
    vault.save_metadata(
        model_name="model-v1",
        stage="testing",
        event_type="test_complete",
        metadata={"accuracy": 0.92}
    )
```

---

## 7. Implementation Roadmap

### Phase 1: Core Interface (Week 1)
- [ ] Create `CloudVaultBackend` abstract base class
- [ ] Add `cloud_backend` parameter to `MetadataStorage`
- [ ] Write unit tests for interface

### Phase 2: Azure Implementation (Week 2)
- [ ] Implement `AzureVaultBackend`
- [ ] Enhance `AzureBlobReceiptStorage` with query capabilities
- [ ] Add Azure Table Storage indexing (optional)
- [ ] Integration tests

### Phase 3: GCP Implementation (Week 3)
- [ ] Implement `GCPVaultBackend`
- [ ] Enhance `CloudStorageReceiptStorage` with filtering
- [ ] Add Firestore indexing (optional)
- [ ] Integration tests

### Phase 4: Factory Functions & Documentation (Week 4)
- [ ] Add `get_azure_vault_backend()` factory
- [ ] Add `get_gcp_vault_backend()` factory
- [ ] Update documentation
- [ ] Create migration guide for existing users

### Phase 5: Advanced Features (Future)
- [ ] Bi-directional sync (local ↔ cloud)
- [ ] Multi-cloud replication
- [ ] Real-time streaming to VaultClient
- [ ] GraphQL API for complex queries

---

## 8. Key Design Decisions

### 8.1 Why Abstract Interface?
- **Portability:** Same code works on Azure, GCP, local SQLite
- **Testing:** Easy to mock for unit tests
- **Migration:** Switch clouds without code changes

### 8.2 Why Extend Existing Storage Classes?
- **Backward Compatibility:** Existing code continues to work
- **Gradual Migration:** Users can opt-in to cloud backends
- **Leverage Existing Features:** Lifecycle management, encryption, etc.

### 8.3 Why Not Build Custom Database?
- **Cost:** Leverage managed services (Blob, GCS) for cheaper storage
- **Scalability:** Cloud storage auto-scales
- **Reliability:** Built-in redundancy, geo-replication
- **Compliance:** SOC2, ISO27001 certifications inherited

---

## 9. Security Considerations

### 9.1 Authentication
- **Azure:** Managed Identity (preferred) or Service Principal
- **GCP:** Application Default Credentials or Service Account

### 9.2 Encryption
- **At Rest:** Automatic with Azure Storage/Cloud Storage
- **In Transit:** HTTPS/TLS 1.2+
- **Signatures:** HSM-backed (FIPS 140-2 Level 3)

### 9.3 Access Control
- **Azure:** RBAC on Storage Account + Key Vault policies
- **GCP:** IAM roles on bucket + KMS key permissions

---

## 10. Next Steps

1. **Review this document** with stakeholders
2. **Prioritize features** (MVP vs nice-to-have)
3. **Create GitHub issues** for each implementation phase
4. **Set up test environments** (Azure + GCP sandbox projects)
5. **Begin Phase 1 implementation**

---

**Questions or Feedback?**  
Contact: Denzil James Greenwood  
Document version: 1.0.0
