"""
CIAF Vault - Centralized Storage Backend

The vault module provides centralized storage for:
- Metadata (training, inference, compliance)
- Receipts and audit trails
- Database backends (SQLite, PostgreSQL)
- Compressed and optimized storage

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .metadata_storage import (
    MetadataStorage,
    get_metadata_storage,
    save_pipeline_metadata,
    get_pipeline_trace,
)

from .metadata_config import (
    MetadataConfig,
    get_metadata_config,
    load_config_from_file,
    create_config_template,
    create_deferred_lcm_config,
    create_high_performance_config,
    create_compliance_first_config,
    create_balanced_config,
)

from .metadata_integration import (
    MetadataCapture,
    ModelMetadataManager,
    ComplianceTracker,
    capture_metadata,
    create_model_manager,
    create_compliance_tracker,
    quick_log,
)

from .metadata_storage_compressed import CompressedMetadataStorage
from .metadata_storage_optimized import HighPerformanceMetadataStorage

# Cloud backend interfaces
from .backends.cloud_backend import CloudVaultBackend


# Factory functions for cloud backends
def get_gcp_vault_backend(
    project_id: str,
    location: str,
    keyring: str,
    key_name: str,
    bucket_name: str,
    use_firestore: bool = False,
) -> "MetadataStorage":
    """
    Create MetadataStorage with GCP cloud backend.
    
    Args:
        project_id: GCP project ID
        location: GCP location (e.g., 'us-central1')
        keyring: Cloud KMS keyring name
        key_name: Cloud KMS key name
        bucket_name: Cloud Storage bucket name
        use_firestore: Enable Firestore indexing
        
    Returns:
        MetadataStorage instance with GCP backend
        
    Example:
        >>> from ciaf.vault import get_gcp_vault_backend
        >>> vault = get_gcp_vault_backend(
        ...     project_id="my-project",
        ...     location="us-central1",
        ...     keyring="ciaf-keyring",
        ...     key_name="ciaf-key",
        ...     bucket_name="ciaf-vault"
        ... )
        >>> vault.save_metadata(...)
    """
    try:
        from ciaf.gcp.vault_backend import GCPVaultBackend
    except ImportError:
        raise ImportError(
            "ciaf-gcp not installed. Install with: pip install ciaf-gcp"
        )
    
    backend = GCPVaultBackend(
        project_id=project_id,
        location=location,
        keyring=keyring,
        key_name=key_name,
        bucket_name=bucket_name,
        use_firestore=use_firestore,
    )
    
    # Return MetadataStorage with cloud backend
    # Note: This requires MetadataStorage to support cloud_backend parameter
    # For now, return the backend directly (it implements the same interface)
    return backend


def get_azure_vault_backend(
    vault_url: str,
    key_name: str,
    storage_account_name: str,
    use_managed_identity: bool = True,
) -> "MetadataStorage":
    """
    Create MetadataStorage with Azure cloud backend.
    
    Args:
        vault_url: Azure Key Vault URL
        key_name: Key Vault key name
        storage_account_name: Storage account name
        use_managed_identity: Use Managed Identity for auth
        
    Returns:
        MetadataStorage instance with Azure backend
        
    Example:
        >>> from ciaf.vault import get_azure_vault_backend
        >>> vault = get_azure_vault_backend(
        ...     vault_url="https://my-vault.vault.azure.net/",
        ...     key_name="ciaf-key",
        ...     storage_account_name="ciafvault"
        ... )
        >>> vault.save_metadata(...)
    """
    try:
        from ciaf.azure.vault_backend import AzureVaultBackend
    except ImportError:
        raise ImportError(
            "ciaf-azure not installed. Install with: pip install ciaf-azure"
        )
    
    backend = AzureVaultBackend(
        vault_url=vault_url,
        key_name=key_name,
        storage_account_name=storage_account_name,
        use_managed_identity=use_managed_identity,
    )
    
    return backend

__all__ = [
    # Core storage
    "MetadataStorage",
    "get_metadata_storage",
    "save_pipeline_metadata",
    "get_pipeline_trace",
    # Configuration
    "MetadataConfig",
    "get_metadata_config",
    "load_config_from_file",
    "create_config_template",
    "create_deferred_lcm_config",
    "create_high_performance_config",
    "create_compliance_first_config",
    "create_balanced_config",
    # Integration
    "MetadataCapture",
    "ModelMetadataManager",
    "ComplianceTracker",
    "capture_metadata",
    "create_model_manager",
    "create_compliance_tracker",
    "quick_log",
    # Specialized storage
    "CompressedMetadataStorage",
    "HighPerformanceMetadataStorage",
    # Cloud backends
    "CloudVaultBackend",
    "get_gcp_vault_backend",
    "get_azure_vault_backend",
]
