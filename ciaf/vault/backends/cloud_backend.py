"""
Abstract base class for cloud-based vault storage backends.

Defines the standard interface that Azure, GCP, and other cloud
implementations must follow for unified vault operations.

Created: April 4, 2026
Author: Denzil James Greenwood
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class CloudVaultBackend(ABC):
    """
    Abstract interface for cloud-based vault storage.
    
    This interface defines the contract that all cloud vault implementations
    must follow, enabling portable code across Azure, GCP, AWS, etc.
    
    Implementations should provide:
    - Cryptographic signing (HSM-backed)
    - Persistent storage (cloud object storage)
    - Query capabilities (indexed search)
    - Integrity verification (signature validation)
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
        signature: Optional[bytes] = None,
        model_version: Optional[str] = None,
    ) -> str:
        """
        Store metadata with cryptographic signature.
        
        Args:
            metadata_id: Unique identifier for this metadata record
            model_name: Name of the ML model
            stage: Pipeline stage (e.g., 'training', 'inference', 'deployment')
            event_type: Type of event (e.g., 'epoch_complete', 'prediction_made')
            metadata: Metadata dictionary containing event-specific data
            timestamp: Timestamp of the event
            signature: Optional pre-computed signature (if None, will be generated)
            model_version: Version of the model
            
        Returns:
            Storage URI or identifier for the stored metadata
        """
        pass
    
    @abstractmethod
    def get_metadata(self, metadata_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata by ID.
        
        Args:
            metadata_id: Unique identifier for the metadata record
            
        Returns:
            Metadata dictionary including signature and timestamp
            
        Raises:
            KeyError: If metadata_id not found
        """
        pass
    
    @abstractmethod
    def query_metadata(
        self,
        model_name: Optional[str] = None,
        stage: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query metadata with filters.
        
        Args:
            model_name: Filter by model name
            stage: Filter by pipeline stage
            event_type: Filter by event type
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)
            limit: Maximum number of results to return
            
        Returns:
            List of metadata records matching the filters
        """
        pass
    
    @abstractmethod
    def store_receipt(
        self,
        receipt_id: str,
        receipt_data: Dict[str, Any],
        signature: bytes,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Store cryptographic receipt.
        
        Args:
            receipt_id: Unique identifier for this receipt
            receipt_data: Receipt data dictionary
            signature: Cryptographic signature of the receipt
            model_name: Optional model name for organization
            
        Returns:
            Storage URI or identifier for the stored receipt
        """
        pass
    
    @abstractmethod
    def get_receipt(self, receipt_id: str) -> Dict[str, Any]:
        """
        Retrieve receipt by ID.
        
        Args:
            receipt_id: Unique identifier for the receipt
            
        Returns:
            Receipt dictionary including signature
            
        Raises:
            KeyError: If receipt_id not found
        """
        pass
    
    @abstractmethod
    def store_audit_event(
        self,
        event_id: str,
        parent_id: str,
        action: str,
        user_id: str,
        timestamp: datetime,
        details: Dict[str, Any],
    ) -> str:
        """
        Store audit trail event.
        
        Args:
            event_id: Unique identifier for this audit event
            parent_id: ID of the parent resource being audited
            action: Action performed (e.g., 'create', 'update', 'delete')
            user_id: User who performed the action
            timestamp: Timestamp of the action
            details: Additional details about the action
            
        Returns:
            Storage URI or identifier for the stored audit event
        """
        pass
    
    @abstractmethod
    def get_audit_trail(self, parent_id: str) -> List[Dict[str, Any]]:
        """
        Get audit trail for a resource.
        
        Args:
            parent_id: ID of the parent resource
            
        Returns:
            List of audit events for the resource, ordered by timestamp
        """
        pass
    
    @abstractmethod
    def verify_integrity(self, metadata_id: str) -> bool:
        """
        Verify cryptographic integrity of stored metadata.
        
        Args:
            metadata_id: ID of the metadata to verify
            
        Returns:
            True if signature is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary containing statistics like:
            - total_metadata_records
            - total_receipts
            - total_audit_events
            - storage_size_bytes
            - oldest_record_date
            - newest_record_date
        """
        pass
