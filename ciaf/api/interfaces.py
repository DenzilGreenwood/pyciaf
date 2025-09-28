"""
CIAF API Protocol Interfaces
==========================

Protocol-based interfaces for the CIAF API system, following the same architectural
patterns as other CIAF modules. These interfaces provide clean separation of concerns
and enable flexible, extensible API implementations.

Created: 2025-09-28
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from abc import abstractmethod
from datetime import datetime


@runtime_checkable
class DatasetAPIHandler(Protocol):
    """Protocol for dataset-related API operations."""
    
    @abstractmethod
    def create_dataset(self, dataset_id: str, metadata: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create a new dataset with metadata."""
        ...
    
    @abstractmethod
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve dataset information by ID."""
        ...
    
    @abstractmethod
    def list_datasets(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List all datasets with optional filtering."""
        ...
    
    @abstractmethod
    def update_dataset(self, dataset_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update dataset metadata."""
        ...
    
    @abstractmethod
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset (if policy allows)."""
        ...


@runtime_checkable
class ModelAPIHandler(Protocol):
    """Protocol for model-related API operations."""
    
    @abstractmethod
    def create_model(self, model_name: str, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create a new model with configuration."""
        ...
    
    @abstractmethod
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve model information by name."""
        ...
    
    @abstractmethod
    def list_models(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List all models with optional filtering."""
        ...
    
    @abstractmethod
    def deploy_model(self, model_name: str, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a model for inference."""
        ...
    
    @abstractmethod
    def update_model(self, model_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update model configuration."""
        ...


@runtime_checkable
class TrainingAPIHandler(Protocol):
    """Protocol for training-related API operations."""
    
    @abstractmethod
    def start_training(self, model_name: str, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a training session."""
        ...
    
    @abstractmethod
    def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """Get status of a training session."""
        ...
    
    @abstractmethod
    def list_training_sessions(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List training sessions with optional filtering."""
        ...
    
    @abstractmethod
    def stop_training(self, training_id: str) -> bool:
        """Stop a training session."""
        ...
    
    @abstractmethod
    def get_training_metrics(self, training_id: str) -> Dict[str, Any]:
        """Get metrics for a training session."""
        ...


@runtime_checkable
class InferenceAPIHandler(Protocol):
    """Protocol for inference-related API operations."""
    
    @abstractmethod
    def perform_inference(self, model_name: str, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Perform inference with a model."""
        ...
    
    @abstractmethod
    def batch_inference(self, model_name: str, batch_data: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """Perform batch inference."""
        ...
    
    @abstractmethod
    def get_inference_history(self, model_name: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get inference history for a model."""
        ...
    
    @abstractmethod
    def validate_input(self, model_name: str, input_data: Any) -> bool:
        """Validate input data for a model."""
        ...


@runtime_checkable
class AuditAPIHandler(Protocol):
    """Protocol for audit-related API operations."""
    
    @abstractmethod
    def get_audit_trail(self, entity_id: str, entity_type: str) -> Dict[str, Any]:
        """Get complete audit trail for an entity."""
        ...
    
    @abstractmethod
    def verify_integrity(self, entity_id: str, entity_type: str) -> Dict[str, Any]:
        """Verify integrity of an entity's audit trail."""
        ...
    
    @abstractmethod
    def generate_compliance_report(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compliance report based on filters."""
        ...
    
    @abstractmethod
    def get_provenance_chain(self, entity_id: str) -> Dict[str, Any]:
        """Get provenance chain for an entity."""
        ...


@runtime_checkable
class ComplianceAPIHandler(Protocol):
    """Protocol for compliance-related API operations."""
    
    @abstractmethod
    def validate_gdpr_compliance(self, entity_id: str) -> Dict[str, Any]:
        """Validate GDPR compliance for an entity."""
        ...
    
    @abstractmethod
    def validate_ai_act_compliance(self, model_name: str) -> Dict[str, Any]:
        """Validate EU AI Act compliance for a model."""
        ...
    
    @abstractmethod
    def get_compliance_status(self, entity_id: str, framework: str) -> Dict[str, Any]:
        """Get compliance status for a specific framework."""
        ...
    
    @abstractmethod
    def generate_audit_log(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate audit log based on filters."""
        ...


@runtime_checkable
class SecurityAPIHandler(Protocol):
    """Protocol for security-related API operations."""
    
    @abstractmethod
    def authenticate_user(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate a user."""
        ...
    
    @abstractmethod
    def authorize_action(self, user_id: str, action: str, resource: str) -> bool:
        """Authorize a user action on a resource."""
        ...
    
    @abstractmethod
    def get_user_permissions(self, user_id: str) -> Dict[str, Any]:
        """Get user permissions."""
        ...
    
    @abstractmethod
    def audit_security_event(self, event: Dict[str, Any]) -> None:
        """Audit a security event."""
        ...


@runtime_checkable
class MetricsAPIHandler(Protocol):
    """Protocol for metrics-related API operations."""
    
    @abstractmethod
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics."""
        ...
    
    @abstractmethod
    def get_model_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get metrics for a specific model."""
        ...
    
    @abstractmethod
    def get_dataset_metrics(self, dataset_id: str) -> Dict[str, Any]:
        """Get metrics for a specific dataset."""
        ...
    
    @abstractmethod
    def get_performance_metrics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get performance metrics with optional filters."""
        ...


@runtime_checkable
class APIResponseHandler(Protocol):
    """Protocol for handling API responses."""
    
    @abstractmethod
    def format_success_response(self, data: Any, message: str = None) -> Dict[str, Any]:
        """Format a successful API response."""
        ...
    
    @abstractmethod
    def format_error_response(self, error: Exception, status_code: int = 500) -> Dict[str, Any]:
        """Format an error API response."""
        ...
    
    @abstractmethod
    def validate_request(self, request_data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate request data against schema."""
        ...


@runtime_checkable
class APIMiddleware(Protocol):
    """Protocol for API middleware components."""
    
    @abstractmethod
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request."""
        ...
    
    @abstractmethod
    def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing response."""
        ...
    
    @abstractmethod
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors in requests."""
        ...


@runtime_checkable
class CIAFAPIFramework(Protocol):
    """Main protocol for the CIAF API framework."""
    
    # Handler protocols
    dataset_handler: DatasetAPIHandler
    model_handler: ModelAPIHandler
    training_handler: TrainingAPIHandler
    inference_handler: InferenceAPIHandler
    audit_handler: AuditAPIHandler
    compliance_handler: ComplianceAPIHandler
    security_handler: SecurityAPIHandler
    metrics_handler: MetricsAPIHandler
    
    # Response and middleware
    response_handler: APIResponseHandler
    middleware: List[APIMiddleware]
    
    @abstractmethod
    def initialize_handlers(self, policy: Any) -> None:
        """Initialize all API handlers with policy configuration."""
        ...
    
    @abstractmethod
    def register_middleware(self, middleware: APIMiddleware) -> None:
        """Register middleware component."""
        ...
    
    @abstractmethod
    def process_api_request(self, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an API request through the full pipeline."""
        ...
    
    @abstractmethod
    def get_api_health(self) -> Dict[str, Any]:
        """Get API health status."""
        ...


# Type aliases for common API patterns
APIRequest = Dict[str, Any]
APIResponse = Dict[str, Any]
APIError = Dict[str, Any]
APIFilters = Dict[str, Any]

# Common response structure
class APIStatus:
    """Standard API status codes and messages."""
    SUCCESS = 200
    CREATED = 201
    ACCEPTED = 202
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    INTERNAL_ERROR = 500
    SERVICE_UNAVAILABLE = 503


__all__ = [
    # Core protocols
    "DatasetAPIHandler",
    "ModelAPIHandler", 
    "TrainingAPIHandler",
    "InferenceAPIHandler",
    "AuditAPIHandler",
    "ComplianceAPIHandler",
    "SecurityAPIHandler",
    "MetricsAPIHandler",
    
    # Framework protocols
    "APIResponseHandler",
    "APIMiddleware", 
    "CIAFAPIFramework",
    
    # Type aliases
    "APIRequest",
    "APIResponse",
    "APIError",
    "APIFilters",
    
    # Utilities
    "APIStatus",
]