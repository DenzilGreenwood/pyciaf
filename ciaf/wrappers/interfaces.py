"""
Protocol-based interfaces for CIAF wrapper system.

This module defines the core protocols for model wrapper functionality,
following the same architectural patterns as core, LCM, compliance, and 
explainability modules. These protocols enable clean dependency injection
and separation of concerns.

Created: 2025-09-27
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, Dict, List, Optional, Tuple, Union
from datetime import datetime


@runtime_checkable
class ModelWrapper(Protocol):
    """Protocol for model wrapper implementations."""
    
    def train(self, dataset_id: str, training_data: List[Dict[str, Any]], 
              master_password: str, training_params: Optional[Dict[str, Any]] = None,
              model_version: str = "1.0.0", fit_model: bool = True) -> Any:
        """Train the wrapped model and create CIAF training metadata."""
        ...
    
    def predict(self, query: Union[str, List, Any], model_version: Optional[str] = None,
                use_model: bool = True) -> Tuple[Any, Any]:
        """Run inference and generate CIAF receipt."""
        ...
    
    def verify(self, receipt: Any) -> Dict[str, Any]:
        """Verify inference receipt integrity and provenance."""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        ...


@runtime_checkable  
class ModelAdapter(Protocol):
    """Protocol for adapting different model types to CIAF wrapper."""
    
    def detect_model_type(self, model: Any) -> str:
        """Detect the type/framework of the provided model."""
        ...
    
    def validate_model_compatibility(self, model: Any) -> Dict[str, Any]:
        """Validate if model is compatible with CIAF wrapper."""
        ...
    
    def extract_model_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract metadata from the model."""
        ...
    
    def prepare_training_data(self, model: Any, training_data: List[Dict[str, Any]]) -> Tuple[Any, Any]:
        """Prepare training data for the specific model type."""
        ...
    
    def handle_model_prediction(self, model: Any, query: Any, preprocessed_query: Any = None) -> Any:
        """Handle prediction for the specific model type."""
        ...


@runtime_checkable
class ModelMetadataProvider(Protocol):
    """Protocol for providing model metadata and information."""
    
    def get_model_signature(self, model: Any) -> Dict[str, Any]:
        """Get model signature information."""
        ...
    
    def get_model_capabilities(self, model: Any) -> List[str]:
        """Get list of model capabilities."""
        ...
    
    def get_model_requirements(self, model: Any) -> Dict[str, Any]:
        """Get model requirements (dependencies, resources, etc.)."""
        ...
    
    def get_training_metadata(self, model: Any, training_data: Any = None) -> Dict[str, Any]:
        """Get metadata about model training."""
        ...
    
    def get_inference_metadata(self, model: Any, query: Any = None) -> Dict[str, Any]:
        """Get metadata about model inference capabilities."""
        ...


@runtime_checkable
class ModelValidator(Protocol):
    """Protocol for validating model compatibility and state."""
    
    def validate_model_structure(self, model: Any) -> Dict[str, Any]:
        """Validate model structure and integrity."""
        ...
    
    def validate_model_readiness(self, model: Any) -> Dict[str, Any]:
        """Validate if model is ready for training/inference."""
        ...
    
    def validate_training_data_compatibility(self, model: Any, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate training data compatibility with model."""
        ...
    
    def validate_inference_input_compatibility(self, model: Any, query: Any) -> Dict[str, Any]:
        """Validate inference input compatibility with model."""
        ...
    
    def validate_ciaf_compliance(self, model: Any, wrapper_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate CIAF compliance for the wrapped model."""
        ...


@runtime_checkable
class ModelTrainingHandler(Protocol):
    """Protocol for handling model training operations."""
    
    def prepare_training_environment(self, model: Any, training_params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare environment for model training."""
        ...
    
    def execute_training(self, model: Any, X: Any, y: Any, training_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual model training."""
        ...
    
    def validate_training_results(self, model: Any, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training results."""
        ...
    
    def create_training_summary(self, model: Any, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of training operation."""
        ...
    
    def handle_training_errors(self, model: Any, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training errors gracefully."""
        ...


@runtime_checkable
class ModelInferenceHandler(Protocol):
    """Protocol for handling model inference operations."""
    
    def prepare_inference_input(self, model: Any, query: Any, preprocessing_enabled: bool = True) -> Any:
        """Prepare input for model inference."""
        ...
    
    def execute_inference(self, model: Any, prepared_input: Any, inference_params: Dict[str, Any] = None) -> Any:
        """Execute model inference."""
        ...
    
    def postprocess_inference_output(self, model: Any, raw_output: Any, query: Any) -> Any:
        """Postprocess inference output."""
        ...
    
    def create_inference_summary(self, model: Any, query: Any, output: Any, 
                                metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create summary of inference operation."""
        ...
    
    def handle_inference_errors(self, model: Any, query: Any, error: Exception, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle inference errors gracefully."""
        ...


@runtime_checkable
class LCMMetadataHandler(Protocol):
    """Protocol for handling LCM metadata operations in wrappers."""
    
    def extract_lcm_metadata_trail(self, wrapper_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract complete LCM metadata trail."""
        ...
    
    def serialize_lcm_metadata(self, metadata: Dict[str, Any], format: str = "json") -> Any:
        """Serialize LCM metadata for storage."""
        ...
    
    def restore_lcm_metadata(self, serialized_metadata: Any) -> Dict[str, Any]:
        """Restore LCM metadata from serialized form."""
        ...
    
    def verify_lcm_integrity(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Verify integrity of LCM metadata."""
        ...
    
    def create_lcm_audit_trail(self, wrapper_state: Dict[str, Any], 
                              operation: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create LCM audit trail entry."""
        ...


@runtime_checkable
class ModelEnhancementProvider(Protocol):
    """Protocol for providing model enhancements (explainability, uncertainty, etc.)."""
    
    def get_available_enhancements(self, model: Any) -> List[str]:
        """Get list of available enhancements for the model."""
        ...
    
    def configure_explainability(self, model: Any, config: Dict[str, Any]) -> Any:
        """Configure explainability for the model."""
        ...
    
    def configure_uncertainty_quantification(self, model: Any, config: Dict[str, Any]) -> Any:
        """Configure uncertainty quantification for the model."""
        ...
    
    def configure_preprocessing(self, model: Any, config: Dict[str, Any]) -> Any:
        """Configure preprocessing for the model."""
        ...
    
    def configure_metadata_tags(self, model: Any, config: Dict[str, Any]) -> Any:
        """Configure metadata tags for the model."""
        ...
    
    def apply_enhancements(self, model: Any, enhancement_configs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configured enhancements to the model."""
        ...


@runtime_checkable
class ComplianceIntegrator(Protocol):
    """Protocol for integrating wrapper with compliance systems."""
    
    def configure_compliance_mode(self, mode: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Configure compliance mode for the wrapper."""
        ...
    
    def validate_regulatory_compliance(self, model: Any, framework: str, 
                                     audit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance with regulatory framework."""
        ...
    
    def create_compliance_report(self, model: Any, wrapper_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create compliance report for the wrapped model."""
        ...
    
    def handle_compliance_alerts(self, alert_type: str, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compliance alerts."""
        ...
    
    def audit_wrapper_operations(self, wrapper_state: Dict[str, Any], 
                                operation: str, timestamp: datetime) -> Dict[str, Any]:
        """Audit wrapper operations for compliance."""
        ...


@runtime_checkable
class PerformanceOptimizer(Protocol):
    """Protocol for optimizing wrapper performance."""
    
    def analyze_performance_characteristics(self, model: Any, 
                                          operations: List[str] = None) -> Dict[str, Any]:
        """Analyze performance characteristics of the model."""
        ...
    
    def optimize_training_performance(self, model: Any, training_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize training performance."""
        ...
    
    def optimize_inference_performance(self, model: Any, inference_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize inference performance."""
        ...
    
    def configure_caching(self, cache_config: Dict[str, Any]) -> Any:
        """Configure caching for improved performance."""
        ...
    
    def monitor_resource_usage(self, operation: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor resource usage during operations."""
        ...