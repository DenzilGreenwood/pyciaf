"""
CIAF Metadata Integration

This module provides easy integration of metadata storage with CIAF models
and provides decorators and utilities for automatic metadata capture.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import functools
import inspect
import json
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .metadata_config import get_metadata_config
from .metadata_storage import MetadataStorage, get_metadata_storage


class MetadataCapture:
    """Context manager and decorator for capturing metadata."""

    def __init__(
        self,
        model_name: str,
        stage: str,
        event_type: str,
        details: Optional[str] = None,
        capture_params: bool = True,
        capture_result: bool = True,
        capture_performance: bool = True,
    ):
        """
        Initialize metadata capture.

        Args:
            model_name: Name of the model
            stage: Pipeline stage
            event_type: Type of event
            details: Additional details
            capture_params: Whether to capture function parameters
            capture_result: Whether to capture function result
            capture_performance: Whether to capture performance metrics
        """
        self.model_name = model_name
        self.stage = stage
        self.event_type = event_type
        self.details = details
        self.capture_params = capture_params
        self.capture_result = capture_result
        self.capture_performance = capture_performance

        self.metadata = {}
        self.start_time = None
        self.storage = get_metadata_storage()

    def __enter__(self):
        """Enter context manager."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and save metadata."""
        if self.capture_performance:
            self.metadata["execution_time"] = time.time() - self.start_time

        if exc_type is not None:
            self.metadata["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_val),
                "traceback": traceback.format_exception(exc_type, exc_val, exc_tb),
            }
            self.event_type = f"{self.event_type}_error"

        self.save()

    def add_metadata(self, key: str, value: Any):
        """Add metadata field."""
        self.metadata[key] = value

    def update_metadata(self, metadata_dict: Dict[str, Any]):
        """Update metadata with dictionary."""
        self.metadata.update(metadata_dict)

    def save(self) -> str:
        """Save metadata and return ID."""
        return self.storage.save_metadata(
            model_name=self.model_name,
            stage=self.stage,
            event_type=self.event_type,
            metadata=self.metadata,
            details=self.details,
        )


def capture_metadata(
    model_name: str,
    stage: str,
    event_type: Optional[str] = None,
    details: Optional[str] = None,
    capture_params: bool = True,
    capture_result: bool = False,
    capture_performance: bool = True,
):
    """
    Decorator to automatically capture metadata for function calls.

    Args:
        model_name: Name of the model
        stage: Pipeline stage
        event_type: Type of event (defaults to function name)
        details: Additional details
        capture_params: Whether to capture function parameters
        capture_result: Whether to capture function result
        capture_performance: Whether to capture performance metrics
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine event type
            func_event_type = event_type or func.__name__

            # Create metadata capture context
            with MetadataCapture(
                model_name=model_name,
                stage=stage,
                event_type=func_event_type,
                details=details,
                capture_params=capture_params,
                capture_result=capture_result,
                capture_performance=capture_performance,
            ) as capture:

                # Capture function parameters
                if capture_params:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    # Filter sensitive parameters
                    filtered_params = {}
                    for name, value in bound_args.arguments.items():
                        if name.lower() in ["password", "token", "key", "secret"]:
                            filtered_params[name] = "[REDACTED]"
                        elif hasattr(value, "__len__") and not isinstance(value, str):
                            try:
                                filtered_params[name] = (
                                    f"<{type(value).__name__}[{len(value)}]>"
                                )
                            except:
                                filtered_params[name] = f"<{type(value).__name__}>"
                        else:
                            try:
                                # Try to serialize to check if it's JSON-serializable
                                json.dumps(value, default=str)
                                filtered_params[name] = value
                            except:
                                filtered_params[name] = str(value)

                    capture.add_metadata("function_parameters", filtered_params)

                # Execute function
                result = func(*args, **kwargs)

                # Capture result if requested
                if capture_result:
                    try:
                        if hasattr(result, "__len__") and not isinstance(result, str):
                            capture.add_metadata(
                                "result_summary",
                                f"<{type(result).__name__}[{len(result)}]>",
                            )
                        else:
                            # Try to serialize result
                            json.dumps(result, default=str)
                            capture.add_metadata("result", result)
                    except:
                        capture.add_metadata(
                            "result_summary", f"<{type(result).__name__}>"
                        )

                # Add function info
                capture.add_metadata("function_name", func.__name__)
                capture.add_metadata("function_module", func.__module__)

                return result

        return wrapper

    return decorator


class ModelMetadataManager:
    """Manages metadata for a specific model throughout its lifecycle."""

    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        """
        Initialize model metadata manager.

        Args:
            model_name: Name of the model
            model_version: Version of the model
        """
        self.model_name = model_name
        self.model_version = model_version
        self.storage = get_metadata_storage()
        self.config = get_metadata_config()

        # Initialize model lifecycle
        self.log_event(
            "model_lifecycle",
            "model_initialized",
            {
                "model_name": model_name,
                "model_version": model_version,
                "initialized_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def log_event(
        self,
        stage: str,
        event_type: str,
        metadata: Dict[str, Any],
        details: Optional[str] = None,
    ) -> str:
        """Log an event for this model."""
        return self.storage.save_metadata(
            model_name=self.model_name,
            stage=stage,
            event_type=event_type,
            metadata=metadata,
            model_version=self.model_version,
            details=details,
        )

    def log_data_ingestion(
        self, data_info: Dict[str, Any], details: Optional[str] = None
    ) -> str:
        """Log data ingestion event."""
        return self.log_event("data_ingestion", "data_loaded", data_info, details)

    def log_data_preprocessing(
        self, preprocessing_info: Dict[str, Any], details: Optional[str] = None
    ) -> str:
        """Log data preprocessing event."""
        return self.log_event(
            "data_preprocessing", "data_preprocessed", preprocessing_info, details
        )

    def log_training_start(
        self, training_config: Dict[str, Any], details: Optional[str] = None
    ) -> str:
        """Log training start event."""
        return self.log_event("training", "training_started", training_config, details)

    def log_training_complete(
        self, training_results: Dict[str, Any], details: Optional[str] = None
    ) -> str:
        """Log training completion event."""
        return self.log_event(
            "training", "training_completed", training_results, details
        )

    def log_validation(
        self, validation_results: Dict[str, Any], details: Optional[str] = None
    ) -> str:
        """Log model validation event."""
        return self.log_event(
            "validation", "model_validated", validation_results, details
        )

    def log_inference(
        self, inference_info: Dict[str, Any], details: Optional[str] = None
    ) -> str:
        """Log inference event."""
        return self.log_event(
            "inference", "inference_performed", inference_info, details
        )

    def log_compliance_check(
        self, framework: str, results: Dict[str, Any], details: Optional[str] = None
    ) -> str:
        """Log compliance check event."""
        metadata_id = self.log_event(
            "compliance", f"{framework.lower()}_compliance", results, details
        )

        # Also add compliance event if score is available
        if "compliance_score" in results:
            score = results["compliance_score"]
            status = (
                "passed"
                if score >= self.config.get("compliance_score_threshold", 0.8)
                else "failed"
            )

            self.storage.add_compliance_event(
                metadata_id=metadata_id,
                framework=framework,
                compliance_score=score,
                validation_status=status,
                details=details,
            )

        return metadata_id

    def log_model_deployment(
        self, deployment_info: Dict[str, Any], details: Optional[str] = None
    ) -> str:
        """Log model deployment event."""
        return self.log_event("deployment", "model_deployed", deployment_info, details)

    def log_model_monitoring(
        self, monitoring_data: Dict[str, Any], details: Optional[str] = None
    ) -> str:
        """Log model monitoring data."""
        return self.log_event("monitoring", "monitoring_data", monitoring_data, details)

    def get_pipeline_trace(self) -> Dict[str, Any]:
        """Get complete pipeline trace for this model."""
        return self.storage.get_pipeline_trace(self.model_name)

    def get_stage_metadata(self, stage: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get metadata for a specific stage."""
        return self.storage.get_model_metadata(self.model_name, stage, limit)

    def export_metadata(self, format: str = "json") -> str:
        """Export all metadata for this model."""
        return self.storage.export_metadata(self.model_name, format)


class ComplianceTracker:
    """Tracks compliance metrics across the model lifecycle."""

    def __init__(self, model_manager: ModelMetadataManager):
        """
        Initialize compliance tracker.

        Args:
            model_manager: Model metadata manager instance
        """
        self.model_manager = model_manager
        self.config = get_metadata_config()
        self.compliance_scores = {}

    def track_gdpr_compliance(
        self,
        data_protection_measures: Dict[str, Any],
        consent_management: Dict[str, Any],
        right_to_explanation: bool = True,
    ) -> str:
        """Track GDPR compliance."""
        gdpr_score = self._calculate_gdpr_score(
            data_protection_measures, consent_management, right_to_explanation
        )

        compliance_data = {
            "compliance_score": gdpr_score,
            "data_protection_measures": data_protection_measures,
            "consent_management": consent_management,
            "right_to_explanation": right_to_explanation,
            "assessment_date": datetime.now(timezone.utc).isoformat(),
        }

        self.compliance_scores["GDPR"] = gdpr_score
        return self.model_manager.log_compliance_check("GDPR", compliance_data)

    def track_fda_compliance(
        self,
        clinical_validation: Dict[str, Any],
        safety_measures: Dict[str, Any],
        quality_management: Dict[str, Any],
    ) -> str:
        """Track FDA compliance for medical AI."""
        fda_score = self._calculate_fda_score(
            clinical_validation, safety_measures, quality_management
        )

        compliance_data = {
            "compliance_score": fda_score,
            "clinical_validation": clinical_validation,
            "safety_measures": safety_measures,
            "quality_management": quality_management,
            "assessment_date": datetime.now(timezone.utc).isoformat(),
        }

        self.compliance_scores["FDA"] = fda_score
        return self.model_manager.log_compliance_check("FDA", compliance_data)

    def track_eeoc_compliance(
        self,
        bias_assessment: Dict[str, Any],
        fairness_metrics: Dict[str, Any],
        protected_classes: List[str],
    ) -> str:
        """Track EEOC compliance for hiring decisions."""
        eeoc_score = self._calculate_eeoc_score(
            bias_assessment, fairness_metrics, protected_classes
        )

        compliance_data = {
            "compliance_score": eeoc_score,
            "bias_assessment": bias_assessment,
            "fairness_metrics": fairness_metrics,
            "protected_classes": protected_classes,
            "assessment_date": datetime.now(timezone.utc).isoformat(),
        }

        self.compliance_scores["EEOC"] = eeoc_score
        return self.model_manager.log_compliance_check("EEOC", compliance_data)

    def track_fcra_compliance(
        self,
        accuracy_measures: Dict[str, Any],
        adverse_action_procedures: Dict[str, Any],
        data_quality: Dict[str, Any],
    ) -> str:
        """Track FCRA compliance for credit decisions."""
        fcra_score = self._calculate_fcra_score(
            accuracy_measures, adverse_action_procedures, data_quality
        )

        compliance_data = {
            "compliance_score": fcra_score,
            "accuracy_measures": accuracy_measures,
            "adverse_action_procedures": adverse_action_procedures,
            "data_quality": data_quality,
            "assessment_date": datetime.now(timezone.utc).isoformat(),
        }

        self.compliance_scores["FCRA"] = fcra_score
        return self.model_manager.log_compliance_check("FCRA", compliance_data)

    def get_overall_compliance_score(self) -> float:
        """Get overall compliance score across all frameworks."""
        if not self.compliance_scores:
            return 0.0

        return sum(self.compliance_scores.values()) / len(self.compliance_scores)

    def _calculate_gdpr_score(
        self, data_protection: Dict, consent: Dict, explanation: bool
    ) -> float:
        """Calculate GDPR compliance score."""
        score = 0.0

        # Data protection measures (40%)
        if data_protection.get("encryption", False):
            score += 0.15
        if data_protection.get("anonymization", False):
            score += 0.15
        if data_protection.get("access_controls", False):
            score += 0.10

        # Consent management (40%)
        if consent.get("explicit_consent", False):
            score += 0.20
        if consent.get("withdrawal_mechanism", False):
            score += 0.20

        # Right to explanation (20%)
        if explanation:
            score += 0.20

        return min(score, 1.0)

    def _calculate_fda_score(
        self, clinical: Dict, safety: Dict, quality: Dict
    ) -> float:
        """Calculate FDA compliance score."""
        score = 0.0

        # Clinical validation (50%)
        if clinical.get("clinical_studies", False):
            score += 0.25
        if clinical.get("performance_validation", False):
            score += 0.25

        # Safety measures (30%)
        if safety.get("risk_assessment", False):
            score += 0.15
        if safety.get("monitoring_plan", False):
            score += 0.15

        # Quality management (20%)
        if quality.get("iso_13485", False):
            score += 0.10
        if quality.get("documentation", False):
            score += 0.10

        return min(score, 1.0)

    def _calculate_eeoc_score(
        self, bias: Dict, fairness: Dict, protected: List
    ) -> float:
        """Calculate EEOC compliance score."""
        score = 0.0

        # Bias assessment (40%)
        if bias.get("disparate_impact", 0) < 0.2:  # Low disparate impact
            score += 0.20
        if bias.get("statistical_parity", 0) > 0.8:  # High statistical parity
            score += 0.20

        # Fairness metrics (40%)
        if fairness.get("equalized_odds", 0) > 0.8:
            score += 0.20
        if fairness.get("calibration", 0) > 0.8:
            score += 0.20

        # Protected class coverage (20%)
        expected_classes = ["race", "gender", "age", "disability"]
        covered = len(set(protected) & set(expected_classes))
        score += (covered / len(expected_classes)) * 0.20

        return min(score, 1.0)

    def _calculate_fcra_score(
        self, accuracy: Dict, adverse: Dict, quality: Dict
    ) -> float:
        """Calculate FCRA compliance score."""
        score = 0.0

        # Accuracy measures (40%)
        if accuracy.get("precision", 0) > 0.8:
            score += 0.20
        if accuracy.get("recall", 0) > 0.8:
            score += 0.20

        # Adverse action procedures (40%)
        if adverse.get("notice_system", False):
            score += 0.20
        if adverse.get("dispute_process", False):
            score += 0.20

        # Data quality (20%)
        if quality.get("completeness", 0) > 0.9:
            score += 0.10
        if quality.get("accuracy", 0) > 0.9:
            score += 0.10

        return min(score, 1.0)


# Convenience functions for common use cases
def create_model_manager(
    model_name: str, model_version: str = "1.0.0"
) -> ModelMetadataManager:
    """Create a model metadata manager."""
    return ModelMetadataManager(model_name, model_version)


def create_compliance_tracker(
    model_name: str, model_version: str = "1.0.0"
) -> ComplianceTracker:
    """Create a compliance tracker for a model."""
    manager = ModelMetadataManager(model_name, model_version)
    return ComplianceTracker(manager)


def quick_log(
    model_name: str,
    stage: str,
    event_type: str,
    metadata: Dict[str, Any],
    details: Optional[str] = None,
) -> str:
    """Quick function to log metadata without creating a manager."""
    storage = get_metadata_storage()
    return storage.save_metadata(
        model_name, stage, event_type, metadata, details=details
    )


if __name__ == "__main__":
    # Example usage

    # Create model manager
    manager = create_model_manager("example_model", "2.0.0")

    # Log data ingestion
    manager.log_data_ingestion(
        {
            "rows": 10000,
            "columns": 20,
            "missing_values": 150,
            "data_quality_score": 0.95,
        },
        "Successfully loaded training dataset",
    )

    # Use decorator for automatic metadata capture
    @capture_metadata("example_model", "training", "train_algorithm")
    def train_model(algorithm: str, epochs: int = 100):
        """Example training function."""
        time.sleep(1)  # Simulate training
        return {"accuracy": 0.95, "loss": 0.05}

    # Train model with automatic metadata capture
    result = train_model("random_forest", epochs=50)

    # Create compliance tracker
    tracker = create_compliance_tracker("example_model", "2.0.0")

    # Track GDPR compliance
    tracker.track_gdpr_compliance(
        data_protection_measures={
            "encryption": True,
            "anonymization": True,
            "access_controls": True,
        },
        consent_management={"explicit_consent": True, "withdrawal_mechanism": True},
        right_to_explanation=True,
    )

    # Get pipeline trace
    trace = manager.get_pipeline_trace()
    print(f"Pipeline trace: {json.dumps(trace, indent=2)}")

    # Export metadata
    export_path = manager.export_metadata("json")
    print(f"Exported metadata to: {export_path}")
