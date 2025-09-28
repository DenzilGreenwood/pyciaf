"""
Consolidated CIAF API Framework
===============================

Modern, consolidated API framework for CIAF that integrates all enhanced modules
(wrappers, LCM, compliance, preprocessing, explainability) with a clean,
protocol-based architecture and comprehensive policy-driven configuration.

Created: 2025-09-28
Author: Denzil James Greenwood
Version: 2.0.0
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from contextlib import asynccontextmanager

from .interfaces import (
    CIAFAPIFramework, DatasetAPIHandler, ModelAPIHandler, TrainingAPIHandler,
    InferenceAPIHandler, AuditAPIHandler, ComplianceAPIHandler, SecurityAPIHandler,
    MetricsAPIHandler, APIResponseHandler, APIMiddleware, APIRequest, APIResponse,
    APIStatus
)
from .policy import APIPolicy, get_default_api_policy, APIMode
from .protocol_implementations import (
    DefaultDatasetAPIHandler, DefaultModelAPIHandler, DefaultTrainingAPIHandler,
    DefaultInferenceAPIHandler, DefaultAPIResponseHandler, DefaultSecurityAPIHandler
)

# Integration imports
try:
    from ..wrappers import create_model_wrapper, UNIVERSAL_ADAPTER_AVAILABLE
    WRAPPER_INTEGRATION_AVAILABLE = True
except ImportError:
    WRAPPER_INTEGRATION_AVAILABLE = False

try:
    from ..lcm import LCMRootManager, get_default_policy as get_lcm_policy
    LCM_INTEGRATION_AVAILABLE = True
except ImportError:
    LCM_INTEGRATION_AVAILABLE = False

try:
    from ..compliance import ComplianceManager
    COMPLIANCE_INTEGRATION_AVAILABLE = True
except ImportError:
    COMPLIANCE_INTEGRATION_AVAILABLE = False


class ConsolidatedAuditAPIHandler:
    """Consolidated audit API handler with full integration."""
    
    def __init__(self, policy: APIPolicy = None):
        self.policy = policy or get_default_api_policy()
        self.logger = logging.getLogger(f"{__name__}.AuditAPI")
        
        # Integration with other handlers
        self.dataset_handler: Optional[DatasetAPIHandler] = None
        self.model_handler: Optional[ModelAPIHandler] = None
        self.inference_handler: Optional[InferenceAPIHandler] = None
    
    def set_handlers(self, dataset_handler, model_handler, inference_handler):
        """Set references to other handlers for comprehensive audit trails."""
        self.dataset_handler = dataset_handler
        self.model_handler = model_handler
        self.inference_handler = inference_handler
    
    def get_audit_trail(self, entity_id: str, entity_type: str) -> Dict[str, Any]:
        """Get complete audit trail for an entity."""
        self.logger.info(f"Generating audit trail for {entity_type}: {entity_id}")
        
        audit_trail = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "generated_at": datetime.now().isoformat(),
            "trail_components": []
        }
        
        if entity_type == "dataset" and self.dataset_handler:
            dataset = self.dataset_handler.get_dataset(entity_id)
            if dataset:
                audit_trail["trail_components"].append({
                    "component": "dataset_creation",
                    "timestamp": dataset.get("created_at"),
                    "data": {
                        "dataset_id": entity_id,
                        "version": dataset.get("version"),
                        "total_samples": dataset.get("total_samples"),
                        "lcm_tracked": dataset.get("lcm_tracked", False)
                    }
                })
        
        elif entity_type == "model" and self.model_handler:
            model = self.model_handler.get_model(entity_id)
            if model:
                audit_trail["trail_components"].append({
                    "component": "model_creation",
                    "timestamp": model.get("created_at"),
                    "data": {
                        "model_name": entity_id,
                        "framework": model.get("framework"),
                        "version": model.get("version"),
                        "authorized_datasets": model.get("authorized_datasets", []),
                        "lcm_tracked": model.get("lcm_tracked", False)
                    }
                })
                
                # Add deployment info if available
                deployment = self.model_handler.deployed_models.get(entity_id)
                if deployment:
                    audit_trail["trail_components"].append({
                        "component": "model_deployment",
                        "timestamp": deployment.get("deployed_at"),
                        "data": {
                            "deployment_id": deployment.get("deployment_id"),
                            "status": deployment.get("status"),
                            "wrapped": deployment.get("wrapped", False)
                        }
                    })
                
                # Add inference history
                if self.inference_handler:
                    inference_history = self.inference_handler.get_inference_history(entity_id)
                    audit_trail["trail_components"].append({
                        "component": "inference_history",
                        "timestamp": datetime.now().isoformat(),
                        "data": {
                            "total_inferences": len(inference_history),
                            "successful_inferences": len([i for i in inference_history if i["status"] == "success"]),
                            "failed_inferences": len([i for i in inference_history if i["status"] == "error"]),
                            "recent_inferences": inference_history[:5]  # Last 5 inferences
                        }
                    })
        
        audit_trail["integrity_hash"] = self._compute_trail_hash(audit_trail)
        
        self.logger.info(f"Audit trail generated with {len(audit_trail['trail_components'])} components")
        return audit_trail
    
    def verify_integrity(self, entity_id: str, entity_type: str) -> Dict[str, Any]:
        """Verify integrity of an entity's audit trail."""
        audit_trail = self.get_audit_trail(entity_id, entity_type)
        
        verification = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "verified_at": datetime.now().isoformat(),
            "integrity_verified": True,
            "verification_details": []
        }
        
        # Verify each component
        for component in audit_trail["trail_components"]:
            component_verification = {
                "component": component["component"],
                "verified": True,
                "issues": []
            }
            
            # Basic verification checks
            if not component.get("timestamp"):
                component_verification["verified"] = False
                component_verification["issues"].append("Missing timestamp")
            
            if not component.get("data"):
                component_verification["verified"] = False
                component_verification["issues"].append("Missing component data")
            
            verification["verification_details"].append(component_verification)
            
            if not component_verification["verified"]:
                verification["integrity_verified"] = False
        
        return verification
    
    def generate_compliance_report(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compliance report based on filters."""
        report = {
            "report_type": "compliance",
            "generated_at": datetime.now().isoformat(),
            "filters": filters,
            "compliance_status": "compliant",
            "findings": []
        }
        
        # GDPR compliance checks
        if "gdpr" in filters.get("frameworks", ["gdpr"]):
            gdpr_findings = self._check_gdpr_compliance(filters)
            report["findings"].extend(gdpr_findings)
        
        # EU AI Act compliance checks
        if "eu_ai_act" in filters.get("frameworks", ["eu_ai_act"]):
            ai_act_findings = self._check_ai_act_compliance(filters)
            report["findings"].extend(ai_act_findings)
        
        # Determine overall compliance status
        critical_findings = [f for f in report["findings"] if f.get("severity") == "critical"]
        if critical_findings:
            report["compliance_status"] = "non_compliant"
        elif any(f.get("severity") == "high" for f in report["findings"]):
            report["compliance_status"] = "needs_attention"
        
        return report
    
    def get_provenance_chain(self, entity_id: str) -> Dict[str, Any]:
        """Get provenance chain for an entity."""
        provenance = {
            "entity_id": entity_id,
            "generated_at": datetime.now().isoformat(),
            "chain": []
        }
        
        # Build provenance chain based on entity relationships
        if self.model_handler:
            model = self.model_handler.get_model(entity_id)
            if model:
                # Add model provenance
                provenance["chain"].append({
                    "step": 1,
                    "entity_type": "model",
                    "entity_id": entity_id,
                    "timestamp": model.get("created_at"),
                    "details": {
                        "framework": model.get("framework"),
                        "version": model.get("version")
                    }
                })
                
                # Add dataset provenance
                for i, dataset_id in enumerate(model.get("authorized_datasets", [])):
                    if self.dataset_handler:
                        dataset = self.dataset_handler.get_dataset(dataset_id)
                        if dataset:
                            provenance["chain"].append({
                                "step": 2 + i,
                                "entity_type": "dataset",
                                "entity_id": dataset_id,
                                "timestamp": dataset.get("created_at"),
                                "details": {
                                    "total_samples": dataset.get("total_samples"),
                                    "features": len(dataset.get("features", []))
                                }
                            })
        
        return provenance
    
    def _compute_trail_hash(self, audit_trail: Dict[str, Any]) -> str:
        """Compute integrity hash for audit trail."""
        import hashlib
        import json
        
        # Create deterministic representation
        trail_data = {
            "entity_id": audit_trail["entity_id"],
            "entity_type": audit_trail["entity_type"],
            "components": audit_trail["trail_components"]
        }
        
        trail_json = json.dumps(trail_data, sort_keys=True, default=str)
        return hashlib.sha256(trail_json.encode()).hexdigest()
    
    def _check_gdpr_compliance(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check GDPR compliance."""
        findings = []
        
        if self.dataset_handler:
            datasets = self.dataset_handler.list_datasets()
            for dataset in datasets:
                compliance_meta = dataset.get("compliance_metadata", {})
                
                if compliance_meta.get("gdpr_applicable"):
                    if not compliance_meta.get("consent_required"):
                        findings.append({
                            "entity_id": dataset["dataset_id"],
                            "entity_type": "dataset",
                            "framework": "GDPR",
                            "finding": "Missing consent requirement for personal data",
                            "severity": "high"
                        })
        
        return findings
    
    def _check_ai_act_compliance(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check EU AI Act compliance."""
        findings = []
        
        if self.model_handler:
            models = self.model_handler.list_models()
            for model in models:
                compliance_meta = model.get("compliance_metadata", {})
                
                if compliance_meta.get("risk_level") == "high":
                    if not compliance_meta.get("human_oversight_required"):
                        findings.append({
                            "entity_id": model["model_name"],
                            "entity_type": "model",
                            "framework": "EU AI Act",
                            "finding": "High-risk AI system missing human oversight",
                            "severity": "critical"
                        })
        
        return findings


class ConsolidatedMetricsAPIHandler:
    """Consolidated metrics API handler."""
    
    def __init__(self, policy: APIPolicy = None):
        self.policy = policy or get_default_api_policy()
        self.logger = logging.getLogger(f"{__name__}.MetricsAPI")
        self.handlers: Dict[str, Any] = {}
    
    def set_handlers(self, **handlers):
        """Set references to other handlers for metrics collection."""
        self.handlers.update(handlers)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "healthy",
            "integrations": {
                "wrapper_integration": WRAPPER_INTEGRATION_AVAILABLE,
                "lcm_integration": LCM_INTEGRATION_AVAILABLE,
                "compliance_integration": COMPLIANCE_INTEGRATION_AVAILABLE,
                "universal_adapter": WRAPPER_INTEGRATION_AVAILABLE and UNIVERSAL_ADAPTER_AVAILABLE
            },
            "api_metrics": {
                "total_datasets": len(self.handlers.get("dataset", {}).datasets) if "dataset" in self.handlers else 0,
                "total_models": len(self.handlers.get("model", {}).models) if "model" in self.handlers else 0,
                "deployed_models": len(self.handlers.get("model", {}).deployed_models) if "model" in self.handlers else 0,
                "training_sessions": len(self.handlers.get("training", {}).training_sessions) if "training" in self.handlers else 0,
                "inference_history": len(self.handlers.get("inference", {}).inference_history) if "inference" in self.handlers else 0
            }
        }
        
        return metrics
    
    def get_model_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get metrics for a specific model."""
        model_handler = self.handlers.get("model")
        inference_handler = self.handlers.get("inference")
        
        if not model_handler:
            raise ValueError("Model handler not available")
        
        model = model_handler.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        metrics = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "framework": model.get("framework"),
                "version": model.get("version"),
                "status": model.get("status"),
                "created_at": model.get("created_at"),
                "lcm_tracked": model.get("lcm_tracked", False)
            }
        }
        
        # Add deployment metrics
        deployment = model_handler.deployed_models.get(model_name)
        if deployment:
            metrics["deployment_info"] = {
                "deployed_at": deployment.get("deployed_at"),
                "deployment_id": deployment.get("deployment_id"),
                "wrapped": deployment.get("wrapped", False)
            }
        
        # Add inference metrics
        if inference_handler:
            inference_history = inference_handler.get_inference_history(model_name)
            metrics["inference_metrics"] = {
                "total_inferences": len(inference_history),
                "successful_inferences": len([i for i in inference_history if i["status"] == "success"]),
                "failed_inferences": len([i for i in inference_history if i["status"] == "error"]),
                "average_duration_ms": sum(i.get("duration_ms", 0) for i in inference_history) / len(inference_history) if inference_history else 0
            }
        
        return metrics
    
    def get_dataset_metrics(self, dataset_id: str) -> Dict[str, Any]:
        """Get metrics for a specific dataset."""
        dataset_handler = self.handlers.get("dataset")
        
        if not dataset_handler:
            raise ValueError("Dataset handler not available")
        
        dataset = dataset_handler.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        metrics = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "total_samples": dataset.get("total_samples"),
                "features": len(dataset.get("features", [])),
                "data_types": dataset.get("data_types", []),
                "created_at": dataset.get("created_at"),
                "lcm_tracked": dataset.get("lcm_tracked", False)
            },
            "usage_metrics": {
                "models_using_dataset": self._count_models_using_dataset(dataset_id)
            }
        }
        
        return metrics
    
    def get_performance_metrics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get performance metrics with optional filters."""
        inference_handler = self.handlers.get("inference")
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "performance_summary": {}
        }
        
        if inference_handler:
            all_inferences = inference_handler.inference_history
            
            # Apply filters
            if filters:
                if "model_name" in filters:
                    all_inferences = [i for i in all_inferences if i["model_name"] == filters["model_name"]]
                if "after" in filters:
                    after_time = datetime.fromisoformat(filters["after"])
                    all_inferences = [i for i in all_inferences if datetime.fromisoformat(i["timestamp"]) > after_time]
            
            if all_inferences:
                durations = [i.get("duration_ms", 0) for i in all_inferences if i.get("duration_ms")]
                
                metrics["performance_summary"] = {
                    "total_requests": len(all_inferences),
                    "successful_requests": len([i for i in all_inferences if i["status"] == "success"]),
                    "failed_requests": len([i for i in all_inferences if i["status"] == "error"]),
                    "average_duration_ms": sum(durations) / len(durations) if durations else 0,
                    "min_duration_ms": min(durations) if durations else 0,
                    "max_duration_ms": max(durations) if durations else 0
                }
        
        return metrics
    
    def _count_models_using_dataset(self, dataset_id: str) -> int:
        """Count models that use a specific dataset."""
        model_handler = self.handlers.get("model")
        if not model_handler:
            return 0
        
        count = 0
        for model in model_handler.models.values():
            if dataset_id in model.get("authorized_datasets", []):
                count += 1
        
        return count


class ConsolidatedCIAFAPIFramework:
    """
    Consolidated CIAF API Framework with full integration.
    
    This framework brings together all enhanced CIAF modules with a clean,
    protocol-based architecture and comprehensive policy-driven configuration.
    """
    
    def __init__(self, policy: APIPolicy = None):
        self.policy = policy or get_default_api_policy()
        self.logger = logging.getLogger(__name__)
        
        # Initialize response handler
        self.response_handler = DefaultAPIResponseHandler(self.policy)
        
        # Middleware stack
        self.middleware: List[APIMiddleware] = []
        
        # Initialize handlers
        self.initialize_handlers(self.policy)
        
        # Integration status
        self.integration_status = {
            "wrapper_integration": WRAPPER_INTEGRATION_AVAILABLE,
            "lcm_integration": LCM_INTEGRATION_AVAILABLE,
            "compliance_integration": COMPLIANCE_INTEGRATION_AVAILABLE
        }
        
        self.logger.info(f"CIAF API Framework initialized in {self.policy.api_mode.value} mode")
        self.logger.info(f"Integrations: {self.integration_status}")
    
    def initialize_handlers(self, policy: APIPolicy) -> None:
        """Initialize all API handlers with policy configuration."""
        # Core handlers
        self.dataset_handler = DefaultDatasetAPIHandler(policy)
        self.model_handler = DefaultModelAPIHandler(policy)
        self.training_handler = DefaultTrainingAPIHandler(policy)
        self.inference_handler = DefaultInferenceAPIHandler(policy, self.model_handler)
        self.security_handler = DefaultSecurityAPIHandler(policy)
        
        # Consolidated handlers
        self.audit_handler = ConsolidatedAuditAPIHandler(policy)
        self.metrics_handler = ConsolidatedMetricsAPIHandler(policy)
        
        # Set cross-references
        self.audit_handler.set_handlers(
            self.dataset_handler, 
            self.model_handler, 
            self.inference_handler
        )
        
        self.metrics_handler.set_handlers(
            dataset=self.dataset_handler,
            model=self.model_handler,
            training=self.training_handler,
            inference=self.inference_handler
        )
        
        self.logger.info("All API handlers initialized successfully")
    
    def register_middleware(self, middleware: APIMiddleware) -> None:
        """Register middleware component."""
        self.middleware.append(middleware)
        self.logger.info(f"Registered middleware: {middleware.__class__.__name__}")
    
    def process_api_request(self, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an API request through the full pipeline."""
        start_time = datetime.now()
        request_id = str(id(data))  # Simple request ID generation
        
        try:
            self.logger.info(f"Processing {method} {endpoint} (ID: {request_id})")
            
            # Apply middleware to request
            processed_request = data
            for middleware in self.middleware:
                processed_request = middleware.process_request(processed_request)
            
            # Route to appropriate handler
            result = self._route_request(endpoint, method, processed_request)
            
            # Format success response
            response = self.response_handler.format_success_response(result)
            
            # Apply middleware to response
            for middleware in reversed(self.middleware):
                response = middleware.process_response(response)
            
            # Add request metadata
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            response["request_metadata"] = {
                "request_id": request_id,
                "endpoint": endpoint,
                "method": method,
                "duration_ms": duration_ms,
                "timestamp": start_time.isoformat()
            }
            
            self.logger.info(f"Request {request_id} completed in {duration_ms:.2f}ms")
            return response
            
        except Exception as e:
            self.logger.error(f"Request {request_id} failed: {e}")
            
            # Format error response
            error_response = self.response_handler.format_error_response(e)
            
            # Apply middleware error handling
            for middleware in reversed(self.middleware):
                error_response = middleware.handle_error(e, {"endpoint": endpoint, "method": method})
            
            return error_response
    
    def _route_request(self, endpoint: str, method: str, data: Dict[str, Any]) -> Any:
        """Route request to appropriate handler."""
        # Dataset endpoints
        if endpoint.startswith("/api/v1/datasets"):
            if method == "POST" and endpoint == "/api/v1/datasets":
                # Extract known parameters to avoid duplication
                dataset_kwargs = {k: v for k, v in data.items() if k not in ["dataset_id", "metadata"]}
                return self.dataset_handler.create_dataset(
                    data["dataset_id"], 
                    data.get("metadata", {}), 
                    **dataset_kwargs
                )
            elif method == "GET" and endpoint == "/api/v1/datasets":
                return self.dataset_handler.list_datasets(data.get("filters"))
            elif method == "GET" and "/datasets/" in endpoint:
                dataset_id = endpoint.split("/datasets/")[1]
                return self.dataset_handler.get_dataset(dataset_id)
            elif method == "PUT" and "/datasets/" in endpoint:
                dataset_id = endpoint.split("/datasets/")[1]
                return self.dataset_handler.update_dataset(dataset_id, data.get("updates", {}))
        
        # Model endpoints
        elif endpoint.startswith("/api/v1/models"):
            if method == "POST" and endpoint == "/api/v1/models":
                # Extract known parameters to avoid duplication
                model_kwargs = {k: v for k, v in data.items() if k not in ["model_name", "config"]}
                return self.model_handler.create_model(
                    data["model_name"], 
                    data.get("config", {}), 
                    **model_kwargs
                )
            elif method == "GET" and endpoint == "/api/v1/models":
                return self.model_handler.list_models(data.get("filters"))
            elif method == "GET" and "/models/" in endpoint and not endpoint.endswith("/predict"):
                model_name = endpoint.split("/models/")[1]
                return self.model_handler.get_model(model_name)
            elif method == "POST" and endpoint.endswith("/deploy"):
                model_name = endpoint.split("/models/")[1].split("/deploy")[0]
                return self.model_handler.deploy_model(model_name, data.get("deployment_config", {}))
            elif method == "POST" and endpoint.endswith("/predict"):
                model_name = endpoint.split("/models/")[1].split("/predict")[0]
                # Extract known parameters to avoid duplication
                inference_kwargs = {k: v for k, v in data.items() if k not in ["input_data"]}
                return self.inference_handler.perform_inference(
                    model_name, 
                    data.get("input_data"), 
                    **inference_kwargs
                )
        
        # Training endpoints
        elif endpoint.startswith("/api/v1/training"):
            if method == "POST" and endpoint == "/api/v1/training":
                return self.training_handler.start_training(
                    data["model_name"], data.get("training_config", {})
                )
            elif method == "GET" and endpoint == "/api/v1/training":
                return self.training_handler.list_training_sessions(data.get("filters"))
            elif method == "GET" and "/training/" in endpoint:
                training_id = endpoint.split("/training/")[1]
                return self.training_handler.get_training_status(training_id)
        
        # Audit endpoints
        elif endpoint.startswith("/api/v1/audit"):
            if "/trail/" in endpoint:
                entity_id = endpoint.split("/trail/")[1]
                entity_type = data.get("entity_type", "model")
                return self.audit_handler.get_audit_trail(entity_id, entity_type)
            elif "/verify/" in endpoint:
                entity_id = endpoint.split("/verify/")[1]
                entity_type = data.get("entity_type", "model")
                return self.audit_handler.verify_integrity(entity_id, entity_type)
            elif endpoint.endswith("/compliance-report"):
                return self.audit_handler.generate_compliance_report(data.get("filters", {}))
        
        # Metrics endpoints
        elif endpoint.startswith("/api/v1/metrics"):
            if endpoint == "/api/v1/metrics/system":
                return self.metrics_handler.get_system_metrics()
            elif "/models/" in endpoint:
                model_name = endpoint.split("/models/")[1]
                return self.metrics_handler.get_model_metrics(model_name)
            elif "/datasets/" in endpoint:
                dataset_id = endpoint.split("/datasets/")[1]
                return self.metrics_handler.get_dataset_metrics(dataset_id)
            elif endpoint.endswith("/performance"):
                return self.metrics_handler.get_performance_metrics(data.get("filters"))
        
        # Health endpoint
        elif endpoint == "/api/v1/health":
            return self.get_api_health()
        
        else:
            raise ValueError(f"Unknown endpoint: {method} {endpoint}")
    
    def get_api_health(self) -> Dict[str, Any]:
        """Get API health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "mode": self.policy.api_mode.value,
            "integrations": self.integration_status,
            "handlers": {
                "dataset": bool(self.dataset_handler),
                "model": bool(self.model_handler),
                "training": bool(self.training_handler),
                "inference": bool(self.inference_handler),
                "audit": bool(self.audit_handler),
                "security": bool(self.security_handler),
                "metrics": bool(self.metrics_handler)
            }
        }
    
    # Async support methods
    async def process_async_request(self, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process API request asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_api_request, endpoint, method, data)
    
    @asynccontextmanager
    async def async_context(self):
        """Async context manager for the API framework."""
        self.logger.info("Starting async API context")
        try:
            yield self
        finally:
            self.logger.info("Closing async API context")


__all__ = [
    "ConsolidatedCIAFAPIFramework",
    "ConsolidatedAuditAPIHandler", 
    "ConsolidatedMetricsAPIHandler",
]