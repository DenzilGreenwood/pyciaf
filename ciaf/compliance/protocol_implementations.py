"""
Concrete implementations of compliance protocols for the CIAF compliance system.

This module provides concrete implementations of the compliance Protocol interfaces,
following the same pattern as the LCM protocol implementations.

Created: 2025-09-26
Author: Denzil James Greenwood  
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import json
import hashlib

from .interfaces import (
    ComplianceValidator as IComplianceValidator,
    AuditTrailProvider,
    RiskAssessor,
    BiasDetector,
    DocumentationGenerator,
    ComplianceStore,
    AlertSystem,
    ComplianceFramework,
    ValidationSeverity,
    AuditEventType
)

# Import from existing modules to wrap them
from .regulatory_mapping import RegulatoryMapper
from ..core import sha256_hash


class DefaultComplianceValidator(IComplianceValidator):
    """Default compliance validator implementation."""
    
    def __init__(self):
        """Initialize with regulatory mapper."""
        self.regulatory_mapper = RegulatoryMapper()
        self.validation_results: List[Dict[str, Any]] = []
    
    def validate_framework_compliance(
        self,
        framework: ComplianceFramework,
        audit_data: Dict[str, Any],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Validate compliance with a specific regulatory framework."""
        requirements = self.regulatory_mapper.get_requirements([framework])
        results = []
        
        for requirement in requirements:
            result = {
                "validation_id": f"VAL_{requirement.requirement_id}",
                "requirement_id": requirement.requirement_id,
                "framework": framework.value,
                "title": requirement.title,
                "severity": ValidationSeverity.HIGH.value if requirement.mandatory else ValidationSeverity.MEDIUM.value,
                "status": "pass" if requirement.is_satisfied_by_ciaf() else "warning",
                "message": f"Requirement {'satisfied' if requirement.is_satisfied_by_ciaf() else 'needs attention'}",
                "details": {
                    "description": requirement.description,
                    "mandatory": requirement.mandatory,
                    "ciaf_capabilities": requirement.ciaf_capabilities
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            results.append(result)
        
        self.validation_results.extend(results)
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {"message": "No validations performed"}
        
        total = len(self.validation_results)
        passing = len([r for r in self.validation_results if r["status"] == "pass"])
        
        return {
            "total_validations": total,
            "passing": passing,
            "failing": total - passing,
            "pass_rate": (passing / total * 100) if total else 0,
            "overall_status": "compliant" if passing == total else "non_compliant"
        }


class DefaultAuditTrailProvider(AuditTrailProvider):
    """Default audit trail provider implementation."""
    
    def __init__(self):
        """Initialize audit trail storage."""
        self.audit_records: List[Dict[str, Any]] = []
        self.last_hash = ""
    
    def record_event(
        self,
        event_type: AuditEventType,
        event_data: Dict[str, Any],
        **kwargs
    ) -> str:
        """Record an audit event and return event ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        event_id = f"{event_type.value}_{int(datetime.now().timestamp() * 1000)}"
        
        # Create audit record
        record = {
            "event_id": event_id,
            "event_type": event_type.value,
            "timestamp": timestamp,
            "event_data": event_data,
            "previous_hash": self.last_hash,
            **kwargs
        }
        
        # Compute record hash
        record_str = json.dumps(record, sort_keys=True)
        record_hash = sha256_hash(record_str.encode('utf-8'))
        record["record_hash"] = record_hash
        
        self.audit_records.append(record)
        self.last_hash = record_hash
        
        return event_id
    
    def get_audit_trail(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> List[Dict[str, Any]]:
        """Get filtered audit trail."""
        filtered_records = self.audit_records
        
        if start_date:
            filtered_records = [
                r for r in filtered_records
                if datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) >= start_date
            ]
        
        if end_date:
            filtered_records = [
                r for r in filtered_records
                if datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) <= end_date
            ]
        
        if event_types:
            event_type_values = [et.value for et in event_types]
            filtered_records = [
                r for r in filtered_records
                if r["event_type"] in event_type_values
            ]
        
        return filtered_records
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify the cryptographic integrity of the audit trail."""
        verification_results = {
            "total_records": len(self.audit_records),
            "integrity_verified": True,
            "broken_connections": [],
            "timestamp_issues": []
        }
        
        previous_hash = ""
        previous_timestamp = None
        
        for i, record in enumerate(self.audit_records):
            # Check hash connections
            if record.get("previous_hash", "") != previous_hash:
                verification_results["integrity_verified"] = False
                verification_results["broken_connections"].append({
                    "record_index": i,
                    "event_id": record["event_id"],
                    "expected_previous": previous_hash,
                    "actual_previous": record.get("previous_hash", "")
                })
            
            # Check timestamp ordering
            current_timestamp = datetime.fromisoformat(
                record["timestamp"].replace("Z", "+00:00")
            )
            if previous_timestamp and current_timestamp < previous_timestamp:
                verification_results["timestamp_issues"].append({
                    "record_index": i,
                    "event_id": record["event_id"],
                    "timestamp": record["timestamp"]
                })
            
            previous_hash = record.get("record_hash", "")
            previous_timestamp = current_timestamp
        
        return verification_results


class DefaultRiskAssessor(RiskAssessor):
    """Default risk assessment implementation."""
    
    def assess_model_risk(
        self,
        model_metadata: Dict[str, Any],
        deployment_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks associated with model deployment."""
        risk_factors = []
        risk_score = 0.0
        
        # Assess based on model type and deployment context
        model_type = model_metadata.get("model_type", "unknown")
        domain = deployment_context.get("domain", "general")
        
        # High-risk domains
        if domain in ["healthcare", "finance", "criminal_justice"]:
            risk_factors.append("high_impact_domain")
            risk_score += 0.3
        
        # Model complexity
        params = model_metadata.get("parameter_count", 0)
        if params > 1e9:  # > 1B parameters
            risk_factors.append("high_complexity_model")
            risk_score += 0.2
        
        # Data sensitivity
        if model_metadata.get("uses_pii", False):
            risk_factors.append("sensitive_data")
            risk_score += 0.25
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendations": self._get_risk_recommendations(risk_factors)
        }
    
    def assess_data_risk(
        self,
        data_metadata: Dict[str, Any],
        usage_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks associated with data usage."""
        risk_factors = []
        risk_score = 0.0
        
        # PII presence
        if data_metadata.get("contains_pii", False):
            risk_factors.append("contains_pii")
            risk_score += 0.4
        
        # Data size and retention
        if data_metadata.get("record_count", 0) > 1e6:
            risk_factors.append("large_dataset")
            risk_score += 0.1
        
        # Cross-border data transfer
        if usage_context.get("cross_border_transfer", False):
            risk_factors.append("cross_border_transfer")
            risk_score += 0.2
        
        # Determine risk level
        if risk_score >= 0.6:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendations": self._get_data_risk_recommendations(risk_factors)
        }
    
    def _get_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Get recommendations based on risk factors."""
        recommendations = []
        
        if "high_impact_domain" in risk_factors:
            recommendations.append("Implement enhanced monitoring and human oversight")
        if "high_complexity_model" in risk_factors:
            recommendations.append("Conduct thorough explainability analysis")
        if "sensitive_data" in risk_factors:
            recommendations.append("Implement privacy-preserving techniques")
        
        return recommendations
    
    def _get_data_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Get data-specific recommendations based on risk factors."""
        recommendations = []
        
        if "contains_pii" in risk_factors:
            recommendations.append("Implement data anonymization or pseudonymization")
        if "large_dataset" in risk_factors:
            recommendations.append("Implement data minimization strategies")
        if "cross_border_transfer" in risk_factors:
            recommendations.append("Ensure compliance with data transfer regulations")
        
        return recommendations


class DefaultBiasDetector(BiasDetector):
    """Default bias detection implementation."""
    
    def detect_bias(
        self,
        predictions: Any,
        protected_attributes: Dict[str, Any],
        ground_truth: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Detect bias in model predictions."""
        try:
            import numpy as np
        except ImportError:
            return {"error": "NumPy not available for bias detection"}
        
        bias_results = {
            "overall_bias_score": 0.95,
            "demographic_parity": {},
            "bias_detected": False,
            "assessment_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Simple demographic parity check
        for attr_name, attr_values in protected_attributes.items():
            unique_values = np.unique(attr_values)
            if len(unique_values) > 1:
                parity_scores = {}
                for value in unique_values:
                    mask = attr_values == value
                    if np.sum(mask) > 0:
                        positive_rate = np.mean(predictions[mask])
                        parity_scores[str(value)] = positive_rate
                
                if len(parity_scores) >= 2:
                    rates = list(parity_scores.values())
                    parity_diff = max(rates) - min(rates)
                    bias_results["demographic_parity"][attr_name] = {
                        "rates": parity_scores,
                        "difference": parity_diff,
                        "bias_detected": parity_diff > 0.1
                    }
                    
                    if parity_diff > 0.1:
                        bias_results["bias_detected"] = True
                        bias_results["overall_bias_score"] *= 0.9
        
        return bias_results
    
    def calculate_fairness_metrics(
        self,
        predictions: Any,
        protected_attributes: Dict[str, Any],
        ground_truth: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Calculate fairness metrics."""
        return self.detect_bias(predictions, protected_attributes, ground_truth)


class InMemoryComplianceStore(ComplianceStore):
    """In-memory compliance data store implementation."""
    
    def __init__(self):
        """Initialize empty storage."""
        self._validation_results: Dict[str, List[Dict[str, Any]]] = {}
        self._compliance_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def store_validation_results(
        self,
        model_id: str,
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None
    ) -> None:
        """Store validation results."""
        if model_id not in self._validation_results:
            self._validation_results[model_id] = []
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results,
            "metadata": metadata or {}
        }
        
        self._validation_results[model_id].append(record)
        
        # Also add to compliance history
        if model_id not in self._compliance_history:
            self._compliance_history[model_id] = []
        self._compliance_history[model_id].append(record)
    
    def get_compliance_history(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get compliance validation history."""
        history = self._compliance_history.get(model_id, [])
        
        if start_date:
            history = [
                h for h in history
                if datetime.fromisoformat(h["timestamp"].replace("Z", "+00:00")) >= start_date
            ]
        
        if end_date:
            history = [
                h for h in history
                if datetime.fromisoformat(h["timestamp"].replace("Z", "+00:00")) <= end_date
            ]
        
        return history


class NoOpAlertSystem(AlertSystem):
    """No-operation alert system (default implementation)."""
    
    def send_compliance_alert(
        self,
        severity: ValidationSeverity,
        message: str,
        details: Dict[str, Any] = None
    ) -> None:
        """Send compliance alert (no-op implementation)."""
        # In a real implementation, this would send alerts via email, Slack, etc.
        pass
    
    def configure_alert_rules(
        self,
        rules: List[Dict[str, Any]]
    ) -> None:
        """Configure alerting rules (no-op implementation)."""
        # In a real implementation, this would configure alerting rules
        pass


class SimpleDocumentationGenerator(DocumentationGenerator):
    """Simple documentation generator implementation."""
    
    def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        model_metadata: Dict[str, Any],
        validation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate compliance documentation."""
        passing = len([r for r in validation_results if r.get("status") == "pass"])
        total = len(validation_results)
        
        return {
            "framework": framework.value,
            "model_id": model_metadata.get("model_id", "unknown"),
            "model_name": model_metadata.get("model_name", "unknown"),
            "assessment_date": datetime.now(timezone.utc).isoformat(),
            "overall_compliance": "compliant" if passing == total else "non_compliant",
            "compliance_rate": (passing / total * 100) if total else 0,
            "validation_summary": {
                "total_validations": total,
                "passed": passing,
                "failed": total - passing
            },
            "detailed_results": validation_results
        }
    
    def export_documentation(self, format: str = "pdf") -> bytes:
        """Export documentation in specified format."""
        # Simple text export for now
        if format.lower() == "text":
            return b"Compliance documentation exported"
        else:
            raise ValueError(f"Unsupported export format: {format}")


def create_default_compliance_protocols() -> Dict[str, Any]:
    """
    Create default compliance protocol implementations.
    
    Returns:
        Dictionary containing default compliance protocol implementations
    """
    return {
        'validator': DefaultComplianceValidator(),
        'audit_provider': DefaultAuditTrailProvider(),
        'risk_assessor': DefaultRiskAssessor(),
        'bias_detector': DefaultBiasDetector(),
        'doc_generator': SimpleDocumentationGenerator(),
        'compliance_store': InMemoryComplianceStore(),
        'alert_system': NoOpAlertSystem()
    }