"""
Uncertainty Quantification Module for CIAF

This module provides uncertainty quantification capabilities for AI models,
meeting NIST AI RMF requirements and EU AI Act uncertainty disclosure mandates.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class UncertaintyMethod(Enum):
    """Uncertainty quantification methods."""

    MONTE_CARLO_DROPOUT = "Monte Carlo Dropout"
    BAYESIAN_NEURAL_NETWORK = "Bayesian Neural Network"
    DEEP_ENSEMBLES = "Deep Ensembles"
    QUANTILE_REGRESSION = "Quantile Regression"
    VARIATIONAL_INFERENCE = "Variational Inference"
    BOOTSTRAP_SAMPLING = "Bootstrap Sampling"


@dataclass
class ConfidenceInterval:
    """Confidence interval for predictions."""

    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)

    def width(self) -> float:
        """Calculate interval width."""
        return self.upper_bound - self.lower_bound


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty metrics for a prediction."""

    prediction_variance: float
    confidence_interval: ConfidenceInterval
    method: UncertaintyMethod
    iterations: int
    epistemic_uncertainty: Optional[float] = None
    aleatoric_uncertainty: Optional[float] = None
    total_uncertainty: Optional[float] = None
    entropy: Optional[float] = None
    mutual_information: Optional[float] = None
    explainability_ref: Optional[str] = None
    calculation_timestamp: str = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.calculation_timestamp is None:
            self.calculation_timestamp = datetime.now(timezone.utc).isoformat()

        # Calculate total uncertainty if components available
        if self.epistemic_uncertainty and self.aleatoric_uncertainty:
            self.total_uncertainty = np.sqrt(
                self.epistemic_uncertainty**2 + self.aleatoric_uncertainty**2
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "prediction_variance": self.prediction_variance,
            "confidence_interval": self.confidence_interval.to_dict(),
            "method": self.method.value,
            "iterations": self.iterations,
            "calculation_timestamp": self.calculation_timestamp,
        }

        # Add optional fields if available
        optional_fields = [
            "epistemic_uncertainty",
            "aleatoric_uncertainty",
            "total_uncertainty",
            "entropy",
            "mutual_information",
            "explainability_ref",
        ]

        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                result[field] = value

        return result

    def get_uncertainty_hash(self) -> str:
        """Generate cryptographic hash of uncertainty data."""
        data_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


class UncertaintyQuantifier:
    """Uncertainty quantification engine for AI models."""

    def __init__(self, model_name: str):
        """Initialize uncertainty quantifier."""
        self.model_name = model_name
        self.uncertainty_cache = {}

    def quantify_monte_carlo_dropout(
        self,
        prediction_samples: List[float],
        confidence_level: float = 0.95,
        explainability_ref: Optional[str] = None,
    ) -> UncertaintyMetrics:
        """Quantify uncertainty using Monte Carlo Dropout."""

        samples = np.array(prediction_samples)
        variance = np.var(samples)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(samples, lower_percentile)
        upper_bound = np.percentile(samples, upper_percentile)

        confidence_interval = ConfidenceInterval(
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            confidence_level=confidence_level,
        )

        # Calculate entropy for discrete predictions
        entropy = None
        if len(np.unique(samples)) > 1:
            hist, _ = np.histogram(samples, bins=min(50, len(samples) // 10 + 1))
            prob = hist / hist.sum()
            prob = prob[prob > 0]  # Remove zero probabilities
            entropy = -np.sum(prob * np.log2(prob))

        return UncertaintyMetrics(
            prediction_variance=float(variance),
            confidence_interval=confidence_interval,
            method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
            iterations=len(prediction_samples),
            entropy=entropy,
            explainability_ref=explainability_ref,
        )

    def quantify_deep_ensembles(
        self,
        ensemble_predictions: List[float],
        confidence_level: float = 0.95,
        explainability_ref: Optional[str] = None,
    ) -> UncertaintyMetrics:
        """Quantify uncertainty using Deep Ensembles."""

        predictions = np.array(ensemble_predictions)

        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = np.var(predictions)

        # Aleatoric uncertainty (data uncertainty) - simplified estimation
        # In practice, this would come from each model's predicted variance
        aleatoric_uncertainty = np.mean(predictions) * 0.1  # Simplified

        # Total variance
        total_variance = epistemic_uncertainty + aleatoric_uncertainty

        # Confidence interval
        alpha = 1 - confidence_level
        lower_bound = np.percentile(predictions, (alpha / 2) * 100)
        upper_bound = np.percentile(predictions, (1 - alpha / 2) * 100)

        confidence_interval = ConfidenceInterval(
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            confidence_level=confidence_level,
        )

        return UncertaintyMetrics(
            prediction_variance=float(total_variance),
            confidence_interval=confidence_interval,
            method=UncertaintyMethod.DEEP_ENSEMBLES,
            iterations=len(ensemble_predictions),
            epistemic_uncertainty=float(epistemic_uncertainty),
            aleatoric_uncertainty=float(aleatoric_uncertainty),
            explainability_ref=explainability_ref,
        )

    def quantify_bayesian_uncertainty(
        self,
        posterior_samples: List[float],
        confidence_level: float = 0.95,
        explainability_ref: Optional[str] = None,
    ) -> UncertaintyMetrics:
        """Quantify uncertainty using Bayesian methods."""

        samples = np.array(posterior_samples)
        variance = np.var(samples)

        # Calculate mutual information as measure of information gain
        # Simplified calculation - in practice would use proper MI estimation
        mutual_info = np.log2(len(samples)) - np.log2(max(1, len(np.unique(samples))))

        # Confidence interval from posterior
        alpha = 1 - confidence_level
        lower_bound = np.percentile(samples, (alpha / 2) * 100)
        upper_bound = np.percentile(samples, (1 - alpha / 2) * 100)

        confidence_interval = ConfidenceInterval(
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            confidence_level=confidence_level,
        )

        return UncertaintyMetrics(
            prediction_variance=float(variance),
            confidence_interval=confidence_interval,
            method=UncertaintyMethod.BAYESIAN_NEURAL_NETWORK,
            iterations=len(posterior_samples),
            mutual_information=float(mutual_info),
            explainability_ref=explainability_ref,
        )

    def generate_uncertainty_receipt(
        self,
        prediction_id: str,
        uncertainty_metrics: UncertaintyMetrics,
        input_hash: str,
    ) -> Dict[str, Any]:
        """Generate cryptographic uncertainty receipt."""

        receipt = {
            "prediction_id": prediction_id,
            "model_name": self.model_name,
            "input_hash": input_hash,
            "uncertainty_metrics": uncertainty_metrics.to_dict(),
            "uncertainty_hash": uncertainty_metrics.get_uncertainty_hash(),
            "receipt_timestamp": datetime.now(timezone.utc).isoformat(),
            "receipt_version": "1.0",
        }

        # Generate receipt hash
        receipt_data = json.dumps(receipt, sort_keys=True)
        receipt["receipt_hash"] = hashlib.sha256(receipt_data.encode()).hexdigest()

        return receipt

    def validate_uncertainty_requirements(
        self, uncertainty_metrics: UncertaintyMetrics, framework: str = "EU_AI_ACT"
    ) -> Dict[str, Any]:
        """Validate uncertainty metrics against regulatory requirements."""

        validation_results = {
            "framework": framework,
            "compliant": True,
            "issues": [],
            "recommendations": [],
        }

        if framework == "EU_AI_ACT":
            # EU AI Act Article 15 - Transparency requirements
            if uncertainty_metrics.confidence_interval.confidence_level < 0.9:
                validation_results["issues"].append(
                    "Confidence level below 90% may not meet transparency requirements"
                )
                validation_results["compliant"] = False

            if uncertainty_metrics.prediction_variance > 0.5:
                validation_results["issues"].append(
                    "High prediction variance may indicate insufficient model reliability"
                )
                validation_results["recommendations"].append(
                    "Consider model retraining or ensemble methods"
                )

        elif framework == "NIST_AI_RMF":
            # NIST AI RMF - Measure function requirements
            if uncertainty_metrics.method not in [
                UncertaintyMethod.MONTE_CARLO_DROPOUT,
                UncertaintyMethod.BAYESIAN_NEURAL_NETWORK,
                UncertaintyMethod.DEEP_ENSEMBLES,
            ]:
                validation_results["recommendations"].append(
                    "Consider using more robust uncertainty quantification methods"
                )

            if uncertainty_metrics.iterations < 100:
                validation_results["recommendations"].append(
                    "Increase number of iterations for more reliable uncertainty estimates"
                )

        return validation_results

    def export_uncertainty_metadata(
        self, uncertainty_metrics: UncertaintyMetrics, format: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export uncertainty metadata in specified format."""

        metadata = {
            "uncertainty_quantification": {
                "enabled": True,
                "method": uncertainty_metrics.method.value,
                "metrics": uncertainty_metrics.to_dict(),
                "regulatory_compliance": {
                    "eu_ai_act": self.validate_uncertainty_requirements(
                        uncertainty_metrics, "EU_AI_ACT"
                    ),
                    "nist_ai_rmf": self.validate_uncertainty_requirements(
                        uncertainty_metrics, "NIST_AI_RMF"
                    ),
                },
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }

        if format == "json":
            return json.dumps(metadata, indent=2)
        else:
            return metadata


# Example usage and demonstration
def demo_uncertainty_quantification():
    """Demonstrate uncertainty quantification capabilities."""

    print("\n UNCERTAINTY QUANTIFICATION DEMO")
    print("=" * 50)

    quantifier = UncertaintyQuantifier("JobClassificationModel_v2.1")

    # Simulate Monte Carlo Dropout predictions
    np.random.seed(42)
    mc_samples = np.random.normal(0.75, 0.08, 100).tolist()

    print("1. Monte Carlo Dropout Uncertainty Quantification")
    mc_uncertainty = quantifier.quantify_monte_carlo_dropout(
        mc_samples, confidence_level=0.95, explainability_ref="shap_values_job_001.json"
    )

    print(f"   Prediction Variance: {mc_uncertainty.prediction_variance:.6f}")
    print(
        f"   Confidence Interval: [{mc_uncertainty.confidence_interval.lower_bound:.3f}, {mc_uncertainty.confidence_interval.upper_bound:.3f}]"
    )
    print(f"   Method: {mc_uncertainty.method.value}")
    print(f"   Entropy: {mc_uncertainty.entropy:.3f}")

    # Simulate Deep Ensemble predictions
    ensemble_samples = np.random.normal(0.73, 0.05, 10).tolist()

    print("\n2. Deep Ensembles Uncertainty Quantification")
    ensemble_uncertainty = quantifier.quantify_deep_ensembles(
        ensemble_samples, explainability_ref="ensemble_shap_job_001.json"
    )

    print(f"   Epistemic Uncertainty: {ensemble_uncertainty.epistemic_uncertainty:.6f}")
    print(f"   Aleatoric Uncertainty: {ensemble_uncertainty.aleatoric_uncertainty:.6f}")
    print(f"   Total Uncertainty: {ensemble_uncertainty.total_uncertainty:.6f}")

    # Generate uncertainty receipt
    print("\n3. Cryptographic Uncertainty Receipt")
    receipt = quantifier.generate_uncertainty_receipt(
        "PRED_20250802_001", mc_uncertainty, "sha256_input_hash_example"
    )

    print(f"   Receipt Hash: {receipt['receipt_hash'][:16]}...")
    print(f"   Uncertainty Hash: {receipt['uncertainty_hash'][:16]}...")

    # Validate against regulations
    print("\n4. Regulatory Compliance Validation")
    eu_validation = quantifier.validate_uncertainty_requirements(
        mc_uncertainty, "EU_AI_ACT"
    )
    nist_validation = quantifier.validate_uncertainty_requirements(
        mc_uncertainty, "NIST_AI_RMF"
    )

    print(f"   EU AI Act Compliant: {eu_validation['compliant']}")
    print(f"   NIST AI RMF Issues: {len(nist_validation['issues'])}")

    # Export metadata
    print("\n5. Uncertainty Metadata Export")
    metadata = quantifier.export_uncertainty_metadata(mc_uncertainty)
    print("    Uncertainty metadata exported for compliance documentation")

    return quantifier, mc_uncertainty, receipt


if __name__ == "__main__":
    demo_uncertainty_quantification()
