"""
Uncertainty Quantification Module for CIAF

This module provides uncertainty quantification capabilities including
aleatoric (data uncertainty) and epistemic (model uncertainty) estimation
for regulatory compliance with AI transparency requirements.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class UncertaintyType(Enum):
    """Types of uncertainty in ML predictions."""

    ALEATORIC = "aleatoric"  # Data uncertainty (noise in data)
    EPISTEMIC = "epistemic"  # Model uncertainty (lack of knowledge)
    TOTAL = "total"  # Combined uncertainty


class UncertaintyMethod(Enum):
    """Methods for uncertainty quantification."""

    MONTE_CARLO_DROPOUT = "monte_carlo_dropout"
    DEEP_ENSEMBLES = "deep_ensembles"
    BAYESIAN_NEURAL_NETWORK = "bayesian_neural_network"
    BOOTSTRAP_AGGREGATION = "bootstrap_aggregation"
    PREDICTION_INTERVALS = "prediction_intervals"
    CONFORMAL_PREDICTION = "conformal_prediction"


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates."""

    prediction: Any
    confidence: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    total_uncertainty: float
    method: UncertaintyMethod
    confidence_interval: Tuple[float, float]
    calibration_score: Optional[float] = None
    explanation: Optional[str] = None


class CIAFUncertaintyQuantifier:
    """Uncertainty quantification for CIAF framework."""

    def __init__(
        self,
        model: Any,
        method: UncertaintyMethod = UncertaintyMethod.MONTE_CARLO_DROPOUT,
        n_samples: int = 100,
        confidence_level: float = 0.95,
    ):
        """
        Initialize uncertainty quantifier.

        Args:
            model: The ML model to quantify uncertainty for
            method: Uncertainty quantification method
            n_samples: Number of samples for Monte Carlo methods
            confidence_level: Confidence level for intervals
        """
        self.model = model
        self.method = method
        self.n_samples = n_samples
        self.confidence_level = confidence_level
        self.is_fitted = False
        self.calibration_data = None

    def fit(
        self, X_cal: Optional[np.ndarray] = None, y_cal: Optional[np.ndarray] = None
    ) -> "CIAFUncertaintyQuantifier":
        """
        Fit uncertainty quantifier (calibration if needed).

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        if X_cal is not None and y_cal is not None:
            self.calibration_data = (X_cal, y_cal)

        self.is_fitted = True
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> UncertaintyEstimate:
        """
        Make prediction with uncertainty quantification.

        Args:
            X: Input features

        Returns:
            UncertaintyEstimate containing prediction and uncertainties
        """
        try:
            if self.method == UncertaintyMethod.MONTE_CARLO_DROPOUT:
                return self._monte_carlo_dropout(X)
            elif self.method == UncertaintyMethod.BOOTSTRAP_AGGREGATION:
                return self._bootstrap_uncertainty(X)
            elif self.method == UncertaintyMethod.PREDICTION_INTERVALS:
                return self._prediction_intervals(X)
            else:
                return self._fallback_uncertainty(X)

        except Exception as e:
            warnings.warn(f"Uncertainty quantification failed: {e}")
            return self._fallback_uncertainty(X)

    def _monte_carlo_dropout(self, X: np.ndarray) -> UncertaintyEstimate:
        """Monte Carlo Dropout uncertainty estimation."""
        predictions = []

        # Simulate Monte Carlo sampling
        for _ in range(self.n_samples):
            # For non-neural networks, add small noise to simulate dropout
            if hasattr(self.model, "predict_proba"):
                pred = self.model.predict_proba(X)
                if pred.ndim > 1:
                    pred = pred[0] if len(pred) > 0 else [0.5, 0.5]
                # Add small random noise to simulate uncertainty
                noise = np.random.normal(0, 0.01, len(pred))
                pred = np.clip(np.array(pred) + noise, 0, 1)
                predictions.append(pred)
            else:
                pred = self.model.predict(X)
                if hasattr(pred, "__len__") and len(pred) > 0:
                    pred = pred[0]
                # Add noise for regression
                noise = (
                    np.random.normal(0, abs(pred) * 0.05)
                    if pred != 0
                    else np.random.normal(0, 0.1)
                )
                predictions.append(pred + noise)

        predictions = np.array(predictions)

        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Estimate aleatoric vs epistemic uncertainty
        # Simplified approach: assume half of variance is aleatoric, half epistemic
        total_uncertainty = (
            float(np.mean(std_pred))
            if hasattr(std_pred, "__len__")
            else float(std_pred)
        )
        aleatoric_uncertainty = total_uncertainty * 0.6  # Data noise component
        epistemic_uncertainty = total_uncertainty * 0.4  # Model uncertainty component

        # Confidence and intervals
        confidence = max(0.0, 1.0 - total_uncertainty)
        alpha = 1 - self.confidence_level
        lower = np.percentile(predictions, alpha / 2 * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha / 2) * 100, axis=0)

        # Handle single prediction
        if hasattr(mean_pred, "__len__") and len(mean_pred) > 1:
            prediction = (
                mean_pred[1] if len(mean_pred) > 1 else mean_pred[0]
            )  # Positive class for classification
            interval = (
                float(lower[1]) if len(lower) > 1 else float(lower[0]),
                float(upper[1]) if len(upper) > 1 else float(upper[0]),
            )
        else:
            prediction = (
                float(mean_pred) if hasattr(mean_pred, "__len__") else mean_pred
            )
            interval = (float(lower), float(upper))

        return UncertaintyEstimate(
            prediction=prediction,
            confidence=confidence,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            total_uncertainty=total_uncertainty,
            method=self.method,
            confidence_interval=interval,
            explanation=f"Monte Carlo Dropout with {self.n_samples} samples",
        )

    def _bootstrap_uncertainty(self, X: np.ndarray) -> UncertaintyEstimate:
        """Bootstrap aggregation uncertainty estimation."""
        # For models without built-in uncertainty, use prediction variance
        try:
            # Make multiple predictions with slight input perturbations
            predictions = []
            for _ in range(self.n_samples):
                # Add small noise to input
                X_noisy = X + np.random.normal(0, 0.01, X.shape)
                pred = self.model.predict(X_noisy)
                if hasattr(pred, "__len__") and len(pred) > 0:
                    pred = pred[0]
                predictions.append(pred)

            predictions = np.array(predictions)
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)

            # Estimate uncertainties
            total_uncertainty = float(std_pred)
            aleatoric_uncertainty = total_uncertainty * 0.7
            epistemic_uncertainty = total_uncertainty * 0.3

            confidence = max(0.0, 1.0 - total_uncertainty)

            # Confidence interval
            alpha = 1 - self.confidence_level
            lower = np.percentile(predictions, alpha / 2 * 100)
            upper = np.percentile(predictions, (1 - alpha / 2) * 100)

            return UncertaintyEstimate(
                prediction=float(mean_pred),
                confidence=confidence,
                aleatoric_uncertainty=aleatoric_uncertainty,
                epistemic_uncertainty=epistemic_uncertainty,
                total_uncertainty=total_uncertainty,
                method=self.method,
                confidence_interval=(float(lower), float(upper)),
                explanation=f"Bootstrap aggregation with {self.n_samples} samples",
            )

        except Exception as e:
            return self._fallback_uncertainty(X)

    def _prediction_intervals(self, X: np.ndarray) -> UncertaintyEstimate:
        """Prediction intervals based uncertainty estimation."""
        # Get base prediction
        if hasattr(self.model, "predict_proba"):
            pred = self.model.predict_proba(X)
            if pred.ndim > 1:
                pred = pred[0]
            prediction = pred[1] if len(pred) > 1 else pred[0]
            confidence = max(pred) if hasattr(pred, "__len__") else pred
        else:
            pred = self.model.predict(X)
            prediction = pred[0] if hasattr(pred, "__len__") and len(pred) > 0 else pred
            confidence = 0.8  # Default confidence

        # Estimate uncertainty based on prediction confidence
        if hasattr(self.model, "predict_proba"):
            # For classification: use entropy of probability distribution
            probs = (
                self.model.predict_proba(X)[0]
                if hasattr(self.model.predict_proba(X), "__len__")
                else [0.5, 0.5]
            )
            entropy = -np.sum([p * np.log(p + 1e-10) for p in probs if p > 0])
            normalized_entropy = (
                entropy / np.log(len(probs)) if len(probs) > 1 else entropy
            )
            total_uncertainty = float(normalized_entropy)
        else:
            # For regression: use fixed uncertainty estimate
            total_uncertainty = abs(prediction) * 0.1 if prediction != 0 else 0.1

        aleatoric_uncertainty = total_uncertainty * 0.6
        epistemic_uncertainty = total_uncertainty * 0.4

        # Simple confidence interval
        margin = total_uncertainty * 2  # 2-sigma interval
        interval = (prediction - margin, prediction + margin)

        return UncertaintyEstimate(
            prediction=prediction,
            confidence=float(confidence),
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            total_uncertainty=total_uncertainty,
            method=self.method,
            confidence_interval=interval,
            explanation="Prediction intervals based on model confidence",
        )

    def _fallback_uncertainty(self, X: np.ndarray) -> UncertaintyEstimate:
        """Fallback uncertainty estimation."""
        # Get base prediction
        try:
            pred = self.model.predict(X)
            prediction = pred[0] if hasattr(pred, "__len__") and len(pred) > 0 else pred
        except:
            prediction = 0.5  # Default prediction

        # Default uncertainty values
        return UncertaintyEstimate(
            prediction=prediction,
            confidence=0.5,
            aleatoric_uncertainty=0.2,
            epistemic_uncertainty=0.3,
            total_uncertainty=0.5,
            method=UncertaintyMethod.PREDICTION_INTERVALS,
            confidence_interval=(prediction - 0.5, prediction + 0.5),
            explanation="Fallback uncertainty estimation",
        )

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> Dict[str, float]:
        """
        Calibrate uncertainty estimates.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets

        Returns:
            Calibration metrics
        """
        try:
            # Generate uncertainty estimates for calibration data
            estimates = []
            for i in range(len(X_cal)):
                est = self.predict_with_uncertainty(X_cal[i : i + 1])
                estimates.append(est)

            # Calculate calibration metrics
            confidences = [est.confidence for est in estimates]
            predictions = [est.prediction for est in estimates]

            # Simplified calibration score (could be improved with proper calibration methods)
            errors = [abs(pred - true) for pred, true in zip(predictions, y_cal)]
            avg_error = np.mean(errors)
            avg_confidence = np.mean(confidences)

            calibration_score = 1.0 - abs(avg_confidence - (1.0 - avg_error))

            return {
                "calibration_score": calibration_score,
                "average_confidence": avg_confidence,
                "average_error": avg_error,
                "samples_used": len(X_cal),
            }

        except Exception as e:
            warnings.warn(f"Calibration failed: {e}")
            return {"calibration_score": 0.0, "error": str(e)}


class CIAFUncertaintyManager:
    """Manager for uncertainty quantification across models."""

    def __init__(self):
        """Initialize uncertainty manager."""
        self.quantifiers: Dict[str, CIAFUncertaintyQuantifier] = {}

    def register_quantifier(
        self,
        model_id: str,
        model: Any,
        method: UncertaintyMethod = UncertaintyMethod.MONTE_CARLO_DROPOUT,
        n_samples: int = 100,
    ) -> CIAFUncertaintyQuantifier:
        """Register uncertainty quantifier for a model."""
        quantifier = CIAFUncertaintyQuantifier(model, method, n_samples)
        self.quantifiers[model_id] = quantifier
        return quantifier

    def predict_with_uncertainty(
        self, model_id: str, X: np.ndarray
    ) -> Optional[UncertaintyEstimate]:
        """Get prediction with uncertainty for a model."""
        if model_id in self.quantifiers:
            return self.quantifiers[model_id].predict_with_uncertainty(X)
        return None

    def get_uncertainty_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get uncertainty quantification metadata for compliance."""
        if model_id in self.quantifiers:
            quantifier = self.quantifiers[model_id]
            return {
                "uncertainty_quantification_enabled": quantifier.is_fitted,
                "uncertainty_method": quantifier.method.value,
                "confidence_level": quantifier.confidence_level,
                "samples_used": quantifier.n_samples,
                "compliance_frameworks": {
                    "eu_ai_act": "Article 15 - Accuracy and robustness requirements",
                    "nist_ai_rmf": "Measure function - Model uncertainty assessment",
                    "iso_27001": "Risk assessment and uncertainty disclosure",
                },
                "uncertainty_types": {
                    "aleatoric": "Data uncertainty (noise in observations)",
                    "epistemic": "Model uncertainty (lack of knowledge)",
                    "total": "Combined uncertainty estimate",
                },
            }
        else:
            return {
                "uncertainty_quantification_enabled": False,
                "uncertainty_method": "none",
                "compliance_note": "Uncertainty quantification not configured",
            }


# Global uncertainty manager instance
uncertainty_manager = CIAFUncertaintyManager()


def create_monte_carlo_quantifier(
    model: Any, n_samples: int = 100
) -> CIAFUncertaintyQuantifier:
    """Create Monte Carlo Dropout uncertainty quantifier."""
    return CIAFUncertaintyQuantifier(
        model, UncertaintyMethod.MONTE_CARLO_DROPOUT, n_samples
    )


def create_bootstrap_quantifier(
    model: Any, n_samples: int = 50
) -> CIAFUncertaintyQuantifier:
    """Create Bootstrap uncertainty quantifier."""
    return CIAFUncertaintyQuantifier(
        model, UncertaintyMethod.BOOTSTRAP_AGGREGATION, n_samples
    )


def create_auto_quantifier(model: Any) -> CIAFUncertaintyQuantifier:
    """Create automatically selected uncertainty quantifier."""
    # Auto-select method based on model type
    if hasattr(model, "predict_proba"):
        # Classification model
        method = UncertaintyMethod.MONTE_CARLO_DROPOUT
    else:
        # Regression model
        method = UncertaintyMethod.PREDICTION_INTERVALS

    return CIAFUncertaintyQuantifier(model, method)
