"""
Explainability Module for CIAF

This module provides explainable AI capabilities including SHAP values,
LIME explanations, and feature attribution for regulatory compliance
with EU AI Act and NIST AI RMF transparency requirements.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    from lime.lime_text import LimeTextExplainer

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")


class ExplanationMethod:
    """Enumeration of explanation methods."""

    SHAP_TREE = "shap_tree"
    SHAP_LINEAR = "shap_linear"
    SHAP_KERNEL = "shap_kernel"
    LIME_TEXT = "lime_text"
    LIME_TABULAR = "lime_tabular"
    FEATURE_IMPORTANCE = "feature_importance"
    GRADIENT_BASED = "gradient_based"


class CIAFExplainer:
    """Base explainer class for CIAF framework."""

    def __init__(
        self,
        model: Any,
        method: str = ExplanationMethod.SHAP_KERNEL,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize explainer.

        Args:
            model: The ML model to explain
            method: Explanation method to use
            feature_names: Names of features for interpretation
        """
        self.model = model
        self.method = method
        self.feature_names = feature_names or []
        self.explainer = None
        self.is_fitted = False

    def fit(
        self, X_train: np.ndarray, training_data: Optional[List[Dict]] = None
    ) -> "CIAFExplainer":
        """Fit the explainer on training data."""
        try:
            if self.method == ExplanationMethod.SHAP_TREE and SHAP_AVAILABLE:
                self.explainer = shap.TreeExplainer(self.model)
                self.is_fitted = True

            elif self.method == ExplanationMethod.SHAP_LINEAR and SHAP_AVAILABLE:
                self.explainer = shap.LinearExplainer(self.model, X_train)
                self.is_fitted = True

            elif self.method == ExplanationMethod.SHAP_KERNEL and SHAP_AVAILABLE:
                # Use subset of training data for kernel explainer (performance)
                background = X_train[: min(100, len(X_train))]
                self.explainer = shap.KernelExplainer(self.model.predict, background)
                self.is_fitted = True

            elif self.method == ExplanationMethod.LIME_TABULAR and LIME_AVAILABLE:
                self.explainer = LimeTabularExplainer(
                    X_train,
                    feature_names=self.feature_names,
                    class_names=["negative", "positive"],
                    mode="classification",
                )
                self.is_fitted = True

            elif self.method == ExplanationMethod.FEATURE_IMPORTANCE:
                # Use model's built-in feature importance if available
                if hasattr(self.model, "feature_importances_"):
                    self.explainer = "feature_importance"
                    self.is_fitted = True
                else:
                    warnings.warn("Model does not have feature_importances_ attribute")

            else:
                warnings.warn(
                    f"Explanation method {self.method} not available or not supported"
                )

        except Exception as e:
            warnings.warn(f"Failed to initialize explainer: {e}")

        return self

    def explain(self, X: np.ndarray, max_features: int = 10) -> Dict[str, Any]:
        """
        Generate explanation for prediction.

        Args:
            X: Input data to explain
            max_features: Maximum number of features in explanation

        Returns:
            Dictionary containing explanation metadata
        """
        if not self.is_fitted:
            return self._fallback_explanation(X)

        try:
            if self.method.startswith("shap") and SHAP_AVAILABLE:
                return self._shap_explain(X, max_features)
            elif self.method.startswith("lime") and LIME_AVAILABLE:
                return self._lime_explain(X, max_features)
            elif self.method == ExplanationMethod.FEATURE_IMPORTANCE:
                return self._feature_importance_explain(max_features)
            else:
                return self._fallback_explanation(X)

        except Exception as e:
            warnings.warn(f"Explanation generation failed: {e}")
            return self._fallback_explanation(X)

    def _shap_explain(self, X: np.ndarray, max_features: int) -> Dict[str, Any]:
        """Generate SHAP explanation."""
        shap_values = self.explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-class case - use positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Ensure we have a 1D array for single prediction
        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        # Get top features
        feature_importance = [(i, abs(val)) for i, val in enumerate(shap_values)]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_importance[:max_features]

        explanation = {
            "method": "SHAP",
            "shap_values": shap_values.tolist(),
            "feature_attributions": [
                {
                    "feature_index": idx,
                    "feature_name": (
                        self.feature_names[idx]
                        if idx < len(self.feature_names)
                        else f"feature_{idx}"
                    ),
                    "attribution_value": float(shap_values[idx]),
                    "importance_rank": rank + 1,
                }
                for rank, (idx, _) in enumerate(top_features)
            ],
            "total_attribution": float(np.sum(shap_values)),
            "explanation_confidence": 0.95,  # SHAP is generally high confidence
        }

        return explanation

    def _lime_explain(self, X: np.ndarray, max_features: int) -> Dict[str, Any]:
        """Generate LIME explanation."""
        # LIME expects single instance
        if X.ndim > 1:
            X_instance = X[0]
        else:
            X_instance = X

        lime_explanation = self.explainer.explain_instance(
            X_instance,
            (
                self.model.predict_proba
                if hasattr(self.model, "predict_proba")
                else self.model.predict
            ),
            num_features=max_features,
        )

        feature_attributions = []
        for feature_idx, importance in lime_explanation.as_list():
            feature_name = (
                self.feature_names[feature_idx]
                if feature_idx < len(self.feature_names)
                else f"feature_{feature_idx}"
            )
            feature_attributions.append(
                {
                    "feature_index": feature_idx,
                    "feature_name": feature_name,
                    "attribution_value": float(importance),
                    "importance_rank": len(feature_attributions) + 1,
                }
            )

        explanation = {
            "method": "LIME",
            "feature_attributions": feature_attributions,
            "explanation_confidence": float(lime_explanation.score),
            "local_prediction": lime_explanation.predict_proba,
        }

        return explanation

    def _feature_importance_explain(self, max_features: int) -> Dict[str, Any]:
        """Generate explanation using model's feature importance."""
        importances = self.model.feature_importances_

        # Get top features
        feature_importance = [(i, val) for i, val in enumerate(importances)]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_importance[:max_features]

        explanation = {
            "method": "Feature Importance",
            "feature_attributions": [
                {
                    "feature_index": idx,
                    "feature_name": (
                        self.feature_names[idx]
                        if idx < len(self.feature_names)
                        else f"feature_{idx}"
                    ),
                    "attribution_value": float(importance),
                    "importance_rank": rank + 1,
                }
                for rank, (idx, importance) in enumerate(top_features)
            ],
            "explanation_confidence": 0.8,  # Model feature importance is moderately confident
        }

        return explanation

    def _fallback_explanation(self, X: np.ndarray) -> Dict[str, Any]:
        """Fallback explanation when explainer fails."""
        return {
            "method": "Fallback",
            "explanation": "Explainability not available for this model",
            "feature_attributions": [],
            "explanation_confidence": 0.0,
            "note": "Install SHAP or LIME for detailed explanations",
        }


class CIAFExplainabilityManager:
    """Manager for explainability across different model types."""

    def __init__(self):
        """Initialize explainability manager."""
        self.explainers: Dict[str, CIAFExplainer] = {}

    def register_explainer(
        self,
        model_id: str,
        model: Any,
        method: str = ExplanationMethod.SHAP_KERNEL,
        feature_names: Optional[List[str]] = None,
    ) -> CIAFExplainer:
        """Register an explainer for a model."""
        explainer = CIAFExplainer(model, method, feature_names)
        self.explainers[model_id] = explainer
        return explainer

    def fit_explainer(
        self,
        model_id: str,
        X_train: np.ndarray,
        training_data: Optional[List[Dict]] = None,
    ) -> bool:
        """Fit an explainer for a model."""
        if model_id in self.explainers:
            self.explainers[model_id].fit(X_train, training_data)
            return self.explainers[model_id].is_fitted
        return False

    def explain_prediction(
        self, model_id: str, X: np.ndarray, prediction: Any, max_features: int = 10
    ) -> Dict[str, Any]:
        """Generate explanation for a prediction."""
        base_explanation = {
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": prediction,
            "input_shape": (
                X.shape
                if hasattr(X, "shape")
                else len(X) if hasattr(X, "__len__") else 1
            ),
        }

        if model_id in self.explainers:
            explanation = self.explainers[model_id].explain(X, max_features)
            base_explanation.update(explanation)
        else:
            base_explanation.update(
                {
                    "method": "None",
                    "explanation": "No explainer registered for this model",
                    "feature_attributions": [],
                    "explanation_confidence": 0.0,
                }
            )

        return base_explanation

    def get_explainability_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get explainability metadata for compliance reporting."""
        if model_id in self.explainers:
            explainer = self.explainers[model_id]
            return {
                "explainability_enabled": explainer.is_fitted,
                "explanation_method": explainer.method,
                "feature_count": len(explainer.feature_names),
                "compliance_frameworks": {
                    "eu_ai_act": "Article 13 - Transparency obligations",
                    "nist_ai_rmf": "Explain function - Understanding AI decisions",
                    "gdpr": "Article 22 - Right to explanation",
                },
            }
        else:
            return {
                "explainability_enabled": False,
                "explanation_method": "none",
                "compliance_note": "Explainability not configured",
            }


# Global explainability manager instance
explainability_manager = CIAFExplainabilityManager()


def create_shap_explainer(
    model: Any, feature_names: Optional[List[str]] = None
) -> CIAFExplainer:
    """Create SHAP explainer for tree-based models."""
    return CIAFExplainer(model, ExplanationMethod.SHAP_TREE, feature_names)


def create_lime_explainer(
    model: Any, feature_names: Optional[List[str]] = None
) -> CIAFExplainer:
    """Create LIME explainer for tabular data."""
    return CIAFExplainer(model, ExplanationMethod.LIME_TABULAR, feature_names)


def create_auto_explainer(
    model: Any, feature_names: Optional[List[str]] = None
) -> CIAFExplainer:
    """Create automatically selected explainer based on model type."""
    # Auto-detect best explainer method
    if hasattr(model, "feature_importances_"):
        # Tree-based model
        method = (
            ExplanationMethod.SHAP_TREE
            if SHAP_AVAILABLE
            else ExplanationMethod.FEATURE_IMPORTANCE
        )
    elif hasattr(model, "coef_"):
        # Linear model
        method = (
            ExplanationMethod.SHAP_LINEAR
            if SHAP_AVAILABLE
            else ExplanationMethod.FEATURE_IMPORTANCE
        )
    else:
        # General model
        method = (
            ExplanationMethod.SHAP_KERNEL
            if SHAP_AVAILABLE
            else ExplanationMethod.LIME_TABULAR
        )

    return CIAFExplainer(model, method, feature_names)
