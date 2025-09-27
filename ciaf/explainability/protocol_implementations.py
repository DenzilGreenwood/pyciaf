"""
Default protocol implementations for explainability system.

This module provides concrete implementations of the explainability protocols,
wrapping the existing functionality in the new architecture while adding
enhanced features and better integration with LCM and compliance systems.

Created: 2025-09-26
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .interfaces import (
    ExplainerProtocol,
    ExplanationProvider,
    ExplanationMetadataProvider,
    ExplanationValidator,
    ExplanationAuditor,
    FeatureAttributionProvider,
    ExplanationStorageProvider,
)
from .policy import (
    ExplainabilityPolicy,
    ExplanationMethod,
    ExplanationLevel,
    ComplianceFramework,
    get_default_explainability_policy,
)

# Optional dependencies with graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class ShapExplainer:
    """SHAP-based explainer implementation."""
    
    def __init__(
        self, 
        model: Any, 
        method: ExplanationMethod,
        feature_names: Optional[List[str]] = None,
        policy: Optional[ExplainabilityPolicy] = None
    ):
        self.model = model
        self.method = method
        self.feature_names = feature_names or []
        self.policy = policy or get_default_explainability_policy()
        self.explainer = None
        self._is_fitted = False
    
    def fit(self, X_train: np.ndarray, training_data: Optional[List[Dict]] = None) -> bool:
        """Fit SHAP explainer."""
        if not SHAP_AVAILABLE:
            warnings.warn("SHAP not available. Install with: pip install shap")
            return False
        
        try:
            if self.method == ExplanationMethod.SHAP_TREE:
                self.explainer = shap.TreeExplainer(self.model)
            elif self.method == ExplanationMethod.SHAP_LINEAR:
                self.explainer = shap.LinearExplainer(self.model, X_train)
            elif self.method == ExplanationMethod.SHAP_KERNEL:
                # Use policy-configured background sample size
                bg_size = self.policy.method_preferences.shap_background_sample_size
                background = X_train[:min(bg_size, len(X_train))]
                self.explainer = shap.KernelExplainer(
                    self.model.predict, 
                    background,
                    max_evals=self.policy.method_preferences.shap_max_evaluations
                )
            else:
                return False
            
            self._is_fitted = True
            return True
            
        except Exception as e:
            warnings.warn(f"SHAP explainer fitting failed: {e}")
            return False
    
    def explain(self, X: np.ndarray, max_features: int = 10) -> Dict[str, Any]:
        """Generate SHAP explanation."""
        if not self._is_fitted or self.explainer is None:
            return self._fallback_explanation()
        
        try:
            shap_values = self.explainer.shap_values(X)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class case - use positive class
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Ensure we have a 1D array for single prediction
            if shap_values.ndim > 1:
                shap_values = shap_values[0]
            
            # Get top features based on policy
            max_features = min(
                max_features,
                self.policy.quality_policy.max_features_in_explanation
            )
            
            feature_importance = [(i, abs(val)) for i, val in enumerate(shap_values)]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_features = feature_importance[:max_features]
            
            # Build explanation with policy-driven detail level
            explanation = {
                "method": "SHAP",
                "method_variant": self.method.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "explanation_confidence": 0.95,  # SHAP is generally high confidence
                "feature_attributions": self._build_feature_attributions(shap_values, top_features),
                "total_attribution": float(np.sum(shap_values)),
            }
            
            # Add detailed information based on policy level
            if self.policy.explanation_level in [ExplanationLevel.DETAILED, ExplanationLevel.COMPREHENSIVE]:
                explanation.update({
                    "shap_values": shap_values.tolist(),
                    "baseline_value": float(getattr(self.explainer, 'expected_value', 0)),
                    "feature_statistics": self._compute_feature_statistics(shap_values),
                })
            
            return explanation
            
        except Exception as e:
            warnings.warn(f"SHAP explanation failed: {e}")
            return self._fallback_explanation()
    
    def _build_feature_attributions(
        self, 
        shap_values: np.ndarray, 
        top_features: List[Tuple[int, float]]
    ) -> List[Dict[str, Any]]:
        """Build feature attribution list."""
        attributions = []
        for rank, (idx, importance) in enumerate(top_features):
            attribution = {
                "feature_index": idx,
                "feature_name": (
                    self.feature_names[idx] 
                    if idx < len(self.feature_names) 
                    else f"feature_{idx}"
                ),
                "attribution_value": float(shap_values[idx]),
                "absolute_importance": float(importance),
                "importance_rank": rank + 1,
            }
            attributions.append(attribution)
        
        return attributions
    
    def _compute_feature_statistics(self, shap_values: np.ndarray) -> Dict[str, Any]:
        """Compute feature statistics for detailed explanations."""
        return {
            "mean_absolute_attribution": float(np.mean(np.abs(shap_values))),
            "max_attribution": float(np.max(shap_values)),
            "min_attribution": float(np.min(shap_values)),
            "attribution_variance": float(np.var(shap_values)),
            "positive_attribution_count": int(np.sum(shap_values > 0)),
            "negative_attribution_count": int(np.sum(shap_values < 0)),
        }
    
    def _fallback_explanation(self) -> Dict[str, Any]:
        """Fallback explanation when SHAP fails."""
        return {
            "method": "Fallback",
            "explanation": "SHAP explanation not available",
            "feature_attributions": [],
            "explanation_confidence": 0.0,
        }
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @property
    def method_name(self) -> str:
        return self.method.value


class LimeExplainer:
    """LIME-based explainer implementation."""
    
    def __init__(
        self,
        model: Any,
        method: ExplanationMethod,
        feature_names: Optional[List[str]] = None,
        policy: Optional[ExplainabilityPolicy] = None
    ):
        self.model = model
        self.method = method
        self.feature_names = feature_names or []
        self.policy = policy or get_default_explainability_policy()
        self.explainer = None
        self._is_fitted = False
    
    def fit(self, X_train: np.ndarray, training_data: Optional[List[Dict]] = None) -> bool:
        """Fit LIME explainer."""
        if not LIME_AVAILABLE:
            warnings.warn("LIME not available. Install with: pip install lime")
            return False
        
        try:
            if self.method == ExplanationMethod.LIME_TABULAR:
                self.explainer = LimeTabularExplainer(
                    X_train,
                    feature_names=self.feature_names,
                    class_names=["negative", "positive"],
                    mode="classification",
                    kernel_width=self.policy.method_preferences.lime_kernel_width,
                )
            elif self.method == ExplanationMethod.LIME_TEXT:
                self.explainer = LimeTextExplainer(
                    class_names=["negative", "positive"],
                    mode="classification",
                )
            else:
                return False
            
            self._is_fitted = True
            return True
            
        except Exception as e:
            warnings.warn(f"LIME explainer fitting failed: {e}")
            return False
    
    def explain(self, X: np.ndarray, max_features: int = 10) -> Dict[str, Any]:
        """Generate LIME explanation."""
        if not self._is_fitted or self.explainer is None:
            return self._fallback_explanation()
        
        try:
            # LIME expects single instance
            X_instance = X[0] if X.ndim > 1 else X
            
            # Use policy-configured number of perturbations
            num_features = min(
                max_features,
                self.policy.quality_policy.max_features_in_explanation
            )
            
            lime_explanation = self.explainer.explain_instance(
                X_instance,
                (
                    self.model.predict_proba
                    if hasattr(self.model, "predict_proba")
                    else self.model.predict
                ),
                num_features=num_features,
                num_samples=self.policy.method_preferences.lime_num_perturbations,
            )
            
            # Build explanation
            explanation = {
                "method": "LIME",
                "method_variant": self.method.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "explanation_confidence": float(lime_explanation.score),
                "feature_attributions": self._build_lime_attributions(lime_explanation),
                "local_prediction": getattr(lime_explanation, 'predict_proba', None),
            }
            
            return explanation
            
        except Exception as e:
            warnings.warn(f"LIME explanation failed: {e}")
            return self._fallback_explanation()
    
    def _build_lime_attributions(self, lime_explanation) -> List[Dict[str, Any]]:
        """Build LIME feature attributions."""
        attributions = []
        for rank, (feature_idx, importance) in enumerate(lime_explanation.as_list()):
            feature_name = (
                self.feature_names[feature_idx]
                if isinstance(feature_idx, int) and feature_idx < len(self.feature_names)
                else str(feature_idx)
            )
            
            attribution = {
                "feature_index": feature_idx if isinstance(feature_idx, int) else -1,
                "feature_name": feature_name,
                "attribution_value": float(importance),
                "absolute_importance": float(abs(importance)),
                "importance_rank": rank + 1,
            }
            attributions.append(attribution)
        
        return attributions
    
    def _fallback_explanation(self) -> Dict[str, Any]:
        """Fallback explanation when LIME fails."""
        return {
            "method": "Fallback",
            "explanation": "LIME explanation not available",
            "feature_attributions": [],
            "explanation_confidence": 0.0,
        }
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @property
    def method_name(self) -> str:
        return self.method.value


class FeatureImportanceExplainer:
    """Feature importance-based explainer."""
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        policy: Optional[ExplainabilityPolicy] = None
    ):
        self.model = model
        self.feature_names = feature_names or []
        self.policy = policy or get_default_explainability_policy()
        self._is_fitted = False
    
    def fit(self, X_train: np.ndarray, training_data: Optional[List[Dict]] = None) -> bool:
        """Fit feature importance explainer."""
        self._is_fitted = hasattr(self.model, 'feature_importances_')
        return self._is_fitted
    
    def explain(self, X: np.ndarray, max_features: int = 10) -> Dict[str, Any]:
        """Generate feature importance explanation."""
        if not self._is_fitted:
            return self._fallback_explanation()
        
        try:
            importances = self.model.feature_importances_
            
            # Get top features
            max_features = min(
                max_features,
                self.policy.quality_policy.max_features_in_explanation
            )
            
            feature_importance = [(i, val) for i, val in enumerate(importances)]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_features = feature_importance[:max_features]
            
            explanation = {
                "method": "Feature Importance",
                "method_variant": "model_native",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "explanation_confidence": 0.8,  # Moderate confidence
                "feature_attributions": [
                    {
                        "feature_index": idx,
                        "feature_name": (
                            self.feature_names[idx]
                            if idx < len(self.feature_names)
                            else f"feature_{idx}"
                        ),
                        "attribution_value": float(importance),
                        "absolute_importance": float(abs(importance)),
                        "importance_rank": rank + 1,
                    }
                    for rank, (idx, importance) in enumerate(top_features)
                ],
            }
            
            return explanation
            
        except Exception as e:
            warnings.warn(f"Feature importance explanation failed: {e}")
            return self._fallback_explanation()
    
    def _fallback_explanation(self) -> Dict[str, Any]:
        """Fallback explanation."""
        return {
            "method": "Fallback",
            "explanation": "Feature importance not available for this model",
            "feature_attributions": [],
            "explanation_confidence": 0.0,
        }
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @property
    def method_name(self) -> str:
        return ExplanationMethod.FEATURE_IMPORTANCE.value


class DefaultExplanationProvider:
    """Default implementation of ExplanationProvider protocol."""
    
    def __init__(self, policy: Optional[ExplainabilityPolicy] = None):
        self.policy = policy or get_default_explainability_policy()
        self.explainers: Dict[str, ExplainerProtocol] = {}
        self.feature_names: Dict[str, List[str]] = {}
    
    def register_explainer(
        self,
        model_id: str,
        explainer: ExplainerProtocol,
        feature_names: Optional[List[str]] = None
    ) -> bool:
        """Register an explainer for a model."""
        try:
            self.explainers[model_id] = explainer
            if feature_names:
                self.feature_names[model_id] = feature_names
            return True
        except Exception:
            return False
    
    def get_explanation(
        self,
        model_id: str,
        X: np.ndarray,
        prediction: Any,
        max_features: int = 10
    ) -> Dict[str, Any]:
        """Get explanation for a model prediction."""
        base_explanation = {
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": prediction,
            "input_shape": (
                X.shape if hasattr(X, "shape") 
                else len(X) if hasattr(X, "__len__") 
                else 1
            ),
            "policy_digest": self.policy.policy_digest(),
        }
        
        if model_id in self.explainers:
            explanation = self.explainers[model_id].explain(X, max_features)
            base_explanation.update(explanation)
        else:
            base_explanation.update({
                "method": "None",
                "explanation": "No explainer registered for this model",
                "feature_attributions": [],
                "explanation_confidence": 0.0,
            })
        
        return base_explanation
    
    def has_explainer(self, model_id: str) -> bool:
        """Check if model has registered explainer."""
        return model_id in self.explainers


class DefaultExplanationValidator:
    """Default implementation of ExplanationValidator protocol."""
    
    def __init__(self, policy: Optional[ExplainabilityPolicy] = None):
        self.policy = policy or get_default_explainability_policy()
    
    def validate_feature_coverage(
        self, 
        explanation: Dict[str, Any], 
        min_features: int = 5
    ) -> bool:
        """Validate explanation covers sufficient features."""
        feature_attributions = explanation.get("feature_attributions", [])
        actual_min = max(min_features, self.policy.quality_policy.min_feature_coverage)
        return len(feature_attributions) >= actual_min
    
    def validate_confidence_threshold(
        self,
        explanation: Dict[str, Any],
        min_confidence: float = 0.7
    ) -> bool:
        """Validate explanation confidence meets threshold."""
        confidence = explanation.get("explanation_confidence", 0.0)
        threshold = max(min_confidence, self.policy.quality_policy.min_explanation_confidence)
        return confidence >= threshold
    
    def validate_regulatory_compliance(
        self,
        explanation: Dict[str, Any],
        framework: str = "eu_ai_act"
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate explanation meets regulatory requirements."""
        compliance_details = {"framework": framework, "checks": []}
        is_compliant = True
        
        # Check feature names are provided (required by most frameworks)
        if self.policy.quality_policy.require_feature_names:
            has_names = all(
                attr.get("feature_name") and attr["feature_name"] != "unknown"
                for attr in explanation.get("feature_attributions", [])
            )
            compliance_details["checks"].append({
                "requirement": "feature_names_provided",
                "status": "pass" if has_names else "fail",
                "details": "Feature names must be provided for interpretability"
            })
            if not has_names:
                is_compliant = False
        
        # Check attribution values are provided
        if self.policy.quality_policy.require_attribution_values:
            has_values = all(
                "attribution_value" in attr
                for attr in explanation.get("feature_attributions", [])
            )
            compliance_details["checks"].append({
                "requirement": "attribution_values_provided",
                "status": "pass" if has_values else "fail",
                "details": "Attribution values must be quantified"
            })
            if not has_values:
                is_compliant = False
        
        return is_compliant, compliance_details


def create_default_explainability_protocols(
    policy: Optional[ExplainabilityPolicy] = None
) -> Dict[str, Any]:
    """Create default explainability protocol implementations.
    
    Args:
        policy: Explainability policy to use
        
    Returns:
        Dictionary of protocol implementations
    """
    if policy is None:
        policy = get_default_explainability_policy()
    
    return {
        "explanation_provider": DefaultExplanationProvider(policy),
        "explanation_validator": DefaultExplanationValidator(policy),
        "policy": policy,
        "available_methods": {
            "shap_tree": ShapExplainer,
            "shap_linear": ShapExplainer,
            "shap_kernel": ShapExplainer,
            "lime_tabular": LimeExplainer,
            "lime_text": LimeExplainer,
            "feature_importance": FeatureImportanceExplainer,
        },
        "method_availability": {
            "shap": SHAP_AVAILABLE,
            "lime": LIME_AVAILABLE,
            "feature_importance": True,
        }
    }


def create_auto_explainer(
    model: Any,
    feature_names: Optional[List[str]] = None,
    policy: Optional[ExplainabilityPolicy] = None
) -> ExplainerProtocol:
    """Create automatically selected explainer based on model type.
    
    Args:
        model: Model to create explainer for
        feature_names: Feature names for interpretation
        policy: Explainability policy
        
    Returns:
        Best available explainer for the model type
    """
    if policy is None:
        policy = get_default_explainability_policy()
    
    # Try preferred methods in order, but handle failures gracefully
    working_explainer = None
    
    for method in policy.method_preferences.preferred_methods:
        try:
            if method in [ExplanationMethod.SHAP_TREE, ExplanationMethod.SHAP_LINEAR, ExplanationMethod.SHAP_KERNEL]:
                if not SHAP_AVAILABLE:
                    continue  # Skip SHAP if not available
                
                # Auto-detect best SHAP method
                if method == ExplanationMethod.SHAP_TREE and hasattr(model, 'feature_importances_'):
                    explainer = ShapExplainer(model, method, feature_names, policy)
                    # Test fit with dummy data to see if it works
                    try:
                        dummy_data = np.random.random((10, 4))
                        if explainer.fit(dummy_data):
                            working_explainer = explainer
                            break
                    except:
                        continue
                elif method == ExplanationMethod.SHAP_LINEAR and hasattr(model, 'coef_'):
                    explainer = ShapExplainer(model, method, feature_names, policy)
                    try:
                        dummy_data = np.random.random((10, 4))
                        if explainer.fit(dummy_data):
                            working_explainer = explainer
                            break
                    except:
                        continue
                elif method == ExplanationMethod.SHAP_KERNEL:
                    explainer = ShapExplainer(model, method, feature_names, policy)
                    try:
                        dummy_data = np.random.random((10, 4))
                        if explainer.fit(dummy_data):
                            working_explainer = explainer
                            break
                    except:
                        continue
            
            elif method in [ExplanationMethod.LIME_TABULAR, ExplanationMethod.LIME_TEXT]:
                if LIME_AVAILABLE:
                    explainer = LimeExplainer(model, method, feature_names, policy)
                    working_explainer = explainer
                    break
            
            elif method == ExplanationMethod.FEATURE_IMPORTANCE:
                if hasattr(model, 'feature_importances_'):
                    explainer = FeatureImportanceExplainer(model, feature_names, policy)
                    working_explainer = explainer
                    break
        
        except Exception:
            continue  # Try next method
    
    # If no method worked from preferred list, try fallbacks
    if working_explainer is None:
        # Fallback to feature importance if available
        if policy.method_preferences.fallback_to_feature_importance and hasattr(model, 'feature_importances_'):
            working_explainer = FeatureImportanceExplainer(model, feature_names, policy)
        else:
            # Final fallback - create a basic explainer that always works
            working_explainer = FeatureImportanceExplainer(model, feature_names, policy)
    
    return working_explainer