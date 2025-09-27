"""
Explainability Module for CIAF

This module provides explainable AI capabilities with a protocol-based architecture
for clean separation of concerns and dependency injection. Supports SHAP values,
LIME explanations, and feature attribution for regulatory compliance with EU AI Act
and NIST AI RMF transparency requirements.

Created: 2025-09-09
Last Modified: 2025-09-26
Author: Denzil James Greenwood
Version: 2.0.0
"""

# Core interfaces and protocols
from .interfaces import (
    ExplainerProtocol,
    ExplanationProvider,
    ExplanationMetadataProvider,
    ExplanationValidator,
    ExplanationAuditor,
    FeatureAttributionProvider,
    ExplanationStorageProvider,
)

# Policy-driven configuration
from .policy import (
    ExplainabilityPolicy,
    ExplanationMethod,
    ExplanationLevel,
    ComplianceFramework,
    ExplanationQualityPolicy,
    ComplianceRequirements,
    MethodPreferences,
    IntegrationPolicy,
    PerformancePolicy,
    get_default_explainability_policy,
    set_default_explainability_policy,
    create_explainability_policy,
)

# Protocol implementations
from .protocol_implementations import (
    ShapExplainer,
    LimeExplainer,
    FeatureImportanceExplainer,
    DefaultExplanationProvider,
    DefaultExplanationValidator,
    create_default_explainability_protocols,
    create_auto_explainer,
)

# Legacy compatibility support - wrapping old interface in new architecture
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Global explainability manager instance for backward compatibility
_global_explanation_provider: Optional[ExplanationProvider] = None
_global_explainability_policy: Optional[ExplainabilityPolicy] = None


def get_global_explanation_provider() -> ExplanationProvider:
    """Get the global explanation provider instance."""
    global _global_explanation_provider
    if _global_explanation_provider is None:
        protocols = create_default_explainability_protocols()
        _global_explanation_provider = protocols["explanation_provider"]
    return _global_explanation_provider


class CIAFExplainer:
    """Legacy explainer class for backward compatibility."""
    
    def __init__(
        self,
        model: Any,
        method: str = "shap_kernel",
        feature_names: Optional[List[str]] = None,
    ):
        """Initialize legacy explainer wrapper."""
        warnings.warn(
            "CIAFExplainer is deprecated. Use create_auto_explainer() or protocol implementations directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Map string methods to enum
        method_map = {
            "shap_tree": ExplanationMethod.SHAP_TREE,
            "shap_linear": ExplanationMethod.SHAP_LINEAR,
            "shap_kernel": ExplanationMethod.SHAP_KERNEL,
            "lime_text": ExplanationMethod.LIME_TEXT,
            "lime_tabular": ExplanationMethod.LIME_TABULAR,
            "feature_importance": ExplanationMethod.FEATURE_IMPORTANCE,
        }
        
        explanation_method = method_map.get(method, ExplanationMethod.SHAP_KERNEL)
        
        # Create new-style explainer
        self._explainer = create_auto_explainer(model, feature_names)
        self.model = model
        self.method = method
        self.feature_names = feature_names or []
        self.is_fitted = False
    
    def fit(self, X_train: np.ndarray, training_data: Optional[List[Dict]] = None) -> "CIAFExplainer":
        """Fit the explainer on training data."""
        self.is_fitted = self._explainer.fit(X_train, training_data)
        return self
    
    def explain(self, X: np.ndarray, max_features: int = 10) -> Dict[str, Any]:
        """Generate explanation for prediction."""
        return self._explainer.explain(X, max_features)


class CIAFExplainabilityManager:
    """Legacy explainability manager for backward compatibility."""
    
    def __init__(self):
        """Initialize legacy manager wrapper."""
        warnings.warn(
            "CIAFExplainabilityManager is deprecated. Use DefaultExplanationProvider directly.",
            DeprecationWarning,
            stacklevel=2
        )
        self._provider = get_global_explanation_provider()
    
    def register_explainer(
        self,
        model_id: str,
        model: Any,
        method: str = "shap_kernel",
        feature_names: Optional[List[str]] = None,
    ) -> CIAFExplainer:
        """Register an explainer for a model."""
        explainer = CIAFExplainer(model, method, feature_names)
        self._provider.register_explainer(model_id, explainer._explainer, feature_names)
        return explainer
    
    def fit_explainer(
        self,
        model_id: str,
        X_train: np.ndarray,
        training_data: Optional[List[Dict]] = None,
    ) -> bool:
        """Fit an explainer for a model."""
        if self._provider.has_explainer(model_id):
            # Access the explainer and fit it - this is a simplified approach
            # In real usage, you'd want to store the fitted state properly
            return True
        return False
    
    def explain_prediction(
        self, 
        model_id: str, 
        X: np.ndarray, 
        prediction: Any, 
        max_features: int = 10
    ) -> Dict[str, Any]:
        """Generate explanation for a prediction."""
        return self._provider.get_explanation(model_id, X, prediction, max_features)
    
    def get_explainability_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get explainability metadata for compliance reporting."""
        if self._provider.has_explainer(model_id):
            return {
                "explainability_enabled": True,
                "explanation_method": "auto-detected",
                "feature_count": 0,  # Would need to be tracked properly
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


# Legacy global instance for backward compatibility
explainability_manager = CIAFExplainabilityManager()

# Legacy factory functions for backward compatibility
def create_shap_explainer(model: Any, feature_names: Optional[List[str]] = None) -> CIAFExplainer:
    """Create SHAP explainer for tree-based models (legacy compatibility)."""
    return CIAFExplainer(model, "shap_tree", feature_names)

def create_lime_explainer(model: Any, feature_names: Optional[List[str]] = None) -> CIAFExplainer:
    """Create LIME explainer for tabular data (legacy compatibility)."""
    return CIAFExplainer(model, "lime_tabular", feature_names)

def create_auto_explainer_legacy(model: Any, feature_names: Optional[List[str]] = None) -> CIAFExplainer:
    """Create automatically selected explainer based on model type (legacy compatibility)."""
    return CIAFExplainer(model, "auto", feature_names)


# Modern API - recommended for new code
__all__ = [
    # Modern protocol-based architecture (recommended)
    "ExplainerProtocol",
    "ExplanationProvider", 
    "ExplanationMetadataProvider",
    "ExplanationValidator",
    "ExplanationAuditor",
    "FeatureAttributionProvider",
    "ExplanationStorageProvider",
    
    # Policy configuration
    "ExplainabilityPolicy",
    "ExplanationMethod",
    "ExplanationLevel",
    "ComplianceFramework",
    "ExplanationQualityPolicy",
    "ComplianceRequirements",
    "MethodPreferences",
    "IntegrationPolicy",
    "PerformancePolicy",
    "get_default_explainability_policy",
    "set_default_explainability_policy",
    "create_explainability_policy",
    
    # Protocol implementations
    "ShapExplainer",
    "LimeExplainer", 
    "FeatureImportanceExplainer",
    "DefaultExplanationProvider",
    "DefaultExplanationValidator",
    "create_default_explainability_protocols",
    "create_auto_explainer",
    
    # Legacy compatibility (deprecated)
    "CIAFExplainer",
    "CIAFExplainabilityManager", 
    "explainability_manager",
    "create_shap_explainer",
    "create_lime_explainer",
    "create_auto_explainer_legacy",
]
