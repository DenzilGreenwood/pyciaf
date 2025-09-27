"""
Protocol-based interfaces for explainability system.

This module defines the core protocols for explainable AI functionality,
following the same architectural patterns as the core and compliance modules.
These protocols enable dependency injection and clean separation of concerns.

Created: 2025-09-26
Author: Denzil James Greenwood  
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
from abc import abstractmethod
import numpy as np


class ExplainerProtocol(Protocol):
    """Protocol for explanation method implementations."""
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, training_data: Optional[List[Dict]] = None) -> bool:
        """Fit the explainer on training data.
        
        Args:
            X_train: Training data features
            training_data: Optional additional training metadata
            
        Returns:
            True if fitting was successful
        """
        ...
    
    @abstractmethod
    def explain(self, X: np.ndarray, max_features: int = 10) -> Dict[str, Any]:
        """Generate explanation for prediction.
        
        Args:
            X: Input data to explain
            max_features: Maximum number of features in explanation
            
        Returns:
            Dictionary containing explanation results
        """
        ...
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if explainer is fitted and ready to use."""
        ...
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Get the name of this explanation method."""
        ...


class ExplanationProvider(Protocol):
    """Protocol for managing explainability across models."""
    
    @abstractmethod
    def register_explainer(
        self, 
        model_id: str, 
        explainer: ExplainerProtocol,
        feature_names: Optional[List[str]] = None
    ) -> bool:
        """Register an explainer for a model.
        
        Args:
            model_id: Unique identifier for the model
            explainer: Explainer implementation to register
            feature_names: Optional feature names for interpretation
            
        Returns:
            True if registration was successful
        """
        ...
    
    @abstractmethod
    def get_explanation(
        self, 
        model_id: str, 
        X: np.ndarray, 
        prediction: Any,
        max_features: int = 10
    ) -> Dict[str, Any]:
        """Get explanation for a model prediction.
        
        Args:
            model_id: Model to explain
            X: Input data
            prediction: Model prediction to explain
            max_features: Maximum features to include
            
        Returns:
            Structured explanation data
        """
        ...
    
    @abstractmethod
    def has_explainer(self, model_id: str) -> bool:
        """Check if model has registered explainer."""
        ...


class ExplanationMetadataProvider(Protocol):
    """Protocol for explainability compliance metadata."""
    
    @abstractmethod
    def get_compliance_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get compliance metadata for explainability.
        
        Args:
            model_id: Model to get metadata for
            
        Returns:
            Compliance metadata including framework mappings
        """
        ...
    
    @abstractmethod
    def get_explanation_capabilities(self, model_id: str) -> Dict[str, Any]:
        """Get explainability capabilities for a model.
        
        Args:
            model_id: Model to check capabilities for
            
        Returns:
            Dictionary describing available explanation features
        """
        ...
    
    @abstractmethod
    def validate_explanation_quality(
        self, 
        explanation: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate explanation meets quality standards.
        
        Args:
            explanation: Explanation to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        ...


class ExplanationValidator(Protocol):
    """Protocol for validating explanation quality and compliance."""
    
    @abstractmethod
    def validate_feature_coverage(
        self,
        explanation: Dict[str, Any], 
        min_features: int = 5
    ) -> bool:
        """Validate explanation covers sufficient features.
        
        Args:
            explanation: Explanation to validate
            min_features: Minimum features required
            
        Returns:
            True if coverage is sufficient
        """
        ...
    
    @abstractmethod
    def validate_confidence_threshold(
        self,
        explanation: Dict[str, Any],
        min_confidence: float = 0.7
    ) -> bool:
        """Validate explanation confidence meets threshold.
        
        Args:
            explanation: Explanation to validate
            min_confidence: Minimum confidence required
            
        Returns:
            True if confidence is sufficient
        """
        ...
    
    @abstractmethod
    def validate_regulatory_compliance(
        self,
        explanation: Dict[str, Any],
        framework: str = "eu_ai_act"
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate explanation meets regulatory requirements.
        
        Args:
            explanation: Explanation to validate
            framework: Regulatory framework to check against
            
        Returns:
            Tuple of (is_compliant, compliance_details)
        """
        ...


class ExplanationAuditor(Protocol):
    """Protocol for auditing explainability implementations."""
    
    @abstractmethod
    def audit_explainer_performance(
        self,
        model_id: str,
        test_data: np.ndarray,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """Audit explainer performance and reliability.
        
        Args:
            model_id: Model to audit
            test_data: Test data for auditing
            num_samples: Number of samples to test
            
        Returns:
            Audit report with performance metrics
        """
        ...
    
    @abstractmethod
    def generate_explainability_report(self, model_id: str) -> Dict[str, Any]:
        """Generate comprehensive explainability report.
        
        Args:
            model_id: Model to generate report for
            
        Returns:
            Comprehensive report on explainability status
        """
        ...


class FeatureAttributionProvider(Protocol):
    """Protocol for feature attribution analysis."""
    
    @abstractmethod
    def get_global_feature_importance(
        self,
        model_id: str,
        importance_type: str = "mean_absolute"
    ) -> Dict[str, float]:
        """Get global feature importance across all predictions.
        
        Args:
            model_id: Model to analyze
            importance_type: Type of importance calculation
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        ...
    
    @abstractmethod
    def get_feature_interaction_analysis(
        self,
        model_id: str,
        X: np.ndarray,
        max_interactions: int = 10
    ) -> List[Dict[str, Any]]:
        """Analyze feature interactions in explanations.
        
        Args:
            model_id: Model to analyze
            X: Input data
            max_interactions: Maximum interactions to return
            
        Returns:
            List of feature interaction analyses
        """
        ...


class ExplanationStorageProvider(Protocol):
    """Protocol for storing and retrieving explanations."""
    
    @abstractmethod
    def store_explanation(
        self,
        model_id: str,
        prediction_id: str,
        explanation: Dict[str, Any]
    ) -> bool:
        """Store explanation for future retrieval.
        
        Args:
            model_id: Model that generated prediction
            prediction_id: Unique prediction identifier
            explanation: Explanation data to store
            
        Returns:
            True if storage was successful
        """
        ...
    
    @abstractmethod
    def retrieve_explanation(
        self,
        model_id: str,
        prediction_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve stored explanation.
        
        Args:
            model_id: Model identifier
            prediction_id: Prediction identifier
            
        Returns:
            Explanation data if found, None otherwise
        """
        ...
    
    @abstractmethod
    def get_explanation_history(
        self,
        model_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get explanation history for a model.
        
        Args:
            model_id: Model identifier
            limit: Maximum number of explanations to return
            
        Returns:
            List of historical explanations
        """
        ...