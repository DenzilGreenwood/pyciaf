"""
Preprocessing and Vectorization Module for CIAF

This module provides vectorization and preprocessing capabilities to ensure
real ML model training with proper feature extraction for text, numerical,
and mixed data types.

Enhanced with protocol-based architecture for consistency with other CIAF modules.

Created: 2025-09-09
Last Modified: 2025-09-27
Author: Denzil James Greenwood
Version: 2.0.0
"""

import warnings
from typing import Any, Dict, List, Optional, Union

# Import new protocol-based architecture
from .interfaces import (
    DataPreprocessor, DataValidator, DataTypeDetector, FeatureExtractor,
    ModelAdapter, PreprocessingPipeline, QualityMonitor,
    DataType, PreprocessingMethod, ValidationSeverity
)

from .policy import (
    PreprocessingPolicy, QualityPolicy, ProcessingPolicy, 
    PerformancePolicy, SecurityPolicy, QualityLevel, PreprocessingIntensity,
    get_default_preprocessing_policy, create_custom_policy
)

from .protocol_implementations import (
    DefaultTextPreprocessor, DefaultNumericalPreprocessor,
    DefaultDataTypeDetector, DefaultDataValidator, DefaultModelAdapter,
    create_default_preprocessing_protocols, create_text_preprocessor,
    create_numerical_preprocessor, create_auto_model_adapter
)

# Import data quality validation (existing)
from .data_quality import DataQualityValidator, ValidationResult, quick_validate, validate_ciaf_dataset

# Protocol-based factory functions
def create_preprocessor(data_type: DataType, policy: Optional[PreprocessingPolicy] = None) -> DataPreprocessor:
    """
    Create a preprocessor for specific data type.
    
    Args:
        data_type: Type of data to preprocess
        policy: Optional preprocessing policy
        
    Returns:
        Appropriate preprocessor implementation
    """
    if data_type == DataType.TEXT:
        return DefaultTextPreprocessor(policy)
    elif data_type == DataType.NUMERICAL:
        return DefaultNumericalPreprocessor(policy)
    else:
        # Default to numerical preprocessor for mixed/unknown types
        return DefaultNumericalPreprocessor(policy)


def create_auto_preprocessor(data: Union[List[Dict[str, Any]], str], 
                            policy: Optional[PreprocessingPolicy] = None) -> DataPreprocessor:
    """
    Automatically detect data type and create appropriate preprocessor.
    
    Args:
        data: Sample data for type detection
        policy: Optional preprocessing policy
        
    Returns:
        Automatically selected preprocessor
    """
    detector = DefaultDataTypeDetector()
    data_type = detector.detect_data_type(data)
    return create_preprocessor(data_type, policy)


def validate_data(data: Union[List[Dict[str, Any]], Any], 
                 policy: Optional[PreprocessingPolicy] = None) -> Dict[str, Any]:
    """
    Validate data quality using policy-driven validation.
    
    Args:
        data: Data to validate
        policy: Optional validation policy
        
    Returns:
        Validation results
    """
    validator = DefaultDataValidator(policy)
    return validator.validate(data)


# ============================================================================
# BACKWARD COMPATIBILITY LAYER
# Legacy classes wrapped with deprecation warnings
# ============================================================================

class CIAFPreprocessor:
    """Legacy base class - DEPRECATED. Use DataPreprocessor protocol instead."""
    
    def __init__(self):
        warnings.warn(
            "CIAFPreprocessor is deprecated. Use the new protocol-based preprocessors instead.",
            DeprecationWarning,
            stacklevel=2
        )


class TextVectorizer:
    """Legacy text vectorizer - DEPRECATED. Use DefaultTextPreprocessor instead."""
    
    def __init__(self, method: str = "tfidf", max_features: int = 1000, 
                 ngram_range: tuple = (1, 2), stop_words: str = "english"):
        warnings.warn(
            "TextVectorizer is deprecated. Use DefaultTextPreprocessor with PreprocessingPolicy instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create new implementation
        policy = PreprocessingPolicy()
        policy.processing_policy.text_vectorization_method = method
        policy.processing_policy.text_max_features = max_features
        policy.processing_policy.text_ngram_range = ngram_range
        policy.processing_policy.text_stop_words = stop_words
        
        self._impl = DefaultTextPreprocessor(policy)
    
    def fit(self, data):
        return self._impl.fit(data)
    
    def transform(self, data):
        return self._impl.transform(data)
    
    def fit_transform(self, data):
        return self._impl.fit_transform(data)
    
    def get_feature_names(self):
        return self._impl.get_feature_names()
    
    @property
    def is_fitted(self):
        return self._impl.is_fitted()


class NumericalPreprocessor:
    """Legacy numerical preprocessor - DEPRECATED. Use DefaultNumericalPreprocessor instead."""
    
    def __init__(self, normalize: bool = True):
        warnings.warn(
            "NumericalPreprocessor is deprecated. Use DefaultNumericalPreprocessor with PreprocessingPolicy instead.",
            DeprecationWarning, 
            stacklevel=2
        )
        
        # Create new implementation
        policy = PreprocessingPolicy()
        policy.processing_policy.numerical_scaling = "standard" if normalize else "none"
        
        self._impl = DefaultNumericalPreprocessor(policy)
    
    def fit(self, data):
        return self._impl.fit(data)
    
    def transform(self, data):
        return self._impl.transform(data)
    
    def fit_transform(self, data):
        return self._impl.fit_transform(data)
    
    @property
    def is_fitted(self):
        return self._impl.is_fitted()


class MixedDataPreprocessor:
    """Legacy mixed data preprocessor - DEPRECATED. Use create_auto_preprocessor instead."""
    
    def __init__(self, text_method: str = "tfidf", normalize_numerical: bool = True, auto_detect: bool = True):
        warnings.warn(
            "MixedDataPreprocessor is deprecated. Use create_auto_preprocessor with PreprocessingPolicy instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create policy-based implementation
        policy = PreprocessingPolicy()
        policy.processing_policy.text_vectorization_method = text_method
        policy.processing_policy.numerical_scaling = "standard" if normalize_numerical else "none"
        
        self._policy = policy
        self._detector = DefaultDataTypeDetector()
        self._impl = None
        self.data_type = None
        self.is_fitted = False
    
    def fit(self, data):
        self.data_type = self._detector.detect_data_type(data)
        self._impl = create_preprocessor(self.data_type, self._policy)
        result = self._impl.fit(data)
        self.is_fitted = self._impl.is_fitted()
        return self
    
    def transform(self, data):
        if self._impl:
            return self._impl.transform(data)
        raise ValueError("Preprocessor not fitted. Call fit() first.")
    
    def fit_transform(self, data):
        return self.fit(data).transform(data)
    
    def get_feature_info(self):
        if self._impl:
            return {
                "type": self.data_type.value if self.data_type else "unknown",
                "features": self._impl.get_feature_names(),
                "preprocessor": type(self._impl).__name__
            }
        return {"type": "none", "features": []}


class CIAFModelAdapter:
    """Legacy model adapter - DEPRECATED. Use DefaultModelAdapter instead."""
    
    def __init__(self, model, preprocessor=None, auto_preprocess: bool = True):
        warnings.warn(
            "CIAFModelAdapter is deprecated. Use DefaultModelAdapter with PreprocessingPolicy instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self._impl = DefaultModelAdapter(model, auto_preprocess=auto_preprocess)
        self.model = model
        self.preprocessor = preprocessor
        self.auto_preprocess = auto_preprocess
        self.is_fitted = False
        self.labels = []
    
    def fit(self, training_data):
        result = self._impl.fit(training_data)
        self.is_fitted = result
        return self
    
    def predict(self, input_data):
        return self._impl.predict(input_data)
    
    def get_preprocessing_info(self):
        return self._impl.get_preprocessing_info()


# Legacy convenience functions with deprecation warnings
def create_text_classifier_adapter(model, method: str = "tfidf"):
    """DEPRECATED: Use create_auto_model_adapter instead."""
    warnings.warn(
        "create_text_classifier_adapter is deprecated. Use create_auto_model_adapter instead.",
        DeprecationWarning,
        stacklevel=2
    )
    policy = PreprocessingPolicy()
    policy.processing_policy.text_vectorization_method = method
    return DefaultModelAdapter(model, policy)


def create_numerical_regressor_adapter(model, normalize: bool = True):
    """DEPRECATED: Use create_auto_model_adapter instead."""
    warnings.warn(
        "create_numerical_regressor_adapter is deprecated. Use create_auto_model_adapter instead.",
        DeprecationWarning,
        stacklevel=2
    )
    policy = PreprocessingPolicy()
    policy.processing_policy.numerical_scaling = "standard" if normalize else "none"
    return DefaultModelAdapter(model, policy)


def create_auto_adapter(model):
    """DEPRECATED: Use create_auto_model_adapter instead."""
    warnings.warn(
        "create_auto_adapter is deprecated. Use create_auto_model_adapter instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_auto_model_adapter(model)


# Legacy auto_preprocess_data function (keep for backward compatibility)
def auto_preprocess_data(X, y=None, store_preprocessor=None):
    """
    DEPRECATED: Use protocol-based preprocessors instead.
    
    Automatically detect and preprocess data for ML models.
    This function is kept for backward compatibility but is deprecated.
    Use create_auto_preprocessor() and PreprocessingPolicy instead.
    """
    warnings.warn(
        "auto_preprocess_data is deprecated. Use create_auto_preprocessor with PreprocessingPolicy instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        if not X:
            return None, None

        # Handle CIAF training data format (list of dicts)
        if isinstance(X, list) and X and isinstance(X[0], dict):
            # Use new protocol-based approach
            preprocessor = create_auto_preprocessor(X)
            X_processed = preprocessor.fit_transform(X)
            
            # Extract targets if available and y is None
            if y is None:
                y = []
                for item in X:
                    if isinstance(item, dict) and "metadata" in item and "target" in item["metadata"]:
                        y.append(item["metadata"]["target"])
            
            # Store preprocessor info for backward compatibility
            if store_preprocessor is not None:
                store_preprocessor.fitted_preprocessor = preprocessor
                store_preprocessor.preprocessing_type = "protocol_based"
            
            return X_processed, y

        # Handle simple data formats
        import numpy as np
        
        # Detect if text or numerical
        if isinstance(X[0], str):
            # Text data
            preprocessor = create_preprocessor(DataType.TEXT)
            ciaf_format = [{"content": text} for text in X]
            X_processed = preprocessor.fit_transform(ciaf_format)
            
            if store_preprocessor is not None:
                store_preprocessor.fitted_preprocessor = preprocessor
                store_preprocessor.preprocessing_type = "text"
        else:
            # Numerical data
            preprocessor = create_preprocessor(DataType.NUMERICAL)
            ciaf_format = [{"content": list(item) if hasattr(item, '__iter__') else [item]} for item in X]
            X_processed = preprocessor.fit_transform(ciaf_format)
            
            if store_preprocessor is not None:
                store_preprocessor.fitted_preprocessor = preprocessor  
                store_preprocessor.preprocessing_type = "numerical"

        # Process targets
        y_processed = y
        if y is not None:
            try:
                y_processed = np.array(y)
            except:
                pass

        return X_processed, y_processed

    except Exception as e:
        print(f"Auto-preprocessing error: {e}")
        return None, None


# ============================================================================
# MODERN API EXPORTS
# ============================================================================

__all__ = [
    # Protocol interfaces
    "DataPreprocessor", "DataValidator", "DataTypeDetector", "FeatureExtractor",
    "ModelAdapter", "PreprocessingPipeline", "QualityMonitor",
    
    # Enums
    "DataType", "PreprocessingMethod", "ValidationSeverity", 
    "QualityLevel", "PreprocessingIntensity",
    
    # Policy classes
    "PreprocessingPolicy", "QualityPolicy", "ProcessingPolicy",
    "PerformancePolicy", "SecurityPolicy",
    
    # Implementation classes
    "DefaultTextPreprocessor", "DefaultNumericalPreprocessor",
    "DefaultDataTypeDetector", "DefaultDataValidator", "DefaultModelAdapter",
    
    # Factory functions (modern API)
    "create_preprocessor", "create_auto_preprocessor", "validate_data",
    "create_default_preprocessing_protocols", "create_text_preprocessor",
    "create_numerical_preprocessor", "create_auto_model_adapter",
    "get_default_preprocessing_policy", "create_custom_policy",
    
    # Data quality (existing)
    "DataQualityValidator", "ValidationResult", "quick_validate", "validate_ciaf_dataset",
    
    # Legacy classes (deprecated but kept for compatibility)
    "TextVectorizer", "NumericalPreprocessor", "MixedDataPreprocessor", 
    "CIAFModelAdapter", "CIAFPreprocessor",
    
    # Legacy functions (deprecated)
    "auto_preprocess_data", "create_text_classifier_adapter",
    "create_numerical_regressor_adapter", "create_auto_adapter",
]
