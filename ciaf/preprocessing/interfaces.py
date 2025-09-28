"""
Preprocessing Interfaces for CIAF

This module defines Protocol interfaces for preprocessing components, following
the same pattern as other CIAF modules to enable clean dependency injection
and testing.

Created: 2025-09-27
Author: CIAF Framework  
Version: 1.0.0
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Optional, List, Dict, Any, Union, Tuple
import numpy as np
from enum import Enum


class DataType(Enum):
    """Supported data types for preprocessing."""
    TEXT = "text"
    NUMERICAL = "numerical"  
    CATEGORICAL = "categorical"
    MIXED = "mixed"
    IMAGE = "image"
    TIME_SERIES = "time_series"


class PreprocessingMethod(Enum):
    """Available preprocessing methods."""
    TFIDF_VECTORIZATION = "tfidf_vectorization"
    COUNT_VECTORIZATION = "count_vectorization"
    STANDARD_SCALING = "standard_scaling"
    MIN_MAX_SCALING = "min_max_scaling"
    LABEL_ENCODING = "label_encoding" 
    ONE_HOT_ENCODING = "one_hot_encoding"
    PCA_REDUCTION = "pca_reduction"
    FEATURE_SELECTION = "feature_selection"


class ValidationSeverity(Enum):
    """Data validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@runtime_checkable
class DataPreprocessor(Protocol):
    """Protocol for data preprocessing components."""
    
    def fit(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> bool:
        """
        Fit the preprocessor on training data.
        
        Args:
            data: Training data in CIAF format or numpy array
            
        Returns:
            True if fitting succeeded, False otherwise
        """
        ...
    
    def transform(self, data: Union[List[Dict[str, Any]], np.ndarray, str]) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data as numpy array
        """
        ...
    
    def fit_transform(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> np.ndarray:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            data: Training data
            
        Returns:
            Transformed training data
        """
        ...
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of features produced by this preprocessor.
        
        Returns:
            List of feature names
        """
        ...
    
    def is_fitted(self) -> bool:
        """
        Check if preprocessor has been fitted.
        
        Returns:
            True if fitted, False otherwise
        """
        ...


@runtime_checkable
class DataValidator(Protocol):
    """Protocol for data quality validation components."""
    
    def validate(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> Dict[str, Any]:
        """
        Validate data quality.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation results dictionary
        """
        ...
    
    def validate_schema(self, 
                       data: Union[List[Dict[str, Any]], np.ndarray],
                       expected_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against expected schema.
        
        Args:
            data: Data to validate
            expected_schema: Expected data structure
            
        Returns:
            Schema validation results
        """
        ...
    
    def generate_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            validation_results: Results from validation
            
        Returns:
            Formatted report string
        """
        ...


@runtime_checkable  
class DataTypeDetector(Protocol):
    """Protocol for automatic data type detection."""
    
    def detect_data_type(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> DataType:
        """
        Automatically detect the primary data type.
        
        Args:
            data: Input data
            
        Returns:
            Detected data type
        """
        ...
    
    def analyze_features(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> Dict[str, DataType]:
        """
        Analyze individual features and detect their types.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary mapping feature names to their detected types
        """
        ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for feature extraction components."""
    
    def extract_features(self, 
                        data: Union[List[Dict[str, Any]], np.ndarray],
                        feature_config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from raw data.
        
        Args:
            data: Raw input data
            feature_config: Optional configuration for feature extraction
            
        Returns:
            Tuple of (extracted_features, feature_names)
        """
        ...
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if available.
        
        Returns:
            Feature importance scores or None
        """
        ...


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for integrating preprocessing with ML models."""
    
    def fit(self, 
            data: List[Dict[str, Any]], 
            targets: Optional[List[Any]] = None) -> bool:
        """
        Fit the adapter with preprocessing and model training.
        
        Args:
            data: Training data in CIAF format
            targets: Optional target labels
            
        Returns:
            True if fitting succeeded
        """
        ...
    
    def predict(self, data: Union[List[Dict[str, Any]], str]) -> Any:
        """
        Make prediction with automatic preprocessing.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Model prediction
        """
        ...
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about applied preprocessing.
        
        Returns:
            Preprocessing metadata
        """
        ...


@runtime_checkable
class PreprocessingPipeline(Protocol):
    """Protocol for preprocessing pipeline management."""
    
    def add_step(self, name: str, preprocessor: DataPreprocessor) -> None:
        """
        Add a preprocessing step to the pipeline.
        
        Args:
            name: Name of the preprocessing step
            preprocessor: Preprocessor component
        """
        ...
    
    def fit_pipeline(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> bool:
        """
        Fit all preprocessing steps in sequence.
        
        Args:
            data: Training data
            
        Returns:
            True if all steps fitted successfully
        """
        ...
    
    def transform_pipeline(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> np.ndarray:
        """
        Apply all preprocessing steps in sequence.
        
        Args:
            data: Data to transform
            
        Returns:
            Final transformed data
        """
        ...
    
    def get_pipeline_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all pipeline steps.
        
        Returns:
            List of step information dictionaries
        """
        ...


@runtime_checkable
class QualityMonitor(Protocol):
    """Protocol for monitoring data quality over time."""
    
    def establish_baseline(self, reference_data: Union[List[Dict[str, Any]], np.ndarray]) -> None:
        """
        Establish quality baseline from reference data.
        
        Args:
            reference_data: Reference/training data for baseline
        """
        ...
    
    def detect_drift(self, 
                    new_data: Union[List[Dict[str, Any]], np.ndarray],
                    threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect data drift compared to baseline.
        
        Args:
            new_data: New data to compare
            threshold: Drift detection threshold
            
        Returns:
            Drift detection results
        """
        ...
    
    def generate_quality_metrics(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> Dict[str, float]:
        """
        Generate quality metrics for data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary of quality metrics
        """
        ...