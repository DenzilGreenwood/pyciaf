"""
Preprocessing Protocol Implementations for CIAF

This module provides concrete implementations of preprocessing protocols,
following the same architectural pattern as other CIAF modules.

Created: 2025-09-27
Author: CIAF Framework
Version: 1.0.0
"""

import json
import warnings
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# Import sklearn components with graceful fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some preprocessing features will be limited.")

from .interfaces import (
    DataPreprocessor, DataValidator, DataTypeDetector, FeatureExtractor,
    ModelAdapter, PreprocessingPipeline, QualityMonitor,
    DataType, PreprocessingMethod, ValidationSeverity
)
from .policy import PreprocessingPolicy, get_default_preprocessing_policy


class DefaultTextPreprocessor(DataPreprocessor):
    """Default implementation of text data preprocessing."""
    
    def __init__(self, policy: Optional[PreprocessingPolicy] = None):
        self.policy = policy or get_default_preprocessing_policy()
        self.vectorizer = None
        self._is_fitted = False
        self.feature_names = []
        
    def fit(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> bool:
        """Fit text preprocessor on training data."""
        try:
            # Extract text content
            texts = self._extract_text_content(data)
            
            if not texts:
                warnings.warn("No text content found for fitting")
                return False
            
            if not SKLEARN_AVAILABLE:
                # Simple fallback: create dummy features
                unique_words = set()
                for text in texts[:100]:  # Limit for memory
                    if isinstance(text, str):
                        unique_words.update(text.lower().split())
                
                self.feature_names = sorted(list(unique_words))[:self.policy.processing_policy.text_max_features]
                self._is_fitted = True
                return True
            
            # Use sklearn vectorizer
            method = self.policy.processing_policy.text_vectorization_method
            max_features = self.policy.processing_policy.text_max_features
            ngram_range = self.policy.processing_policy.text_ngram_range
            
            if method == "tfidf":
                self.vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words=self.policy.processing_policy.text_stop_words,
                    lowercase=self.policy.processing_policy.text_lowercase
                )
            else:
                self.vectorizer = CountVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words=self.policy.processing_policy.text_stop_words,
                    lowercase=self.policy.processing_policy.text_lowercase
                )
            
            self.vectorizer.fit(texts)
            self.feature_names = self.vectorizer.get_feature_names_out().tolist()
            self._is_fitted = True
            
            return True
            
        except Exception as e:
            warnings.warn(f"Text preprocessor fitting failed: {e}")
            return False
    
    def transform(self, data: Union[List[Dict[str, Any]], np.ndarray, str]) -> np.ndarray:
        """Transform text data to numerical vectors."""
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        texts = self._extract_text_content(data)
        
        if not SKLEARN_AVAILABLE:
            # Simple fallback: binary word presence
            result = []
            for text in texts:
                if isinstance(text, str):
                    words = set(text.lower().split())
                    features = [1.0 if word in words else 0.0 for word in self.feature_names]
                else:
                    features = [0.0] * len(self.feature_names)
                result.append(features)
            return np.array(result)
        
        if self.vectorizer:
            return self.vectorizer.transform(texts).toarray()
        
        return np.array([[0.0] * len(self.feature_names)] * len(texts))
    
    def fit_transform(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> np.ndarray:
        """Fit and transform text data."""
        if self.fit(data):
            return self.transform(data)
        return np.array([])
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for text vectorization."""
        return self.feature_names.copy()
    
    def is_fitted(self) -> bool:
        """Check if preprocessor is fitted."""
        return self._is_fitted
    
    def _extract_text_content(self, data: Union[List[Dict[str, Any]], np.ndarray, str]) -> List[str]:
        """Extract text content from various data formats."""
        if isinstance(data, str):
            return [data]
        
        if isinstance(data, list):
            texts = []
            for item in data:
                if isinstance(item, dict):
                    # CIAF format
                    content = item.get("content", "")
                    if isinstance(content, str):
                        texts.append(content)
                    else:
                        texts.append(str(content))
                elif isinstance(item, str):
                    texts.append(item)
                else:
                    texts.append(str(item))
            return texts
        
        if isinstance(data, np.ndarray):
            return [str(x) for x in data.flatten()]
        
        return [str(data)]


class DefaultNumericalPreprocessor(DataPreprocessor):
    """Default implementation of numerical data preprocessing."""
    
    def __init__(self, policy: Optional[PreprocessingPolicy] = None):
        self.policy = policy or get_default_preprocessing_policy()
        self.scaler = None
        self._is_fitted = False
        self.feature_names = []
        self.n_features = 0
        
    def fit(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> bool:
        """Fit numerical preprocessor on training data."""
        try:
            # Extract numerical data
            numerical_data = self._extract_numerical_data(data)
            
            if len(numerical_data) == 0:
                warnings.warn("No numerical data found for fitting")
                return False
            
            numerical_array = np.array(numerical_data)
            
            # Ensure 2D array
            if numerical_array.ndim == 1:
                numerical_array = numerical_array.reshape(-1, 1)
            
            self.n_features = numerical_array.shape[1]
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
            
            if not SKLEARN_AVAILABLE:
                # Simple normalization fallback
                self._is_fitted = True
                return True
            
            # Use sklearn scaler
            scaling_method = self.policy.processing_policy.numerical_scaling
            
            if scaling_method == "standard":
                self.scaler = StandardScaler()
            elif scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                # No scaling
                self.scaler = None
                
            if self.scaler:
                self.scaler.fit(numerical_array)
            
            self._is_fitted = True
            return True
            
        except Exception as e:
            warnings.warn(f"Numerical preprocessor fitting failed: {e}")
            return False
    
    def transform(self, data: Union[List[Dict[str, Any]], np.ndarray, str]) -> np.ndarray:
        """Transform numerical data."""
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        numerical_data = self._extract_numerical_data(data)
        numerical_array = np.array(numerical_data)
        
        # Ensure 2D array
        if numerical_array.ndim == 1:
            numerical_array = numerical_array.reshape(-1, 1)
        
        # Ensure correct number of features
        if numerical_array.shape[1] != self.n_features:
            warnings.warn(f"Feature count mismatch: expected {self.n_features}, got {numerical_array.shape[1]}")
            # Pad or truncate as needed
            if numerical_array.shape[1] < self.n_features:
                padding = np.zeros((numerical_array.shape[0], self.n_features - numerical_array.shape[1]))
                numerical_array = np.hstack([numerical_array, padding])
            else:
                numerical_array = numerical_array[:, :self.n_features]
        
        if not SKLEARN_AVAILABLE or self.scaler is None:
            return numerical_array
        
        return self.scaler.transform(numerical_array)
    
    def fit_transform(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> np.ndarray:
        """Fit and transform numerical data."""
        if self.fit(data):
            return self.transform(data)
        return np.array([])
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for numerical data."""
        return self.feature_names.copy()
    
    def is_fitted(self) -> bool:
        """Check if preprocessor is fitted."""
        return self._is_fitted
    
    def _extract_numerical_data(self, data: Union[List[Dict[str, Any]], np.ndarray, str]) -> List[List[float]]:
        """Extract numerical data from various formats."""
        if isinstance(data, np.ndarray):
            return data.tolist() if data.ndim > 1 else [[float(x)] for x in data]
        
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, (list, tuple)):
                    return [list(map(float, parsed))]
                else:
                    return [[float(parsed)]]
            except:
                return [[0.0]]
        
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, dict):
                    # CIAF format
                    content = item.get("content", [])
                    if isinstance(content, (list, tuple)):
                        try:
                            result.append(list(map(float, content)))
                        except:
                            result.append([0.0] * len(content) if content else [0.0])
                    elif isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, (list, tuple)):
                                result.append(list(map(float, parsed)))
                            else:
                                result.append([float(parsed)])
                        except:
                            result.append([0.0])
                    else:
                        try:
                            result.append([float(content)])
                        except:
                            result.append([0.0])
                elif isinstance(item, (list, tuple)):
                    try:
                        result.append(list(map(float, item)))
                    except:
                        result.append([0.0] * len(item) if item else [0.0])
                else:
                    try:
                        result.append([float(item)])
                    except:
                        result.append([0.0])
            return result
        
        return [[0.0]]


class DefaultDataTypeDetector(DataTypeDetector):
    """Default implementation of data type detection."""
    
    def detect_data_type(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> DataType:
        """Automatically detect the primary data type."""
        if isinstance(data, np.ndarray):
            if np.issubdtype(data.dtype, np.number):
                return DataType.NUMERICAL
            else:
                return DataType.TEXT
        
        if not isinstance(data, list) or not data:
            return DataType.MIXED
        
        # Sample first few items for detection
        sample_size = min(10, len(data))
        text_count = 0
        numerical_count = 0
        
        for item in data[:sample_size]:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str) and not self._is_numeric_string(content):
                    text_count += 1
                elif isinstance(content, (list, tuple, int, float)):
                    numerical_count += 1
                else:
                    text_count += 1
            elif isinstance(item, str) and not self._is_numeric_string(item):
                text_count += 1
            elif isinstance(item, (int, float, list, tuple)):
                numerical_count += 1
            else:
                text_count += 1
        
        if text_count > numerical_count * 1.5:
            return DataType.TEXT
        elif numerical_count > text_count * 1.5:
            return DataType.NUMERICAL
        else:
            return DataType.MIXED
    
    def analyze_features(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> Dict[str, DataType]:
        """Analyze individual features and detect their types."""
        feature_types = {}
        
        if isinstance(data, np.ndarray):
            for i in range(data.shape[1] if data.ndim > 1 else 1):
                feature_types[f"feature_{i}"] = DataType.NUMERICAL if np.issubdtype(data.dtype, np.number) else DataType.TEXT
            return feature_types
        
        # For structured data, analyze each feature
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                # CIAF format - analyze content structure
                sample_item = data[0]
                content = sample_item.get("content", {})
                
                if isinstance(content, dict):
                    for key, value in content.items():
                        if isinstance(value, (int, float)):
                            feature_types[key] = DataType.NUMERICAL
                        elif isinstance(value, str) and self._is_numeric_string(value):
                            feature_types[key] = DataType.NUMERICAL
                        else:
                            feature_types[key] = DataType.TEXT
                else:
                    feature_types["content"] = self.detect_data_type(data)
        
        return feature_types
    
    def _is_numeric_string(self, s: str) -> bool:
        """Check if string represents a numeric value."""
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False


class DefaultDataValidator(DataValidator):
    """Default implementation of data quality validation."""
    
    def __init__(self, policy: Optional[PreprocessingPolicy] = None):
        self.policy = policy or get_default_preprocessing_policy()
    
    def validate(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> Dict[str, Any]:
        """Validate data quality."""
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Convert to DataFrame for analysis
            df = self._prepare_data_frame(data)
            
            # Basic structure validation
            self._validate_structure(df, results)
            
            # Content quality validation
            self._validate_content_quality(df, results)
            
            # Statistical validation
            self._validate_statistical_properties(df, results)
            
            # Calculate overall quality score
            self._calculate_quality_score(results)
            
        except Exception as e:
            results["errors"].append(f"Validation failed: {str(e)}")
            results["is_valid"] = False
        
        return results
    
    def validate_schema(self, 
                       data: Union[List[Dict[str, Any]], np.ndarray],
                       expected_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against expected schema."""
        results = {
            "schema_valid": True,
            "missing_fields": [],
            "extra_fields": [],
            "type_mismatches": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            df = self._prepare_data_frame(data)
            expected_columns = expected_schema.get("columns", [])
            
            if expected_columns:
                missing = set(expected_columns) - set(df.columns)
                extra = set(df.columns) - set(expected_columns)
                
                results["missing_fields"] = list(missing)
                results["extra_fields"] = list(extra)
                
                if missing:
                    results["schema_valid"] = False
                
        except Exception as e:
            results["schema_valid"] = False
            results["validation_error"] = str(e)
        
        return results
    
    def generate_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall status
        status = "✅ PASSED" if validation_results.get("is_valid", False) else "❌ FAILED"
        report.append(f"\nOverall Status: {status}")
        
        # Metrics
        metrics = validation_results.get("metrics", {})
        if metrics:
            report.append("\nQuality Metrics:")
            report.append("-" * 30)
            for key, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"{key}: {value:.3f}")
                else:
                    report.append(f"{key}: {value}")
        
        # Errors
        errors = validation_results.get("errors", [])
        if errors:
            report.append("\nErrors:")
            report.append("-" * 30)
            for i, error in enumerate(errors, 1):
                report.append(f"{i}. {error}")
        
        # Warnings
        warnings = validation_results.get("warnings", [])
        if warnings:
            report.append("\nWarnings:")
            report.append("-" * 30)
            for i, warning in enumerate(warnings, 1):
                report.append(f"{i}. {warning}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)
    
    def _prepare_data_frame(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> pd.DataFrame:
        """Convert data to pandas DataFrame for analysis."""
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.DataFrame({'values': data})
            else:
                columns = [f'feature_{i}' for i in range(data.shape[1])]
                return pd.DataFrame(data, columns=columns)
        
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                records = []
                for item in data:
                    record = {}
                    content = item.get('content', item)
                    
                    if isinstance(content, dict):
                        record.update(content)
                    elif isinstance(content, (list, tuple)):
                        for i, val in enumerate(content):
                            record[f'feature_{i}'] = val
                    else:
                        record['content'] = content
                    
                    # Add metadata
                    metadata = item.get('metadata', {})
                    for key, value in metadata.items():
                        record[f'meta_{key}'] = value
                    
                    records.append(record)
                
                return pd.DataFrame(records)
            else:
                return pd.DataFrame({'values': data})
        
        return pd.DataFrame()
    
    def _validate_structure(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Validate basic data structure."""
        min_samples = self.policy.quality_policy.min_samples
        
        if len(df) < min_samples:
            results["errors"].append(f"Insufficient samples: {len(df)} < {min_samples}")
            results["is_valid"] = False
        
        results["metrics"]["sample_count"] = len(df)
        results["metrics"]["feature_count"] = len(df.columns)
        
        if df.empty:
            results["errors"].append("Dataset is empty")
            results["is_valid"] = False
    
    def _validate_content_quality(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Validate content quality."""
        max_missing_ratio = self.policy.quality_policy.max_missing_ratio
        
        missing_ratios = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_ratio = missing_count / len(df) if len(df) > 0 else 0
            missing_ratios[column] = missing_ratio
            
            if missing_ratio > max_missing_ratio:
                results["errors"].append(f"Column '{column}' has too many missing values: {missing_ratio:.2%}")
                results["is_valid"] = False
            elif missing_ratio > 0:
                results["warnings"].append(f"Column '{column}' has {missing_ratio:.2%} missing values")
        
        results["metrics"]["missing_ratios"] = missing_ratios
        results["metrics"]["avg_missing_ratio"] = np.mean(list(missing_ratios.values())) if missing_ratios else 0
    
    def _validate_statistical_properties(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Validate statistical properties."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            # Check for outliers if enabled
            if self.policy.quality_policy.enable_outlier_detection:
                outlier_ratios = {}
                for column in numeric_columns:
                    col_data = df[column].dropna()
                    if len(col_data) > 0:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                        outlier_ratio = len(outliers) / len(col_data)
                        outlier_ratios[column] = outlier_ratio
                        
                        if outlier_ratio > self.policy.quality_policy.outlier_threshold:
                            results["warnings"].append(f"Column '{column}' has {outlier_ratio:.2%} outliers")
                
                results["metrics"]["outlier_ratios"] = outlier_ratios
        
        # Check for duplicates if enabled
        if self.policy.quality_policy.check_duplicates:
            duplicate_count = df.duplicated().sum()
            duplicate_ratio = duplicate_count / len(df) if len(df) > 0 else 0
            
            results["metrics"]["duplicate_count"] = duplicate_count
            results["metrics"]["duplicate_ratio"] = duplicate_ratio
            
            if duplicate_ratio > self.policy.quality_policy.duplicate_threshold:
                results["warnings"].append(f"High duplicate ratio: {duplicate_ratio:.2%}")
    
    def _calculate_quality_score(self, results: Dict[str, Any]):
        """Calculate overall quality score."""
        score = 100.0
        
        # Penalize based on errors and warnings
        score -= len(results["errors"]) * 20
        score -= len(results["warnings"]) * 10
        
        # Penalize based on missing values
        avg_missing = results["metrics"].get("avg_missing_ratio", 0)
        score -= avg_missing * 30
        
        # Penalize based on duplicates
        duplicate_ratio = results["metrics"].get("duplicate_ratio", 0)
        score -= duplicate_ratio * 15
        
        score = max(0, min(100, score))
        results["metrics"]["quality_score"] = round(score, 2)


class DefaultModelAdapter(ModelAdapter):
    """Default implementation of model adapter with preprocessing."""
    
    def __init__(self, 
                 model: Any,
                 policy: Optional[PreprocessingPolicy] = None,
                 auto_preprocess: bool = True):
        self.model = model
        self.policy = policy or get_default_preprocessing_policy()
        self.auto_preprocess = auto_preprocess
        self.preprocessor = None
        self.data_type_detector = DefaultDataTypeDetector()
        self._is_fitted = False
        self.preprocessing_info = {}
    
    def fit(self, 
            data: List[Dict[str, Any]], 
            targets: Optional[List[Any]] = None) -> bool:
        """Fit the adapter with preprocessing and model training."""
        try:
            # Detect data type and select appropriate preprocessor
            if self.auto_preprocess:
                data_type = self.data_type_detector.detect_data_type(data)
                
                if data_type == DataType.TEXT:
                    self.preprocessor = DefaultTextPreprocessor(self.policy)
                elif data_type == DataType.NUMERICAL:
                    self.preprocessor = DefaultNumericalPreprocessor(self.policy)
                else:
                    # Mixed data - try numerical first, fallback to text
                    self.preprocessor = DefaultNumericalPreprocessor(self.policy)
                    if not self.preprocessor.fit(data):
                        self.preprocessor = DefaultTextPreprocessor(self.policy)
            
            # Fit preprocessor and transform data
            if self.preprocessor and self.preprocessor.fit(data):
                X = self.preprocessor.transform(data)
                self.preprocessing_info = {
                    "data_type": data_type.value if 'data_type' in locals() else "unknown",
                    "preprocessor_type": type(self.preprocessor).__name__,
                    "feature_count": len(self.preprocessor.get_feature_names()),
                    "feature_names": self.preprocessor.get_feature_names()
                }
            else:
                # No preprocessing - use raw data
                X = [item.get("content", item) for item in data]
                self.preprocessing_info = {"data_type": "raw", "preprocessor_type": "none"}
            
            # Extract targets
            if targets is None:
                targets = []
                for item in data:
                    if isinstance(item, dict) and "metadata" in item and "target" in item["metadata"]:
                        targets.append(item["metadata"]["target"])
            
            # Fit model if it has a fit method
            if hasattr(self.model, "fit") and targets:
                try:
                    self.model.fit(X, targets)
                    self._is_fitted = True
                    return True
                except Exception as e:
                    warnings.warn(f"Model fitting failed: {e}")
                    return False
            else:
                # Model doesn't support training or no targets
                self._is_fitted = False
                return True
                
        except Exception as e:
            warnings.warn(f"Adapter fitting failed: {e}")
            return False
    
    def predict(self, data: Union[List[Dict[str, Any]], str]) -> Any:
        """Make prediction with automatic preprocessing."""
        try:
            # Apply preprocessing if available
            if self.preprocessor and self.preprocessor.is_fitted():
                X = self.preprocessor.transform(data)
            else:
                # Handle raw data
                if isinstance(data, str):
                    X = [[data]]
                elif isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict):
                        X = [[item.get("content", str(item))] for item in data]
                    else:
                        X = [[str(item)] for item in data]
                else:
                    X = [[str(data)]]
            
            # Make prediction
            if hasattr(self.model, "predict") and self._is_fitted:
                prediction = self.model.predict(X)
                return prediction[0] if len(prediction) == 1 else prediction
            else:
                # Fallback prediction
                return self._fallback_prediction(data)
                
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")
            return self._fallback_prediction(data)
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about applied preprocessing."""
        return self.preprocessing_info.copy()
    
    def _fallback_prediction(self, data: Any) -> Any:
        """Simple fallback prediction logic."""
        if isinstance(data, str):
            # Simple sentiment analysis fallback
            positive_words = ["good", "great", "excellent", "positive", "happy", "love"]
            negative_words = ["bad", "terrible", "horrible", "negative", "sad", "hate"]
            
            data_lower = data.lower()
            positive_count = sum(1 for word in positive_words if word in data_lower)
            negative_count = sum(1 for word in negative_words if word in data_lower)
            
            if positive_count > negative_count:
                return 1  # Positive
            else:
                return 0  # Negative/Neutral
        
        # Default fallback
        return 0


def create_default_preprocessing_protocols(policy: Optional[PreprocessingPolicy] = None) -> Dict[str, Any]:
    """Create default preprocessing protocol implementations."""
    if policy is None:
        policy = get_default_preprocessing_policy()
    
    return {
        "text_preprocessor": DefaultTextPreprocessor(policy),
        "numerical_preprocessor": DefaultNumericalPreprocessor(policy),
        "data_type_detector": DefaultDataTypeDetector(),
        "data_validator": DefaultDataValidator(policy),
        "model_adapter_factory": lambda model: DefaultModelAdapter(model, policy)
    }


def create_text_preprocessor(policy: Optional[PreprocessingPolicy] = None) -> DefaultTextPreprocessor:
    """Create a text preprocessor with given policy."""
    return DefaultTextPreprocessor(policy)


def create_numerical_preprocessor(policy: Optional[PreprocessingPolicy] = None) -> DefaultNumericalPreprocessor:
    """Create a numerical preprocessor with given policy."""
    return DefaultNumericalPreprocessor(policy)


def create_auto_model_adapter(model: Any, policy: Optional[PreprocessingPolicy] = None) -> DefaultModelAdapter:
    """Create a model adapter with automatic preprocessing."""
    return DefaultModelAdapter(model, policy, auto_preprocess=True)