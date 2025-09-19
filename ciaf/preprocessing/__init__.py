"""
Preprocessing and Vectorization Module for CIAF

This module provides vectorization and preprocessing capabilities to ensure
real ML model training with proper feature extraction for text, numerical,
and mixed data types.

Created: 2025-09-09
Last PYPIModified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler


class CIAFPreprocessor(ABC):
    """Base class for CIAF preprocessing components."""

    @abstractmethod
    def fit(self, data: List[Dict[str, Any]]) -> "CIAFPreprocessor":
        """Fit the preprocessor on training data."""
        pass

    @abstractmethod
    def transform(self, data: Union[List[Dict[str, Any]], str, List]) -> np.ndarray:
        """Transform data to numerical format."""
        pass

    @abstractmethod
    def fit_transform(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Fit and transform training data."""
        pass


class TextVectorizer(CIAFPreprocessor):
    """Text vectorization using TF-IDF or Count Vectorization."""

    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 2),
        stop_words: str = "english",
    ):
        """
        Initialize text vectorizer.

        Args:
            method: "tfidf" or "count"
            max_features: Maximum number of features
            ngram_range: N-gram range for feature extraction
            stop_words: Stop words to remove
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.vectorizer = None
        self.is_fitted = False

    def fit(self, data: List[Dict[str, Any]]) -> "TextVectorizer":
        """Fit vectorizer on text data."""
        texts = [item["content"] for item in data if isinstance(item["content"], str)]

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words=self.stop_words,
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words=self.stop_words,
            )

        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, data: Union[List[Dict[str, Any]], str, List]) -> np.ndarray:
        """Transform text data to numerical vectors."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")

        # Handle different input types
        if isinstance(data, str):
            texts = [data]
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                texts = [
                    item["content"] for item in data if isinstance(item["content"], str)
                ]
            else:
                texts = [str(item) for item in data]
        else:
            texts = [str(data)]

        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Fit and transform text data."""
        return self.fit(data).transform(data)

    def get_feature_names(self) -> List[str]:
        """Get feature names for explainability."""
        if self.is_fitted:
            return self.vectorizer.get_feature_names_out().tolist()
        return []


class NumericalPreprocessor(CIAFPreprocessor):
    """Preprocessing for numerical data."""

    def __init__(self, normalize: bool = True):
        """Initialize numerical preprocessor."""
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.is_fitted = False

    def fit(self, data: List[Dict[str, Any]]) -> "NumericalPreprocessor":
        """Fit preprocessor on numerical data."""
        # Extract numerical features from content
        numerical_data = []
        for item in data:
            content = item["content"]
            if isinstance(content, (list, tuple)):
                numerical_data.append(content)
            elif isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, (list, tuple)):
                        numerical_data.append(parsed)
                except:
                    # Skip non-numerical content
                    continue

        if numerical_data and self.scaler:
            self.scaler.fit(numerical_data)

        self.is_fitted = True
        return self

    def transform(self, data: Union[List[Dict[str, Any]], str, List]) -> np.ndarray:
        """Transform numerical data."""
        # Handle different input types
        if isinstance(data, str):
            try:
                numerical_data = [json.loads(data)]
            except:
                raise ValueError(f"Cannot parse numerical data from string: {data}")
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                numerical_data = []
                for item in data:
                    content = item["content"]
                    if isinstance(content, (list, tuple)):
                        numerical_data.append(content)
                    elif isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                            numerical_data.append(parsed)
                        except:
                            continue
            else:
                numerical_data = [data] if isinstance(data[0], (int, float)) else data
        else:
            numerical_data = [data]

        numerical_array = np.array(numerical_data)

        if self.scaler and self.is_fitted:
            return self.scaler.transform(numerical_array)

        return numerical_array

    def fit_transform(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Fit and transform numerical data."""
        return self.fit(data).transform(data)


class MixedDataPreprocessor(CIAFPreprocessor):
    """Preprocessor for mixed text and numerical data."""

    def __init__(
        self,
        text_method: str = "tfidf",
        normalize_numerical: bool = True,
        auto_detect: bool = True,
    ):
        """Initialize mixed data preprocessor."""
        self.text_vectorizer = TextVectorizer(method=text_method)
        self.numerical_preprocessor = NumericalPreprocessor(
            normalize=normalize_numerical
        )
        self.auto_detect = auto_detect
        self.data_type = None
        self.is_fitted = False

    def _detect_data_type(self, data: List[Dict[str, Any]]) -> str:
        """Automatically detect data type."""
        text_count = 0
        numerical_count = 0

        for item in data[:10]:  # Sample first 10 items
            content = item["content"]
            if isinstance(content, str):
                try:
                    json.loads(content)
                    numerical_count += 1
                except:
                    text_count += 1
            elif isinstance(content, (list, tuple, int, float)):
                numerical_count += 1
            else:
                text_count += 1

        if text_count > numerical_count:
            return "text"
        else:
            return "numerical"

    def fit(self, data: List[Dict[str, Any]]) -> "MixedDataPreprocessor":
        """Fit preprocessor on mixed data."""
        if self.auto_detect:
            self.data_type = self._detect_data_type(data)

        if self.data_type == "text":
            self.text_vectorizer.fit(data)
        else:
            self.numerical_preprocessor.fit(data)

        self.is_fitted = True
        return self

    def transform(self, data: Union[List[Dict[str, Any]], str, List]) -> np.ndarray:
        """Transform mixed data."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        if self.data_type == "text":
            return self.text_vectorizer.transform(data)
        else:
            return self.numerical_preprocessor.transform(data)

    def fit_transform(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Fit and transform mixed data."""
        return self.fit(data).transform(data)

    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature information for explainability."""
        if self.data_type == "text":
            return {
                "type": "text",
                "features": self.text_vectorizer.get_feature_names(),
                "method": self.text_vectorizer.method,
            }
        else:
            return {
                "type": "numerical",
                "features": (
                    [
                        "feature_" + str(i)
                        for i in range(
                            self.numerical_preprocessor.scaler.n_features_in_
                        )
                    ]
                    if self.numerical_preprocessor.scaler
                    else []
                ),
                "normalized": self.numerical_preprocessor.normalize,
            }


class CIAFModelAdapter:
    """Adapter to integrate preprocessing with ML models for CIAF."""

    def __init__(
        self,
        model: Any,
        preprocessor: Optional[CIAFPreprocessor] = None,
        auto_preprocess: bool = True,
    ):
        """
        Initialize model adapter.

        Args:
            model: The ML model to wrap
            preprocessor: Custom preprocessor, None for auto-detection
            auto_preprocess: Whether to automatically select preprocessing
        """
        self.model = model
        self.preprocessor = preprocessor
        self.auto_preprocess = auto_preprocess
        self.is_fitted = False
        self.labels = []

    def fit(self, training_data: List[Dict[str, Any]]) -> "CIAFModelAdapter":
        """Fit the model with preprocessing."""
        # Auto-select preprocessor if not provided
        if self.preprocessor is None and self.auto_preprocess:
            self.preprocessor = MixedDataPreprocessor()

        # Fit preprocessor and transform data
        if self.preprocessor:
            X = self.preprocessor.fit_transform(training_data)
        else:
            # Fallback: assume model can handle raw data
            X = [item["content"] for item in training_data]

        # Extract labels
        y = [
            item["metadata"]["target"]
            for item in training_data
            if "target" in item["metadata"]
        ]
        self.labels = list(set(y))

        # Fit model
        if hasattr(self.model, "fit"):
            try:
                self.model.fit(X, y)
                self.is_fitted = True
                print(
                    f"   ✅ Model fitted successfully with {len(training_data)} samples"
                )
            except Exception as e:
                warnings.warn(f"Model fitting failed: {e}. Using fallback mode.")
                self.is_fitted = False
        else:
            warnings.warn("Model does not have fit method. Using prediction-only mode.")
            self.is_fitted = False

        return self

    def predict(self, input_data: Union[str, List, Dict]) -> Any:
        """Make prediction with preprocessing."""
        if self.preprocessor:
            # Transform input data
            X = self.preprocessor.transform(input_data)
        else:
            X = input_data

        # Make prediction
        if hasattr(self.model, "predict") and self.is_fitted:
            try:
                prediction = self.model.predict(X)
                # Handle single prediction
                if hasattr(prediction, "__len__") and len(prediction) == 1:
                    return prediction[0]
                return prediction
            except Exception as e:
                warnings.warn(f"Prediction failed: {e}. Using fallback.")
                return self._fallback_prediction(input_data)
        else:
            return self._fallback_prediction(input_data)

    def _fallback_prediction(self, input_data: Any) -> Any:
        """Fallback prediction when model fails."""
        if self.labels:
            # Return first label as fallback
            return self.labels[0]
        # Default fallback
        if isinstance(input_data, str) and "positive" in input_data.lower():
            return 1
        return 0

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get preprocessing information for metadata."""
        if self.preprocessor and hasattr(self.preprocessor, "get_feature_info"):
            return self.preprocessor.get_feature_info()
        return {"type": "none", "features": []}


# Example usage functions
def create_text_classifier_adapter(model, method: str = "tfidf") -> CIAFModelAdapter:
    """Create adapter for text classification."""
    preprocessor = TextVectorizer(method=method)
    return CIAFModelAdapter(model, preprocessor)


def create_numerical_regressor_adapter(
    model, normalize: bool = True
) -> CIAFModelAdapter:
    """Create adapter for numerical regression."""
    preprocessor = NumericalPreprocessor(normalize=normalize)
    return CIAFModelAdapter(model, preprocessor)


def create_auto_adapter(model) -> CIAFModelAdapter:
    """Create auto-detecting adapter."""
    return CIAFModelAdapter(model, auto_preprocess=True)


def auto_preprocess_data(X, y=None, store_preprocessor=None):
    """
    Automatically detect and preprocess data for ML models.

    Args:
        X: Input features (list of strings or numerical data)
        y: Target labels (optional)
        store_preprocessor: Object to store the fitted preprocessor (optional)

    Returns:
        Tuple of (processed_X, processed_y)
    """
    try:
        if not X:
            return None, None

        # Handle CIAF training data format (list of dicts)
        if isinstance(X, list) and X and isinstance(X[0], dict):
            # Extract content from CIAF format
            X_content = [item.get("content", item) for item in X]

            # Extract targets if available
            if y is None and all(
                "metadata" in item and "target" in item["metadata"] for item in X
            ):
                y = [item["metadata"]["target"] for item in X]

            X = X_content

        # Detect data type and preprocess
        if isinstance(X[0], str):
            # Text data - create simple TF-IDF vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
            X_processed = vectorizer.fit_transform(X)
            # Convert sparse matrix to dense for better compatibility
            X_processed = X_processed.toarray()

            # Store the fitted vectorizer for inference
            if store_preprocessor is not None:
                store_preprocessor.fitted_vectorizer = vectorizer
                store_preprocessor.preprocessing_type = "text"

        else:
            # Numerical data
            import numpy as np
            from sklearn.preprocessing import StandardScaler

            # Convert to numpy array and preserve original shape
            X_array = np.array(X)
            
            # Ensure we have a 2D array for sklearn
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
            
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_array)

            # Store the fitted scaler
            if store_preprocessor is not None:
                store_preprocessor.fitted_preprocessor = scaler
                store_preprocessor.preprocessing_type = "numerical"

        # Process targets if provided
        y_processed = y
        if y is not None:
            try:
                import numpy as np

                y_processed = np.array(y)
            except:
                pass

        return X_processed, y_processed

    except Exception as e:
        print(f"Auto-preprocessing error: {e}")
        return None, None


# Enhanced imports for model wrapper integration
__all__ = [
    "TextVectorizer",
    "NumericalPreprocessor",
    "MixedDataPreprocessor",
    "CIAFModelAdapter",
    "auto_preprocess_data",
    "create_text_classifier_adapter",
    "create_numerical_regressor_adapter",
    "create_auto_adapter",
]
