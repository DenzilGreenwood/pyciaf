"""
Universal Model Adapter for CIAF Wrapper System
===============================================

This module provides comprehensive model type detection, validation, and adaptation
support for all major ML frameworks and custom models. It consolidates and enhances
the existing model adapter functionality to work reliably with any model type.

Created: 2025-09-27
Author: Denzil James Greenwood
Version: 1.0.0
"""

import warnings
import inspect
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TYPE_CHECKING
from datetime import datetime
from abc import ABC, abstractmethod

from .interfaces import ModelAdapter, ModelMetadataProvider
from .policy import ModelType

if TYPE_CHECKING:
    from .policy import DataType

# Import ModelType from policy for framework compatibility
try:
    from .policy import ModelType
except ImportError:
    # Fallback enum if policy not available
    from enum import Enum
    class ModelType(Enum):
        SCIKIT_LEARN = "scikit_learn"
        PYTORCH = "pytorch"
        TENSORFLOW = "tensorflow"
        HUGGINGFACE = "huggingface"
        XGBOOST = "xgboost"
        LIGHTGBM = "lightgbm"
        CUSTOM = "custom"
        AUTO_DETECT = "auto_detect"
from .policy import ModelType


class UniversalModelDetector:
    """Advanced model type detection for all ML frameworks."""
    
    def __init__(self):
        self.framework_signatures = {
            # Scikit-learn
            'sklearn': {
                'modules': ['sklearn', 'scikit_learn'],
                'methods': ['fit', 'predict'],
                'optional_methods': ['score', 'predict_proba', 'predict_log_proba'],
                'attributes': ['get_params', 'set_params'],
                'type': ModelType.SCIKIT_LEARN
            },
            
            # PyTorch
            'pytorch': {
                'modules': ['torch', 'pytorch'],
                'methods': ['forward'],
                'optional_methods': ['parameters', 'train', 'eval', 'state_dict'],
                'attributes': ['training', 'device'],
                'type': ModelType.PYTORCH
            },
            
            # TensorFlow/Keras
            'tensorflow': {
                'modules': ['tensorflow', 'keras'],
                'methods': ['call'],
                'optional_methods': ['fit', 'predict', 'evaluate', 'compile'],
                'attributes': ['trainable_variables', 'layers'],
                'type': ModelType.TENSORFLOW
            },
            
            # HuggingFace Transformers
            'huggingface': {
                'modules': ['transformers', 'diffusers', 'datasets'],
                'methods': ['forward'],
                'optional_methods': ['generate', 'encode', 'decode'],
                'attributes': ['config', 'tokenizer'],
                'type': ModelType.HUGGINGFACE
            },
            
            # XGBoost
            'xgboost': {
                'modules': ['xgboost'],
                'methods': ['fit', 'predict'],
                'optional_methods': ['predict_proba', 'save_model', 'load_model'],
                'attributes': ['booster', 'feature_importances_'],
                'type': ModelType.XGBOOST
            },
            
            # LightGBM
            'lightgbm': {
                'modules': ['lightgbm', 'lgb'],
                'methods': ['fit', 'predict'],
                'optional_methods': ['predict_proba', 'save_model'],
                'attributes': ['feature_importances_', 'booster_'],
                'type': ModelType.LIGHTGBM
            }
        }
    
    def detect_model_framework(self, model: Any) -> Tuple[str, ModelType]:
        """Detect the ML framework and return framework name and ModelType."""
        try:
            model_class = model.__class__
            module_name = model_class.__module__.lower()
            class_name = model_class.__name__.lower()
            
            # Check module-based detection first
            for framework, signature in self.framework_signatures.items():
                if any(module in module_name for module in signature['modules']):
                    return framework, signature['type']
            
            # Check method-based detection
            for framework, signature in self.framework_signatures.items():
                required_methods = signature['methods']
                optional_methods = signature.get('optional_methods', [])
                
                has_required = all(hasattr(model, method) for method in required_methods)
                has_optional = sum(1 for method in optional_methods if hasattr(model, method))
                
                # If has required methods and at least 50% of optional methods
                if has_required and has_optional >= len(optional_methods) * 0.5:
                    return framework, signature['type']
            
            # Check for callable models (custom functions)
            if callable(model) and not hasattr(model, '__class__'):
                return 'custom_function', ModelType.CUSTOM
            
            return 'custom', ModelType.CUSTOM
            
        except Exception as e:
            warnings.warn(f"Framework detection failed: {e}")
            return 'unknown', ModelType.CUSTOM
    
    def get_model_capabilities(self, model: Any) -> Dict[str, bool]:
        """Determine what capabilities the model has."""
        capabilities = {
            'training': False,
            'prediction': False,
            'probability_prediction': False,
            'batch_prediction': False,
            'online_learning': False,
            'feature_importance': False,
            'model_explanation': False,
            'serialization': False
        }
        
        try:
            # Training capability
            capabilities['training'] = hasattr(model, 'fit') or hasattr(model, 'train')
            
            # Prediction capability
            capabilities['prediction'] = (
                hasattr(model, 'predict') or 
                hasattr(model, 'forward') or 
                hasattr(model, '__call__') or
                callable(model)
            )
            
            # Probability prediction
            capabilities['probability_prediction'] = (
                hasattr(model, 'predict_proba') or 
                hasattr(model, 'predict_log_proba')
            )
            
            # Batch prediction (most models support this)
            capabilities['batch_prediction'] = capabilities['prediction']
            
            # Online learning
            capabilities['online_learning'] = hasattr(model, 'partial_fit')
            
            # Feature importance
            capabilities['feature_importance'] = (
                hasattr(model, 'feature_importances_') or
                hasattr(model, 'coef_') or
                hasattr(model, 'feature_importance')
            )
            
            # Model explanation (built-in interpretability)
            capabilities['model_explanation'] = (
                hasattr(model, 'decision_path') or
                hasattr(model, 'tree_') or
                capabilities['feature_importance']
            )
            
            # Serialization support
            capabilities['serialization'] = (
                hasattr(model, 'save') or 
                hasattr(model, 'save_model') or
                hasattr(model, '__getstate__')
            )
            
        except Exception as e:
            warnings.warn(f"Capability detection failed: {e}")
        
        return capabilities


class UniversalDataProcessor:
    """Universal data preprocessing for different model types and data formats."""
    
    def __init__(self):
        self.text_processors = {}
        self.numeric_processors = {}
    
    def process_data(self, data: Any, data_type: 'DataType') -> Any:
        """
        Process data based on the specified data type.
        
        Args:
            data: Input data to process
            data_type: DataType enum specifying how to process the data
            
        Returns:
            Processed data ready for model consumption
        """
        try:
            from .policy import DataType
            
            if data_type == DataType.NUMERICAL:
                return self._process_numeric_simple(data)
            elif data_type == DataType.TEXT:
                return self._process_text_simple(data)
            elif data_type == DataType.MIXED:
                return self._process_mixed_simple(data)
            elif data_type == DataType.CATEGORICAL:
                return self._process_categorical_simple(data)
            else:
                # Default processing
                return self._process_default(data)
                
        except Exception as e:
            warnings.warn(f"Data processing failed: {e}")
            return data
    
    def _process_numeric_simple(self, data):
        """Simple numeric data processing."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        else:
            return np.array([data])
    
    def _process_text_simple(self, data):
        """Simple text data processing."""
        if isinstance(data, str):
            data = [data]
        
        # Simple character-level encoding
        max_length = 100
        encoded = []
        for text in data:
            text_encoded = [ord(c) for c in str(text)[:max_length]]
            text_encoded += [0] * (max_length - len(text_encoded))  # Padding
            encoded.append(text_encoded)
        
        return np.array(encoded)
    
    def _process_mixed_simple(self, data):
        """Simple mixed data processing."""
        if isinstance(data, dict):
            # Try to concatenate numeric parts
            numeric_parts = []
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    numeric_parts.append(value)
                elif isinstance(value, np.ndarray):
                    numeric_parts.extend(value.flatten())
                elif isinstance(value, (list, tuple)):
                    numeric_parts.extend([float(x) if isinstance(x, (int, float)) else 0.0 for x in value])
            
            return np.array(numeric_parts) if numeric_parts else np.array([0.0])
        else:
            return self._process_default(data)
    
    def _process_categorical_simple(self, data):
        """Simple categorical data processing."""
        if isinstance(data, (list, tuple)):
            # Simple label encoding
            unique_vals = list(set(data))
            return np.array([unique_vals.index(x) for x in data])
        else:
            return np.array([0])
    
    def _process_default(self, data):
        """Default data processing."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        else:
            return np.array([data])

    def process_training_data(self, model: Any, training_data: List[Dict[str, Any]], 
                            model_type: ModelType) -> Tuple[Any, Any]:
        """Process training data for any model type."""
        try:
            # Extract content and targets
            X, y = self._extract_xy_from_ciaf_data(training_data)
            
            if not X:
                return None, None
            
            # Determine data type and process accordingly
            data_type = self._determine_data_type(X)
            
            if data_type == 'text':
                return self._process_text_data(X, y, model_type)
            elif data_type == 'numeric':
                return self._process_numeric_data(X, y, model_type)
            elif data_type == 'mixed':
                return self._process_mixed_data(X, y, model_type)
            else:
                return self._process_custom_data(X, y, model_type)
                
        except Exception as e:
            warnings.warn(f"Training data processing failed: {e}")
            return None, None
    
    def process_inference_input(self, model: Any, query: Any, 
                               model_type: ModelType) -> Any:
        """Process inference input for any model type."""
        try:
            if isinstance(query, str):
                return self._process_text_query(query, model_type)
            elif isinstance(query, (int, float)):
                return self._process_numeric_query(query, model_type)
            elif isinstance(query, (list, tuple)):
                return self._process_array_query(query, model_type)
            elif isinstance(query, np.ndarray):
                return self._process_numpy_query(query, model_type)
            elif hasattr(query, 'shape'):  # Tensor-like objects
                return self._process_tensor_query(query, model_type)
            else:
                return self._process_custom_query(query, model_type)
                
        except Exception as e:
            warnings.warn(f"Inference input processing failed: {e}")
            return query
    
    def _extract_xy_from_ciaf_data(self, training_data: List[Dict[str, Any]]) -> Tuple[List, List]:
        """Extract X and y from CIAF format data."""
        X = []
        y = []
        
        for item in training_data:
            # Extract content
            if "content" in item:
                X.append(item["content"])
            else:
                X.append(str(item))
            
            # Extract target
            if "metadata" in item and "target" in item["metadata"]:
                y.append(item["metadata"]["target"])
            elif "target" in item:
                y.append(item["target"])
            elif "label" in item:
                y.append(item["label"])
        
        return X, y if y else None
    
    def _determine_data_type(self, X: List) -> str:
        """Determine the primary data type in the dataset."""
        if not X:
            return 'unknown'
        
        text_count = sum(1 for x in X[:10] if isinstance(x, str))  # Sample first 10
        numeric_count = sum(1 for x in X[:10] if isinstance(x, (int, float)))
        array_count = sum(1 for x in X[:10] if isinstance(x, (list, tuple, np.ndarray)))
        
        if text_count > numeric_count and text_count > array_count:
            return 'text'
        elif numeric_count > text_count and numeric_count > array_count:
            return 'numeric'
        elif array_count > 0:
            return 'numeric'  # Treat arrays as numeric data
        else:
            return 'mixed'
    
    def _process_text_data(self, X: List, y: List, model_type: ModelType) -> Tuple[Any, Any]:
        """Process text data for different model types."""
        try:
            if model_type == ModelType.HUGGINGFACE:
                # For HuggingFace models, return text as-is (tokenization happens in model)
                return X, y
            
            elif model_type in [ModelType.SCIKIT_LEARN, ModelType.XGBOOST, ModelType.LIGHTGBM]:
                # Use TF-IDF vectorization for traditional ML models
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                    X_vectorized = vectorizer.fit_transform(X).toarray()
                    self.text_processors[id(vectorizer)] = vectorizer  # Cache for inference
                    return X_vectorized, y
                except ImportError:
                    # Fallback to simple text features
                    X_features = np.array([[len(str(x)), str(x).count(' '), 
                                          hash(str(x)) % 1000] for x in X])
                    return X_features, y
            
            elif model_type in [ModelType.PYTORCH, ModelType.TENSORFLOW]:
                # For deep learning models, return processed text
                # This would ideally use proper tokenization, but we'll use simple encoding
                max_length = min(512, max(len(str(x)) for x in X) if X else 100)
                X_encoded = []
                for text in X:
                    # Simple character-level encoding
                    encoded = [ord(c) for c in str(text)[:max_length]]
                    encoded += [0] * (max_length - len(encoded))  # Padding
                    X_encoded.append(encoded)
                return np.array(X_encoded), y
            
            else:
                return X, y
                
        except Exception as e:
            warnings.warn(f"Text processing failed: {e}")
            return X, y
    
    def _process_numeric_data(self, X: List, y: List, model_type: ModelType) -> Tuple[Any, Any]:
        """Process numeric data for different model types."""
        try:
            # Convert to numpy array
            if all(isinstance(x, (int, float)) for x in X):
                X_array = np.array(X).reshape(-1, 1)
            elif all(isinstance(x, (list, tuple)) for x in X):
                X_array = np.array(X)
            else:
                # Mixed numeric data - try to convert
                X_converted = []
                for x in X:
                    if isinstance(x, (int, float)):
                        X_converted.append([float(x)])
                    elif isinstance(x, (list, tuple)):
                        X_converted.append([float(v) if isinstance(v, (int, float)) else 0.0 for v in x])
                    else:
                        X_converted.append([0.0])  # Default value
                X_array = np.array(X_converted)
            
            # Convert y to appropriate format
            if y:
                try:
                    y_array = np.array(y)
                except:
                    y_array = y
            else:
                y_array = None
            
            return X_array, y_array
            
        except Exception as e:
            warnings.warn(f"Numeric processing failed: {e}")
            return X, y
    
    def _process_mixed_data(self, X: List, y: List, model_type: ModelType) -> Tuple[Any, Any]:
        """Process mixed data types."""
        try:
            # Create feature matrix from mixed data
            X_features = []
            for x in X:
                if isinstance(x, str):
                    # Text features
                    features = [len(x), x.count(' '), hash(x) % 1000]
                elif isinstance(x, (int, float)):
                    # Numeric features
                    features = [float(x), 0, 0]
                elif isinstance(x, (list, tuple)):
                    # Array features
                    features = [len(x), sum(x) if all(isinstance(v, (int, float)) for v in x) else 0, 0]
                else:
                    # Default features
                    features = [0, 0, 0]
                
                X_features.append(features)
            
            return np.array(X_features), y
            
        except Exception as e:
            warnings.warn(f"Mixed data processing failed: {e}")
            return X, y
    
    def _process_custom_data(self, X: List, y: List, model_type: ModelType) -> Tuple[Any, Any]:
        """Process custom data formats."""
        return X, y
    
    def _process_text_query(self, query: str, model_type: ModelType) -> Any:
        """Process text query for inference."""
        if model_type == ModelType.HUGGINGFACE:
            return query
        elif model_type in [ModelType.SCIKIT_LEARN, ModelType.XGBOOST, ModelType.LIGHTGBM]:
            # Simple text features for traditional ML
            return np.array([[len(query), query.count(' '), hash(query) % 1000]])
        elif model_type in [ModelType.PYTORCH, ModelType.TENSORFLOW]:
            # Character-level encoding
            encoded = [ord(c) for c in query[:512]]
            encoded += [0] * (512 - len(encoded))  # Padding
            return np.array([encoded])
        else:
            return query
    
    def _process_numeric_query(self, query: Union[int, float], model_type: ModelType) -> Any:
        """Process numeric query for inference."""
        return np.array([[float(query)]])
    
    def _process_array_query(self, query: Union[List, Tuple], model_type: ModelType) -> Any:
        """Process array query for inference."""
        try:
            return np.array([query])
        except:
            return np.array([[len(query), sum(query) if all(isinstance(v, (int, float)) for v in query) else 0]])
    
    def _process_numpy_query(self, query: np.ndarray, model_type: ModelType) -> Any:
        """Process numpy array query for inference."""
        if query.ndim == 1:
            return query.reshape(1, -1)
        return query
    
    def _process_tensor_query(self, query: Any, model_type: ModelType) -> Any:
        """Process tensor-like query for inference."""
        try:
            # Try to convert to numpy
            if hasattr(query, 'numpy'):
                return query.numpy()
            elif hasattr(query, 'detach'):
                return query.detach().numpy()
            else:
                return query
        except:
            return query
    
    def _process_custom_query(self, query: Any, model_type: ModelType) -> Any:
        """Process custom query formats."""
        return query


class UniversalModelAdapter(ModelAdapter):
    """Universal model adapter supporting all ML frameworks."""
    
    def __init__(self, policy=None):
        """Initialize with optional policy parameter."""
        self.policy = policy
        self.detector = UniversalModelDetector()
        self.processor = UniversalDataProcessor()
    
    def detect_model_type(self, model: Any) -> str:
        """Detect the type/framework of the provided model."""
        framework, model_type = self.detector.detect_model_framework(model)
        return model_type.value
    
    def predict(self, model: Any, input_data: Any) -> Any:
        """Universal predict method for any model type."""
        try:
            # Detect framework and use appropriate prediction method
            framework, _ = self.detector.detect_model_framework(model)
            
            if framework == 'sklearn':
                return model.predict(input_data)
            elif framework == 'pytorch':
                return self._pytorch_predict(model, input_data)
            elif framework == 'tensorflow':
                return self._tensorflow_predict(model, input_data)
            elif framework == 'huggingface':
                return self._huggingface_predict(model, input_data)
            else:
                # Fallback to generic prediction
                if hasattr(model, 'predict'):
                    return model.predict(input_data)
                elif hasattr(model, '__call__'):
                    return model(input_data)
                else:
                    raise AttributeError(f"Model {type(model)} has no predict method")
                    
        except Exception as e:
            warnings.warn(f"Universal prediction failed: {e}")
            return f"Universal fallback prediction for {type(model)}"
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get comprehensive model information."""
        framework, model_type = self.detector.detect_model_framework(model)
        return {
            'framework': framework,
            'model_type': model_type.value,
            'class_name': model.__class__.__name__,
            'module': model.__class__.__module__,
        }
    
    def get_model_metadata(self, model: Any) -> Dict[str, Any]:
        """Get universal model metadata."""
        try:
            metadata = {}
            framework, model_type = self.detector.detect_model_framework(model)
            
            metadata['framework'] = framework
            metadata['model_type'] = model_type.value
            metadata['class_name'] = model.__class__.__name__
            metadata['module'] = model.__class__.__module__
            
            # Add framework-specific metadata
            if framework == 'sklearn':
                self._extract_sklearn_metadata(model, metadata)
            elif framework == 'pytorch':
                self._extract_pytorch_metadata(model, metadata)
            elif framework == 'tensorflow':
                self._extract_tensorflow_metadata(model, metadata)
            elif framework == 'huggingface':
                self._extract_huggingface_metadata(model, metadata)
            elif framework in ['xgboost', 'lightgbm']:
                self._extract_boosting_metadata(model, metadata)
                
            return metadata
        except Exception as e:
            return {'error': str(e), 'framework': 'unknown'}
    
    def validate_model_compatibility(self, model: Any) -> Dict[str, Any]:
        """Validate if model is compatible with CIAF wrapper."""
        try:
            framework, model_type = self.detector.detect_model_framework(model)
            capabilities = self.detector.get_model_capabilities(model)
            
            result = {
                "is_compatible": True,
                "framework": framework,
                "model_type": model_type.value,
                "capabilities": capabilities,
                "warnings": [],
                "errors": []
            }
            
            # Check minimum requirements
            if not capabilities['prediction']:
                result["errors"].append("Model must support prediction")
                result["is_compatible"] = False
            
            # Add framework-specific warnings
            if framework == 'custom':
                result["warnings"].append("Custom model detected - some features may not work")
            
            if not capabilities['training']:
                result["warnings"].append("Model does not support training")
            
            return result
            
        except Exception as e:
            return {
                "is_compatible": False,
                "errors": [f"Validation failed: {e}"],
                "warnings": [],
                "framework": "unknown",
                "model_type": "custom",
                "capabilities": {}
            }
    
    def extract_model_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract comprehensive metadata from any model type."""
        metadata = {
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        try:
            framework, model_type = self.detector.detect_model_framework(model)
            capabilities = self.detector.get_model_capabilities(model)
            
            metadata.update({
                "framework": framework,
                "model_type": model_type.value,
                "capabilities": capabilities
            })
            
            # Framework-specific metadata extraction
            if framework == 'sklearn':
                self._extract_sklearn_metadata(model, metadata)
            elif framework == 'pytorch':
                self._extract_pytorch_metadata(model, metadata)
            elif framework == 'tensorflow':
                self._extract_tensorflow_metadata(model, metadata)
            elif framework == 'huggingface':
                self._extract_huggingface_metadata(model, metadata)
            elif framework in ['xgboost', 'lightgbm']:
                self._extract_boosting_metadata(model, metadata)
            
            return metadata
            
        except Exception as e:
            metadata["extraction_error"] = str(e)
            return metadata
    
    def prepare_training_data(self, model: Any, training_data: List[Dict[str, Any]]) -> Tuple[Any, Any]:
        """Prepare training data for any model type."""
        framework, model_type = self.detector.detect_model_framework(model)
        return self.processor.process_training_data(model, training_data, model_type)
    
    def handle_model_prediction(self, model: Any, query: Any, preprocessed_query: Any = None) -> Any:
        """Handle prediction for any model type."""
        try:
            framework, model_type = self.detector.detect_model_framework(model)
            
            # Use preprocessed query if available
            input_data = preprocessed_query if preprocessed_query is not None else query
            
            # Process the input for the specific model type
            processed_input = self.processor.process_inference_input(model, input_data, model_type)
            
            # Make prediction based on model type
            if framework == 'sklearn' or hasattr(model, 'predict'):
                return model.predict(processed_input)
            elif framework == 'pytorch':
                return self._pytorch_predict(model, processed_input)
            elif framework == 'tensorflow':
                return self._tensorflow_predict(model, processed_input)
            elif framework == 'huggingface':
                return self._huggingface_predict(model, processed_input)
            elif callable(model):
                return model(processed_input)
            else:
                warnings.warn(f"Unknown prediction method for framework: {framework}")
                return f"Fallback prediction for: {query}"
                
        except Exception as e:
            warnings.warn(f"Model prediction failed: {e}")
            return f"Fallback prediction for: {query}"
    
    def _extract_sklearn_metadata(self, model: Any, metadata: Dict[str, Any]) -> None:
        """Extract sklearn-specific metadata."""
        if hasattr(model, 'get_params'):
            metadata['parameters'] = model.get_params()
        if hasattr(model, 'feature_importances_'):
            metadata['has_feature_importance'] = True
        if hasattr(model, 'n_features_in_'):
            metadata['n_features'] = model.n_features_in_
    
    def _extract_pytorch_metadata(self, model: Any, metadata: Dict[str, Any]) -> None:
        """Extract PyTorch-specific metadata."""
        if hasattr(model, 'parameters'):
            metadata['parameter_count'] = sum(p.numel() for p in model.parameters())
        if hasattr(model, 'training'):
            metadata['is_training'] = model.training
        if hasattr(model, 'device'):
            metadata['device'] = str(model.device)
    
    def _extract_tensorflow_metadata(self, model: Any, metadata: Dict[str, Any]) -> None:
        """Extract TensorFlow/Keras-specific metadata."""
        if hasattr(model, 'count_params'):
            metadata['parameter_count'] = model.count_params()
        if hasattr(model, 'layers'):
            metadata['layer_count'] = len(model.layers)
    
    def _extract_huggingface_metadata(self, model: Any, metadata: Dict[str, Any]) -> None:
        """Extract HuggingFace-specific metadata."""
        if hasattr(model, 'config'):
            metadata['model_config'] = str(type(model.config))
        if hasattr(model, 'name_or_path'):
            metadata['model_name'] = model.name_or_path
    
    def _extract_boosting_metadata(self, model: Any, metadata: Dict[str, Any]) -> None:
        """Extract XGBoost/LightGBM-specific metadata."""
        if hasattr(model, 'feature_importances_'):
            metadata['has_feature_importance'] = True
        if hasattr(model, 'n_features_in_'):
            metadata['n_features'] = model.n_features_in_
    
    def _pytorch_predict(self, model: Any, input_data: Any) -> Any:
        """Handle PyTorch model prediction."""
        try:
            import torch
            model.eval()
            with torch.no_grad():
                if not isinstance(input_data, torch.Tensor):
                    input_data = torch.tensor(input_data, dtype=torch.float32)
                return model(input_data).numpy()
        except Exception as e:
            warnings.warn(f"PyTorch prediction failed: {e}")
            return f"PyTorch fallback prediction"
    
    def _tensorflow_predict(self, model: Any, input_data: Any) -> Any:
        """Handle TensorFlow model prediction."""
        try:
            if hasattr(model, 'predict'):
                return model.predict(input_data)
            else:
                return model(input_data).numpy()
        except Exception as e:
            warnings.warn(f"TensorFlow prediction failed: {e}")
            return f"TensorFlow fallback prediction"
    
    def _huggingface_predict(self, model: Any, input_data: Any) -> Any:
        """Handle HuggingFace model prediction."""
        try:
            # This is a simplified approach - would need proper tokenization in practice
            if hasattr(model, 'generate'):
                return model.generate(input_data)
            elif hasattr(model, 'forward'):
                return model.forward(input_data)
            else:
                return model(input_data)
        except Exception as e:
            warnings.warn(f"HuggingFace prediction failed: {e}")
            return f"HuggingFace fallback prediction"
    
    def adapt_model(self, model: Any, model_name: str, **kwargs) -> Any:
        """
        Adapt a model to work with CIAF wrapper system.
        
        This method wraps the model with universal adapter capabilities.
        """
        # Create a wrapper class that contains the model and adapter functionality
        class UniversalModelWrapper:
            def __init__(self, model, adapter, model_name):
                self.model = model
                self.adapter = adapter
                self.model_name = model_name
                
            def predict(self, input_data):
                """Universal predict method."""
                return self.adapter.predict(self.model, input_data)
                
            def get_metadata(self):
                """Get model metadata."""
                return self.adapter.get_model_metadata(self.model)
                
            def __getattr__(self, name):
                """Forward attribute access to the original model."""
                return getattr(self.model, name)
        
        return UniversalModelWrapper(model, self, model_name)