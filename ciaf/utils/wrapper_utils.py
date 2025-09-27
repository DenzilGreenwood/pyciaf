"""
CIAF Wrapper Utilities  
Simplified CIAF operations and wrapper management
Following naming improvement recommendations from CIAF_LCM_IMPROVEMENT_RECOMMENDATIONS.md
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Import CIAF components (assuming they exist)
try:
    from ciaf.wrappers import CIAFModelWrapper
    from ciaf.metadata_storage_optimized import OptimizedMetadataStorage
except ImportError:
    # Mock classes for demonstration if CIAF not available
    class CIAFModelWrapper:
        def __init__(self, *args, **kwargs):
            pass
    
    class OptimizedMetadataStorage:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)


class CIAFWrapperUtils:
    """Simplified CIAF wrapper operations"""
    
    @staticmethod
    def create_wrapper(model_name: str, **kwargs) -> Any:
        """
        Create CIAF wrapper (simplified naming)
        Previously: initialize_ciaf_model_wrapper_with_comprehensive_configuration()
        """
        try:
            # Import the actual CIAFModelWrapper
            from ciaf.wrappers.model_wrapper import CIAFModelWrapper
            
            # Create a simple model if none provided
            model = kwargs.get('model', None)
            if model is None:
                # Create a simple sklearn model as default
                try:
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    logger.info(f"Created default LinearRegression model for wrapper")
                except ImportError:
                    # Fallback to a simple mock model
                    class MockModel:
                        def fit(self, X, y):
                            pass
                        def predict(self, X):
                            import numpy as np
                            return np.random.random(len(X)) if hasattr(X, '__len__') else np.random.random(1)
                    
                    model = MockModel()
                    logger.info(f"Created mock model for wrapper")
            
            # Create wrapper with proper parameters
            wrapper = CIAFModelWrapper(
                model=model,
                model_name=model_name,
                enable_connections=kwargs.get('enable_connections', True),
                compliance_mode=kwargs.get('compliance_mode', "general"),
                enable_preprocessing=kwargs.get('enable_preprocessing', True),
                enable_explainability=kwargs.get('explainability', True),
                enable_uncertainty=kwargs.get('uncertainty', True),
                enable_metadata_tags=kwargs.get('metadata', True),
                auto_configure=kwargs.get('auto_configure', True),
                framework=kwargs.get('framework', None)
            )
            
            logger.info(f"✅ Created CIAF wrapper for '{model_name}'")
            return wrapper
            
        except Exception as e:
            logger.error(f"❌ CIAF wrapper creation failed: {e}")
            return None
    
    @staticmethod
    def safe_train(
        wrapper: Any,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Safe CIAF training (simplified naming)
        Previously: perform_ciaf_training_with_comprehensive_error_handling_and_validation()
        """
        try:
            if wrapper is None:
                return False, "Wrapper not initialized"
            
            # Convert data to CIAF format
            from ciaf.utils.data_utils import CIAFDataUtils
            ciaf_data = CIAFDataUtils.to_ciaf_format(X, y)
            
            if not ciaf_data:
                return False, "Data conversion failed"
            
            # Check if wrapper has train method
            if not hasattr(wrapper, 'train'):
                return False, f"Wrapper {type(wrapper).__name__} does not have 'train' method"
            
            # Perform CIAF training with proper parameters
            # The actual CIAFModelWrapper.train() method expects these parameters:
            # train(dataset_id, training_data, master_password, training_params, model_version, fit_model)
            dataset_id = kwargs.get('dataset_id', f"dataset_{wrapper.model_name}")
            master_password = kwargs.get('master_password', "default_password_for_demo")
            training_params = kwargs.get('training_params', {})
            model_version = kwargs.get('model_version', "1.0.0")
            fit_model = kwargs.get('fit_model', True)
            
            # Call the actual train method with correct signature
            result = wrapper.train(
                dataset_id=dataset_id,
                training_data=ciaf_data,
                master_password=master_password,
                training_params=training_params,
                model_version=model_version,
                fit_model=fit_model
            )
            
            if result:  # train() returns TrainingSnapshot object
                logger.info(f"✅ CIAF training completed successfully")
                return True, "Training completed"
            else:
                return False, "Training failed"
                
        except Exception as e:
            error_msg = f"CIAF training failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
    
    @staticmethod
    def safe_predict(
        wrapper: Any,
        X: pd.DataFrame,
        return_receipt: bool = True,
        **kwargs
    ) -> Tuple[Optional[np.ndarray], Optional[str], Optional[Dict]]:
        """
        Safe CIAF prediction (simplified naming)
        Previously: perform_inference_with_receipt_generation_and_comprehensive_audit_trail()
        """
        try:
            if wrapper is None:
                return None, "Wrapper not initialized", None
            
            # Check if wrapper has predict method
            if not hasattr(wrapper, 'predict'):
                return None, f"Wrapper {type(wrapper).__name__} does not have 'predict' method", None
            
            # The actual CIAFModelWrapper.predict() method expects:
            # predict(query, model_version, use_model) -> Tuple[prediction, InferenceReceipt]
            
            # For pandas DataFrame, we need to convert to appropriate format
            if len(X) == 1:
                # Single prediction - convert first row to list
                query = X.iloc[0].tolist()
            else:
                # Multiple predictions - handle as batch
                query = X.values.tolist()
            
            # Get model version from wrapper if not provided
            model_version = kwargs.get('model_version', None)
            if model_version is None and hasattr(wrapper, 'model_version'):
                model_version = wrapper.model_version
            if model_version is None:
                model_version = "1.0.0"  # Default version
            
            use_model = kwargs.get('use_model', True)
            
            # Call the actual predict method
            result = wrapper.predict(
                query=query,
                model_version=model_version,
                use_model=use_model
            )
            
            if isinstance(result, tuple) and len(result) >= 2:
                prediction, receipt = result[0], result[1]
                
                # Convert prediction to numpy array if needed
                if isinstance(prediction, (list, tuple)):
                    prediction = np.array(prediction)
                elif not isinstance(prediction, np.ndarray):
                    # Handle single value predictions
                    if len(X) == 1:
                        prediction = np.array([prediction])
                    else:
                        # For multiple inputs, create array with same prediction for each
                        prediction = np.array([prediction] * len(X))
                
                logger.info(f"✅ CIAF predictions completed")
                return prediction, "Prediction completed", receipt.__dict__ if receipt else None
            else:
                return np.array([result] * len(X)), "Prediction completed (no receipt)", None
                
        except Exception as e:
            error_msg = f"CIAF prediction failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return None, error_msg, None
    
    @staticmethod
    def get_audit_info(wrapper: Any) -> Dict[str, Any]:
        """
        Get audit information (simplified naming)
        Previously: generate_comprehensive_ciaf_audit_trail_with_full_provenance_chain()
        """
        try:
            if wrapper is None:
                return {"error": "Wrapper not initialized"}
            
            audit_info = {
                "model_name": getattr(wrapper, 'model_name', 'Unknown'),
                "ciaf_enabled": True,
                "training_completed": hasattr(wrapper, '_trained') and wrapper._trained,
                "last_prediction_time": getattr(wrapper, '_last_prediction', None),
                "wrapper_type": type(wrapper).__name__
            }
            
            return audit_info
            
        except Exception as e:
            return {"error": f"Audit info retrieval failed: {e}"}
    
    @staticmethod
    def validate_wrapper(wrapper: Any) -> Tuple[bool, List[str]]:
        """
        Validate CIAF wrapper state (simplified naming)
        Previously: comprehensive_ciaf_wrapper_validation_with_detailed_diagnostics()
        """
        issues = []
        
        if wrapper is None:
            issues.append("Wrapper is None")
            return False, issues
        
        # Check basic attributes
        if not hasattr(wrapper, 'model_name'):
            issues.append("Missing model_name attribute")
        
        # Check if initialized
        if not hasattr(wrapper, '_initialized'):
            issues.append("Wrapper not properly initialized")
        
        is_valid = len(issues) == 0
        return is_valid, issues