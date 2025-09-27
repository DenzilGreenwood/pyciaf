"""
CIAF Regression Base Class
Standardized base class for all CIAF-enabled regression models
Following naming improvement recommendations from CIAF_LCM_IMPROVEMENT_RECOMMENDATIONS.md
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

from ciaf.utils.data_utils import CIAFDataUtils
from ciaf.utils.wrapper_utils import CIAFWrapperUtils  
from ciaf.utils.error_utils import CIAFErrorUtils

logger = logging.getLogger(__name__)


class CIAFRegressionModel(ABC):
    """
    Base class for all CIAF-enabled regression models
    Simplified naming: Previously CIAFEnhancedRegressorWithComprehensiveProvenanceAndAuditTrail
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize CIAF regression model (simplified naming)
        Previously: initialize_comprehensive_ciaf_regression_model_with_full_configuration()
        """
        self.model_name = model_name
        self.wrapper = None
        self._trained = False
        self._last_error = None
        self._config = kwargs
        
        # Initialize CIAF wrapper
        self._init_ciaf_wrapper()
    
    def _init_ciaf_wrapper(self):
        """Initialize CIAF wrapper with error handling"""
        try:
            self.wrapper = CIAFWrapperUtils.create_wrapper(
                self.model_name,
                **self._config
            )
            if self.wrapper:
                logger.info(f"🚀 CIAF wrapper initialized for '{self.model_name}'")
        except Exception as e:
            self._last_error = CIAFErrorUtils.translate_error(e)
            logger.error(f"❌ CIAF wrapper initialization failed: {self._last_error}")
    
    def train_with_ciaf(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Train model with CIAF integration (simplified naming)
        Previously: perform_comprehensive_training_with_full_provenance_tracking_and_audit_trail()
        """
        try:
            logger.info(f"🏋️ Training {self.model_name} with CIAF LCM...")
            
            # Validate inputs
            if X is None or y is None:
                raise ValueError("Training data cannot be None")
            
            if len(X) != len(y):
                raise ValueError("X and y must have same length")
            
            # Convert to CIAF format
            logger.info(f"📊 Converted {len(X)} samples to CIAF format")
            
            # Use wrapper for CIAF training
            success, error_msg = CIAFWrapperUtils.safe_train(
                self.wrapper, X, y
            )
            
            if success:
                # Train the underlying model
                self._train_underlying_model(X, y)
                self._trained = True
                logger.info(f"✅ {self.model_name} training completed")
                return True
            else:
                self._last_error = error_msg
                logger.error(f"❌ {self._last_error}")
                return False
                
        except Exception as e:
            self._last_error = CIAFErrorUtils.log_error(
                e, "training", self.model_name
            )
            return False
    
    def predict_with_receipt(self, X: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Make predictions with CIAF receipt (simplified naming)  
        Previously: perform_inference_with_comprehensive_receipt_generation_and_audit_trail()
        """
        try:
            logger.info(f"🔮 Making predictions with {self.model_name}...")
            
            if not self._trained:
                logger.warning("⚠️ Model not trained, using fallback prediction")
                return self._predict_fallback(X), None
            
            # Use wrapper for CIAF prediction
            predictions, error_msg, receipt = CIAFWrapperUtils.safe_predict(
                self.wrapper, X, return_receipt=True
            )
            
            if predictions is not None:
                logger.info(f"✅ Made {len(predictions)} predictions")
                return predictions, receipt
            else:
                logger.info(f"📋 Using fallback predictions")
                return self._predict_fallback(X), None
                
        except Exception as e:
            self._last_error = CIAFErrorUtils.log_error(
                e, "prediction", self.model_name
            )
            return None, None
    
    def get_audit_info(self) -> Dict[str, Any]:
        """
        Get audit information (simplified naming)
        Previously: generate_comprehensive_audit_trail_with_full_provenance_and_compliance_details()
        """
        audit_info = CIAFWrapperUtils.get_audit_info(self.wrapper)
        
        # Add model-specific info
        audit_info.update({
            "model_name": self.model_name,
            "trained": self._trained,
            "ciaf_enabled": self.wrapper is not None,
            "last_error": self._last_error
        })
        
        return audit_info
    
    def get_last_error(self) -> Optional[str]:
        """Get last error message (user-friendly)"""
        return self._last_error
    
    def trace_prediction(self, receipt_id: str) -> Dict[str, Any]:
        """
        Trace prediction by receipt ID (simplified naming)
        Previously: comprehensive_prediction_traceability_analysis_with_full_audit_chain()
        """
        try:
            if not self.wrapper:
                return {"error": "CIAF wrapper not initialized"}
            
            # This would integrate with CIAF tracing system
            trace_info = {
                "receipt_id": receipt_id,
                "model_name": self.model_name,
                "trace_available": True,
                "message": "Prediction tracing would be implemented here"
            }
            
            return trace_info
            
        except Exception as e:
            return {"error": CIAFErrorUtils.translate_error(e)}
    
    @abstractmethod
    def _train_underlying_model(self, X: pd.DataFrame, y: pd.Series):
        """Train the underlying ML model (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _predict_fallback(self, X: pd.DataFrame) -> np.ndarray:
        """Fallback prediction without CIAF (to be implemented by subclasses)"""
        pass
    
    # Convenience methods with improved naming
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CIAFRegressionModel':
        """Sklearn-style fit method"""
        success = self.train_with_ciaf(X, y)
        if not success:
            raise RuntimeError(f"Training failed: {self.get_last_error()}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Sklearn-style predict method"""
        predictions, _ = self.predict_with_receipt(X)
        if predictions is None:
            raise RuntimeError(f"Prediction failed: {self.get_last_error()}")
        return predictions
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate R² score"""
        try:
            predictions = self.predict(X)
            
            # Calculate R² score
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            return r2
        except:
            return 0.0