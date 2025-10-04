"""
Enhanced CIAF Model Wrapper with Deferred LCM Support
=====================================================

This module extends the existing CIAFModelWrapper to support deferred LCM processing
for dramatically improved inference performance while maintaining full compliance.
"""

import time
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from ciaf
sys.path.insert(0, str(Path(__file__).parent.parent))

# CIAF imports - deferred to avoid circular imports
CIAF_AVAILABLE = True
_ciaf_model_wrapper = None
_adaptive_lcm_wrapper = None
_deferred_lcm_processor = None
_lcm_mode = None
_inference_priority = None
_adaptive_lcm_config = None

def _get_ciaf_components():
    """Lazy import of CIAF components to avoid circular imports."""
    global _ciaf_model_wrapper, _adaptive_lcm_wrapper, _deferred_lcm_processor
    global _lcm_mode, _inference_priority, _adaptive_lcm_config
    
    if _ciaf_model_wrapper is None:
        try:
            from ciaf.wrappers.model_wrapper import CIAFModelWrapper
            from ciaf.adaptive_lcm import AdaptiveLCMWrapper, AdaptiveLCMConfig, LCMMode, InferencePriority
            from ciaf.deferred_lcm import DeferredLCMProcessor
            _ciaf_model_wrapper = CIAFModelWrapper
            _adaptive_lcm_wrapper = AdaptiveLCMWrapper
            _deferred_lcm_processor = DeferredLCMProcessor
            _lcm_mode = LCMMode
            _inference_priority = InferencePriority
            _adaptive_lcm_config = AdaptiveLCMConfig
            return True
        except ImportError:
            global CIAF_AVAILABLE
            CIAF_AVAILABLE = False
            # Create fallback enums and classes
            from enum import Enum
            
            class LCMMode(Enum):
                IMMEDIATE = "immediate"
                DEFERRED = "deferred"
                ADAPTIVE = "adaptive"
            
            class InferencePriority(Enum):
                LOW = "low"
                NORMAL = "normal"
                HIGH = "high"
                
            class MockAdaptiveLCMConfig:
                def __init__(self, **kwargs):
                    self.default_mode = kwargs.get('default_mode', LCMMode.ADAPTIVE)
            
            class MockCIAFModelWrapper:
                def __init__(self, *args, **kwargs):
                    pass
            
            _lcm_mode = LCMMode
            _inference_priority = InferencePriority
            _adaptive_lcm_config = MockAdaptiveLCMConfig
            _ciaf_model_wrapper = MockCIAFModelWrapper
            return False
    return CIAF_AVAILABLE

# Initialize fallback components
_get_ciaf_components()

# Create aliases for the components to avoid direct global access
def get_components():
    _get_ciaf_components()
    return (_ciaf_model_wrapper, _adaptive_lcm_wrapper, _deferred_lcm_processor,
            _lcm_mode, _inference_priority, _adaptive_lcm_config)

CIAFModelWrapper, AdaptiveLCMWrapper, DeferredLCMProcessor, LCMMode, InferencePriority, AdaptiveLCMConfig = get_components()

try:
    from ciaf.metadata_tags import create_classification_tag
except ImportError:
    def create_classification_tag(**kwargs):
        return type('Tag', (), {'to_dict': lambda: kwargs})()

class EnhancedCIAFModelWrapper(CIAFModelWrapper):
    """
    Enhanced CIAF wrapper with deferred LCM support.
    
    This wrapper extends the standard CIAFModelWrapper to provide:
    - Deferred LCM processing for faster inference
    - Adaptive mode switching based on system load
    - Priority-based inference handling
    - Performance optimization while maintaining compliance
    """
    
    def __init__(self,
                 model: Any,
                 model_name: str,
                 enable_connections: bool = True,
                 compliance_mode: str = "general",
                 enable_preprocessing: bool = True,
                 enable_explainability: bool = True,
                 enable_uncertainty: bool = True,
                 enable_metadata_tags: bool = True,
                 auto_configure: bool = True,
                 # New deferred LCM parameters
                 enable_deferred_lcm: bool = True,
                 lcm_config: Optional[Any] = None,
                 default_lcm_mode: Any = None):
        """
        Initialize Enhanced CIAF wrapper with deferred LCM support.
        
        Args:
            All standard CIAFModelWrapper args plus:
            enable_deferred_lcm: Enable deferred LCM processing
            lcm_config: Configuration for adaptive LCM behavior
            default_lcm_mode: Default LCM processing mode
        """
        
        # Initialize base wrapper
        super().__init__(
            model=model,
            model_name=model_name,
            enable_connections=enable_connections,
            compliance_mode=compliance_mode,
            enable_preprocessing=enable_preprocessing,
            enable_explainability=enable_explainability,
            enable_uncertainty=enable_uncertainty,
            enable_metadata_tags=enable_metadata_tags,
            auto_configure=auto_configure
        )
        
        # Deferred LCM setup
        self.enable_deferred_lcm = enable_deferred_lcm
        self.lcm_config = lcm_config or AdaptiveLCMConfig(default_mode=default_lcm_mode)
        self.adaptive_lcm = None
        
        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'deferred_predictions': 0,
            'immediate_predictions': 0,
            'total_inference_time': 0.0,
            'total_lcm_time': 0.0,
            'average_inference_time': 0.0,
            'performance_improvement': 0.0
        }
        
        print(f"[INIT] Enhanced CIAF Model Wrapper initialized for '{model_name}'")
        if enable_deferred_lcm:
            print(f"  [LCM] Deferred LCM enabled (mode: {default_lcm_mode.value})")
        
    def train(self,
              X_train: Union[List, np.ndarray, Any],
              y_train: Union[List, np.ndarray, Any],
              version: str = "1.0.0",
              **kwargs) -> str:
        """
        Train the model with CIAF provenance tracking.
        
        This extends the base train method and sets up the adaptive LCM
        wrapper after training is complete.
        """
        
        print(f"[TRAIN] [{self.model_name}] Starting verifiable training for version '{version}'...")
        
        # Convert numpy arrays to the format expected by base CIAF
        training_data = self._prepare_training_data(X_train, y_train)
        
        # Generate a dataset ID
        dataset_id = f"{self.model_name}_dataset_{version}"
        
        # Use a default master password (in production, this should be properly managed)
        master_password = kwargs.get('master_password', 'ciaf_demo_password')
        
        # Call parent train method with correct signature
        training_snapshot = super().train(
            dataset_id=dataset_id,
            training_data=training_data,
            master_password=master_password,
            model_version=version,
            **{k: v for k, v in kwargs.items() if k != 'master_password'}
        )
        
        # Set up adaptive LCM wrapper after training
        if self.enable_deferred_lcm:
            self._setup_adaptive_lcm()
            
        return training_snapshot.snapshot_id if hasattr(training_snapshot, 'snapshot_id') else str(training_snapshot)
        
    def _prepare_training_data(self, X_train, y_train) -> List[Dict[str, Any]]:
        """Convert numpy arrays to CIAF training data format"""
        training_data = []
        
        # Convert to numpy if not already
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
            
        # Create training data in expected format
        for i in range(len(X_train)):
            training_data.append({
                'content': X_train[i].tolist() if isinstance(X_train[i], np.ndarray) else X_train[i],
                'metadata': {
                    'id': f'sample_{i}',  # Required id field
                    'target': y_train[i].item() if isinstance(y_train[i], np.ndarray) else y_train[i],
                    'sample_id': i,
                    'enhanced_wrapper': True
                }
            })
            
        return training_data
        
    def _setup_adaptive_lcm(self):
        """Set up adaptive LCM wrapper after training"""
        if not self.training_snapshot:
            print("WARNING: No training snapshot available for LCM setup")
            return
            
        self.adaptive_lcm = AdaptiveLCMWrapper(
            base_model=self.model,
            config=self.lcm_config,
            model_ref=self.training_snapshot.snapshot_id,
            model_version=self.model_version or "1.0.0"
        )
        
        print(f"[LCM] Adaptive LCM wrapper configured")
        
    def predict(self,
                X: Union[List, np.ndarray, Any],
                priority: Any = None,
                enable_fast_mode: bool = None,
                return_enhanced_info: bool = True,
                **kwargs) -> Union[Any, Dict[str, Any]]:
        """
        Make predictions with optional deferred LCM processing.
        
        Args:
            X: Input data for prediction
            priority: Priority level for LCM processing
            enable_fast_mode: Override to force fast/deferred mode
            return_enhanced_info: Return detailed prediction info
            **kwargs: Additional arguments
            
        Returns:
            Prediction result (raw or enhanced dict based on return_enhanced_info)
        """
        
        if not self.enable_deferred_lcm or self.adaptive_lcm is None:
            # Fall back to standard CIAF prediction
            return self._standard_predict(X, return_enhanced_info, **kwargs)
            
        # Use adaptive LCM prediction
        return self._adaptive_predict(X, priority, enable_fast_mode, return_enhanced_info, **kwargs)
        
    def _standard_predict(self, X, return_enhanced_info: bool, **kwargs):
        """Standard CIAF prediction (original behavior)"""
        start_time = time.time()
        
        # Call parent predict method
        result = super().predict(X, **kwargs)
        
        inference_time = time.time() - start_time
        
        # Update stats
        self.performance_stats['total_predictions'] += 1
        self.performance_stats['immediate_predictions'] += 1
        self.performance_stats['total_inference_time'] += inference_time
        
        if not return_enhanced_info:
            return result
            
        return {
            'prediction': result,
            'inference_time': inference_time,
            'lcm_mode': 'immediate',
            'model_version': self.model_version,
            'timestamp': datetime.now().isoformat(),
            'receipt_id': self.last_receipt.receipt_hash if self.last_receipt else None
        }
        
    def _adaptive_predict(self, X, priority, enable_fast_mode, return_enhanced_info, **kwargs):
        """Adaptive LCM prediction with performance optimization"""
        
        # Determine if we should force a specific mode
        if enable_fast_mode is True:
            # Force deferred mode
            original_mode = self.lcm_config.default_mode
            self.adaptive_lcm.set_mode(LCMMode.DEFERRED)
        elif enable_fast_mode is False:
            # Force immediate mode
            original_mode = self.lcm_config.default_mode
            self.adaptive_lcm.set_mode(LCMMode.IMMEDIATE)
        else:
            original_mode = None
            
        try:
            # Preprocess if needed (reuse existing preprocessing logic)
            processed_X = self._preprocess_input(X)
            
            # Make prediction with adaptive LCM
            result = self.adaptive_lcm.predict(
                input_data=processed_X,
                priority=priority,
                include_receipts=self.enable_connections,
                **kwargs
            )
            
            # Update performance stats
            self._update_performance_stats(result)
            
            # Add enhanced info if requested
            if return_enhanced_info:
                result = self._enhance_prediction_result(result, processed_X)
                
            return result['prediction'] if not return_enhanced_info else result
            
        finally:
            # Restore original mode if we forced a change
            if original_mode is not None:
                self.adaptive_lcm.set_mode(original_mode)
                
    def _preprocess_input(self, X):
        """Preprocess input using existing CIAF preprocessing logic"""
        if not self.enable_preprocessing or not hasattr(self, 'fitted_vectorizer'):
            return X
            
        # Reuse existing preprocessing logic from parent class
        if self.fitted_vectorizer is not None:
            if hasattr(X, 'shape') and len(X.shape) == 1:
                X = [X]
            return self.fitted_vectorizer.transform(X)
        return X
        
    def _enhance_prediction_result(self, result: Dict, input_data) -> Dict:
        """Add enhanced information to prediction result"""
        enhanced_result = result.copy()
        
        # Add explainability if enabled
        if self.enable_explainability and self.explainer:
            try:
                explanation = self.explainer.explain(input_data, result['prediction'])
                enhanced_result['explainability'] = explanation
            except Exception as e:
                enhanced_result['explainability'] = {'error': str(e)}
                
        # Add uncertainty quantification if enabled
        if self.enable_uncertainty and self.uncertainty_quantifier:
            try:
                uncertainty = self.uncertainty_quantifier.quantify(input_data, result['prediction'])
                enhanced_result['uncertainty'] = uncertainty
            except Exception as e:
                enhanced_result['uncertainty'] = {'error': str(e)}
                
        # Add metadata tags if enabled
        if self.enable_metadata_tags:
            try:
                tag = create_classification_tag(
                    prediction=result['prediction'],
                    confidence=enhanced_result.get('uncertainty', {}).get('confidence', 0.75),
                    compliance_level="STANDARD"
                )
                enhanced_result['metadata_tag'] = tag.to_dict()
            except Exception as e:
                enhanced_result['metadata_tag'] = {'error': str(e)}
                
        return enhanced_result
        
    def _update_performance_stats(self, result: Dict):
        """Update performance statistics"""
        self.performance_stats['total_predictions'] += 1
        
        inference_time = result.get('inference_time', 0)
        lcm_time = result.get('lcm_time', 0)
        lcm_mode = result.get('lcm_mode', 'unknown')
        
        self.performance_stats['total_inference_time'] += inference_time
        self.performance_stats['total_lcm_time'] += lcm_time
        
        if lcm_mode == 'deferred':
            self.performance_stats['deferred_predictions'] += 1
        else:
            self.performance_stats['immediate_predictions'] += 1
            
        # Calculate averages
        total_preds = self.performance_stats['total_predictions']
        self.performance_stats['average_inference_time'] = (
            self.performance_stats['total_inference_time'] / total_preds
        )
        
    def predict_batch(self,
                     X_batch: List[Any],
                     priority: Any = None,
                     enable_fast_mode: bool = True,
                     show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Efficient batch prediction with deferred LCM.
        
        Args:
            X_batch: List of inputs for batch prediction
            priority: Priority for LCM processing
            enable_fast_mode: Use deferred LCM for better performance
            show_progress: Show progress during batch processing
            
        Returns:
            List of prediction results
        """
        
        results = []
        batch_start = time.time()
        
        if show_progress:
            print(f"[BATCH] Processing batch of {len(X_batch)} predictions...")
            
        for i, x in enumerate(X_batch):
            result = self.predict(
                x,
                priority=priority,
                enable_fast_mode=enable_fast_mode,
                return_enhanced_info=True
            )
            results.append(result)
            
            if show_progress and (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(X_batch)} predictions...")
                
        batch_time = time.time() - batch_start
        
        if show_progress:
            avg_time = batch_time / len(X_batch)
            print(f"[SUCCESS] Batch complete: {batch_time:.3f}s total, {avg_time:.4f}s avg per prediction")
            
        return results
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.performance_stats.copy()
        
        # Add LCM-specific stats if available
        if self.adaptive_lcm:
            lcm_stats = self.adaptive_lcm.get_stats()
            stats['lcm_details'] = lcm_stats
            
        # Calculate performance improvement
        total_preds = stats['total_predictions']
        if total_preds > 0:
            deferred_ratio = stats['deferred_predictions'] / total_preds
            # Estimate improvement based on deferred ratio
            # (deferred LCM is ~90% faster than immediate)
            stats['estimated_speedup'] = 1 + (deferred_ratio * 0.9)
            
        return stats
        
    def set_lcm_mode(self, mode: Any):
        """Set LCM processing mode"""
        if self.adaptive_lcm:
            self.adaptive_lcm.set_mode(mode)
            print(f"[MODE] LCM mode set to: {mode.value}")
        else:
            self.lcm_config.default_mode = mode
            print(f"[CONFIG] LCM mode configured for next initialization: {mode.value}")
            
    def enable_fast_inference(self):
        """Enable fast inference mode (primarily deferred LCM)"""
        self.set_lcm_mode(LCMMode.DEFERRED)
        
    def enable_compliance_mode(self):
        """Enable compliance mode (immediate LCM for all requests)"""
        self.set_lcm_mode(LCMMode.IMMEDIATE)
        
    def shutdown(self):
        """Gracefully shutdown the enhanced wrapper"""
        if self.adaptive_lcm:
            self.adaptive_lcm.shutdown()
        print(f"[SHUTDOWN] Enhanced CIAF wrapper '{self.model_name}' shutdown complete")

# Convenience function for quick setup
def create_enhanced_ciaf_wrapper(model: Any, 
                                model_name: str,
                                fast_inference: bool = True,
                                **kwargs) -> EnhancedCIAFModelWrapper:
    """
    Convenience function to create an enhanced CIAF wrapper.
    
    Args:
        model: The ML model to wrap
        model_name: Unique name for the model
        fast_inference: Enable fast inference mode by default
        **kwargs: Additional arguments for EnhancedCIAFModelWrapper
        
    Returns:
        Configured EnhancedCIAFModelWrapper
    """
    
    # Handle default_lcm_mode conflict - prioritize explicit kwargs over fast_inference
    if 'default_lcm_mode' not in kwargs:
        kwargs['default_lcm_mode'] = LCMMode.DEFERRED if fast_inference else LCMMode.IMMEDIATE
    
    return EnhancedCIAFModelWrapper(
        model=model,
        model_name=model_name,
        **kwargs
    )