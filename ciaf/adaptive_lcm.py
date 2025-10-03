"""
Adaptive LCM Wrapper
===================

This module provides an adaptive wrapper that can switch between immediate
and deferred LCM processing based on priority, configuration, and system load.
"""

import time
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from .deferred_lcm import (
    DeferredLCMProcessor, 
    LightweightReceipt, 
    ReceiptHasher
)

class LCMMode(Enum):
    """LCM processing modes"""
    IMMEDIATE = "immediate"
    DEFERRED = "deferred"
    ADAPTIVE = "adaptive"

class InferencePriority(Enum):
    """Priority levels for inference requests"""
    CRITICAL = "critical"     # Always immediate LCM
    HIGH = "high"            # Prefer immediate, fallback to deferred
    NORMAL = "normal"        # Use configured default
    LOW = "low"              # Always deferred

class AdaptiveLCMConfig:
    """Configuration for adaptive LCM behavior"""
    
    def __init__(self,
                 default_mode: LCMMode = LCMMode.ADAPTIVE,
                 immediate_threshold_ms: float = 50.0,
                 queue_size_threshold: int = 1000,
                 cpu_threshold_percent: float = 80.0,
                 memory_threshold_mb: float = 500.0,
                 batch_size: int = 50,
                 processing_interval: float = 2.0,
                 enable_persistence: bool = True):
        self.default_mode = default_mode
        self.immediate_threshold_ms = immediate_threshold_ms
        self.queue_size_threshold = queue_size_threshold
        self.cpu_threshold_percent = cpu_threshold_percent
        self.memory_threshold_mb = memory_threshold_mb
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.enable_persistence = enable_persistence

class SystemMonitor:
    """Simple system resource monitoring"""
    
    def __init__(self):
        self.last_check = 0
        self.check_interval = 5.0  # Check every 5 seconds
        self.cached_metrics = {}
        
    def get_system_load(self) -> Dict[str, float]:
        """Get current system load metrics"""
        current_time = time.time()
        
        if current_time - self.last_check < self.check_interval:
            return self.cached_metrics
            
        try:
            import psutil
            
            self.cached_metrics = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_mb': psutil.virtual_memory().used / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent
            }
        except ImportError:
            # Fallback if psutil not available
            self.cached_metrics = {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'memory_percent': 0.0
            }
            
        self.last_check = current_time
        return self.cached_metrics

class AdaptiveLCMWrapper:
    """
    Adaptive wrapper that intelligently chooses between immediate and deferred LCM
    based on priority, system load, and configuration.
    """
    
    def __init__(self, 
                 base_model: Any,
                 config: Optional[AdaptiveLCMConfig] = None,
                 model_ref: str = "adaptive_model",
                 model_version: str = "1.0.0"):
        self.base_model = base_model
        self.config = config or AdaptiveLCMConfig()
        self.model_ref = model_ref
        self.model_version = model_version
        self.current_mode = self.config.default_mode  # Add missing current_mode attribute
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
        # Deferred processing
        self.deferred_processor = None
        if self.config.default_mode in [LCMMode.DEFERRED, LCMMode.ADAPTIVE]:
            self.deferred_processor = DeferredLCMProcessor(
                batch_size=self.config.batch_size,
                processing_interval=self.config.processing_interval
            )
            self.deferred_processor.start_background_processing()
            
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'immediate_lcm_count': 0,
            'deferred_lcm_count': 0,
            'total_inference_time': 0.0,
            'total_lcm_time': 0.0,
            'mode_switches': 0
        }
        
    def predict(self, 
                input_data: Any, 
                priority: InferencePriority = InferencePriority.NORMAL,
                include_receipts: bool = True,
                **kwargs) -> Dict[str, Any]:
        """
        Make prediction with adaptive LCM processing
        
        Args:
            input_data: Input for model prediction
            priority: Priority level for this inference
            include_receipts: Whether to generate LCM receipts
            **kwargs: Additional arguments for model
            
        Returns:
            Dictionary containing prediction and metadata
        """
        start_time = time.time()
        
        # Make the actual prediction (always fast)
        prediction = self.base_model.predict([input_data])[0] if hasattr(self.base_model, 'predict') else input_data
        
        inference_time = time.time() - start_time
        
        # Update basic stats
        self.stats['total_predictions'] += 1
        self.stats['total_inference_time'] += inference_time
        
        result = {
            'prediction': prediction,
            'inference_time': inference_time,
            'model_version': self.model_version,
            'timestamp': datetime.now().isoformat(),
            'request_id': f"req_{uuid.uuid4().hex[:8]}"
        }
        
        # Handle LCM processing if requested
        if include_receipts:
            lcm_mode = self._determine_lcm_mode(priority, inference_time)
            receipt_info = self._process_lcm(input_data, prediction, lcm_mode, result['request_id'])
            result.update(receipt_info)
            
        return result
        
    def _determine_lcm_mode(self, priority: InferencePriority, inference_time: float) -> LCMMode:
        """Determine which LCM mode to use for this request"""
        
        # Priority-based decisions
        if priority == InferencePriority.CRITICAL:
            return LCMMode.IMMEDIATE
        elif priority == InferencePriority.LOW:
            return LCMMode.DEFERRED
            
        # If not adaptive mode, use configured default
        if self.config.default_mode != LCMMode.ADAPTIVE:
            return self.config.default_mode
            
        # Adaptive logic
        if not self.deferred_processor:
            return LCMMode.IMMEDIATE
            
        # Check system load
        system_load = self.system_monitor.get_system_load()
        
        # Check queue size
        queue_size = self.deferred_processor.receipt_queue.size()
        
        # Decision logic
        should_defer = (
            # Fast inference suggests we can afford deferred processing
            inference_time * 1000 < self.config.immediate_threshold_ms and
            # System not overloaded
            system_load.get('cpu_percent', 0) < self.config.cpu_threshold_percent and
            # Queue not too full
            queue_size < self.config.queue_size_threshold
        )
        
        if priority == InferencePriority.HIGH and not should_defer:
            return LCMMode.IMMEDIATE
            
        return LCMMode.DEFERRED if should_defer else LCMMode.IMMEDIATE
        
    def _process_lcm(self, input_data: Any, prediction: Any, mode: LCMMode, request_id: str) -> Dict[str, Any]:
        """Process LCM based on determined mode"""
        lcm_start = time.time()
        
        # Generate basic receipt info
        receipt_id = ReceiptHasher.generate_receipt_id()
        input_hash = ReceiptHasher.hash_data(input_data)
        output_hash = ReceiptHasher.hash_data(prediction)
        input_commitment = ReceiptHasher.create_commitment(input_data)
        output_commitment = ReceiptHasher.create_commitment(prediction)
        
        if mode == LCMMode.IMMEDIATE:
            return self._immediate_lcm(
                receipt_id, request_id, input_hash, output_hash, 
                input_commitment, output_commitment, lcm_start
            )
        else:
            return self._deferred_lcm(
                receipt_id, request_id, input_hash, output_hash,
                input_commitment, output_commitment, input_data, prediction, lcm_start
            )
            
    def _immediate_lcm(self, receipt_id: str, request_id: str, input_hash: str, 
                      output_hash: str, input_commitment: str, output_commitment: str, 
                      lcm_start: float) -> Dict[str, Any]:
        """Process immediate LCM (original heavy processing)"""
        
        # Simulate full cryptographic processing
        receipt_content = f"{receipt_id}{datetime.now().isoformat()}{input_hash}{output_hash}"
        receipt_digest = hashlib.sha256(receipt_content.encode()).hexdigest()
        
        connections_content = f"{self.model_ref}{receipt_digest}"
        connections_digest = hashlib.sha256(connections_content.encode()).hexdigest()
        
        full_receipt = {
            "receipt_id": receipt_id,
            "model_anchor_ref": self.model_ref,
            "deployment_anchor_ref": "Production_Model",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "receipt_digest": receipt_digest,
            "connections_digest": connections_digest,
            "anchor_id": f"r_{receipt_digest[:8]}...",
            "input_commitment": {
                "commitment_type": "CommitmentType.SALTED",
                "commitment_value": input_commitment
            },
            "output_commitment": {
                "commitment_type": "CommitmentType.SALTED",
                "commitment_value": output_commitment
            }
        }
        
        lcm_time = time.time() - lcm_start
        
        # Update stats
        self.stats['immediate_lcm_count'] += 1
        self.stats['total_lcm_time'] += lcm_time
        
        return {
            'lcm_mode': 'immediate',
            'lcm_time': lcm_time,
            'receipt': full_receipt,
            'receipt_id': receipt_id
        }
        
    def _deferred_lcm(self, receipt_id: str, request_id: str, input_hash: str,
                     output_hash: str, input_commitment: str, output_commitment: str,
                     input_data: Any, prediction: Any, lcm_start: float) -> Dict[str, Any]:
        """Process deferred LCM (lightweight receipt)"""
        
        # Create lightweight receipt
        light_receipt = LightweightReceipt(
            receipt_id=receipt_id,
            timestamp=datetime.now().isoformat(),
            model_ref=self.model_ref,
            model_version=self.model_version,
            request_id=request_id,
            input_hash=input_hash,
            output_hash=output_hash,
            input_commitment=input_commitment,
            output_commitment=output_commitment,
            raw_input=str(input_data),
            raw_output=str(prediction),
            metadata={"deferred": True}
        )
        
        # Queue for background processing
        success = self.deferred_processor.add_receipt(light_receipt)
        
        lcm_time = time.time() - lcm_start
        
        # Update stats
        self.stats['deferred_lcm_count'] += 1
        self.stats['total_lcm_time'] += lcm_time
        
        return {
            'lcm_mode': 'deferred',
            'lcm_time': lcm_time,
            'receipt_queued': success,
            'receipt_id': receipt_id,
            'light_receipt': light_receipt.to_dict()
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = self.stats.copy()
        
        if self.stats['total_predictions'] > 0:
            stats['avg_inference_time'] = self.stats['total_inference_time'] / self.stats['total_predictions']
            stats['avg_lcm_time'] = self.stats['total_lcm_time'] / self.stats['total_predictions']
            stats['immediate_percentage'] = (self.stats['immediate_lcm_count'] / self.stats['total_predictions']) * 100
            stats['deferred_percentage'] = (self.stats['deferred_lcm_count'] / self.stats['total_predictions']) * 100
            
        if self.deferred_processor:
            stats['deferred_processor'] = self.deferred_processor.get_stats()
            
        stats['system_load'] = self.system_monitor.get_system_load()
        
        return stats
        
    def set_mode(self, mode: LCMMode):
        """Manually set LCM mode"""
        old_mode = self.config.default_mode
        self.config.default_mode = mode
        
        if old_mode != mode:
            self.stats['mode_switches'] += 1
            print(f"🔄 LCM mode changed: {old_mode.value} → {mode.value}")
            
    def shutdown(self):
        """Gracefully shutdown the wrapper"""
        if self.deferred_processor:
            self.deferred_processor.stop_background_processing()
            
        print("✅ Adaptive LCM wrapper shutdown complete")

if __name__ == "__main__":
    # Simple test of adaptive LCM
    class MockModel:
        def predict(self, data):
            time.sleep(0.001)  # Simulate inference
            return [1]  # Mock prediction
            
    model = MockModel()
    wrapper = AdaptiveLCMWrapper(
        model, 
        config=AdaptiveLCMConfig(default_mode=LCMMode.ADAPTIVE),
        model_ref="test_model_hash"
    )
    
    print("🧪 Testing Adaptive LCM Wrapper")
    
    # Test different priorities
    priorities = [InferencePriority.CRITICAL, InferencePriority.HIGH, 
                 InferencePriority.NORMAL, InferencePriority.LOW]
    
    for i in range(20):
        priority = priorities[i % len(priorities)]
        input_data = [i * 0.1] * 20
        
        result = wrapper.predict(input_data, priority=priority)
        print(f"Prediction {i}: Mode={result.get('lcm_mode', 'none')}, Time={result['inference_time']:.4f}s")
        
        time.sleep(0.1)  # Small delay
        
    # Show final stats
    stats = wrapper.get_stats()
    print(f"\n📊 Final Statistics:")
    for key, value in stats.items():
        if key != 'deferred_processor':
            print(f"  {key}: {value}")
            
    wrapper.shutdown()