"""
Deferred LCM Performance Demonstration
====================================

This script demonstrates the performance improvements achieved with
deferred LCM processing compared to traditional immediate LCM.

Features demonstrated:
- Performance comparison between immediate and deferred LCM
- Adaptive mode switching based on system load
- Batch processing capabilities
- Real-world fraud detection scenario
- Comprehensive performance metrics
"""

import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Mock imports for demonstration (replace with actual imports)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available, using mock model")

try:
    # Import enhanced wrapper from tools folder
    from enhanced_model_wrapper import EnhancedCIAFModelWrapper, create_enhanced_ciaf_wrapper
    # Add the parent directory to the path so we can import from ciaf
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ciaf.adaptive_lcm import LCMMode, InferencePriority
    from ciaf.metadata_config import create_high_performance_config, create_compliance_first_config, create_balanced_config
    CIAF_AVAILABLE = True
except ImportError:
    CIAF_AVAILABLE = False
    print("WARNING: CIAF modules not available, using mock implementations")

class MockModel:
    """Mock ML model for demonstration when sklearn not available"""
    def __init__(self):
        self.is_fitted = False
        
    def fit(self, X, y):
        # Simulate training time
        time.sleep(0.1)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        # Simulate inference with some randomness
        if hasattr(X, 'shape'):
            n_samples = X.shape[0] if len(X.shape) > 1 else 1
        else:
            n_samples = len(X) if isinstance(X, list) else 1
        return np.random.randint(0, 2, n_samples)

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for deferred LCM"""
    
    def __init__(self):
        self.results = {}
        self.test_data = None
        self.models = {}
        
    def generate_test_data(self, n_samples: int = 1000, n_features: int = 20):
        """Generate synthetic fraud detection dataset"""
        print(f"[DATA] Generating test dataset: {n_samples} samples, {n_features} features")
        
        if SKLEARN_AVAILABLE:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.8),
                n_redundant=int(n_features * 0.1),
                n_clusters_per_class=1,
                class_sep=0.8,
                random_state=42
            )
        else:
            # Mock data
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 2, n_samples)
            
        self.test_data = {
            'X_train': X[:int(0.8 * n_samples)],
            'y_train': y[:int(0.8 * n_samples)],
            'X_test': X[int(0.8 * n_samples):],
            'y_test': y[int(0.8 * n_samples):],
        }
        
        print(f"[SUCCESS] Dataset ready: {self.test_data['X_train'].shape[0]} training, {self.test_data['X_test'].shape[0]} test samples")
        
    def create_models(self):
        """Create different model configurations for comparison"""
        print("[SETUP] Creating model configurations...")
        
        # Base ML model
        if SKLEARN_AVAILABLE:
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            base_model = MockModel()
            
        if CIAF_AVAILABLE:
            # Standard CIAF model (immediate LCM)
            self.models['standard_ciaf'] = create_enhanced_ciaf_wrapper(
                model=base_model,
                model_name="Standard_CIAF_Model",
                fast_inference=False,
                enable_deferred_lcm=False
            )
            
            # High performance model (deferred LCM)
            self.models['high_performance'] = create_enhanced_ciaf_wrapper(
                model=base_model,
                model_name="High_Performance_Model",
                fast_inference=True,
                default_lcm_mode=LCMMode.DEFERRED
            )
            
            # Adaptive model (intelligent switching)
            self.models['adaptive'] = create_enhanced_ciaf_wrapper(
                model=base_model,
                model_name="Adaptive_Model",
                fast_inference=True,
                default_lcm_mode=LCMMode.ADAPTIVE
            )
        else:
            # Mock models for demo
            self.models = {
                'standard_ciaf': base_model,
                'high_performance': base_model,
                'adaptive': base_model
            }
            
        print(f"[SUCCESS] Created {len(self.models)} model configurations")
        
    def train_models(self):
        """Train all model configurations"""
        print("[TRAINING] Training models...")
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            start_time = time.time()
            
            if hasattr(model, 'train'):
                # CIAF wrapper
                model.train(
                    self.test_data['X_train'],
                    self.test_data['y_train'],
                    version="1.0.0-demo"
                )
            else:
                # Standard model
                model.fit(self.test_data['X_train'], self.test_data['y_train'])
                
            training_time = time.time() - start_time
            self.results[f'{name}_training_time'] = training_time
            print(f"    [SUCCESS] {name} trained in {training_time:.3f}s")
            
    def benchmark_single_predictions(self, n_predictions: int = 100):
        """Benchmark single prediction performance"""
        print(f"[BENCHMARK] Benchmarking single predictions ({n_predictions} samples)...")
        
        test_samples = self.test_data['X_test'][:n_predictions]
        
        for name, model in self.models.items():
            print(f"  Testing {name}...")
            times = []
            
            for i, sample in enumerate(test_samples):
                start_time = time.time()
                
                if hasattr(model, 'predict') and hasattr(model, 'train'):
                    # Enhanced CIAF wrapper
                    result = model.predict(sample, return_enhanced_info=True)
                    prediction = result.get('prediction') if isinstance(result, dict) else result
                else:
                    # Standard model
                    prediction = model.predict([sample])[0]
                    
                inference_time = time.time() - start_time
                times.append(inference_time)
                
                if (i + 1) % 20 == 0:
                    avg_time = np.mean(times[-20:])
                    print(f"    Progress: {i+1}/{n_predictions}, Avg: {avg_time:.4f}s")
                    
            self.results[f'{name}_single_predictions'] = {
                'total_time': sum(times),
                'average_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times),
                'samples': n_predictions
            }
            
            print(f"    [SUCCESS] {name}: {np.mean(times):.4f}s avg, {sum(times):.3f}s total")
            
    def benchmark_batch_predictions(self, batch_size: int = 50):
        """Benchmark batch prediction performance"""
        print(f"[BATCH] Benchmarking batch predictions (batch size: {batch_size})...")
        
        test_samples = self.test_data['X_test'][:batch_size]
        
        for name, model in self.models.items():
            print(f"  Testing {name}...")
            start_time = time.time()
            
            if hasattr(model, 'predict_batch'):
                # Enhanced CIAF wrapper with batch support
                results = model.predict_batch(
                    test_samples,
                    priority=InferencePriority.NORMAL,
                    enable_fast_mode=True,
                    show_progress=False
                )
            elif hasattr(model, 'predict') and hasattr(model, 'train'):
                # CIAF wrapper without batch support
                results = []
                for sample in test_samples:
                    result = model.predict(sample, return_enhanced_info=True)
                    results.append(result)
            else:
                # Standard model
                results = model.predict(test_samples)
                
            batch_time = time.time() - start_time
            avg_per_sample = batch_time / len(test_samples)
            
            self.results[f'{name}_batch_predictions'] = {
                'total_time': batch_time,
                'average_per_sample': avg_per_sample,
                'batch_size': batch_size,
                'throughput': len(test_samples) / batch_time
            }
            
            print(f"    [SUCCESS] {name}: {batch_time:.3f}s total, {avg_per_sample:.4f}s avg, {len(test_samples)/batch_time:.1f} samples/sec")
            
    def test_priority_handling(self):
        """Test priority-based LCM mode switching"""
        print("[PRIORITY] Testing priority-based mode switching...")
        
        adaptive_model = self.models.get('adaptive')
        if not hasattr(adaptive_model, 'predict'):
            print("  WARNING: Adaptive model not available, skipping priority test")
            return
            
        priorities = [
            InferencePriority.CRITICAL,
            InferencePriority.HIGH,
            InferencePriority.NORMAL,
            InferencePriority.LOW
        ]
        
        priority_results = {}
        test_sample = self.test_data['X_test'][0]
        
        for priority in priorities:
            times = []
            modes = []
            
            for _ in range(10):  # Test each priority 10 times
                start_time = time.time()
                result = adaptive_model.predict(
                    test_sample,
                    priority=priority,
                    return_enhanced_info=True
                )
                inference_time = time.time() - start_time
                
                times.append(inference_time)
                if isinstance(result, dict):
                    modes.append(result.get('lcm_mode', 'unknown'))
                    
            priority_results[priority.value] = {
                'average_time': np.mean(times),
                'modes_used': list(set(modes)) if modes else ['unknown'],
                'samples': len(times)
            }
            
            print(f"    {priority.value}: {np.mean(times):.4f}s avg, modes: {list(set(modes))}")
            
        self.results['priority_handling'] = priority_results
        
    def collect_performance_stats(self):
        """Collect detailed performance statistics from models"""
        print("[STATS] Collecting performance statistics...")
        
        for name, model in self.models.items():
            if hasattr(model, 'get_performance_stats'):
                stats = model.get_performance_stats()
                self.results[f'{name}_stats'] = stats
                
                print(f"  {name} stats:")
                print(f"    Total predictions: {stats.get('total_predictions', 0)}")
                print(f"    Deferred: {stats.get('deferred_predictions', 0)}")
                print(f"    Immediate: {stats.get('immediate_predictions', 0)}")
                if stats.get('total_predictions', 0) > 0:
                    deferred_pct = (stats.get('deferred_predictions', 0) / stats['total_predictions']) * 100
                    print(f"    Deferred percentage: {deferred_pct:.1f}%")
                    
    def calculate_improvements(self):
        """Calculate performance improvements between configurations"""
        print("[ANALYSIS] Calculating performance improvements...")
        
        baseline = 'standard_ciaf'
        comparisons = ['high_performance', 'adaptive']
        
        if baseline not in self.results or not any(comp in self.results for comp in comparisons):
            print("  WARNING: Insufficient data for comparison")
            return
            
        improvements = {}
        
        # Single prediction improvements
        if f'{baseline}_single_predictions' in self.results:
            baseline_time = self.results[f'{baseline}_single_predictions']['average_time']
            
            for comp in comparisons:
                if f'{comp}_single_predictions' in self.results:
                    comp_time = self.results[f'{comp}_single_predictions']['average_time']
                    improvement = ((baseline_time - comp_time) / baseline_time) * 100
                    improvements[f'{comp}_single_improvement'] = improvement
                    print(f"  {comp} single prediction improvement: {improvement:.1f}%")
                    
        # Batch prediction improvements
        if f'{baseline}_batch_predictions' in self.results:
            baseline_throughput = self.results[f'{baseline}_batch_predictions']['throughput']
            
            for comp in comparisons:
                if f'{comp}_batch_predictions' in self.results:
                    comp_throughput = self.results[f'{comp}_batch_predictions']['throughput']
                    improvement = ((comp_throughput - baseline_throughput) / baseline_throughput) * 100
                    improvements[f'{comp}_batch_improvement'] = improvement
                    print(f"  {comp} batch throughput improvement: {improvement:.1f}%")
                    
        self.results['performance_improvements'] = improvements
        
    def save_results(self, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deferred_lcm_benchmark_{timestamp}.json"
            
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
            
        # Clean results for JSON serialization
        clean_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                clean_results[key] = {k: convert_numpy(v) for k, v in value.items()}
            else:
                clean_results[key] = convert_numpy(value)
                
        with open(filename, 'w') as f:
            json.dump({
                'benchmark_timestamp': datetime.now().isoformat(),
                'sklearn_available': SKLEARN_AVAILABLE,
                'ciaf_available': CIAF_AVAILABLE,
                'results': clean_results
            }, f, indent=2)
            
        print(f"[SAVE] Results saved to: {filename}")
        return filename
        
    def print_summary(self):
        """Print a comprehensive summary of results"""
        print("\n" + "="*80)
        print("[BENCHMARK] DEFERRED LCM PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        if not self.results:
            print("[ERROR] No results available")
            return
            
        # Training times
        print("\n[TRAINING] TRAINING PERFORMANCE:")
        for key, value in self.results.items():
            if key.endswith('_training_time'):
                model_name = key.replace('_training_time', '')
                print(f"  {model_name}: {value:.3f}s")
                
        # Single prediction performance
        print("\n[SINGLE] SINGLE PREDICTION PERFORMANCE:")
        for key, value in self.results.items():
            if key.endswith('_single_predictions'):
                model_name = key.replace('_single_predictions', '')
                print(f"  {model_name}:")
                print(f"    Average: {value['average_time']:.4f}s")
                print(f"    Min: {value['min_time']:.4f}s")
                print(f"    Max: {value['max_time']:.4f}s")
                
        # Batch prediction performance
        print("\n[BATCH] BATCH PREDICTION PERFORMANCE:")
        for key, value in self.results.items():
            if key.endswith('_batch_predictions'):
                model_name = key.replace('_batch_predictions', '')
                print(f"  {model_name}:")
                print(f"    Throughput: {value['throughput']:.1f} samples/sec")
                print(f"    Avg per sample: {value['average_per_sample']:.4f}s")
                
        # Performance improvements
        if 'performance_improvements' in self.results:
            print("\n[IMPROVEMENTS] PERFORMANCE IMPROVEMENTS:")
            for key, value in self.results['performance_improvements'].items():
                print(f"  {key}: {value:.1f}%")
                
        # Priority handling
        if 'priority_handling' in self.results:
            print("\n[PRIORITY] PRIORITY HANDLING:")
            for priority, stats in self.results['priority_handling'].items():
                print(f"  {priority}: {stats['average_time']:.4f}s avg, modes: {stats['modes_used']}")
                
        print("\n[SUCCESS] Benchmark complete!")

def run_comprehensive_benchmark():
    """Run the complete deferred LCM benchmark"""
    print("[START] Starting Comprehensive Deferred LCM Benchmark")
    print("="*60)
    
    benchmark = PerformanceBenchmark()
    
    try:
        # Setup
        benchmark.generate_test_data(n_samples=1000, n_features=20)
        benchmark.create_models()
        
        # Training
        benchmark.train_models()
        
        # Performance testing
        benchmark.benchmark_single_predictions(n_predictions=100)
        benchmark.benchmark_batch_predictions(batch_size=50)
        
        if CIAF_AVAILABLE:
            benchmark.test_priority_handling()
            
        # Analysis
        benchmark.collect_performance_stats()
        benchmark.calculate_improvements()
        
        # Results
        filename = benchmark.save_results()
        benchmark.print_summary()
        
        return filename
        
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    # Run the comprehensive benchmark
    try:
        result_file = run_comprehensive_benchmark()
        print(f"\n[SUCCESS] Benchmark completed successfully!")
        print(f"[RESULTS] Results saved to: {result_file}")
        
        # Cleanup
        print("\n[CLEANUP] Cleaning up...")
        
    except KeyboardInterrupt:
        print("\n[STOP] Benchmark interrupted by user")
    except Exception as e:
        print(f"\n[FAILURE] Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()