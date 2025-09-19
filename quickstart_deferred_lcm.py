"""
Deferred LCM Quick Start
=======================

This script provides a simple example of using the deferred LCM architecture
for improved inference performance while maintaining full audit compliance.

Run this script to see the performance difference between traditional CIAF
and the new deferred LCM approach.
"""

import numpy as np
import time
from datetime import datetime

# Try importing the enhanced CIAF components
try:
    from ciaf.enhanced_model_wrapper import create_enhanced_ciaf_wrapper
    from ciaf.adaptive_lcm import LCMMode, InferencePriority
    from ciaf.metadata_config import create_high_performance_config, create_balanced_config
    CIAF_AVAILABLE = True
    print("✅ CIAF deferred LCM components available")
except ImportError as e:
    CIAF_AVAILABLE = False
    print(f"❌ CIAF components not available: {e}")
    print("Please ensure the CIAF framework is properly installed")
    exit(1)

# Try importing scikit-learn for demo model
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available, using mock model")

class MockModel:
    """Simple mock model when sklearn is not available"""
    def __init__(self):
        self.is_fitted = False
        
    def fit(self, X, y):
        time.sleep(0.1)  # Simulate training
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        n_samples = X.shape[0] if hasattr(X, 'shape') and len(X.shape) > 1 else 1
        return np.random.randint(0, 2, n_samples)

def generate_demo_data():
    """Generate sample fraud detection data"""
    print("📊 Generating demo fraud detection dataset...")
    
    if SKLEARN_AVAILABLE:
        X, y = make_classification(
            n_samples=500,
            n_features=15,
            n_informative=12,
            n_redundant=2,
            n_clusters_per_class=1,
            class_sep=0.8,
            random_state=42
        )
    else:
        # Create mock data
        X = np.random.randn(500, 15)
        y = np.random.randint(0, 2, 500)
    
    # Split into train/test
    split_idx = int(0.8 * len(X))
    return {
        'X_train': X[:split_idx],
        'y_train': y[:split_idx],
        'X_test': X[split_idx:],
        'y_test': y[split_idx:]
    }

def create_demo_models():
    """Create standard and enhanced CIAF models for comparison"""
    print("🏭 Creating demo models...")
    
    # Base ML model
    if SKLEARN_AVAILABLE:
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        base_model = MockModel()
    
    # Standard CIAF model (immediate LCM)
    standard_model = create_enhanced_ciaf_wrapper(
        model=base_model,
        model_name="Standard_CIAF_Demo",
        fast_inference=False,
        enable_deferred_lcm=False
    )
    
    # High-performance model (deferred LCM)
    fast_model = create_enhanced_ciaf_wrapper(
        model=base_model,
        model_name="Fast_CIAF_Demo",
        fast_inference=True,
        default_lcm_mode=LCMMode.DEFERRED
    )
    
    return standard_model, fast_model

def train_models(models, data):
    """Train both models"""
    print("🚀 Training models...")
    
    training_times = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        start_time = time.time()
        
        model.train(
            data['X_train'],
            data['y_train'],
            version="1.0.0-quickstart"
        )
        
        training_time = time.time() - start_time
        training_times[name] = training_time
        print(f"    ✅ Completed in {training_time:.3f}s")
    
    return training_times

def compare_inference_performance(models, data, n_samples=50):
    """Compare inference performance between models"""
    print(f"🔮 Comparing inference performance ({n_samples} samples)...")
    
    test_samples = data['X_test'][:n_samples]
    results = {}
    
    for name, model in models.items():
        print(f"  Testing {name}...")
        
        times = []
        predictions = []
        
        for i, sample in enumerate(test_samples):
            start_time = time.time()
            
            # Make prediction with enhanced info
            result = model.predict(sample, return_enhanced_info=True)
            
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            if isinstance(result, dict):
                predictions.append(result.get('prediction'))
            else:
                predictions.append(result)
                
            if (i + 1) % 10 == 0:
                avg_time = np.mean(times[-10:])
                print(f"    Progress: {i+1}/{n_samples}, Recent avg: {avg_time:.4f}s")
        
        results[name] = {
            'total_time': sum(times),
            'average_time': np.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'predictions': predictions,
            'samples': len(times)
        }
        
        print(f"    ✅ {name}: {np.mean(times):.4f}s avg")
    
    return results

def demonstrate_priority_handling(adaptive_model, data):
    """Demonstrate priority-based LCM mode switching"""
    print("🎯 Demonstrating priority handling...")
    
    test_sample = data['X_test'][0]
    priorities = [
        InferencePriority.CRITICAL,
        InferencePriority.HIGH,
        InferencePriority.NORMAL,
        InferencePriority.LOW
    ]
    
    priority_results = {}
    
    for priority in priorities:
        print(f"  Testing {priority.value} priority...")
        
        times = []
        modes = []
        
        for _ in range(5):  # Test each priority 5 times
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
        
        avg_time = np.mean(times)
        unique_modes = list(set(modes))
        
        priority_results[priority.value] = {
            'average_time': avg_time,
            'modes_used': unique_modes
        }
        
        print(f"    {priority.value}: {avg_time:.4f}s avg, modes: {unique_modes}")
    
    return priority_results

def print_performance_summary(training_times, inference_results, priority_results=None):
    """Print a comprehensive summary of the demo results"""
    print("\n" + "="*60)
    print("🎯 DEFERRED LCM QUICKSTART RESULTS")
    print("="*60)
    
    # Training performance
    print("\n📚 Training Performance:")
    for model, time_taken in training_times.items():
        print(f"  {model}: {time_taken:.3f}s")
    
    # Inference performance
    print("\n🔮 Inference Performance:")
    for model, results in inference_results.items():
        print(f"  {model}:")
        print(f"    Average time: {results['average_time']:.4f}s")
        print(f"    Total time: {results['total_time']:.3f}s")
        print(f"    Samples: {results['samples']}")
    
    # Calculate improvement
    if len(inference_results) == 2:
        models = list(inference_results.keys())
        baseline = models[0]
        enhanced = models[1]
        
        baseline_time = inference_results[baseline]['average_time']
        enhanced_time = inference_results[enhanced]['average_time']
        
        improvement = ((baseline_time - enhanced_time) / baseline_time) * 100
        print(f"\n📈 Performance Improvement:")
        print(f"  {enhanced} is {improvement:.1f}% faster than {baseline}")
    
    # Priority handling
    if priority_results:
        print("\n🎯 Priority Handling:")
        for priority, stats in priority_results.items():
            print(f"  {priority}: {stats['average_time']:.4f}s, modes: {stats['modes_used']}")
    
    print("\n✅ Demo completed successfully!")

def main():
    """Main quickstart demonstration"""
    print("🚀 CIAF Deferred LCM Quick Start Demo")
    print("="*50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Generate demo data
        data = generate_demo_data()
        
        # Create models
        standard_model, fast_model = create_demo_models()
        models = {
            'Standard CIAF': standard_model,
            'Deferred LCM': fast_model
        }
        
        # Train models
        training_times = train_models(models, data)
        
        # Compare inference performance
        inference_results = compare_inference_performance(models, data, n_samples=30)
        
        # Demonstrate priority handling with adaptive model
        print("\n🔄 Creating adaptive model for priority demo...")
        adaptive_model = create_enhanced_ciaf_wrapper(
            model=RandomForestClassifier(n_estimators=50, random_state=42) if SKLEARN_AVAILABLE else MockModel(),
            model_name="Adaptive_Demo",
            fast_inference=True,
            default_lcm_mode=LCMMode.ADAPTIVE
        )
        
        # Train adaptive model
        adaptive_model.train(data['X_train'], data['y_train'], version="1.0.0-adaptive")
        
        # Test priority handling
        priority_results = demonstrate_priority_handling(adaptive_model, data)
        
        # Print summary
        print_performance_summary(training_times, inference_results, priority_results)
        
        # Show how to use in production
        print("\n💡 Production Usage Example:")
        print("```python")
        print("# Create high-performance model")
        print("model = create_enhanced_ciaf_wrapper(")
        print("    model=your_ml_model,")
        print("    model_name='Production_Model',")
        print("    fast_inference=True,")
        print("    default_lcm_mode=LCMMode.DEFERRED")
        print(")")
        print("")
        print("# Train with full audit trail")
        print("model.train(X_train, y_train, version='1.0.0')")
        print("")
        print("# Fast inference with deferred LCM")
        print("prediction = model.predict(sample, priority=InferencePriority.HIGH)")
        print("```")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()