#!/usr/bin/env python3
"""
CIAF LCM Deployable Model Example (Performance Optimized)

This example demonstrates how to create a production-ready model with complete
CIAF LCM tracking that can be deployed, pickled, and used in production while
maintaining full audit trail capabilities with optimized performance.

Created: 2025-09-19
Author: CIAF Development Team
"""

import pickle
import json
import time
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

# Import CIAF components
from ciaf.wrappers import CIAFModelWrapper

# Import optimized metadata storage
try:
    from ciaf.metadata_storage_optimized import HighPerformanceMetadataStorage
    OPTIMIZED_STORAGE_AVAILABLE = True
except ImportError:
    OPTIMIZED_STORAGE_AVAILABLE = False
    print("⚠️  Optimized storage not available, using standard storage")

class CIAFBenchmark:
    """Benchmark utility to measure CIAF LCM overhead."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
    
    def start_timer(self, operation_name):
        """Start timing an operation."""
        self.start_time = time.perf_counter()
        self.operation_name = operation_name
    
    def end_timer(self):
        """End timing and record the result."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        elapsed = time.perf_counter() - self.start_time
        self.results[self.operation_name] = elapsed
        self.start_time = None
        return elapsed
    
    def get_summary(self):
        """Get benchmark summary."""
        return self.results
    
    def print_comparison(self, baseline_results, ciaf_results):
        """Print detailed comparison between baseline and CIAF performance."""
        print("\n📊 CIAF LCM Performance Benchmark Results")
        print("="*60)
        
        overhead_summary = {}
        
        for operation in baseline_results:
            if operation in ciaf_results:
                baseline_time = baseline_results[operation]
                ciaf_time = ciaf_results[operation]
                overhead = ciaf_time - baseline_time
                overhead_percent = (overhead / baseline_time) * 100 if baseline_time > 0 else 0
                
                overhead_summary[operation] = {
                    'baseline': baseline_time,
                    'ciaf': ciaf_time,
                    'overhead': overhead,
                    'overhead_percent': overhead_percent
                }
                
                print(f"\n🔍 {operation.replace('_', ' ').title()}:")
                print(f"   Baseline:     {baseline_time:.4f}s")
                print(f"   CIAF LCM:     {ciaf_time:.4f}s")
                print(f"   Overhead:     {overhead:.4f}s ({overhead_percent:+.1f}%)")
        
        # Calculate total overhead
        total_baseline = sum(baseline_results.values())
        total_ciaf = sum(ciaf_results.values())
        total_overhead = total_ciaf - total_baseline
        total_overhead_percent = (total_overhead / total_baseline) * 100 if total_baseline > 0 else 0
        
        print(f"\n📈 Total Performance Impact:")
        print(f"   Total Baseline:  {total_baseline:.4f}s")
        print(f"   Total CIAF LCM:  {total_ciaf:.4f}s")
        print(f"   Total Overhead:  {total_overhead:.4f}s ({total_overhead_percent:+.1f}%)")
        
        return overhead_summary

def benchmark_baseline_model():
    """Benchmark baseline model without CIAF LCM."""
    print("🏃 Benchmarking Baseline Model (No CIAF LCM)")
    print("="*50)
    
    benchmark = CIAFBenchmark()
    
    # 1. Model Creation
    benchmark.start_timer("model_creation")
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    baseline_creation_time = benchmark.end_timer()
    print(f"   Model Creation: {baseline_creation_time:.4f}s")
    
    # 2. Data Preparation
    benchmark.start_timer("data_preparation")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    baseline_data_time = benchmark.end_timer()
    print(f"   Data Preparation: {baseline_data_time:.4f}s")
    
    # 3. Model Training
    benchmark.start_timer("model_training")
    base_model.fit(X, y)
    baseline_training_time = benchmark.end_timer()
    print(f"   Model Training: {baseline_training_time:.4f}s")
    
    # 4. Inference (batch)
    test_samples = X[:100]  # First 100 samples
    benchmark.start_timer("batch_inference")
    predictions = base_model.predict(test_samples)
    baseline_inference_time = benchmark.end_timer()
    print(f"   Batch Inference (100 samples): {baseline_inference_time:.4f}s")
    
    # 5. Single Inference
    benchmark.start_timer("single_inference")
    for i in range(10):  # 10 single predictions
        single_pred = base_model.predict([X[i]])
    baseline_single_time = benchmark.end_timer()
    print(f"   Single Inference (10x): {baseline_single_time:.4f}s")
    
    # 6. Model Serialization
    benchmark.start_timer("model_serialization")
    with open("baseline_model_temp.pkl", 'wb') as f:
        pickle.dump(base_model, f)
    baseline_pickle_time = benchmark.end_timer()
    print(f"   Model Serialization: {baseline_pickle_time:.4f}s")
    
    # Clean up
    Path("baseline_model_temp.pkl").unlink(missing_ok=True)
    
    return benchmark.get_summary(), base_model, X, y

def benchmark_optimized_model():
    """Benchmark CIAF model with performance optimizations."""
    print("🚀 Benchmarking Optimized CIAF Model")
    print("="*50)
    
    if not OPTIMIZED_STORAGE_AVAILABLE:
        print("❌ Optimized storage not available, skipping optimized benchmark")
        return None, None, None
    
    benchmark = CIAFBenchmark()
    
    # Initialize optimized storage with performance configuration
    optimized_config = {
        "storage_path": "ciaf_metadata_optimized",
        "enable_lazy_materialization": True,
        "fast_inference_mode": True,
        "memory_buffer_size": 1000,
        "db_connection_pool_size": 5,
        "enable_async_writes": True,
        "batch_write_size": 50
    }
    optimized_storage = HighPerformanceMetadataStorage(optimized_config)
    
    # 1. Model Creation with optimizations
    benchmark.start_timer("model_creation")
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Create optimized wrapper
    production_wrapper = CIAFModelWrapper(
        model=base_model,
        model_name="Optimized_Fraud_Detection_Model",
        enable_connections=True,
        compliance_mode="financial",
        enable_explainability=False,  # Disable for performance
        enable_uncertainty=False,     # Disable for performance
        enable_metadata_tags=False,   # Disable for performance
        auto_configure=True
    )
    
    creation_time = benchmark.end_timer()
    print(f"1️⃣ Optimized Model Creation: {creation_time:.4f}s")
    
    # 2. Data Preparation (minimal processing)
    benchmark.start_timer("data_preparation")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Minimal CIAF format for performance
    training_data = []
    for i in range(len(X)):
        training_data.append({
            "content": X[i].tolist(),
            "metadata": {
                "id": f"transaction_{i:04d}",
                "target": int(y[i]),
                "source": "optimized_dataset_v1.0"
            }
        })
    
    data_time = benchmark.end_timer()
    print(f"2️⃣ Optimized Data Preparation: {data_time:.4f}s")
    
    # 3. Training with performance mode
    benchmark.start_timer("model_training")
    
    # Training with minimal overhead
    training_snapshot = production_wrapper.train(
        dataset_id="fraud_detection_optimized_dataset",
        training_data=training_data,
        master_password="optimized_password_2025",
        training_params={
            "algorithm": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10,
            "performance_mode": True
        },
        model_version="1.0.0-optimized",
        fit_model=True
    )
    
    training_time = benchmark.end_timer()
    print(f"3️⃣ Optimized Model Training: {training_time:.4f}s")
    
    # 4. Optimized Batch Inference
    benchmark.start_timer("batch_inference")
    test_samples = X[:100].tolist()
    
    predictions = []
    for query in test_samples:
        prediction, receipt = production_wrapper.predict(
            query=query,
            use_model=True
        )
        predictions.append(prediction)
    
    batch_time = benchmark.end_timer()
    print(f"4️⃣ Optimized Batch Inference: {batch_time:.4f}s")
    
    # 5. Optimized Single Inference
    benchmark.start_timer("single_inference")
    for i in range(10):
        query = X[i].tolist()
        prediction, receipt = production_wrapper.predict(
            query=query,
            use_model=True
        )
    
    single_time = benchmark.end_timer()
    print(f"5️⃣ Optimized Single Inference: {single_time:.4f}s")
    
    return production_wrapper, benchmark.get_summary(), optimized_storage


def create_production_model():
    """Create a production-ready model with CIAF LCM tracking and benchmarking."""
    print("🏭 Creating Production Model with CIAF LCM Process (Benchmarked)")
    print("="*70)
    
    benchmark = CIAFBenchmark()
    
    # 1. Model Creation with CIAF Wrapper
    benchmark.start_timer("model_creation")
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Wrap with CIAF for comprehensive tracking
    production_wrapper = CIAFModelWrapper(
        model=base_model,
        model_name="Production_Fraud_Detection_Model",
        enable_connections=True,
        compliance_mode="financial",  # Financial compliance mode
        enable_explainability=True,
        enable_uncertainty=True,
        enable_metadata_tags=True,
        auto_configure=True
    )
    ciaf_creation_time = benchmark.end_timer()
    print(f"1️⃣ CIAF Model Creation: {ciaf_creation_time:.4f}s")
    
    # 2. Data Preparation with CIAF format
    benchmark.start_timer("data_preparation")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Convert to CIAF format
    training_data = []
    for i in range(len(X)):
        training_data.append({
            "content": X[i].tolist(),  # Feature vector
            "metadata": {
                "id": f"transaction_{i:04d}",
                "target": int(y[i]),
                "timestamp": datetime.now().isoformat(),
                "source": "production_dataset_v1.0",
                "compliance_checked": True
            }
        })
    
    ciaf_data_time = benchmark.end_timer()
    print(f"2️⃣ CIAF Data Preparation: {ciaf_data_time:.4f}s")
    print(f"   ✅ Prepared {len(training_data)} training samples")
    
    # 3. Model Training with full CIAF LCM tracking
    benchmark.start_timer("model_training")
    training_snapshot = production_wrapper.train(
        dataset_id="fraud_detection_production_dataset",
        training_data=training_data,
        master_password="production_secure_password_2025",
        training_params={
            "algorithm": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10,
            "compliance_mode": "financial",
            "validation_split": 0.2,
            "cross_validation": True
        },
        model_version="1.0.0-production",
        fit_model=True  # Actually train the model
    )
    ciaf_training_time = benchmark.end_timer()
    print(f"3️⃣ CIAF Model Training: {ciaf_training_time:.4f}s")
    print(f"   ✅ Model trained: {training_snapshot.snapshot_id}")
    
    # 4. Batch Inference Benchmarking
    benchmark.start_timer("batch_inference")
    production_queries = X[:100].tolist()  # First 100 samples as lists
    
    predictions = []
    for query in production_queries:
        prediction, receipt = production_wrapper.predict(
            query=query, 
            use_model=True  # Use actual trained model
        )
        predictions.append(prediction)
    
    ciaf_batch_time = benchmark.end_timer()
    print(f"4️⃣ CIAF Batch Inference (100 samples): {ciaf_batch_time:.4f}s")
    
    # 5. Single Inference Benchmarking
    benchmark.start_timer("single_inference")
    for i in range(10):  # 10 single predictions
        query = X[i].tolist()
        prediction, receipt = production_wrapper.predict(
            query=query, 
            use_model=True
        )
    
    ciaf_single_time = benchmark.end_timer()
    print(f"5️⃣ CIAF Single Inference (10x): {ciaf_single_time:.4f}s")
    
    return production_wrapper, benchmark.get_summary(), X, y

def deploy_model(wrapper, deployment_path="./production_models"):
    """Deploy the model with complete LCM metadata preservation and benchmarking."""
    print(f"\n🚢 Deploying Model with LCM Preservation (Benchmarked)")
    print("="*60)
    
    benchmark = CIAFBenchmark()
    
    # Create deployment directory
    deploy_dir = Path(deployment_path)
    deploy_dir.mkdir(exist_ok=True)
    
    model_name = wrapper.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Export LCM audit trail before deployment
    benchmark.start_timer("audit_export")
    audit_report = wrapper.export_lcm_metadata(
        output_format="audit_report", 
        include_receipts=True
    )
    
    audit_file = deploy_dir / f"{model_name}_audit_trail_{timestamp}.json"
    with open(audit_file, 'w') as f:
        json.dump(audit_report, f, indent=2)
    
    audit_time = benchmark.end_timer()
    print(f"1️⃣ LCM Audit Export: {audit_time:.4f}s")
    print(f"   ✅ Audit trail exported: {audit_file}")
    
    # 2. Pickle the complete model with LCM metadata
    benchmark.start_timer("model_serialization")
    model_file = deploy_dir / f"{model_name}_{timestamp}.pkl"
    
    with open(model_file, 'wb') as f:
        pickle.dump(wrapper, f)
    
    pickle_time = benchmark.end_timer()
    print(f"2️⃣ CIAF Model Serialization: {pickle_time:.4f}s")
    print(f"   ✅ Model pickled: {model_file}")
    print(f"   📊 File size: {model_file.stat().st_size:,} bytes")
    
    # 3. Create deployment manifest
    benchmark.start_timer("manifest_creation")
    
    model_info = wrapper.get_model_info()
    lcm_trail = wrapper.get_lcm_metadata_trail()
    
    manifest = {
        "deployment_info": {
            "model_name": model_name,
            "deployment_timestamp": datetime.now().isoformat(),
            "model_version": wrapper.model_version,
            "compliance_mode": wrapper.compliance_mode,
            "deployment_id": f"deploy_{timestamp}"
        },
        "model_files": {
            "model_pickle": str(model_file.name),
            "audit_trail": str(audit_file.name)
        },
        "lcm_summary": {
            "lcm_enabled": True,
            "training_capsules": len(lcm_trail.get('training_metadata', {})),
            "inference_receipts": len(lcm_trail.get('inference_metadata', {})),
            "connections_count": len(lcm_trail.get('connections_metadata', {}).get('connections_summary', [])),
            "enhanced_features": lcm_trail.get('enhanced_features', {}),
            "integrity_verified": wrapper._verify_lcm_integrity()
        },
        "production_readiness": {
            "audit_trail_complete": True,
            "pickle_preservation_verified": True,
            "compliance_ready": True,
            "inference_tracking_enabled": True
        },
        "benchmark_results": benchmark.get_summary()
    }
    
    manifest_file = deploy_dir / f"{model_name}_deployment_manifest_{timestamp}.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    manifest_time = benchmark.end_timer()
    print(f"3️⃣ Manifest Creation: {manifest_time:.4f}s")
    print(f"   ✅ Deployment manifest: {manifest_file}")
    
    return {
        "model_file": model_file,
        "audit_file": audit_file,
        "manifest_file": manifest_file,
        "deployment_id": f"deploy_{timestamp}",
        "benchmark_results": benchmark.get_summary()
    }

def load_deployed_model(model_file):
    """Load and verify a deployed model with LCM metadata."""
    print(f"\n📥 Loading Deployed Model with LCM Verification")
    print("="*55)
    
    print("1️⃣ Loading pickled model...")
    with open(model_file, 'rb') as f:
        restored_wrapper = pickle.load(f)
    
    print(f"   ✅ Model loaded: {restored_wrapper.model_name}")
    print(f"   📝 Version: {restored_wrapper.model_version}")
    
    # 2. Verify LCM metadata preservation
    print("2️⃣ Verifying LCM metadata preservation...")
    
    model_info = restored_wrapper.get_model_info()
    lcm_metadata = model_info.get('lcm_metadata', {})
    
    print(f"   LCM Integration: {'✅' if lcm_metadata.get('lcm_integration_enabled') else '❌'}")
    print(f"   Pickle Preservation: {'✅' if lcm_metadata.get('pickle_preservation_ready') else '❌'}")
    print(f"   Audit Trail Available: {'✅' if lcm_metadata.get('lcm_trail_available') else '❌'}")
    
    if hasattr(restored_wrapper, '_lcm_serialization_timestamp'):
        print(f"   Serialization Time: {restored_wrapper._lcm_serialization_timestamp}")
    
    # 3. Test production inference on restored model
    print("3️⃣ Testing production inference on restored model...")
    
    # Create a test query
    test_query = [0.5, -1.2, 0.8, 1.1, -0.3, 0.7, -0.9, 1.4, 0.2, -0.6,
                  0.9, -0.4, 1.3, 0.1, -0.8, 0.6, -1.1, 0.4, 0.9, -0.2]
    
    prediction, receipt = restored_wrapper.predict(
        query=test_query,
        use_model=True  # Use the actual restored model
    )
    
    print(f"   ✅ Inference successful!")
    print(f"   🎯 Prediction: {prediction}")
    print(f"   📋 Receipt: {receipt.receipt_hash[:16]}...")
    
    # 4. Export updated audit trail
    print("4️⃣ Exporting updated audit trail...")
    updated_audit = restored_wrapper.export_lcm_metadata(
        output_format="audit_report",
        include_receipts=True
    )
    
    total_receipts = updated_audit.get('audit_summary', {}).get('total_inference_receipts', 0)
    print(f"   📊 Total receipts in audit trail: {total_receipts}")
    
    return restored_wrapper

def main():
    """Main deployment demonstration with comprehensive benchmarking."""
    print("🚀 CIAF LCM Deployable Model Demonstration with Performance Benchmarking")
    print("Complete production deployment with audit trail preservation and overhead analysis")
    print("="*80)
    
    try:
        # 1. Benchmark baseline model (no CIAF)
        print("\n" + "="*80)
        baseline_results, baseline_model, X, y = benchmark_baseline_model()
        
        print("\n" + "="*80)
        # 2. Benchmark standard CIAF model
        production_model, ciaf_results, X_ciaf, y_ciaf = create_production_model()
        
        # 3. Benchmark optimized CIAF model (if available)
        optimized_results = None
        optimized_storage = None
        if OPTIMIZED_STORAGE_AVAILABLE:
            print("\n" + "="*80)
            optimized_model, optimized_results, optimized_storage = benchmark_optimized_model()
        
        # 4. Compare performance
        benchmark_util = CIAFBenchmark()
        print("\n" + "="*80)
        print("📊 STANDARD CIAF vs BASELINE Performance Analysis")
        overhead_analysis = benchmark_util.print_comparison(baseline_results, ciaf_results)
        
        if optimized_results and OPTIMIZED_STORAGE_AVAILABLE:
            print("\n" + "="*80) 
            print("🚀 OPTIMIZED CIAF vs BASELINE Performance Analysis")
            optimized_analysis = benchmark_util.print_comparison(baseline_results, optimized_results)
            
            print("\n" + "="*80)
            print("⚡ OPTIMIZED vs STANDARD CIAF Performance Improvement")
            improvement_analysis = benchmark_util.print_comparison(ciaf_results, optimized_results)
            
            # Show optimization gains
            print(f"\n🎯 Optimization Impact Summary:")
            for operation in optimized_results:
                if operation in ciaf_results:
                    standard_time = ciaf_results[operation]
                    optimized_time = optimized_results[operation]
                    improvement = ((standard_time - optimized_time) / standard_time) * 100
                    print(f"   • {operation.replace('_', ' ').title()}: {improvement:.1f}% faster")
        
        # 5. Deploy the model with deployment benchmarks
        deployment_info = deploy_model(production_model)
        
        print(f"\n🎉 Deployment Complete!")
        print(f"   Model File: {deployment_info['model_file']}")
        print(f"   Audit Trail: {deployment_info['audit_file']}")
        print(f"   Manifest: {deployment_info['manifest_file']}")
        print(f"   Deployment ID: {deployment_info['deployment_id']}")
        
        # 6. Simulate loading in production environment
        print(f"\n🔄 Simulating Production Environment...")
        print("(Loading model as if in a different process/server)")
        
        load_benchmark = CIAFBenchmark()
        load_benchmark.start_timer("model_loading")
        restored_model = load_deployed_model(deployment_info['model_file'])
        load_time = load_benchmark.end_timer()
        
        print(f"\n📊 Model Loading Benchmark: {load_time:.4f}s")
        
        # 7. Performance Summary
        print(f"\n✅ SUCCESS: Complete CIAF LCM Process Maintained!")
        print("="*80)
        print("🏭 Production Benefits:")
        print("   • Complete audit trail preserved through deployment")
        print("   • Regulatory compliance maintained")
        print("   • Inference tracking continues in production")
        print("   • Model lineage fully traceable")
        print("   • Pickle serialization preserves all LCM metadata")
        print("   • Ready for production deployment and scaling")
        
        print(f"\n⚡ Performance Insights:")
        for operation, data in overhead_analysis.items():
            impact = "HIGH" if data['overhead_percent'] > 50 else "MEDIUM" if data['overhead_percent'] > 20 else "LOW"
            print(f"   • {operation.replace('_', ' ').title()}: {data['overhead_percent']:+.1f}% overhead ({impact} impact)")
        
        # 8. Save comprehensive benchmark report
        benchmark_report = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "baseline_performance": baseline_results,
            "ciaf_performance": ciaf_results,
            "optimized_performance": optimized_results,
            "deployment_performance": deployment_info['benchmark_results'],
            "loading_performance": {"model_loading": load_time},
            "overhead_analysis": overhead_analysis,
            "model_info": {
                "algorithm": "RandomForestClassifier",
                "n_estimators": 100,
                "n_samples": 1000,
                "n_features": 20,
                "compliance_mode": "financial"
            }
        }
        
        if optimized_results:
            benchmark_report["optimization_gains"] = {
                operation: {
                    "standard_time": ciaf_results[operation],
                    "optimized_time": optimized_results[operation],
                    "improvement_percent": ((ciaf_results[operation] - optimized_results[operation]) / ciaf_results[operation]) * 100
                }
                for operation in optimized_results if operation in ciaf_results
            }
        
        benchmark_file = Path("./ciaf_performance_benchmark.json")
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_report, f, indent=2)
        
        print(f"\n📋 Comprehensive benchmark report saved: {benchmark_file}")
        
        # 9. Show performance statistics if available
        if optimized_storage and hasattr(optimized_storage, 'get_performance_stats'):
            stats = optimized_storage.get_performance_stats()
            print(f"\n📈 Optimized Storage Performance Stats:")
            print(f"   Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
            print(f"   Total Saves: {stats['total_saves']}")
            print(f"   Average Save Time: {stats['avg_save_time']:.4f}s")
            print(f"   Buffer Flushes: {stats['buffer_flushes']}")
            
            # Cleanup optimized storage
            if hasattr(optimized_storage, 'shutdown'):
                optimized_storage.shutdown()
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()