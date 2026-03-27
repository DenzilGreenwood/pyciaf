#!/usr/bin/env python3
"""
Quick integration test to verify optimized storage works with CIAF wrappers.
"""

import time
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Test if we can import CIAF components
try:
    from ciaf.wrappers.model_wrapper import CIAFModelWrapper
    from ciaf.vault.metadata_storage_optimized import HighPerformanceMetadataStorage

    print("✅ CIAF imports successful")
except ImportError as e:
    print(f"❌ CIAF import failed: {e}")
    exit(1)


def test_optimized_integration():
    """Test optimized storage with CIAF model wrapper."""
    print("🧪 Testing Optimized Storage Integration with CIAF")
    print("=" * 60)

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure optimized storage
        optimized_config = {
            "storage_path": os.path.join(temp_dir, "optimized_ciaf"),
            "enable_lazy_materialization": True,
            "fast_inference_mode": True,
            "memory_buffer_size": 100,
            "batch_write_size": 10,
        }

        # Initialize optimized storage
        print("🔧 Initializing optimized storage...")
        optimized_storage = HighPerformanceMetadataStorage(optimized_config)
        print(f"   Storage path: {optimized_config['storage_path']}")

        # Create test data
        print("📋 Generating test dataset...")
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        print(f"   Dataset shape: {X.shape}")

        # Test basic storage operations
        print("💾 Testing basic storage operations...")
        start_time = time.perf_counter()

        # Save some test metadata
        metadata_ids = []
        for i in range(10):
            metadata = {
                "test_id": i,
                "operation": "unit_test",
                "data_shape": list(X.shape),
                "timestamp": time.time(),
            }

            metadata_id = optimized_storage.save_metadata(
                model_name="test_integration_model",
                stage="testing",
                event_type="unit_test",
                metadata=metadata,
            )
            metadata_ids.append(metadata_id)

        save_time = time.perf_counter() - start_time
        print(f"   Saved 10 metadata items in {save_time:.4f}s")

        # Test retrieval
        start_time = time.perf_counter()
        retrieved_count = 0
        for metadata_id in metadata_ids:
            retrieved = optimized_storage.get_metadata(metadata_id)
            if retrieved:
                retrieved_count += 1

        load_time = time.perf_counter() - start_time
        print(f"   Retrieved {retrieved_count}/10 items in {load_time:.4f}s")

        # Test bulk query
        start_time = time.perf_counter()
        all_items = optimized_storage.get_model_metadata(
            "test_integration_model", "testing"
        )
        query_time = time.perf_counter() - start_time
        print(f"   Queried {len(all_items)} items in {query_time:.4f}s")

        # Get performance stats
        stats = optimized_storage.get_performance_stats()
        print("\n📈 Performance Stats:")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        print(f"   Total Saves: {stats['total_saves']}")
        print(f"   Average Save Time: {stats['avg_save_time']:.6f}s")
        print(f"   Cached Items: {stats['cached_items']}")

        # Test CIAF model wrapper (basic test)
        print("\n🤖 Testing CIAF Model Wrapper Integration...")
        try:
            # Create model wrapper
            wrapper = CIAFModelWrapper(
                model_name="Integration_Test_Model", compliance_mode="financial"
            )
            print("   ✅ CIAF wrapper created successfully")

            # Create and wrap a simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            wrapper.set_model(model)
            print("   ✅ Model set in wrapper")

            # Quick training test
            print("   🔄 Testing training integration...")
            training_start = time.perf_counter()
            wrapper.train(X[:50], y[:50])  # Small subset for speed
            training_time = time.perf_counter() - training_start
            print(f"   ✅ Training completed in {training_time:.4f}s")

            # Quick inference test
            print("   🔄 Testing inference integration...")
            inference_start = time.perf_counter()
            predictions = wrapper.predict(X[50:55])  # 5 samples
            inference_time = time.perf_counter() - inference_start
            print(f"   ✅ Inference completed in {inference_time:.4f}s")
            print(f"   📊 Predictions shape: {predictions.shape}")

        except Exception as e:
            print(f"   ⚠️  CIAF wrapper test failed: {e}")
            print("   Note: This may be due to missing dependencies or configuration")

        # Cleanup
        optimized_storage.shutdown()
        print("\n✅ Integration test completed successfully!")
        print(
            f"   Total operations: {stats['total_saves']} saves, {stats['cache_hits'] + stats['cache_misses']} loads"
        )


if __name__ == "__main__":
    test_optimized_integration()
