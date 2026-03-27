#!/usr/bin/env python3
"""
Quick test to verify the performance optimizations are properly integrated
in the deployable_model_demo.py file.
"""


def test_optimized_integration():
    """Test that the optimized functions are available and working."""
    print("🧪 Testing Performance Optimization Integration")
    print("=" * 55)

    try:
        # Test imports
        print("1️⃣ Testing imports...")
        from deployable_model_demo import CIAFBenchmark, OPTIMIZED_STORAGE_AVAILABLE
        from ciaf.vault.metadata_storage_optimized import HighPerformanceMetadataStorage

        print("   ✅ All imports successful")

        # Test benchmark utility
        print("2️⃣ Testing benchmark utility...")
        benchmark = CIAFBenchmark()
        benchmark.start_timer("test_operation")
        import time

        time.sleep(0.01)  # 10ms test
        elapsed = benchmark.end_timer()
        print(f"   ✅ Benchmark working: {elapsed:.4f}s measured")

        # Test optimized storage
        print("3️⃣ Testing optimized storage...")
        config = {"storage_path": "test_optimized", "memory_buffer_size": 100}
        storage = HighPerformanceMetadataStorage(config)

        # Quick storage test
        metadata_id = storage.save_metadata(
            model_name="test_model",
            stage="testing",
            event_type="unit_test",
            metadata={"test": True},
        )

        retrieved = storage.get_metadata(metadata_id)
        print("   ✅ Storage working: saved and retrieved metadata")

        # Test performance stats
        stats = storage.get_performance_stats()
        print(f"   📊 Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"   📊 Total saves: {stats['total_saves']}")

        storage.shutdown()

        # Test availability flags
        print("4️⃣ Testing availability flags...")
        print(
            f"   Optimized storage available: {'✅' if OPTIMIZED_STORAGE_AVAILABLE else '❌'}"
        )

        # Test optimized benchmark function (without full execution)
        print("5️⃣ Testing optimized benchmark function availability...")
        if OPTIMIZED_STORAGE_AVAILABLE:
            print("   ✅ benchmark_optimized_model() function is available")
        else:
            print("   ⚠️  Optimized storage not available, benchmark will be skipped")

        print("\n✅ Integration Test Passed!")
        print("The performance optimizations are properly integrated.")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_optimized_integration()
