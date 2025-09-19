#!/usr/bin/env python3
"""
Quick test of the optimized storage performance improvements.
This script tests the optimized storage in isolation.
"""

import time
import json
import tempfile
import os
from pathlib import Path

# Import our optimized storage
try:
    from ciaf.metadata_storage_optimized import HighPerformanceMetadataStorage
    OPTIMIZED_AVAILABLE = True
except ImportError as e:
    print(f"❌ Optimized storage not available: {e}")
    OPTIMIZED_AVAILABLE = False

# Import standard storage for comparison
from ciaf.metadata_storage import MetadataStorage

def generate_test_metadata(count: int = 100):
    """Generate test metadata for benchmarking."""
    metadata_items = []
    for i in range(count):
        metadata = {
            "id": f"test_item_{i:04d}",
            "timestamp": time.time(),
            "operation": "test_operation",
            "data": {
                "sample_size": 1000,
                "features": list(range(20)),
                "target": f"prediction_{i}",
                "confidence": 0.85 + (i % 15) * 0.01,
                "metadata": {
                    "nested_field_1": f"value_{i}",
                    "nested_field_2": [1, 2, 3, 4, 5] * (i % 5 + 1),
                    "nested_field_3": {"deep": {"nested": {"value": i}}}
                }
            }
        }
        metadata_items.append(metadata)
    return metadata_items

def benchmark_storage(storage, metadata_items, name: str):
    """Benchmark a storage implementation."""
    print(f"\n📊 Benchmarking {name}...")
    
    # Test save performance
    start_time = time.perf_counter()
    saved_ids = []
    for item in metadata_items:
        metadata_id = storage.save_metadata(
            model_name="test_model",
            stage="benchmark",
            event_type="test_event",
            metadata=item,
            model_version="1.0.0"
        )
        saved_ids.append(metadata_id)
    save_time = time.perf_counter() - start_time
    
    # Test load performance by metadata ID
    start_time = time.perf_counter()
    loaded_items = []
    for metadata_id in saved_ids[:50]:  # Test first 50 for speed
        loaded = storage.get_metadata(metadata_id)
        if loaded:
            loaded_items.append(loaded)
    load_time = time.perf_counter() - start_time
    
    # Test bulk query performance
    start_time = time.perf_counter()
    all_records = storage.get_model_metadata(
        model_name="test_model",
        stage="benchmark",
        limit=len(metadata_items)
    )
    list_time = time.perf_counter() - start_time
    
    results = {
        "save_time": save_time,
        "load_time": load_time,
        "list_time": list_time,
        "total_time": save_time + load_time + list_time,
        "items_count": len(metadata_items),
        "avg_save_time": save_time / len(metadata_items),
        "avg_load_time": load_time / min(50, len(saved_ids)) if saved_ids else 0,
        "items_saved": len(saved_ids),
        "items_loaded": len(loaded_items),
        "items_listed": len(all_records) if all_records else 0
    }
    
    print(f"   💾 Save Time: {save_time:.4f}s ({results['avg_save_time']:.6f}s per item)")
    print(f"   📖 Load Time: {load_time:.4f}s ({results['avg_load_time']:.6f}s per item)")
    print(f"   📋 List Time: {list_time:.4f}s")
    print(f"   🔢 Total Time: {results['total_time']:.4f}s")
    print(f"   ✅ Items Saved: {results['items_saved']}")
    print(f"   📥 Items Loaded: {results['items_loaded']}")
    print(f"   📋 Items Listed: {results['items_listed']}")
    
    return results

def main():
    """Main benchmarking function."""
    print("🚀 Optimized Storage Performance Test")
    print("="*60)
    
    if not OPTIMIZED_AVAILABLE:
        print("❌ Cannot run test - optimized storage not available")
        return
    
    # Generate test data
    test_count = 100  # Reduced for faster testing
    print(f"📋 Generating {test_count} test metadata items...")
    metadata_items = generate_test_metadata(test_count)
    
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        standard_config = {
            "storage_path": os.path.join(temp_dir, "standard"),
            "use_compression": False,
            "backup_enabled": False
        }
        
        optimized_config = {
            "storage_path": os.path.join(temp_dir, "optimized"),
            "enable_lazy_materialization": True,
            "fast_inference_mode": True,
            "memory_buffer_size": 1000,
            "db_connection_pool_size": 5,
            "enable_async_writes": True,
            "batch_write_size": 50
        }
        
        # Test standard storage
        print(f"\n🔧 Setting up Standard Storage...")
        standard_storage = MetadataStorage(
            storage_path=standard_config["storage_path"],
            backend="json",
            use_compression=standard_config["use_compression"]
        )
        standard_results = benchmark_storage(standard_storage, metadata_items, "Standard Storage")
        
        # Test optimized storage
        print(f"\n⚡ Setting up Optimized Storage...")
        optimized_storage = HighPerformanceMetadataStorage(optimized_config)
        optimized_results = benchmark_storage(optimized_storage, metadata_items, "Optimized Storage")
        
        # Show performance statistics if available
        if hasattr(optimized_storage, 'get_performance_stats'):
            stats = optimized_storage.get_performance_stats()
            print(f"\n📈 Optimized Storage Performance Stats:")
            print(f"   Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
            print(f"   Total Saves: {stats['total_saves']}")
            print(f"   Average Save Time: {stats['avg_save_time']:.6f}s")
            print(f"   Buffer Flushes: {stats['buffer_flushes']}")
        
        # Calculate improvement
        print(f"\n🎯 Performance Comparison:")
        print("="*60)
        
        save_improvement = ((standard_results['save_time'] - optimized_results['save_time']) / standard_results['save_time']) * 100
        load_improvement = ((standard_results['load_time'] - optimized_results['load_time']) / standard_results['load_time']) * 100
        total_improvement = ((standard_results['total_time'] - optimized_results['total_time']) / standard_results['total_time']) * 100
        
        print(f"💾 Save Performance: {save_improvement:+.1f}% {'improvement' if save_improvement > 0 else 'regression'}")
        print(f"📖 Load Performance: {load_improvement:+.1f}% {'improvement' if load_improvement > 0 else 'regression'}")
        print(f"🔢 Total Performance: {total_improvement:+.1f}% {'improvement' if total_improvement > 0 else 'regression'}")
        
        # Save detailed results
        benchmark_report = {
            "test_timestamp": time.time(),
            "test_config": {
                "item_count": test_count,
                "standard_config": standard_config,
                "optimized_config": optimized_config
            },
            "standard_results": standard_results,
            "optimized_results": optimized_results,
            "improvements": {
                "save_improvement_percent": save_improvement,
                "load_improvement_percent": load_improvement,
                "total_improvement_percent": total_improvement
            }
        }
        
        if hasattr(optimized_storage, 'get_performance_stats'):
            benchmark_report["optimized_stats"] = optimized_storage.get_performance_stats()
        
        report_file = Path("./optimized_storage_benchmark.json")
        with open(report_file, 'w') as f:
            json.dump(benchmark_report, f, indent=2)
        
        print(f"\n📋 Detailed benchmark report saved: {report_file}")
        
        # Cleanup
        if hasattr(optimized_storage, 'shutdown'):
            optimized_storage.shutdown()
        
        print(f"\n✅ Storage Performance Test Complete!")

if __name__ == "__main__":
    main()