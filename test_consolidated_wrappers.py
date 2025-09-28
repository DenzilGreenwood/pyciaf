#!/usr/bin/env python3
"""
Test script for consolidated wrapper implementations.

This script tests the new universal model wrapper system that works
with all ML frameworks and consolidates functionality from legacy wrappers.
"""

import sys
import traceback
import warnings
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all new consolidated imports work."""
    print("Testing consolidated wrapper imports...")
    
    try:
        from ciaf.wrappers import (
            # Universal model support
            UniversalModelDetector,
            UniversalDataProcessor, 
            UniversalModelAdapter,
            
            # Consolidated implementations
            ConsolidatedModelAdapter,
            EnhancedModelMetadataProvider,
            RobustModelValidator,
            ConsolidatedModelTrainingHandler,
            ConsolidatedModelInferenceHandler,
            ConsolidatedLCMMetadataHandler,
            ConsolidatedEnhancementProvider,
            ConsolidatedComplianceIntegrator,
            ConsolidatedPerformanceOptimizer,
            
            # Enhanced enums and policies
            ModelType,
            DataType,
            WrapperPolicy,
            
            # Factory functions
            create_model_wrapper,
            create_auto_wrapper,
            create_consolidated_wrapper_protocols,
            create_universal_model_wrapper,
        )
        print("✅ All consolidated wrapper imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False


def test_model_type_expansion():
    """Test the expanded ModelType enum."""
    print("\nTesting expanded ModelType enum...")
    
    try:
        from ciaf.wrappers import ModelType
        
        # Test new model types
        new_types = [
            ModelType.CATBOOST,
            ModelType.JAX, 
            ModelType.ONNX,
            ModelType.KERAS
        ]
        
        print(f"✅ New model types available: {[t.value for t in new_types]}")
        return True
        
    except Exception as e:
        print(f"❌ ModelType expansion test failed: {e}")
        return False


def test_data_type_enum():
    """Test the new DataType enum."""
    print("\nTesting DataType enum...")
    
    try:
        from ciaf.wrappers import DataType
        
        # Test data types
        data_types = [
            DataType.TABULAR,
            DataType.TEXT,
            DataType.IMAGE,
            DataType.NUMERICAL,
            DataType.CATEGORICAL,
            DataType.MULTIMODAL
        ]
        
        print(f"✅ DataType enum available: {[t.value for t in data_types]}")
        return True
        
    except Exception as e:
        print(f"❌ DataType enum test failed: {e}")
        return False


def test_sklearn_model_wrapper():
    """Test consolidated wrapper with sklearn model."""
    print("\nTesting consolidated wrapper with sklearn model...")
    
    try:
        # Create a simple sklearn model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        # Create and fit model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test with consolidated wrapper
        from ciaf.wrappers import create_universal_model_wrapper
        
        wrapped_model = create_universal_model_wrapper(
            model=model,
            model_name="test_sklearn_model"
        )
        
        print("✅ Successfully created universal wrapper for sklearn model!")
        print(f"   Wrapped model type: {type(wrapped_model)}")
        
        # Test prediction
        predictions = wrapped_model.predict(X[:5])
        print(f"   Predictions shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sklearn wrapper test failed: {e}")
        traceback.print_exc()
        return False


def test_universal_model_detector():
    """Test the universal model detection capabilities."""
    print("\nTesting universal model detector...")
    
    try:
        from ciaf.wrappers import UniversalModelDetector
        from sklearn.ensemble import RandomForestClassifier
        
        # Create test model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Test detection
        detector = UniversalModelDetector()
        framework, model_type = detector.detect_model_framework(model)
        
        print(f"✅ Model detection successful!")
        print(f"   Framework: {framework}")
        print(f"   Model type: {model_type.value if hasattr(model_type, 'value') else model_type}")
        
        # Test capabilities detection
        capabilities = []
        if hasattr(model, 'fit'):
            capabilities.append('trainable')
        if hasattr(model, 'predict'):
            capabilities.append('predictive')
        if hasattr(model, 'predict_proba'):
            capabilities.append('probabilistic')
        
        print(f"   Capabilities: {capabilities}")
        
        return True
        
    except Exception as e:
        print(f"❌ Universal model detector test failed: {e}")
        traceback.print_exc()
        return False


def test_consolidated_protocols():
    """Test consolidated protocol creation."""
    print("\nTesting consolidated protocol creation...")
    
    try:
        from ciaf.wrappers import (
            create_consolidated_wrapper_protocols,
            get_default_wrapper_policy
        )
        
        # Create default policy
        policy = get_default_wrapper_policy()
        
        # Create consolidated protocols
        protocols = create_consolidated_wrapper_protocols(policy)
        
        expected_protocols = [
            'model_adapter',
            'metadata_provider', 
            'model_validator',
            'training_handler',
            'inference_handler',
            'lcm_metadata_handler',
            'enhancement_provider',
            'compliance_integrator',
            'performance_optimizer'
        ]
        
        for protocol_name in expected_protocols:
            if protocol_name not in protocols:
                print(f"❌ Missing protocol: {protocol_name}")
                return False
        
        print(f"✅ All {len(protocols)} consolidated protocols created successfully!")
        print(f"   Available protocols: {list(protocols.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consolidated protocols test failed: {e}")
        traceback.print_exc()
        return False


def test_framework_compatibility():
    """Test compatibility flags for different frameworks."""
    print("\nTesting framework compatibility flags...")
    
    try:
        from ciaf.wrappers import (
            CONSOLIDATED_IMPLEMENTATIONS_AVAILABLE,
            UNIVERSAL_ADAPTER_AVAILABLE,
            PROTOCOL_IMPLEMENTATIONS_AVAILABLE,
            MODERN_WRAPPER_AVAILABLE
        )
        
        print(f"✅ Framework compatibility status:")
        print(f"   Consolidated implementations: {CONSOLIDATED_IMPLEMENTATIONS_AVAILABLE}")
        print(f"   Universal adapter: {UNIVERSAL_ADAPTER_AVAILABLE}")
        print(f"   Protocol implementations: {PROTOCOL_IMPLEMENTATIONS_AVAILABLE}")
        print(f"   Modern wrapper: {MODERN_WRAPPER_AVAILABLE}")
        
        # Test that at least consolidated implementations are available
        if not CONSOLIDATED_IMPLEMENTATIONS_AVAILABLE:
            print("❌ Consolidated implementations should be available!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Framework compatibility test failed: {e}")
        return False


def run_all_tests():
    """Run all consolidated wrapper tests."""
    print("=" * 60)
    print("CIAF Consolidated Wrapper System Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_type_expansion,
        test_data_type_enum,
        test_universal_model_detector,
        test_consolidated_protocols,
        test_framework_compatibility,
        test_sklearn_model_wrapper,  # This might have dependencies
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Consolidated wrapper system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)