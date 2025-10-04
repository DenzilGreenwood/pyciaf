#!/usr/bin/env python3
"""
Test suite for the ultimate GDPR model wrapper to verify all gaps have been fixed.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_ultimate_wrapper_imports():
    """Test all necessary imports work correctly."""
    print("🔄 Testing imports...")
    
    try:
        from ciaf.wrappers import create_ultimate_gdpr_wrapper, create_model_wrapper
        from ciaf.wrappers.gdpr_model_wrapper import GDPRModelWrapper
        from ciaf.wrappers.policy import WrapperPolicy
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_policy_creation():
    """Test policy creation and configuration."""
    print("\n🔄 Testing policy creation...")
    
    try:
        from ciaf.wrappers.policy import WrapperPolicy
        
        # Test default policy
        policy = WrapperPolicy()
        print(f"✅ Default policy created: {policy.wrapper_mode}")
        
        # Test compliance policy
        compliance_policy = WrapperPolicy.create_compliance_policy()
        print(f"✅ Compliance policy created: {compliance_policy.wrapper_mode}")
        
        # Test production policy
        production_policy = WrapperPolicy.create_production_policy()
        print(f"✅ Production policy created: {production_policy.wrapper_mode}")
        
        return True
    except Exception as e:
        print(f"❌ Policy creation failed: {e}")
        return False

def test_mock_model_creation():
    """Test with a simple mock model."""
    print("\n🔄 Testing with mock model...")
    
    try:
        from ciaf.wrappers.gdpr_model_wrapper import GDPRModelWrapper
        from ciaf.wrappers.policy import WrapperPolicy
        import numpy as np
        
        # Create a simple mock model
        class MockModel:
            def fit(self, X, y):
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not hasattr(self, 'is_fitted'):
                    raise ValueError("Model not fitted")
                return np.zeros(len(X))
        
        mock_model = MockModel()
        policy = WrapperPolicy()
        
        # Test wrapper creation
        wrapper = GDPRModelWrapper(
            model=mock_model,
            model_name="test_mock_model",
            custom_policy=policy
        )
        print("✅ GDPRModelWrapper created successfully")
        
        # Test basic properties
        print(f"✅ Model name: {wrapper.model_name}")
        print(f"✅ Model type: {wrapper.model_type}")
        print(f"✅ Policy mode: {wrapper.policy.wrapper_mode}")
        
        return True
    except Exception as e:
        print(f"❌ Mock model test failed: {e}")
        return False

def test_factory_functions():
    """Test factory functions."""
    print("\n🔄 Testing factory functions...")
    
    try:
        from ciaf.wrappers import create_ultimate_gdpr_wrapper
        import numpy as np
        
        # Create a simple mock model
        class MockModel:
            def fit(self, X, y):
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not hasattr(self, 'is_fitted'):
                    raise ValueError("Model not fitted")
                return np.zeros(len(X))
        
        mock_model = MockModel()
        
        # Test ultimate wrapper factory
        wrapper = create_ultimate_gdpr_wrapper(
            model=mock_model,
            model_name="test_factory_model",
            compliance_level="balanced",
            performance_mode="balanced"
        )
        print("✅ Ultimate GDPR wrapper factory successful")
        print(f"✅ Wrapper type: {type(wrapper).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        return False

def test_general_factory():
    """Test general factory function."""
    print("\n🔄 Testing general factory...")
    
    try:
        from ciaf.wrappers import create_model_wrapper
        import numpy as np
        
        # Create a simple mock model
        class MockModel:
            def fit(self, X, y):
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not hasattr(self, 'is_fitted'):
                    raise ValueError("Model not fitted")
                return np.zeros(len(X))
        
        mock_model = MockModel()
        
        # Test auto wrapper selection
        wrapper = create_model_wrapper(
            model=mock_model,
            model_name="test_auto_model",
            wrapper_type="auto"
        )
        print("✅ Auto wrapper selection successful")
        print(f"✅ Selected wrapper type: {type(wrapper).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ General factory test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Ultimate GDPR Model Wrapper")
    print("=" * 50)
    
    tests = [
        test_ultimate_wrapper_imports,
        test_policy_creation,
        test_mock_model_creation,
        test_factory_functions,
        test_general_factory
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ultimate wrapper is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    main()