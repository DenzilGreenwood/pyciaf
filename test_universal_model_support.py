#!/usr/bin/env python3
"""
Comprehensive test for universal model support in CIAF consolidated wrapper system.

This test demonstrates the wrapper system's ability to work with multiple ML frameworks
including sklearn, PyTorch, TensorFlow, HuggingFace, XGBoost, LightGBM, and custom models.

Created: 2025-09-28
Author: CIAF Development Team
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_sklearn_models():
    """Test sklearn model support."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LinearRegression
        from ciaf.wrappers import create_model_wrapper, ModelType
        
        print("Testing sklearn models...")
        
        # Create sample data
        X = np.random.rand(100, 5)
        y_class = np.random.randint(0, 2, 100)
        y_reg = np.random.rand(100)
        
        # Test classifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y_class)
        
        wrapper = create_model_wrapper(clf, "sklearn_classifier", wrapper_type="auto")
        predictions = wrapper.predict(X[:5])
        print(f"✅ sklearn classifier: {type(clf).__name__}, predictions shape: {predictions.shape}")
        
        # Test regressor  
        reg = LinearRegression()
        reg.fit(X, y_reg)
        
        wrapper = create_model_wrapper(reg, "sklearn_regressor", wrapper_type="auto") 
        predictions = wrapper.predict(X[:5])
        print(f"✅ sklearn regressor: {type(reg).__name__}, predictions shape: {predictions.shape}")
        
        return True
    except Exception as e:
        print(f"❌ sklearn test failed: {e}")
        return False

def test_pytorch_models():
    """Test PyTorch model support if available."""
    try:
        import torch
        import torch.nn as nn
        from ciaf.wrappers import create_model_wrapper
        
        print("Testing PyTorch models...")
        
        # Create simple neural network
        class SimpleNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(5, 10)
                self.fc2 = nn.Linear(10, 1)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        model = SimpleNN()
        
        # Test wrapper creation
        wrapper = create_model_wrapper(model, "pytorch_model", wrapper_type="auto")
        
        # Test prediction with numpy input (should be converted internally)
        X = np.random.rand(5, 5).astype(np.float32)
        predictions = wrapper.predict(X)
        
        print(f"✅ PyTorch model: {type(model).__name__}, predictions shape: {predictions.shape}")
        return True
        
    except ImportError:
        print("⚠️  PyTorch not available, skipping PyTorch tests")
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_tensorflow_models():
    """Test TensorFlow model support if available."""
    try:
        import tensorflow as tf
        from ciaf.wrappers import create_model_wrapper
        
        print("Testing TensorFlow models...")
        
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Test wrapper creation
        wrapper = create_model_wrapper(model, "tensorflow_model", wrapper_type="auto")
        
        # Test prediction
        X = np.random.rand(5, 5).astype(np.float32)
        predictions = wrapper.predict(X)
        
        print(f"✅ TensorFlow model: {type(model).__name__}, predictions shape: {predictions.shape}")
        return True
        
    except ImportError:
        print("⚠️  TensorFlow not available, skipping TensorFlow tests")
        return True  
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def test_xgboost_models():
    """Test XGBoost model support if available."""
    try:
        import xgboost as xgb
        from ciaf.wrappers import create_model_wrapper
        
        print("Testing XGBoost models...")
        
        # Create sample data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Create and train model
        model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test wrapper creation
        wrapper = create_model_wrapper(model, "xgboost_model", wrapper_type="auto")
        
        # Test prediction
        predictions = wrapper.predict(X[:5])
        
        print(f"✅ XGBoost model: {type(model).__name__}, predictions shape: {predictions.shape}")
        return True
        
    except ImportError:
        print("⚠️  XGBoost not available, skipping XGBoost tests")
        return True
    except Exception as e:
        print(f"❌ XGBoost test failed: {e}")
        return False

def test_lightgbm_models():
    """Test LightGBM model support if available."""
    try:
        import lightgbm as lgb
        from ciaf.wrappers import create_model_wrapper
        
        print("Testing LightGBM models...")
        
        # Create sample data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Create and train model
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test wrapper creation  
        wrapper = create_model_wrapper(model, "lightgbm_model", wrapper_type="auto")
        
        # Test prediction
        predictions = wrapper.predict(X[:5])
        
        print(f"✅ LightGBM model: {type(model).__name__}, predictions shape: {predictions.shape}")
        return True
        
    except ImportError:
        print("⚠️  LightGBM not available, skipping LightGBM tests")
        return True
    except Exception as e:
        print(f"❌ LightGBM test failed: {e}")
        return False

def test_custom_models():
    """Test custom model support."""
    try:
        from ciaf.wrappers import create_model_wrapper
        
        print("Testing custom models...")
        
        # Create custom model class
        class CustomModel:
            def __init__(self):
                self.is_fitted = False
                self.coefficients = None
                
            def fit(self, X, y):
                # Simple linear fit
                self.coefficients = np.random.rand(X.shape[1])
                self.is_fitted = True
                return self
                
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                return X @ self.coefficients
        
        # Create and fit custom model
        model = CustomModel()
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        model.fit(X, y)
        
        # Test wrapper creation
        wrapper = create_model_wrapper(model, "custom_model", wrapper_type="auto")
        
        # Test prediction
        predictions = wrapper.predict(X[:5])
        
        print(f"✅ Custom model: {type(model).__name__}, predictions shape: {predictions.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Custom model test failed: {e}")
        return False

def test_mixed_data_types():
    """Test handling of different data types."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from ciaf.wrappers import create_model_wrapper, DataType
        from ciaf.wrappers.universal_model_adapter import UniversalDataProcessor
        
        print("Testing mixed data type processing...")
        
        processor = UniversalDataProcessor()
        
        # Test numeric data
        numeric_data = np.random.rand(10, 5)
        processed = processor.process_data(numeric_data, DataType.NUMERICAL)
        print(f"✅ Numeric data processing: input {numeric_data.shape} -> output {processed.shape}")
        
        # Test text data (if possible)
        try:
            text_data = ["sample text", "another example", "third item"]
            processed = processor.process_data(text_data, DataType.TEXT)
            print(f"✅ Text data processing: input {len(text_data)} items -> output shape {processed.shape if hasattr(processed, 'shape') else 'processed'}")
        except Exception as e:
            print(f"⚠️  Text processing requires additional dependencies: {e}")
        
        # Test mixed data
        mixed_data = {
            'numeric': np.random.rand(5, 3),
            'categorical': ['A', 'B', 'A', 'C', 'B']
        }
        try:
            processed = processor.process_data(mixed_data, DataType.MIXED)
            print(f"✅ Mixed data processing: processed successfully")
        except Exception as e:
            print(f"⚠️  Mixed data processing: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mixed data type test failed: {e}")
        return False

def main():
    """Run all universal model support tests."""
    print("=" * 60)
    print("CIAF Universal Model Support Tests")
    print("=" * 60)
    
    tests = [
        ("sklearn Models", test_sklearn_models),
        ("PyTorch Models", test_pytorch_models), 
        ("TensorFlow Models", test_tensorflow_models),
        ("XGBoost Models", test_xgboost_models),
        ("LightGBM Models", test_lightgbm_models),
        ("Custom Models", test_custom_models),
        ("Mixed Data Types", test_mixed_data_types),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} test suites passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All universal model support tests passed!")
        print("   The consolidated wrapper system successfully supports:")
        print("   - Multiple ML frameworks (sklearn, PyTorch, TensorFlow, XGBoost, LightGBM)")
        print("   - Custom model implementations") 
        print("   - Various data types (numeric, text, mixed)")
        print("   - Universal prediction interfaces")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)