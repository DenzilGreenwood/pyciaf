"""
Simple debug test for protocol issue.
"""

try:
    print("Testing protocol imports...")
    from ciaf.wrappers.interfaces import ModelWrapper
    print("✅ Interfaces imported")
    
    from ciaf.wrappers.policy import WrapperPolicy
    print("✅ Policy imported")
    
    from ciaf.wrappers.protocol_implementations import DefaultModelAdapter
    print("✅ Protocol implementations imported")
    
    # Test creating policy
    policy = WrapperPolicy()
    print("✅ Policy created")
    print(f"Policy protocols available: {policy.model_adapter is not None}")
    
    # Test creating modern wrapper
    from ciaf.wrappers.modern_wrapper import ModernCIAFModelWrapper
    print("✅ Modern wrapper imported")
    
    # Test creating a simple model and wrapper
    class TestModel:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return [1, 0, 1]
    
    model = TestModel()
    print("✅ Test model created")
    
    # Create wrapper
    wrapper = ModernCIAFModelWrapper(
        model=model,
        model_name="debug_test",
        policy=policy
    )
    print("✅ Modern wrapper created successfully!")
    
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    print(traceback.format_exc())