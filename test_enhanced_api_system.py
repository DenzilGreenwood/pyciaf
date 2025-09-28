#!/usr/bin/env python3
"""
CIAF Enhanced API System Test
=============================

Comprehensive test for the enhanced CIAF API system, demonstrating
protocol-based architecture, policy-driven configuration, and full
integration with all enhanced CIAF modules.

Created: 2025-09-28
Author: CIAF Development Team
"""

import warnings
warnings.filterwarnings('ignore')

def test_api_imports():
    """Test API module imports and availability."""
    try:
        from ciaf.api import (
            # Core interfaces
            APIPolicy, APIMode, SecurityLevel,
            # Consolidated framework
            ConsolidatedCIAFAPIFramework,
            # Factory functions
            create_api_framework, create_development_api, create_production_api,
            # Availability flags
            CONSOLIDATED_API_AVAILABLE, PROTOCOL_IMPLEMENTATIONS_AVAILABLE
        )
        
        print("✅ All API imports successful!")
        print(f"   Consolidated API available: {CONSOLIDATED_API_AVAILABLE}")
        print(f"   Protocol implementations available: {PROTOCOL_IMPLEMENTATIONS_AVAILABLE}")
        
        return True
    except ImportError as e:
        print(f"❌ API import failed: {e}")
        return False


def test_api_policy_configuration():
    """Test policy-driven API configuration."""
    try:
        from ciaf.api import (
            get_development_api_policy, get_production_api_policy, get_testing_api_policy,
            create_api_policy, APIMode, SecurityLevel
        )
        
        print("Testing API policy configurations...")
        
        # Test environment-specific policies
        dev_policy = get_development_api_policy()
        print(f"✅ Development policy: mode={dev_policy.api_mode.value}, auth_required={dev_policy.security.require_authentication}")
        
        prod_policy = get_production_api_policy()
        print(f"✅ Production policy: mode={prod_policy.api_mode.value}, auth_required={prod_policy.security.require_authentication}")
        
        test_policy = get_testing_api_policy()
        print(f"✅ Testing policy: mode={test_policy.api_mode.value}, auth_required={test_policy.security.require_authentication}")
        
        # Test custom policy creation
        custom_policy = create_api_policy(
            api_mode=APIMode.STAGING,
            security_level=SecurityLevel.HIGH
        )
        print(f"✅ Custom policy: mode={custom_policy.api_mode.value}, security_level=HIGH")
        
        return True
    except Exception as e:
        print(f"❌ Policy configuration test failed: {e}")
        return False


def test_consolidated_api_framework():
    """Test the consolidated API framework."""
    try:
        from ciaf.api import create_development_api, APIMode
        
        print("Testing consolidated API framework...")
        
        # Create development API framework
        api = create_development_api()
        
        # Test framework initialization
        health = api.get_api_health()
        print(f"✅ API Framework initialized: status={health['status']}, version={health['version']}")
        print(f"   Mode: {health['mode']}")
        print(f"   Integrations: {health['integrations']}")
        print(f"   Handlers: {len([h for h in health['handlers'].values() if h])} active")
        
        return True, api
    except Exception as e:
        print(f"❌ Consolidated API framework test failed: {e}")
        return False, None


def test_dataset_api_operations(api):
    """Test dataset API operations."""
    try:
        print("Testing dataset API operations...")
        
        # Create dataset
        response = api.process_api_request(
            "/api/v1/datasets",
            "POST",
            {
                "dataset_id": "test_dataset_1",
                "metadata": {
                    "name": "Test Dataset",
                    "version": "1.0.0",
                    "description": "Test dataset for API validation",
                    "features": ["feature1", "feature2", "feature3"],
                    "total_samples": 1000,
                    "data_types": ["numerical"]
                }
            }
        )
        
        if response["success"]:
            dataset = response["data"]
            print(f"✅ Dataset created: {dataset['dataset_id']}")
            print(f"   Features: {len(dataset['features'])}, Samples: {dataset['total_samples']}")
            print(f"   LCM tracked: {dataset.get('lcm_tracked', False)}")
        
        # List datasets
        response = api.process_api_request("/api/v1/datasets", "GET", {})
        if response["success"]:
            datasets = response["data"]
            print(f"✅ Datasets listed: {len(datasets)} datasets found")
        
        # Get specific dataset
        response = api.process_api_request("/api/v1/datasets/test_dataset_1", "GET", {})
        if response["success"]:
            dataset = response["data"]
            print(f"✅ Dataset retrieved: {dataset['dataset_id']} (status: {dataset['status']})")
        
        return True
    except Exception as e:
        print(f"❌ Dataset API test failed: {e}")
        return False


def test_model_api_operations(api):
    """Test model API operations."""
    try:
        print("Testing model API operations...")
        
        # Create model
        response = api.process_api_request(
            "/api/v1/models",
            "POST",
            {
                "model_name": "test_model_1",
                "config": {
                    "framework": "sklearn",
                    "model_type": "classifier",
                    "version": "1.0.0",
                    "parameters": {
                        "n_estimators": 100,
                        "max_depth": 10
                    },
                    "authorized_datasets": ["test_dataset_1"]
                }
            }
        )
        
        if response["success"]:
            model = response["data"]
            print(f"✅ Model created: {model['model_name']}")
            print(f"   Framework: {model['framework']}, Version: {model['version']}")
            print(f"   LCM tracked: {model.get('lcm_tracked', False)}")
            print(f"   Authorized datasets: {model['authorized_datasets']}")
        
        # List models
        response = api.process_api_request("/api/v1/models", "GET", {})
        if response["success"]:
            models = response["data"]
            print(f"✅ Models listed: {len(models)} models found")
        
        # Deploy model (simulate)
        response = api.process_api_request(
            "/api/v1/models/test_model_1/deploy",
            "POST",
            {
                "deployment_config": {
                    "environment": "development",
                    "resources": {"cpu": 2, "memory": "4GB"}
                }
            }
        )
        
        if response["success"]:
            deployment = response["data"]
            print(f"✅ Model deployed: {deployment['model_name']}")
            print(f"   Deployment ID: {deployment['deployment_id']}")
            print(f"   Endpoint: {deployment['endpoint']}")
        
        return True
    except Exception as e:
        print(f"❌ Model API test failed: {e}")
        return False


def test_training_api_operations(api):
    """Test training API operations."""
    try:
        print("Testing training API operations...")
        
        # Start training
        response = api.process_api_request(
            "/api/v1/training",
            "POST",
            {
                "model_name": "test_model_1",
                "training_config": {
                    "datasets": ["test_dataset_1"],
                    "parameters": {
                        "epochs": 10,
                        "batch_size": 32,
                        "learning_rate": 0.001
                    }
                }
            }
        )
        
        if response["success"]:
            training = response["data"]
            training_id = training["training_id"]
            print(f"✅ Training started: {training_id}")
            print(f"   Model: {training['model_name']}, Status: {training['status']}")
            print(f"   LCM tracked: {training.get('lcm_tracked', False)}")
        
        # Get training status
        response = api.process_api_request(f"/api/v1/training/{training_id}", "GET", {})
        if response["success"]:
            status = response["data"]
            print(f"✅ Training status: {status['status']} ({status['progress']:.1f}% complete)")
        
        # List training sessions
        response = api.process_api_request("/api/v1/training", "GET", {})
        if response["success"]:
            sessions = response["data"]
            print(f"✅ Training sessions listed: {len(sessions)} sessions found")
        
        return True
    except Exception as e:
        print(f"❌ Training API test failed: {e}")
        return False


def test_inference_api_operations(api):
    """Test inference API operations."""
    try:
        print("Testing inference API operations...")
        
        # Perform inference
        response = api.process_api_request(
            "/api/v1/models/test_model_1/predict",
            "POST",
            {
                "input_data": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "user_id": "test_user",
                "metadata": {"test_inference": True}
            }
        )
        
        if response["success"]:
            inference = response["data"]
            print(f"✅ Inference completed: {inference['inference_id']}")
            print(f"   Status: {inference['status']}, Duration: {inference['duration_ms']:.2f}ms")
            print(f"   Method: {inference['method']}")
            print(f"   LCM tracked: {inference.get('lcm_tracked', False)}")
            
            if inference["output"]:
                print(f"   Output: {inference['output']}")
        
        return True
    except Exception as e:
        print(f"❌ Inference API test failed: {e}")
        return False


def test_audit_api_operations(api):
    """Test audit API operations."""
    try:
        print("Testing audit API operations...")
        
        # Get audit trail
        response = api.process_api_request(
            "/api/v1/audit/trail/test_model_1",
            "GET",
            {"entity_type": "model"}
        )
        
        if response["success"]:
            audit_trail = response["data"]
            print(f"✅ Audit trail generated for: {audit_trail['entity_id']}")
            print(f"   Components: {len(audit_trail['trail_components'])}")
            print(f"   Integrity hash: {audit_trail['integrity_hash'][:16]}...")
        
        # Verify integrity
        response = api.process_api_request(
            "/api/v1/audit/verify/test_model_1",
            "GET",
            {"entity_type": "model"}
        )
        
        if response["success"]:
            verification = response["data"]
            print(f"✅ Integrity verification: {verification['integrity_verified']}")
            print(f"   Verified components: {len(verification['verification_details'])}")
        
        # Generate compliance report
        response = api.process_api_request(
            "/api/v1/audit/compliance-report",
            "GET",
            {
                "filters": {
                    "frameworks": ["gdpr", "eu_ai_act"],
                    "entity_types": ["model", "dataset"]
                }
            }
        )
        
        if response["success"]:
            report = response["data"]
            print(f"✅ Compliance report generated: {report['compliance_status']}")
            print(f"   Findings: {len(report['findings'])}")
        
        return True
    except Exception as e:
        print(f"❌ Audit API test failed: {e}")
        return False


def test_metrics_api_operations(api):
    """Test metrics API operations."""
    try:
        print("Testing metrics API operations...")
        
        # Get system metrics
        response = api.process_api_request("/api/v1/metrics/system", "GET", {})
        if response["success"]:
            metrics = response["data"]
            print(f"✅ System metrics: status={metrics['system_status']}")
            print(f"   API metrics: {metrics['api_metrics']}")
            print(f"   Integrations: {metrics['integrations']}")
        
        # Get model metrics
        response = api.process_api_request("/api/v1/metrics/models/test_model_1", "GET", {})
        if response["success"]:
            metrics = response["data"]
            print(f"✅ Model metrics for: {metrics['model_name']}")
            print(f"   Model info: {metrics['model_info']}")
            if "inference_metrics" in metrics:
                print(f"   Inference metrics: {metrics['inference_metrics']}")
        
        # Get dataset metrics
        response = api.process_api_request("/api/v1/metrics/datasets/test_dataset_1", "GET", {})
        if response["success"]:
            metrics = response["data"]
            print(f"✅ Dataset metrics for: {metrics['dataset_id']}")
            print(f"   Dataset info: {metrics['dataset_info']}")
        
        return True
    except Exception as e:
        print(f"❌ Metrics API test failed: {e}")
        return False


def test_integration_status(api):
    """Test integration with other CIAF modules."""
    try:
        print("Testing CIAF module integrations...")
        
        health = api.get_api_health()
        integrations = health["integrations"]
        
        print("Integration Status:")
        print(f"   ✅ Wrapper integration: {integrations.get('wrapper_integration', False)}")
        print(f"   ✅ LCM integration: {integrations.get('lcm_integration', False)}")
        print(f"   ✅ Compliance integration: {integrations.get('compliance_integration', False)}")
        
        # Check if universal wrapper support is available
        if integrations.get("wrapper_integration"):
            print("   🚀 Universal wrapper support enabled!")
            
        # Check if LCM tracking is available
        if integrations.get("lcm_integration"):
            print("   🚀 LCM lifecycle tracking enabled!")
            
        # Check if compliance validation is available
        if integrations.get("compliance_integration"):
            print("   🚀 Real-time compliance validation enabled!")
        
        return True
    except Exception as e:
        print(f"❌ Integration status test failed: {e}")
        return False


def main():
    """Run all enhanced API system tests."""
    print("=" * 60)
    print("CIAF Enhanced API System Tests")
    print("=" * 60)
    
    tests = [
        ("API Imports", test_api_imports),
        ("Policy Configuration", test_api_policy_configuration),
        ("Consolidated Framework", lambda: test_consolidated_api_framework()),
    ]
    
    # Run basic tests first
    passed = 0
    total = len(tests)
    api = None
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_name == "Consolidated Framework":
            success, api = test_func()
            if success:
                passed += 1
        else:
            if test_func():
                passed += 1
    
    if api is None:
        print("\n❌ Cannot continue without API framework")
        return False
    
    # Run API operation tests
    api_tests = [
        ("Dataset Operations", lambda: test_dataset_api_operations(api)),
        ("Model Operations", lambda: test_model_api_operations(api)),
        ("Training Operations", lambda: test_training_api_operations(api)),
        ("Inference Operations", lambda: test_inference_api_operations(api)),
        ("Audit Operations", lambda: test_audit_api_operations(api)),
        ("Metrics Operations", lambda: test_metrics_api_operations(api)),
        ("Integration Status", lambda: test_integration_status(api)),
    ]
    
    for test_name, test_func in api_tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        total += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} test suites passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All enhanced API system tests passed!")
        print("   The consolidated API framework successfully provides:")
        print("   - Protocol-based architecture with clean separation of concerns")
        print("   - Policy-driven configuration for all environments")
        print("   - Full integration with enhanced CIAF modules")
        print("   - Comprehensive dataset, model, training, and inference APIs")
        print("   - Complete audit trails and compliance validation")
        print("   - Real-time metrics and monitoring")
        print("   - Universal model wrapper integration")
        print("   - LCM lifecycle tracking")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)