"""
Test CIAF Vault Integration

Quick test to verify the vault client and examples are working.

Run: python test_vault_client.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all imports work."""
    print("🧪 Testing imports...")
    
    try:
        from ciaf import VaultClient, VAULT_CLIENT_AVAILABLE
        print("   ✅ VaultClient import successful")
        
        if not VAULT_CLIENT_AVAILABLE or VaultClient is None:
            print("   ⚠️  VaultClient not available (missing requests?)")
            return False
        
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False


def test_client_creation():
    """Test client creation."""
    print("\n🧪 Testing client creation...")
    
    try:
        from ciaf import VaultClient
        
        client = VaultClient("http://localhost:3000")
        print("   ✅ Client created successfully")
        print(f"   📍 Vault URL: {client.config.vault_url}")
        
        return True
    except Exception as e:
        print(f"   ❌ Client creation failed: {e}")
        return False


def test_methods():
    """Test client methods exist."""
    print("\n🧪 Testing client methods...")
    
    try:
        from ciaf import VaultClient
        
        client = VaultClient()
        
        methods = [
            'send_core_event',
            'send_inference_event',
            'send_training_event',
            'send_web_event',
            'register_agent',
            'execute_agent_action',
            'grant_elevation',
            'get_agent',
            'list_agents',
            'health_check'
        ]
        
        missing = []
        for method in methods:
            if not hasattr(client, method):
                missing.append(method)
        
        if missing:
            print(f"   ❌ Missing methods: {', '.join(missing)}")
            return False
        
        print(f"   ✅ All {len(methods)} methods present")
        return True
    except Exception as e:
        print(f"   ❌ Method check failed: {e}")
        return False


def test_connectivity():
    """Test vault connectivity (optional, may fail if vault not running)."""
    print("\n🧪 Testing vault connectivity...")
    
    try:
        from ciaf import VaultClient
        
        client = VaultClient("http://localhost:3000")
        
        if client.health_check():
            print("   ✅ Vault is accessible at http://localhost:3000")
            return True
        else:
            print("   ⚠️  Vault not accessible (this is OK if vault isn't running)")
            print("      Start vault with: cd ciaf_vault && npm run dev")
            return True  # Not a failure, just informational
    except Exception as e:
        print(f"   ⚠️  Connectivity test skipped: {e}")
        return True  # Not a failure


def test_example_files():
    """Test that example files exist."""
    print("\n🧪 Testing example files...")
    
    examples_dir = os.path.join(os.path.dirname(__file__))
    
    files = [
        'quick_start_vault.py',
        'vault_integration_complete.py',
        'VAULT_INTEGRATION_README.md'
    ]
    
    missing = []
    for file in files:
        path = os.path.join(examples_dir, file)
        if not os.path.exists(path):
            missing.append(file)
    
    if missing:
        print(f"   ❌ Missing files: {', '.join(missing)}")
        return False
    
    print(f"   ✅ All {len(files)} example files present")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CIAF VAULT CLIENT - TEST SUITE")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Client Creation", test_client_creation),
        ("Client Methods", test_methods),
        ("Example Files", test_example_files),
        ("Connectivity", test_connectivity),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to use CIAF Vault integration.")
        print("\nNext steps:")
        print("  1. Start vault: cd ciaf_vault && npm run dev")
        print("  2. Run demo:    python examples/quick_start_vault.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
