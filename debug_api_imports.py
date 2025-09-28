#!/usr/bin/env python3
"""Debug API imports"""

print("Testing individual imports...")

try:
    from ciaf.api.policy import APIPolicy
    print("✅ Policy import OK")
except Exception as e:
    print(f"❌ Policy import failed: {e}")

try:
    from ciaf.api.interfaces import DatasetAPIHandler
    print("✅ Interfaces import OK")
except Exception as e:
    print(f"❌ Interfaces import failed: {e}")

try:
    from ciaf.api.protocol_implementations import DefaultDatasetAPIHandler
    print("✅ Protocol implementations import OK")
except Exception as e:
    print(f"❌ Protocol implementations import failed: {e}")

try:
    from ciaf.api.consolidated_api import ConsolidatedCIAFAPIFramework
    print("✅ Consolidated API import OK")
except Exception as e:
    print(f"❌ Consolidated API import failed: {e}")

print("\nTesting full API module import...")
try:
    from ciaf.api import CONSOLIDATED_API_AVAILABLE, PROTOCOL_IMPLEMENTATIONS_AVAILABLE
    print(f"Protocol implementations available: {PROTOCOL_IMPLEMENTATIONS_AVAILABLE}")
    print(f"Consolidated API available: {CONSOLIDATED_API_AVAILABLE}")
except Exception as e:
    print(f"❌ Full API import failed: {e}")