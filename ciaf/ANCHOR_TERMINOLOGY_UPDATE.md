# CIAF Anchor Terminology Update Summary

## Overview
This document summarizes the comprehensive update from "key" terminology to "anchor" terminology throughout the CIAF codebase, completed on September 9, 2025.

## Core Architecture Changes

### New Anchor System (`core/base_anchor.py`)
- **`derive_master_anchor()`**: Root cryptographic foundation for the entire system
- **`derive_dataset_anchor()`**: Dataset-specific anchor derived from master anchor
- **`derive_capsule_anchor()`**: Individual capsule anchor for fine-grained security
- **`derive_model_anchor()`**: Model-specific anchor for training authorization
- **`BaseAnchorManager`**: Central management class for hierarchical anchor operations

### Updated Terminology Mapping
| Old Term | New Term | Purpose |
|----------|----------|---------|
| `master_key` | `master_anchor` | Root cryptographic foundation |
| `dataset_key` | `dataset_anchor` | Dataset-specific derivation |
| `capsule_key` | `capsule_anchor` | Individual capsule security |
| `model_key` | `model_anchor` | Model training authorization |
| Key Management | Anchor Management | Security architecture |
| Key Rotation | Anchor Rotation | Lifecycle management |
| Key Derivation | Anchor Derivation | Cryptographic foundation |

## Files Updated

### Core Modules
- ✅ **`core/base_anchor.py`**: New anchor management system (created)
- ✅ **`core/__init__.py`**: Updated imports with backwards compatibility
- ✅ **`core/crypto.py`**: Updated documentation to reference anchor derivation

### Anchoring System
- ✅ **`anchoring/dataset_anchor.py`**: Updated docstrings and comments
- ✅ **`anchoring/true_lazy_manager.py`**: Updated all references to anchor terminology
- ✅ **`anchoring/simple_lazy_manager.py`**: Updated comments and documentation

### API Framework
- ✅ **`api/framework.py`**: Updated to use BaseAnchorManager and anchor terminology
- ✅ **`api/framework_new.py`**: Updated docstrings for anchor derivation

### Provenance System
- ✅ **`provenance/snapshots.py`**: Updated docstrings for anchor derivation
- ✅ **`provenance/capsules.py`**: Updated documentation for anchor-based operations

### Wrappers
- ✅ **`wrappers/model_wrapper.py`**: Updated training documentation

### Compliance
- ✅ **`compliance/cybersecurity.py`**: Updated security controls to reference anchor management

### Documentation
- ✅ **`README.md`**: Comprehensive update to anchor terminology
- ✅ **`LICENSE`**: Updated compliance notices to reference anchor management

## Backwards Compatibility

### Legacy Function Aliases
All legacy function names are preserved as aliases in `core/base_anchor.py`:
```python
# Backwards compatibility aliases
derive_key = derive_anchor
derive_master_key = derive_master_anchor
derive_dataset_key = derive_dataset_anchor
derive_capsule_key = derive_capsule_anchor
AnchorManager = BaseAnchorManager
```

### Legacy Method Names
Dataset anchor methods maintain old names for compatibility:
```python
# Legacy method aliases in DatasetAnchor class
def derive_item_key(self, item_id: str) -> str:
    """Legacy alias for derive_item_anchor"""
    return self.derive_item_anchor(item_id)

def derive_capsule_key(self, capsule_id: str) -> str:
    """Legacy alias for derive_capsule_anchor"""
    return self.derive_capsule_anchor(capsule_id)
```

## Security Improvements

### Binary-First Operations
- Anchors are derived as binary data, not hex strings
- More secure than previous hex-based approach
- Includes utility functions for hex conversion when needed

### Enhanced Documentation
- All docstrings now use consistent anchor terminology
- Clear distinction between different anchor types
- Better explanation of hierarchical derivation system

## Migration Guide

### For New Code
Use the new anchor terminology:
```python
from ciaf.core import BaseAnchorManager, derive_master_anchor

# Create anchor manager
anchor_manager = BaseAnchorManager()

# Derive anchors
master_anchor = derive_master_anchor("password", salt)
dataset_anchor = derive_dataset_anchor(master_anchor, "dataset_id")
```

### For Existing Code
No changes required - all legacy function names continue to work:
```python
# New preferred way:
from ciaf.core import AnchorManager, derive_master_anchor

# Legacy code continues to function:
from ciaf.core import KeyManager, derive_master_key  # Still works

# Legacy code continues to function
key_manager = KeyManager()  # Actually creates AnchorManager instance
master_key = derive_master_key("password", salt)
```

## Documentation Philosophy

The anchor terminology better reflects the cryptographic architecture:
- **Anchors** provide cryptographic foundations, not just encryption keys
- **Hierarchical derivation** creates a tree of security contexts
- **Dataset anchors** establish security boundaries for data collections
- **Model anchors** authorize specific training operations
- **Capsule anchors** provide fine-grained data protection

This terminology makes the security model more intuitive and aligns with the framework's goal of providing verifiable AI training pipelines.

## Testing Status
- ✅ All backwards compatibility aliases tested
- ✅ Core anchor derivation functions tested
- ✅ No breaking changes introduced
- ✅ Documentation updated consistently

## Future Considerations
- Consider deprecation warnings for legacy terminology in v3.0
- Enhanced anchor rotation automation
- Integration with hardware security modules (HSMs)
- Advanced anchor policies for multi-tenant environments
