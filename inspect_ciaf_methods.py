#!/usr/bin/env python3
"""
CIAF Package Method Inspector

This script provides comprehensive inspection of all methods available in the CIAF package.
"""

import inspect
import ciaf

def inspect_class_methods(cls, class_name):
    """Inspect methods of a specific class."""
    print(f"\n{class_name} Methods:")
    print("-" * (len(class_name) + 9))
    
    methods = []
    for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
        methods.append(f"  {name}()")
    
    # Also get functions that aren't bound methods
    for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith('_'):  # Skip private methods
            sig = inspect.signature(func)
            methods.append(f"  {name}{sig}")
    
    # Get properties
    for name, prop in inspect.getmembers(cls, predicate=lambda x: isinstance(x, property)):
        methods.append(f"  {name} (property)")
    
    if methods:
        for method in sorted(set(methods)):
            print(method)
    else:
        print("  No public methods found")

def inspect_module_functions(module, module_name):
    """Inspect functions in a module."""
    print(f"\n{module_name} Functions:")
    print("-" * (len(module_name) + 11))
    
    functions = []
    for name, func in inspect.getmembers(module, predicate=inspect.isfunction):
        if not name.startswith('_'):  # Skip private functions
            try:
                sig = inspect.signature(func)
                functions.append(f"  {name}{sig}")
            except (ValueError, TypeError):
                functions.append(f"  {name}()")
    
    if functions:
        for func in sorted(functions):
            print(func)
    else:
        print("  No public functions found")

def main():
    print("CIAF Package Method Inspector")
    print("=" * 50)
    print(f"CIAF Version: {ciaf.__version__}")
    
    # Inspect main classes
    key_classes = [
        ('CIAFFramework', ciaf.CIAFFramework),
        ('CIAFModelWrapper', ciaf.CIAFModelWrapper),
        ('EnhancedCIAFModelWrapper', ciaf.EnhancedCIAFModelWrapper),
        ('CryptoUtils', ciaf.CryptoUtils),
        ('MerkleTree', ciaf.MerkleTree),
        ('DatasetAnchor', ciaf.DatasetAnchor),
        ('ProvenanceCapsule', ciaf.ProvenanceCapsule),
        ('TrainingSnapshot', ciaf.TrainingSnapshot),
        ('InferenceReceipt', ciaf.InferenceReceipt),
        ('MetadataConfig', ciaf.MetadataConfig),
        ('MetadataStorage', ciaf.MetadataStorage),
    ]
    
    for class_name, cls in key_classes:
        if cls is not None:  # Handle optional components
            inspect_class_methods(cls, class_name)
    
    # Inspect deferred LCM classes if available
    if ciaf.DEFERRED_LCM_AVAILABLE:
        print("\nDeferred LCM Components:")
        print("=" * 25)
        deferred_classes = [
            ('LightweightReceipt', ciaf.LightweightReceipt),
            ('DeferredLCMProcessor', ciaf.DeferredLCMProcessor),
            ('AdaptiveLCMWrapper', ciaf.AdaptiveLCMWrapper),
        ]
        
        for class_name, cls in deferred_classes:
            if cls is not None:
                inspect_class_methods(cls, class_name)
    
    # Inspect module-level functions
    inspect_module_functions(ciaf, "CIAF Package")
    
    # Show submodule functions
    submodules = [
        ('ciaf.core', ciaf.core),
        ('ciaf.anchoring', ciaf.anchoring),
        ('ciaf.provenance', ciaf.provenance),
        ('ciaf.inference', ciaf.inference),
    ]
    
    for mod_name, module in submodules:
        inspect_module_functions(module, mod_name)
    
    print(f"\nFeature Availability:")
    print("=" * 20)
    print(f"Enhanced Wrapper: {ciaf.ENHANCED_WRAPPER_AVAILABLE}")
    print(f"Deferred LCM: {ciaf.DEFERRED_LCM_AVAILABLE}")
    print(f"Compliance: {ciaf.COMPLIANCE_AVAILABLE}")
    print(f"Enterprise Compliance: {ciaf.ENTERPRISE_COMPLIANCE_AVAILABLE}")
    print(f"Explainability: {ciaf.EXPLAINABILITY_AVAILABLE}")
    print(f"Uncertainty: {ciaf.UNCERTAINTY_AVAILABLE}")
    print(f"Preprocessing: {ciaf.PREPROCESSING_AVAILABLE}")
    print(f"Metadata Tags: {ciaf.METADATA_TAGS_AVAILABLE}")

if __name__ == "__main__":
    main()