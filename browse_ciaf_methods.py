#!/usr/bin/env python3
"""
Simple CIAF Method Browser

Run this script to interactively explore CIAF methods.
"""

import ciaf
import inspect

def show_class_methods(cls, class_name):
    """Show methods for a specific class."""
    print(f"\n{class_name} ({cls.__module__}):")
    print("=" * (len(class_name) + len(cls.__module__) + 4))
    
    # Get all methods and functions
    methods = []
    for name in dir(cls):
        if not name.startswith('_'):  # Skip private methods
            attr = getattr(cls, name)
            if callable(attr):
                try:
                    if inspect.ismethod(attr) or inspect.isfunction(attr):
                        sig = inspect.signature(attr)
                        methods.append(f"  {name}{sig}")
                    else:
                        methods.append(f"  {name}()")
                except (ValueError, TypeError):
                    methods.append(f"  {name}()")
            elif isinstance(attr, property):
                methods.append(f"  {name} (property)")
    
    for method in sorted(methods):
        print(method)

def main():
    print("CIAF Method Browser")
    print("=" * 20)
    print(f"CIAF Version: {ciaf.__version__}")
    
    # Define key classes to explore
    classes = [
        ("CIAFFramework", ciaf.CIAFFramework),
        ("CIAFModelWrapper", ciaf.CIAFModelWrapper),
        ("EnhancedCIAFModelWrapper", ciaf.EnhancedCIAFModelWrapper),
        ("CryptoUtils", ciaf.CryptoUtils),
        ("MerkleTree", ciaf.MerkleTree),
        ("DatasetAnchor", ciaf.DatasetAnchor),
        ("ProvenanceCapsule", ciaf.ProvenanceCapsule),
        ("TrainingSnapshot", ciaf.TrainingSnapshot),
        ("InferenceReceipt", ciaf.InferenceReceipt),
        ("MetadataConfig", ciaf.MetadataConfig),
    ]
    
    # Show methods for each class
    for class_name, cls in classes:
        if cls is not None:
            show_class_methods(cls, class_name)
    
    # Show available package-level functions
    print(f"\nPackage-Level Functions:")
    print("=" * 25)
    functions = []
    for name in dir(ciaf):
        if not name.startswith('_') and not name.isupper():  # Skip private and constants
            attr = getattr(ciaf, name)
            if callable(attr) and not inspect.isclass(attr):
                try:
                    sig = inspect.signature(attr)
                    functions.append(f"  ciaf.{name}{sig}")
                except (ValueError, TypeError):
                    functions.append(f"  ciaf.{name}()")
    
    for func in sorted(functions):
        print(func)
    
    print(f"\nTo get detailed help on any method:")
    print(f"  python -c \"import ciaf; help(ciaf.ClassName.method_name)\"")
    print(f"  python -c \"import ciaf; help(ciaf.function_name)\"")
    
    print(f"\nExample usage:")
    print(f"  python -c \"import ciaf; help(ciaf.CIAFFramework.create_dataset_anchor)\"")

if __name__ == "__main__":
    main()