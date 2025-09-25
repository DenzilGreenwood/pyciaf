#!/usr/bin/env python3
"""
Test script to demonstrate DataQualityValidator functionality
"""

import numpy as np
import pandas as pd
from ciaf.preprocessing import DataQualityValidator, quick_validate, validate_ciaf_dataset

def test_basic_validation():
    """Test basic data validation functionality."""
    print("🧪 Testing DataQualityValidator...")
    
    # Create test data in CIAF format
    good_data = [
        {"content": "This is a positive example", "metadata": {"target": 1}},
        {"content": "This is a negative example", "metadata": {"target": 0}},
        {"content": "Another positive case", "metadata": {"target": 1}},
        {"content": "Another negative case", "metadata": {"target": 0}},
        {"content": "Mixed example here", "metadata": {"target": 1}},
        {"content": "Final test case", "metadata": {"target": 0}},
        {"content": "More data for validation", "metadata": {"target": 1}},
        {"content": "Quality check example", "metadata": {"target": 0}},
        {"content": "Robust testing data", "metadata": {"target": 1}},
        {"content": "Comprehensive validation", "metadata": {"target": 0}},
    ]
    
    # Test with good data
    print("\n📊 Validating good quality data:")
    result = quick_validate(good_data)
    print(result)
    
    # Create problematic data
    problematic_data = [
        {"content": "Same text", "metadata": {"target": 1}},
        {"content": "Same text", "metadata": {"target": 1}},  # Duplicate
        {"content": "", "metadata": {"target": 0}},  # Empty content
        {"content": "Only example", "metadata": {"target": 1}},
    ]
    
    print("\n⚠️  Validating problematic data:")
    result = quick_validate(problematic_data)
    print(result)
    
    # Test numerical data
    print("\n🔢 Validating numerical data:")
    numerical_data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [1000, 2000, 3000],  # Outlier
    ])
    
    result = quick_validate(numerical_data)
    print(result)

def test_comprehensive_validation():
    """Test comprehensive validation features."""
    print("\n🔍 Testing comprehensive validation...")
    
    # Create mixed quality dataset
    mixed_data = [
        {"content": [1.0, 2.0, 3.0], "metadata": {"target": "class_a"}},
        {"content": [2.0, 3.0, 4.0], "metadata": {"target": "class_b"}},
        {"content": [1.0, 2.0, 3.0], "metadata": {"target": "class_a"}},  # Duplicate
        {"content": [3.0, None, 5.0], "metadata": {"target": "class_a"}},  # Missing value
        {"content": [4.0, 5.0, 6.0], "metadata": {"target": "class_b"}},
        {"content": [100.0, 200.0, 300.0], "metadata": {"target": "class_a"}},  # Outlier
        {"content": [6.0, 7.0, 8.0], "metadata": {"target": "class_b"}},
        {"content": [7.0, 8.0, 9.0], "metadata": {"target": "class_a"}},
        {"content": [8.0, 9.0, 10.0], "metadata": {"target": "class_b"}},
        {"content": [9.0, 10.0, 11.0], "metadata": {"target": "class_a"}},
    ]
    
    # Create validator with custom settings
    validator = DataQualityValidator(
        min_samples=8,
        max_missing_ratio=0.2,
        check_duplicates=True,
        check_outliers=True
    )
    
    result = validator.validate(mixed_data)
    print(result)
    
    # Generate detailed report
    print("\n📋 Generating quality report...")
    report = validator.generate_quality_report(mixed_data)
    print(report)

def test_ciaf_specific_validation():
    """Test CIAF-specific validation features."""
    print("\n🎯 Testing CIAF-specific validation...")
    
    # Test with CIAF dataset
    ciaf_data = [
        {"content": "Good training example", "metadata": {"target": 1, "source": "dataset_v1"}},
        {"content": "Another example", "metadata": {"target": 0, "source": "dataset_v1"}},
        {"content": "Third example", "metadata": {"target": 1, "source": "dataset_v2"}},
        {"content": "Fourth example", "metadata": {"target": 0, "source": "dataset_v1"}},
        {"content": "Fifth example", "metadata": {"target": 1, "source": "dataset_v2"}},
        {"content": "Sixth example", "metadata": {"target": 0, "source": "dataset_v1"}},
        {"content": "Seventh example", "metadata": {"target": 1, "source": "dataset_v2"}},
        {"content": "Eighth example", "metadata": {"target": 0, "source": "dataset_v1"}},
        {"content": "Ninth example", "metadata": {"target": 1, "source": "dataset_v2"}},
        {"content": "Tenth example", "metadata": {"target": 0, "source": "dataset_v1"}},
    ]
    
    result = validate_ciaf_dataset(ciaf_data, min_samples=8, require_targets=True)
    print(result)

if __name__ == "__main__":
    print("🚀 DataQualityValidator Test Suite")
    print("=" * 50)
    
    test_basic_validation()
    test_comprehensive_validation()
    test_ciaf_specific_validation()
    
    print("\n✅ All tests completed!")