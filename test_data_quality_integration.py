#!/usr/bin/env python3
"""
Test DataQualityValidator integration with CIAF framework
"""

import unittest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from ciaf.preprocessing import DataQualityValidator, ValidationResult, quick_validate, validate_ciaf_dataset
import numpy as np

class TestDataQualityValidator(unittest.TestCase):
    """Test cases for DataQualityValidator integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataQualityValidator()
        
        self.good_ciaf_data = [
            {"content": "Good training example", "metadata": {"target": 1}},
            {"content": "Another example", "metadata": {"target": 0}},
            {"content": "Third example", "metadata": {"target": 1}},
            {"content": "Fourth example", "metadata": {"target": 0}},
            {"content": "Fifth example", "metadata": {"target": 1}},
            {"content": "Sixth example", "metadata": {"target": 0}},
            {"content": "Seventh example", "metadata": {"target": 1}},
            {"content": "Eighth example", "metadata": {"target": 0}},
            {"content": "Ninth example", "metadata": {"target": 1}},
            {"content": "Tenth example", "metadata": {"target": 0}},
        ]

    def test_import_successful(self):
        """Test that DataQualityValidator can be imported."""
        from ciaf.preprocessing import DataQualityValidator
        validator = DataQualityValidator()
        self.assertIsInstance(validator, DataQualityValidator)

    def test_validation_result_structure(self):
        """Test ValidationResult structure."""
        result = ValidationResult()
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 0)
        
        result.add_error("Test error")
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)

    def test_good_data_validation(self):
        """Test validation of good quality data."""
        result = self.validator.validate(self.good_ciaf_data)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertIn("quality_score", result.metrics)
        self.assertGreaterEqual(result.metrics["quality_score"], 90)

    def test_problematic_data_detection(self):
        """Test detection of data quality issues."""
        # Create problematic dataset
        bad_data = [
            {"content": "Same content", "metadata": {"target": 1}},
            {"content": "Same content", "metadata": {"target": 1}},  # Duplicate
            {"content": "", "metadata": {"target": 0}},  # Empty
        ]
        
        result = self.validator.validate(bad_data)
        
        # Should detect insufficient samples and duplicates
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)

    def test_numerical_data_validation(self):
        """Test validation of numerical data."""
        numerical_data = np.array([
            [1, 2, 3],
            [4, 5, 6], 
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
            [28, 29, 30],
        ])
        
        result = self.validator.validate(numerical_data)
        self.assertIsInstance(result, ValidationResult)
        self.assertIn("quality_score", result.metrics)

    def test_quick_validate_function(self):
        """Test quick_validate convenience function."""
        result = quick_validate(self.good_ciaf_data)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)

    def test_ciaf_specific_validation(self):
        """Test CIAF-specific validation features."""
        result = validate_ciaf_dataset(self.good_ciaf_data)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)

    def test_quality_report_generation(self):
        """Test quality report generation."""
        report = self.validator.generate_quality_report(self.good_ciaf_data)
        
        self.assertIsInstance(report, str)
        self.assertIn("DATA QUALITY REPORT", report)
        self.assertIn("QUALITY METRICS", report)

    def test_schema_validation(self):
        """Test schema validation functionality."""
        result = self.validator.validate_schema(
            self.good_ciaf_data,
            expected_columns=["text_content", "target"],
        )
        
        self.assertIsInstance(result, ValidationResult)

    def test_custom_validator_settings(self):
        """Test validator with custom settings."""
        custom_validator = DataQualityValidator(
            min_samples=5,
            max_missing_ratio=0.5,
            check_duplicates=False
        )
        
        result = custom_validator.validate(self.good_ciaf_data[:6])
        self.assertIsInstance(result, ValidationResult)

    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        result = self.validator.validate([])
        
        self.assertFalse(result.is_valid)
        self.assertIn("empty", str(result).lower())

    def test_validation_with_missing_targets(self):
        """Test validation of data without targets."""
        no_target_data = [
            {"content": "Content without target"},
            {"content": "Another content"},
            {"content": "Third content"},
            {"content": "Fourth content"},
            {"content": "Fifth content"},
            {"content": "Sixth content"},
            {"content": "Seventh content"},
            {"content": "Eighth content"},
            {"content": "Ninth content"},
            {"content": "Tenth content"},
        ]
        
        result = validate_ciaf_dataset(no_target_data, require_targets=False)
        self.assertTrue(result.is_valid)
        
        result = validate_ciaf_dataset(no_target_data, require_targets=True)
        # Should have warning about missing targets
        self.assertGreater(len(result.warnings), 0)


def run_tests():
    """Run all tests."""
    print("🧪 Running DataQualityValidator Integration Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDataQualityValidator)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All {result.testsRun} tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)