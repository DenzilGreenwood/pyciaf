"""
Data Quality Validation Module for CIAF

This module provides comprehensive data quality validation capabilities
for ML workflows, ensuring data integrity and identifying potential issues
before model training and inference.

Created: 2025-09-25
Author: CIAF Framework
Version: 1.0.0
"""

import json
import re
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd


class ValidationResult:
    """Container for validation results."""

    def __init__(self, is_valid: bool = True, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.metrics = {}

    def add_error(self, error: str):
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning."""
        self.warnings.append(warning)

    def add_metric(self, name: str, value: Any):
        """Add a quality metric."""
        self.metrics[name] = value

    def __str__(self):
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        result = f"Validation Result: {status}\n"
        
        if self.errors:
            result += f"Errors ({len(self.errors)}):\n"
            for error in self.errors:
                result += f"  - {error}\n"
        
        if self.warnings:
            result += f"Warnings ({len(self.warnings)}):\n"
            for warning in self.warnings:
                result += f"  - {warning}\n"
        
        if self.metrics:
            result += f"Quality Metrics:\n"
            for name, value in self.metrics.items():
                result += f"  - {name}: {value}\n"
        
        return result


class DataQualityValidator:
    """Comprehensive data quality validator for CIAF datasets."""

    def __init__(self, 
                 min_samples: int = 10,
                 max_missing_ratio: float = 0.3,
                 min_unique_ratio: float = 0.01,
                 check_duplicates: bool = True,
                 check_outliers: bool = True,
                 outlier_method: str = "iqr"):
        """
        Initialize data quality validator.

        Args:
            min_samples: Minimum number of samples required
            max_missing_ratio: Maximum allowed ratio of missing values
            min_unique_ratio: Minimum ratio of unique values for categorical data
            check_duplicates: Whether to check for duplicate entries
            check_outliers: Whether to check for statistical outliers
            outlier_method: Method for outlier detection ('iqr', 'zscore')
        """
        self.min_samples = min_samples
        self.max_missing_ratio = max_missing_ratio
        self.min_unique_ratio = min_unique_ratio
        self.check_duplicates = check_duplicates
        self.check_outliers = check_outliers
        self.outlier_method = outlier_method

    def validate(self, data: Union[List[Dict[str, Any]], pd.DataFrame, np.ndarray]) -> ValidationResult:
        """
        Perform comprehensive data quality validation.

        Args:
            data: Input data in CIAF format, DataFrame, or numpy array

        Returns:
            ValidationResult with detailed quality assessment
        """
        result = ValidationResult()

        try:
            # Convert data to standardized format
            df = self._prepare_data(data)
            
            # Basic structure validation
            self._validate_structure(df, result)
            
            # Content validation
            self._validate_content(df, result)
            
            # Statistical validation
            if self.check_outliers:
                self._validate_statistical_quality(df, result)
            
            # Duplicate validation
            if self.check_duplicates:
                self._validate_duplicates(df, result)
            
            # Target validation (for supervised learning)
            self._validate_targets(df, result)
            
            # Calculate overall quality score
            self._calculate_quality_score(df, result)

        except Exception as e:
            result.add_error(f"Validation failed with exception: {str(e)}")

        return result

    def _prepare_data(self, data: Union[List[Dict[str, Any]], pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input data to pandas DataFrame for analysis."""
        
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        elif isinstance(data, np.ndarray):
            # Convert numpy array to DataFrame with generic column names
            if data.ndim == 1:
                return pd.DataFrame({'feature_0': data})
            else:
                columns = [f'feature_{i}' for i in range(data.shape[1])]
                return pd.DataFrame(data, columns=columns)
        
        elif isinstance(data, list):
            # Handle empty list
            if not data:
                return pd.DataFrame()  # Return empty DataFrame
            
            if isinstance(data[0], dict):
                # CIAF format - extract content and metadata
                records = []
                for item in data:
                    record = {}
                    
                    # Extract content
                    content = item.get('content', item)
                    if isinstance(content, str):
                        try:
                            # Try to parse JSON content
                            parsed = json.loads(content)
                            if isinstance(parsed, dict):
                                record.update(parsed)
                            elif isinstance(parsed, list):
                                for i, val in enumerate(parsed):
                                    record[f'feature_{i}'] = val
                            else:
                                record['text_content'] = content
                        except:
                            record['text_content'] = content
                    elif isinstance(content, (list, tuple)):
                        for i, val in enumerate(content):
                            record[f'feature_{i}'] = val
                    else:
                        record['content'] = content
                    
                    # Extract metadata
                    metadata = item.get('metadata', {})
                    for key, value in metadata.items():
                        record[f'meta_{key}'] = value
                    
                    # Extract target if available
                    if 'target' in metadata:
                        record['target'] = metadata['target']
                    
                    records.append(record)
                
                return pd.DataFrame(records)
            else:
                # Simple list of values
                return pd.DataFrame({'values': data})
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _validate_structure(self, df: pd.DataFrame, result: ValidationResult):
        """Validate basic data structure."""
        
        # Check minimum sample size
        if len(df) < self.min_samples:
            result.add_error(f"Insufficient samples: {len(df)} < {self.min_samples}")
        else:
            result.add_metric("sample_count", len(df))
        
        # Check for empty DataFrame
        if df.empty:
            result.add_error("Dataset is empty")
            return
        
        # Check column count
        if len(df.columns) == 0:
            result.add_error("No features found in dataset")
        else:
            result.add_metric("feature_count", len(df.columns))
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            result.add_warning(f"Completely empty columns found: {empty_columns}")

    def _validate_content(self, df: pd.DataFrame, result: ValidationResult):
        """Validate data content quality."""
        
        missing_stats = {}
        
        for column in df.columns:
            col_data = df[column]
            
            # Missing value analysis
            missing_count = col_data.isnull().sum()
            missing_ratio = missing_count / len(df)
            missing_stats[column] = missing_ratio
            
            if missing_ratio > self.max_missing_ratio:
                result.add_error(f"Column '{column}' has {missing_ratio:.2%} missing values (>{self.max_missing_ratio:.1%})")
            elif missing_ratio > 0:
                result.add_warning(f"Column '{column}' has {missing_ratio:.2%} missing values")
            
            # Unique value analysis
            if not col_data.isnull().all():
                unique_count = col_data.nunique()
                unique_ratio = unique_count / len(df)
                
                # Check for constant columns
                if unique_count == 1:
                    result.add_warning(f"Column '{column}' has constant value")
                elif unique_ratio < self.min_unique_ratio:
                    result.add_warning(f"Column '{column}' has low diversity: {unique_ratio:.3%} unique values")
        
        result.add_metric("missing_value_ratios", missing_stats)
        result.add_metric("overall_missing_ratio", np.mean(list(missing_stats.values())))

    def _validate_statistical_quality(self, df: pd.DataFrame, result: ValidationResult):
        """Validate statistical properties and detect outliers."""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_stats = {}
        
        for column in numeric_columns:
            col_data = df[column].dropna()
            
            if len(col_data) < 3:
                continue
            
            outliers = []
            
            if self.outlier_method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            elif self.outlier_method == "zscore":
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outliers = col_data[z_scores > 3]
            
            outlier_ratio = len(outliers) / len(col_data)
            outlier_stats[column] = outlier_ratio
            
            if outlier_ratio > 0.1:  # More than 10% outliers
                result.add_warning(f"Column '{column}' has {outlier_ratio:.2%} potential outliers")
        
        if outlier_stats:
            result.add_metric("outlier_ratios", outlier_stats)

    def _validate_duplicates(self, df: pd.DataFrame, result: ValidationResult):
        """Check for duplicate records."""
        
        # Full duplicate check
        duplicate_count = df.duplicated().sum()
        duplicate_ratio = duplicate_count / len(df)
        
        if duplicate_count > 0:
            if duplicate_ratio > 0.05:  # More than 5% duplicates
                result.add_error(f"High duplicate ratio: {duplicate_ratio:.2%} ({duplicate_count} duplicates)")
            else:
                result.add_warning(f"Found {duplicate_count} duplicate records ({duplicate_ratio:.2%})")
        
        result.add_metric("duplicate_count", duplicate_count)
        result.add_metric("duplicate_ratio", duplicate_ratio)

    def _validate_targets(self, df: pd.DataFrame, result: ValidationResult):
        """Validate target variables for supervised learning."""
        
        target_columns = [col for col in df.columns if 'target' in col.lower() or col == 'y']
        
        for target_col in target_columns:
            if target_col not in df.columns:
                continue
            
            target_data = df[target_col].dropna()
            
            if len(target_data) == 0:
                result.add_error(f"Target column '{target_col}' is completely empty")
                continue
            
            # Check target distribution
            if target_data.dtype in [np.number]:
                # Regression target
                result.add_metric(f"{target_col}_mean", target_data.mean())
                result.add_metric(f"{target_col}_std", target_data.std())
            else:
                # Classification target
                value_counts = target_data.value_counts()
                class_balance = value_counts.min() / value_counts.max()
                
                result.add_metric(f"{target_col}_classes", len(value_counts))
                result.add_metric(f"{target_col}_balance_ratio", class_balance)
                
                if class_balance < 0.1:  # Highly imbalanced
                    result.add_warning(f"Target '{target_col}' is highly imbalanced (ratio: {class_balance:.3f})")

    def _calculate_quality_score(self, df: pd.DataFrame, result: ValidationResult):
        """Calculate an overall data quality score (0-100)."""
        
        score = 100.0
        
        # Penalize missing values
        missing_ratio = result.metrics.get("overall_missing_ratio", 0)
        score -= missing_ratio * 30
        
        # Penalize duplicates
        duplicate_ratio = result.metrics.get("duplicate_ratio", 0)
        score -= duplicate_ratio * 20
        
        # Penalize errors more heavily
        score -= len(result.errors) * 15
        
        # Small penalty for warnings
        score -= len(result.warnings) * 5
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        result.add_metric("quality_score", round(score, 2))

    def validate_schema(self, data: Union[List[Dict[str, Any]], pd.DataFrame], 
                       expected_columns: List[str] = None,
                       column_types: Dict[str, type] = None) -> ValidationResult:
        """
        Validate data schema against expected structure.

        Args:
            data: Input data
            expected_columns: List of expected column names
            column_types: Expected data types for columns

        Returns:
            ValidationResult for schema validation
        """
        result = ValidationResult()
        
        try:
            df = self._prepare_data(data)
            
            if expected_columns:
                missing_columns = set(expected_columns) - set(df.columns)
                extra_columns = set(df.columns) - set(expected_columns)
                
                if missing_columns:
                    result.add_error(f"Missing expected columns: {list(missing_columns)}")
                
                if extra_columns:
                    result.add_warning(f"Unexpected columns found: {list(extra_columns)}")
            
            if column_types:
                for column, expected_type in column_types.items():
                    if column in df.columns:
                        actual_type = df[column].dtype
                        # Simple type compatibility check
                        compatible = (
                            (expected_type in [int, float] and np.issubdtype(actual_type, np.number)) or
                            (expected_type == str and actual_type == object) or
                            str(actual_type) == str(expected_type)
                        )
                        
                        if not compatible:
                            result.add_warning(f"Column '{column}' type mismatch: expected {expected_type}, got {actual_type}")
            
        except Exception as e:
            result.add_error(f"Schema validation failed: {str(e)}")
        
        return result

    def generate_quality_report(self, data: Union[List[Dict[str, Any]], pd.DataFrame]) -> str:
        """Generate a comprehensive data quality report."""
        
        validation_result = self.validate(data)
        
        report = "="*60 + "\n"
        report += "DATA QUALITY REPORT\n"
        report += "="*60 + "\n\n"
        
        # Overall status
        status = "✅ PASSED" if validation_result.is_valid else "❌ FAILED"
        report += f"Overall Status: {status}\n\n"
        
        # Quality metrics
        if validation_result.metrics:
            report += "QUALITY METRICS:\n"
            report += "-"*30 + "\n"
            for metric, value in validation_result.metrics.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        report += f"{metric}: {value:.4f}\n"
                    else:
                        report += f"{metric}: {value}\n"
                else:
                    report += f"{metric}: {value}\n"
            report += "\n"
        
        # Errors
        if validation_result.errors:
            report += "ERRORS:\n"
            report += "-"*30 + "\n"
            for i, error in enumerate(validation_result.errors, 1):
                report += f"{i}. {error}\n"
            report += "\n"
        
        # Warnings
        if validation_result.warnings:
            report += "WARNINGS:\n"
            report += "-"*30 + "\n"
            for i, warning in enumerate(validation_result.warnings, 1):
                report += f"{i}. {warning}\n"
            report += "\n"
        
        # Recommendations
        report += "RECOMMENDATIONS:\n"
        report += "-"*30 + "\n"
        
        if validation_result.errors:
            report += "• Address all errors before proceeding with model training\n"
        
        if validation_result.warnings:
            report += "• Review warnings and consider data preprocessing steps\n"
        
        quality_score = validation_result.metrics.get("quality_score", 0)
        if quality_score < 70:
            report += "• Data quality score is low - consider data cleaning\n"
        elif quality_score < 85:
            report += "• Data quality is moderate - some preprocessing recommended\n"
        else:
            report += "• Data quality is good - minimal preprocessing needed\n"
        
        report += "\n" + "="*60
        
        return report


# Convenience functions
def quick_validate(data: Union[List[Dict[str, Any]], pd.DataFrame, np.ndarray]) -> ValidationResult:
    """Perform quick data quality validation with default settings."""
    validator = DataQualityValidator()
    return validator.validate(data)


def validate_ciaf_dataset(data: List[Dict[str, Any]], 
                         min_samples: int = 10,
                         require_targets: bool = True) -> ValidationResult:
    """Validate CIAF format dataset with appropriate defaults."""
    validator = DataQualityValidator(min_samples=min_samples)
    result = validator.validate(data)
    
    # Additional CIAF-specific validation
    if require_targets:
        has_targets = any('metadata' in item and 'target' in item['metadata'] for item in data[:5])
        if not has_targets:
            result.add_warning("No target labels found in metadata - unsupervised learning assumed")
    
    return result