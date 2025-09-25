"""
Pre-Ingestion Data Validation System

This module provides comprehensive data validation before train/test split
to catch bias, quality issues, and compliance violations early in the pipeline.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    issue_id: str
    severity: ValidationSeverity
    message: str
    column: Optional[str] = None
    affected_rows: Optional[int] = None
    recommendation: Optional[str] = None


@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis."""

    protected_attribute: str
    bias_detected: bool
    bias_score: float
    statistical_significance: float
    affected_groups: List[str]
    recommendation: str


class PreIngestionValidator:
    """
    Comprehensive pre-ingestion data validation system.

    Validates data for bias, quality, completeness, and compliance
    before any train/test splitting occurs.
    """

    def __init__(self, compliance_framework: str = "GDPR"):
        """
        Initialize the pre-ingestion validator.

        Args:
            compliance_framework: Primary compliance framework to validate against
        """
        self.compliance_framework = compliance_framework
        self.validation_issues: List[ValidationIssue] = []
        self.bias_results: List[BiasDetectionResult] = []

    def validate_dataset(
        self,
        data: pd.DataFrame,
        target_column: str,
        protected_attributes: List[str],
        sensitive_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive dataset validation before ingestion.

        Args:
            data: The complete dataset to validate
            target_column: Name of the target/label column
            protected_attributes: List of protected attribute columns (gender, race, etc.)
            sensitive_columns: List of sensitive data columns for privacy validation

        Returns:
            Validation report with issues, bias analysis, and recommendations
        """
        print(
            f"Starting pre-ingestion validation for dataset with {len(data)} samples..."
        )

        validation_start = datetime.now()

        # Clear previous results
        self.validation_issues.clear()
        self.bias_results.clear()

        # 1. Data Quality Validation
        self._validate_data_quality(data, target_column)

        # 2. Bias Detection
        self._detect_bias(data, target_column, protected_attributes)

        # 3. Privacy and Sensitivity Validation
        if sensitive_columns:
            self._validate_privacy_compliance(data, sensitive_columns)

        # 4. Statistical Validation
        self._validate_statistical_properties(data, target_column)

        # 5. Compliance Validation
        self._validate_compliance_requirements(data, protected_attributes)

        validation_duration = (datetime.now() - validation_start).total_seconds()

        # Generate comprehensive report
        report = self._generate_validation_report(data, validation_duration)

        # Print summary
        self._print_validation_summary(report)

        return report

    def _validate_data_quality(self, data: pd.DataFrame, target_column: str):
        """Validate basic data quality metrics."""
        print("Validating data quality...")

        # Check for missing values
        missing_counts = data.isnull().sum()
        total_rows = len(data)

        for column, missing_count in missing_counts.items():
            if missing_count > 0:
                missing_percentage = (missing_count / total_rows) * 100

                if missing_percentage > 50:
                    severity = ValidationSeverity.CRITICAL
                elif missing_percentage > 20:
                    severity = ValidationSeverity.ERROR
                elif missing_percentage > 5:
                    severity = ValidationSeverity.WARNING
                else:
                    severity = ValidationSeverity.INFO

                self.validation_issues.append(
                    ValidationIssue(
                        issue_id=f"missing_values_{column}",
                        severity=severity,
                        message=f"Column '{column}' has {missing_percentage:.1f}% missing values",
                        column=column,
                        affected_rows=missing_count,
                        recommendation="Consider imputation or removal of samples with missing values",
                    )
                )

        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / total_rows) * 100
            severity = (
                ValidationSeverity.WARNING
                if duplicate_percentage < 10
                else ValidationSeverity.ERROR
            )

            self.validation_issues.append(
                ValidationIssue(
                    issue_id="duplicate_rows",
                    severity=severity,
                    message=f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)",
                    affected_rows=duplicate_count,
                    recommendation="Remove duplicate rows before training",
                )
            )

        # Check target column distribution
        if target_column in data.columns:
            target_distribution = data[target_column].value_counts(normalize=True)
            min_class_percentage = target_distribution.min() * 100

            if min_class_percentage < 5:
                self.validation_issues.append(
                    ValidationIssue(
                        issue_id="class_imbalance",
                        severity=ValidationSeverity.WARNING,
                        message=f"Severe class imbalance detected. Smallest class: {min_class_percentage:.1f}%",
                        column=target_column,
                        recommendation="Consider class balancing techniques or stratified sampling",
                    )
                )

    def _detect_bias(
        self, data: pd.DataFrame, target_column: str, protected_attributes: List[str]
    ):
        """Detect potential bias in the dataset."""
        print("Detecting bias in protected attributes...")

        if target_column not in data.columns:
            self.validation_issues.append(
                ValidationIssue(
                    issue_id="missing_target",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Target column '{target_column}' not found in dataset",
                    recommendation="Ensure target column is present and correctly named",
                )
            )
            return

        for protected_attr in protected_attributes:
            if protected_attr not in data.columns:
                self.validation_issues.append(
                    ValidationIssue(
                        issue_id=f"missing_protected_attr_{protected_attr}",
                        severity=ValidationSeverity.ERROR,
                        message=f"Protected attribute '{protected_attr}' not found in dataset",
                        column=protected_attr,
                        recommendation="Ensure all protected attributes are present for bias analysis",
                    )
                )
                continue

            bias_result = self._analyze_bias_for_attribute(
                data, target_column, protected_attr
            )
            self.bias_results.append(bias_result)

            if bias_result.bias_detected:
                severity = (
                    ValidationSeverity.ERROR
                    if bias_result.bias_score > 0.3
                    else ValidationSeverity.WARNING
                )

                self.validation_issues.append(
                    ValidationIssue(
                        issue_id=f"bias_detected_{protected_attr}",
                        severity=severity,
                        message=f"Bias detected in '{protected_attr}' (score: {bias_result.bias_score:.3f})",
                        column=protected_attr,
                        recommendation=bias_result.recommendation,
                    )
                )

    def _analyze_bias_for_attribute(
        self, data: pd.DataFrame, target_column: str, protected_attr: str
    ) -> BiasDetectionResult:
        """Analyze bias for a specific protected attribute."""
        try:
            # Calculate outcome rates by group
            group_outcomes = data.groupby(protected_attr)[target_column].agg(
                ["mean", "count"]
            )

            # Calculate demographic parity (difference in positive outcome rates)
            outcome_rates = group_outcomes["mean"]
            max_rate = outcome_rates.max()
            min_rate = outcome_rates.min()
            bias_score = max_rate - min_rate

            # Determine if bias is statistically significant
            # Use chi-square test for independence
            from scipy.stats import chi2_contingency

            contingency_table = pd.crosstab(data[protected_attr], data[target_column])
            chi2, p_value, _, _ = chi2_contingency(contingency_table)

            bias_detected = (
                bias_score > 0.1 and p_value < 0.05
            )  # 10% difference threshold

            # Identify most and least advantaged groups
            sorted_groups = outcome_rates.sort_values(ascending=False)
            affected_groups = [
                f"{group}: {rate:.3f}" for group, rate in sorted_groups.items()
            ]

            # Generate recommendation
            if bias_detected:
                if bias_score > 0.3:
                    recommendation = "CRITICAL: Severe bias detected. Consider data rebalancing, algorithmic debiasing, or additional data collection."
                elif bias_score > 0.2:
                    recommendation = "Significant bias detected. Apply bias mitigation techniques during training."
                else:
                    recommendation = "Moderate bias detected. Monitor bias metrics during training and validation."
            else:
                recommendation = "No significant bias detected for this attribute."

            return BiasDetectionResult(
                protected_attribute=protected_attr,
                bias_detected=bias_detected,
                bias_score=bias_score,
                statistical_significance=p_value,
                affected_groups=affected_groups,
                recommendation=recommendation,
            )

        except Exception as e:
            # Fallback for when scipy is not available or other errors
            outcome_rates = data.groupby(protected_attr)[target_column].mean()
            bias_score = outcome_rates.max() - outcome_rates.min()
            bias_detected = bias_score > 0.1

            affected_groups = [
                f"{group}: {rate:.3f}" for group, rate in outcome_rates.items()
            ]

            return BiasDetectionResult(
                protected_attribute=protected_attr,
                bias_detected=bias_detected,
                bias_score=bias_score,
                statistical_significance=0.05,  # Conservative assumption
                affected_groups=affected_groups,
                recommendation="Bias analysis completed with limited statistical testing. Manual review recommended.",
            )

    def _validate_privacy_compliance(
        self, data: pd.DataFrame, sensitive_columns: List[str]
    ):
        """Validate privacy and sensitive data handling."""
        print("Validating privacy compliance...")

        for column in sensitive_columns:
            if column not in data.columns:
                continue

            # Check for potential PII exposure
            if data[column].dtype == "object":
                # Look for patterns that might be PII
                unique_ratio = data[column].nunique() / len(data)

                if unique_ratio > 0.95:  # Very high uniqueness could indicate PII
                    self.validation_issues.append(
                        ValidationIssue(
                            issue_id=f"potential_pii_{column}",
                            severity=ValidationSeverity.WARNING,
                            message=f"Column '{column}' has {unique_ratio:.1%} unique values (potential PII)",
                            column=column,
                            recommendation="Consider anonymization or pseudonymization of this column",
                        )
                    )

    def _validate_statistical_properties(self, data: pd.DataFrame, target_column: str):
        """Validate statistical properties of the dataset."""
        print("Validating statistical properties...")

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if column == target_column:
                continue

            col_data = data[column].dropna()

            # Check for extreme outliers
            if len(col_data) > 0:
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr

                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_percentage = len(outliers) / len(col_data) * 100

                if outlier_percentage > 5:
                    severity = (
                        ValidationSeverity.WARNING
                        if outlier_percentage < 15
                        else ValidationSeverity.ERROR
                    )

                    self.validation_issues.append(
                        ValidationIssue(
                            issue_id=f"outliers_{column}",
                            severity=severity,
                            message=f"Column '{column}' has {outlier_percentage:.1f}% extreme outliers",
                            column=column,
                            affected_rows=len(outliers),
                            recommendation="Consider outlier treatment or robust scaling methods",
                        )
                    )

    def _validate_compliance_requirements(
        self, data: pd.DataFrame, protected_attributes: List[str]
    ):
        """Validate compliance with regulatory requirements."""
        print(f"Validating {self.compliance_framework} compliance...")

        if self.compliance_framework == "GDPR":
            # Check for consent tracking
            consent_columns = [col for col in data.columns if "consent" in col.lower()]
            if not consent_columns:
                self.validation_issues.append(
                    ValidationIssue(
                        issue_id="missing_consent_tracking",
                        severity=ValidationSeverity.WARNING,
                        message="No consent tracking columns found",
                        recommendation="Consider adding consent tracking for GDPR compliance",
                    )
                )

        elif self.compliance_framework == "EEOC":
            # Ensure protected attributes are properly represented
            for attr in protected_attributes:
                if attr in data.columns:
                    group_sizes = data[attr].value_counts()
                    min_group_size = group_sizes.min()

                    if min_group_size < 30:  # Statistical significance threshold
                        self.validation_issues.append(
                            ValidationIssue(
                                issue_id=f"small_group_size_{attr}",
                                severity=ValidationSeverity.WARNING,
                                message=f"Small group size detected in '{attr}' (min: {min_group_size})",
                                column=attr,
                                recommendation="Ensure adequate representation of all protected groups",
                            )
                        )

    def _generate_validation_report(
        self, data: pd.DataFrame, validation_duration: float
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report."""

        # Categorize issues by severity
        issues_by_severity = {}
        for severity in ValidationSeverity:
            issues_by_severity[severity.value] = [
                issue for issue in self.validation_issues if issue.severity == severity
            ]

        # Calculate overall data quality score
        total_issues = len(self.validation_issues)
        critical_issues = len(issues_by_severity["critical"])
        error_issues = len(issues_by_severity["error"])
        warning_issues = len(issues_by_severity["warning"])

        # Quality score calculation (0-100)
        if critical_issues > 0:
            quality_score = max(0, 40 - critical_issues * 10)
        elif error_issues > 0:
            quality_score = max(40, 70 - error_issues * 5)
        elif warning_issues > 0:
            quality_score = max(70, 85 - warning_issues * 2)
        else:
            quality_score = 95 if total_issues == 0 else 85

        # Overall recommendation
        if quality_score < 50:
            overall_recommendation = (
                "Dataset requires significant cleanup before use"
            )
        elif quality_score < 70:
            overall_recommendation = (
                "Dataset needs attention - address critical issues before proceeding"
            )
        elif quality_score < 85:
            overall_recommendation = (
                "Dataset is acceptable with minor issues to address"
            )
        else:
            overall_recommendation = "Dataset passes validation - ready for use"

        return {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_duration_seconds": validation_duration,
            "dataset_summary": {
                "total_samples": len(data),
                "total_features": len(data.columns),
                "numeric_features": len(
                    data.select_dtypes(include=[np.number]).columns
                ),
                "categorical_features": len(
                    data.select_dtypes(include=["object"]).columns
                ),
            },
            "data_quality_score": quality_score,
            "overall_recommendation": overall_recommendation,
            "validation_issues": {
                "total_issues": total_issues,
                "by_severity": {
                    severity: len(issues)
                    for severity, issues in issues_by_severity.items()
                },
                "details": [
                    {
                        "issue_id": issue.issue_id,
                        "severity": issue.severity.value,
                        "message": issue.message,
                        "column": issue.column,
                        "affected_rows": issue.affected_rows,
                        "recommendation": issue.recommendation,
                    }
                    for issue in self.validation_issues
                ],
            },
            "bias_analysis": {
                "attributes_analyzed": len(self.bias_results),
                "bias_detected": any(
                    result.bias_detected for result in self.bias_results
                ),
                "results": [
                    {
                        "protected_attribute": result.protected_attribute,
                        "bias_detected": result.bias_detected,
                        "bias_score": result.bias_score,
                        "statistical_significance": result.statistical_significance,
                        "affected_groups": result.affected_groups,
                        "recommendation": result.recommendation,
                    }
                    for result in self.bias_results
                ],
            },
            "compliance_framework": self.compliance_framework,
            "ready_for_training": quality_score >= 70
            and not any(
                issue.severity == ValidationSeverity.CRITICAL
                for issue in self.validation_issues
            ),
        }

    def _print_validation_summary(self, report: Dict[str, Any]):
        """Print a human-readable validation summary."""
        print("\n" + "=" * 60)
        print("PRE-INGESTION VALIDATION SUMMARY")
        print("=" * 60)

        print(
            f"Dataset: {report['dataset_summary']['total_samples']:,} samples, "
            f"{report['dataset_summary']['total_features']} features"
        )
        print(f"Quality Score: {report['data_quality_score']}/100")
        print(f"Validation Duration: {report['validation_duration_seconds']:.2f}s")
        print(f"{report['overall_recommendation']}")

        # Issues summary
        issues = report["validation_issues"]
        if issues["total_issues"] > 0:
            print(f"\n Issues Found: {issues['total_issues']}")
            for severity, count in issues["by_severity"].items():
                if count > 0:
                    emoji = {
                        "critical": "critical",
                        "error": "error",
                        "warning": "warning",
                        "info": "info",
                    }
                    print(f"  {emoji.get(severity, '•')} {severity.title()}: {count}")
        else:
            print("\nNo issues found!")

        # Bias summary
        bias = report["bias_analysis"]
        if bias["attributes_analyzed"] > 0:
            print(
                f"\nBias Analysis: {bias['attributes_analyzed']} attributes analyzed"
            )
            if bias["bias_detected"]:
                biased_attrs = [
                    result["protected_attribute"]
                    for result in bias["results"]
                    if result["bias_detected"]
                ]
                print(f"  Bias detected in: {', '.join(biased_attrs)}")
            else:
                print("  No significant bias detected")

        print(
            f"\nReady for Training: {'Yes' if report['ready_for_training'] else 'No'}"
        )
        print("=" * 60)


def validate_before_split(
    data: pd.DataFrame,
    target_column: str,
    protected_attributes: List[str],
    compliance_framework: str = "GDPR",
    sensitive_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function for pre-ingestion validation.

    Args:
        data: Complete dataset to validate
        target_column: Name of the target/label column
        protected_attributes: List of protected attribute columns
        compliance_framework: Compliance framework to validate against
        sensitive_columns: List of sensitive data columns

    Returns:
        Validation report dictionary
    """
    validator = PreIngestionValidator(compliance_framework)
    return validator.validate_dataset(
        data=data,
        target_column=target_column,
        protected_attributes=protected_attributes,
        sensitive_columns=sensitive_columns,
    )


if __name__ == "__main__":
    # Example usage
    print("Pre-Ingestion Data Validation System")
    print(
        "Run this module with your dataset for comprehensive validation before train/test split"
    )
