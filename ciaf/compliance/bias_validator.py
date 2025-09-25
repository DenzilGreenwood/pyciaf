"""
Bias Detection and Validation Module

This module provides bias detection capabilities to support CIAF's
compliance with fairness and non-discrimination requirements in
EU AI Act Articles 10 and 15, and NIST AI RMF MEASURE functions.

Created: 2025-09-12
Author: Denzil James Greenwood
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class BiasMetric(Enum):
    """Supported bias metrics for fairness assessment."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"

@dataclass
class BiasResult:
    """Result of bias assessment for a protected group."""
    protected_attribute: str
    group_value: str
    metric_name: str
    metric_value: float
    threshold: float
    is_fair: bool
    sample_size: int
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class BiasAssessment:
    """Comprehensive bias assessment results."""
    overall_fairness_score: float
    individual_results: List[BiasResult]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]
    compliance_status: str

class BiasValidator:
    """
    Comprehensive bias validation for AI systems.
    
    Supports EU AI Act Article 10 (Data Governance) and Article 15 
    (Accuracy, Robustness, Cybersecurity) requirements for bias mitigation.
    """
    
    def __init__(self, fairness_threshold: float = 0.8):
        """
        Initialize bias validator.
        
        Args:
            fairness_threshold: Minimum acceptable fairness score (0-1)
        """
        self.fairness_threshold = fairness_threshold
        self.logger = logging.getLogger(__name__)
    
    def validate_predictions(
        self,
        predictions: np.ndarray,
        protected_attributes: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
        metrics: List[BiasMetric] = None
    ) -> BiasAssessment:
        """
        Validate predictions for bias across protected attributes.
        
        Args:
            predictions: Model predictions (probabilities or classes)
            protected_attributes: Dict mapping attribute names to values
            labels: True labels (required for some metrics)
            metrics: List of bias metrics to compute
            
        Returns:
            BiasAssessment with detailed results
        """
        if metrics is None:
            metrics = [BiasMetric.DEMOGRAPHIC_PARITY]
            if labels is not None:
                metrics.extend([BiasMetric.EQUALIZED_ODDS, BiasMetric.EQUAL_OPPORTUNITY])
        
        individual_results = []
        
        for attribute_name, attribute_values in protected_attributes.items():
            for metric in metrics:
                results = self._compute_bias_metric(
                    predictions=predictions,
                    protected_attribute=attribute_values,
                    attribute_name=attribute_name,
                    labels=labels,
                    metric=metric
                )
                individual_results.extend(results)
        
        # Calculate overall fairness score
        fairness_scores = [r.metric_value for r in individual_results if r.metric_value is not None]
        overall_fairness = np.mean(fairness_scores) if fairness_scores else 0.0
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(individual_results, protected_attributes)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(individual_results, overall_fairness)
        
        # Determine compliance status
        compliance_status = self._assess_compliance_status(overall_fairness, individual_results)
        
        return BiasAssessment(
            overall_fairness_score=overall_fairness,
            individual_results=individual_results,
            summary_statistics=summary_stats,
            recommendations=recommendations,
            compliance_status=compliance_status
        )
    
    def _compute_bias_metric(
        self,
        predictions: np.ndarray,
        protected_attribute: np.ndarray,
        attribute_name: str,
        labels: Optional[np.ndarray],
        metric: BiasMetric
    ) -> List[BiasResult]:
        """Compute specific bias metric for each group in protected attribute."""
        results = []
        unique_groups = np.unique(protected_attribute)
        
        for group in unique_groups:
            group_mask = protected_attribute == group
            group_predictions = predictions[group_mask]
            group_labels = labels[group_mask] if labels is not None else None
            
            try:
                if metric == BiasMetric.DEMOGRAPHIC_PARITY:
                    metric_value = self._demographic_parity(group_predictions)
                elif metric == BiasMetric.EQUALIZED_ODDS and group_labels is not None:
                    metric_value = self._equalized_odds(group_predictions, group_labels)
                elif metric == BiasMetric.EQUAL_OPPORTUNITY and group_labels is not None:
                    metric_value = self._equal_opportunity(group_predictions, group_labels)
                elif metric == BiasMetric.CALIBRATION and group_labels is not None:
                    metric_value = self._calibration(group_predictions, group_labels)
                else:
                    metric_value = None
                
                if metric_value is not None:
                    is_fair = metric_value >= self.fairness_threshold
                    
                    results.append(BiasResult(
                        protected_attribute=attribute_name,
                        group_value=str(group),
                        metric_name=metric.value,
                        metric_value=metric_value,
                        threshold=self.fairness_threshold,
                        is_fair=is_fair,
                        sample_size=len(group_predictions)
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Failed to compute {metric.value} for {attribute_name}={group}: {e}")
        
        return results
    
    def _demographic_parity(self, predictions: np.ndarray) -> float:
        """
        Compute demographic parity (statistical parity).
        
        Returns the positive prediction rate for the group.
        """
        if len(predictions) == 0:
            return 0.0
        
        # Convert probabilities to binary predictions if needed
        if predictions.dtype == float and np.all((predictions >= 0) & (predictions <= 1)):
            binary_predictions = (predictions > 0.5).astype(int)
        else:
            binary_predictions = predictions
        
        return np.mean(binary_predictions)
    
    def _equalized_odds(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute equalized odds metric.
        
        Returns the average of TPR and TNR for the group.
        """
        if len(predictions) == 0 or len(labels) == 0:
            return 0.0
        
        # Convert probabilities to binary predictions if needed
        if predictions.dtype == float and np.all((predictions >= 0) & (predictions <= 1)):
            binary_predictions = (predictions > 0.5).astype(int)
        else:
            binary_predictions = predictions
        
        # True Positive Rate
        positive_mask = labels == 1
        if np.sum(positive_mask) > 0:
            tpr = np.mean(binary_predictions[positive_mask])
        else:
            tpr = 0.0
        
        # True Negative Rate
        negative_mask = labels == 0
        if np.sum(negative_mask) > 0:
            tnr = np.mean(1 - binary_predictions[negative_mask])
        else:
            tnr = 0.0
        
        return (tpr + tnr) / 2
    
    def _equal_opportunity(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute equal opportunity metric (TPR).
        
        Returns the True Positive Rate for the group.
        """
        if len(predictions) == 0 or len(labels) == 0:
            return 0.0
        
        # Convert probabilities to binary predictions if needed
        if predictions.dtype == float and np.all((predictions >= 0) & (predictions <= 1)):
            binary_predictions = (predictions > 0.5).astype(int)
        else:
            binary_predictions = predictions
        
        positive_mask = labels == 1
        if np.sum(positive_mask) > 0:
            return np.mean(binary_predictions[positive_mask])
        else:
            return 0.0
    
    def _calibration(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute calibration metric.
        
        Returns how well predicted probabilities match actual outcomes.
        """
        if len(predictions) == 0 or len(labels) == 0:
            return 0.0
        
        # Only meaningful for probability predictions
        if not (predictions.dtype == float and np.all((predictions >= 0) & (predictions <= 1))):
            return 0.0
        
        # Bin predictions and compute calibration error
        n_bins = min(10, len(predictions) // 10)
        if n_bins < 2:
            return 0.0
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return 1.0 - calibration_error  # Convert error to score
    
    def _generate_summary_statistics(
        self,
        results: List[BiasResult],
        protected_attributes: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Generate summary statistics for bias assessment."""
        stats = {
            "total_groups_assessed": len(results),
            "fair_groups": len([r for r in results if r.is_fair]),
            "unfair_groups": len([r for r in results if not r.is_fair]),
            "protected_attributes": list(protected_attributes.keys()),
            "metrics_computed": list(set(r.metric_name for r in results)),
            "sample_sizes": {
                attr: len(values) for attr, values in protected_attributes.items()
            }
        }
        
        if results:
            stats["fairness_scores"] = {
                "min": min(r.metric_value for r in results if r.metric_value is not None),
                "max": max(r.metric_value for r in results if r.metric_value is not None),
                "mean": np.mean([r.metric_value for r in results if r.metric_value is not None]),
                "std": np.std([r.metric_value for r in results if r.metric_value is not None])
            }
        
        return stats
    
    def _generate_recommendations(
        self,
        results: List[BiasResult],
        overall_fairness: float
    ) -> List[str]:
        """Generate bias mitigation recommendations."""
        recommendations = []
        
        if overall_fairness < self.fairness_threshold:
            recommendations.append(
                f"Overall fairness score ({overall_fairness:.3f}) below threshold ({self.fairness_threshold})"
            )
        
        unfair_groups = [r for r in results if not r.is_fair]
        if unfair_groups:
            recommendations.append(
                f"Consider bias mitigation for {len(unfair_groups)} underperforming groups"
            )
            
            # Group-specific recommendations
            by_attribute = {}
            for result in unfair_groups:
                if result.protected_attribute not in by_attribute:
                    by_attribute[result.protected_attribute] = []
                by_attribute[result.protected_attribute].append(result)
            
            for attr, attr_results in by_attribute.items():
                recommendations.append(
                    f"Attribute '{attr}': {len(attr_results)} groups below fairness threshold"
                )
        
        # Sample size warnings
        small_groups = [r for r in results if r.sample_size < 30]
        if small_groups:
            recommendations.append(
                f"Warning: {len(small_groups)} groups have small sample sizes (<30)"
            )
        
        if overall_fairness >= self.fairness_threshold:
            recommendations.append("All groups meet fairness threshold")
        
        return recommendations
    
    def _assess_compliance_status(
        self,
        overall_fairness: float,
        results: List[BiasResult]
    ) -> str:
        """Assess compliance status based on fairness metrics."""
        unfair_groups = [r for r in results if not r.is_fair]
        
        if overall_fairness >= self.fairness_threshold and len(unfair_groups) == 0:
            return "compliant"
        elif overall_fairness >= 0.7:  # Partial compliance
            return "partial_compliance_with_monitoring_required"
        else:
            return "non_compliant_requires_mitigation"

def generate_bias_report(assessment: BiasAssessment) -> str:
    """Generate a human-readable bias assessment report."""
    report = f"""
BIAS ASSESSMENT REPORT
=====================

Overall Fairness Score: {assessment.overall_fairness_score:.3f}
Compliance Status: {assessment.compliance_status}

DETAILED RESULTS:
{'-' * 50}
"""
    
    for result in assessment.individual_results:
        status = "FAIR" if result.is_fair else "UNFAIR"
        report += f"""
{result.protected_attribute} = {result.group_value}:
  Metric: {result.metric_name}
  Score: {result.metric_value:.3f} (threshold: {result.threshold})
  Status: {status}
  Sample Size: {result.sample_size}
"""
    
    report += f"""
SUMMARY STATISTICS:
{'-' * 50}
Total Groups: {assessment.summary_statistics['total_groups_assessed']}
Fair Groups: {assessment.summary_statistics['fair_groups']}
Unfair Groups: {assessment.summary_statistics['unfair_groups']}

RECOMMENDATIONS:
{'-' * 50}
"""
    
    for i, rec in enumerate(assessment.recommendations, 1):
        report += f"{i}. {rec}\n"
    
    return report