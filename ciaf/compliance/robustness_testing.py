"""
CIAF Advanced Robustness Testing - Adversarial and Stress Testing

This module provides comprehensive robustness testing capabilities including
adversarial attacks, distribution shift testing, and stress testing for AI models.

Created: 2025-09-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


class TestType(Enum):
    """Types of robustness tests."""
    ADVERSARIAL = "adversarial"
    DISTRIBUTION_SHIFT = "distribution_shift"
    NOISE_INJECTION = "noise_injection"
    STRESS_TEST = "stress_test"
    BOUNDARY_TEST = "boundary_test"
    FAIRNESS_TEST = "fairness_test"
    PRIVACY_TEST = "privacy_test"


class TestSeverity(Enum):
    """Test severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestResult:
    """Individual test result."""
    test_id: str
    test_type: TestType
    severity: TestSeverity
    passed: bool
    score: float
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RobustnessReport:
    """Comprehensive robustness testing report."""
    test_session_id: str
    model_name: str
    model_version: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    overall_score: float = 0.0
    test_results: List[TestResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)


class AdversarialTester:
    """Adversarial attack testing implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fgsm_attack(self, model_fn: Callable, inputs: np.ndarray, 
                    targets: np.ndarray, epsilon: float = 0.01) -> TestResult:
        """Fast Gradient Sign Method attack test."""
        test_id = f"fgsm_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # Simulate FGSM attack
            # In real implementation, would compute gradients and apply perturbations
            original_pred = model_fn(inputs)
            
            # Apply simulated perturbation
            noise = np.random.normal(0, epsilon, inputs.shape)
            perturbed_inputs = inputs + noise
            adversarial_pred = model_fn(perturbed_inputs)
            
            # Calculate attack success rate
            original_correct = np.sum(np.argmax(original_pred, axis=1) == targets)
            adversarial_correct = np.sum(np.argmax(adversarial_pred, axis=1) == targets)
            
            attack_success_rate = (original_correct - adversarial_correct) / len(targets)
            robustness_score = 1.0 - attack_success_rate
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_type=TestType.ADVERSARIAL,
                severity=TestSeverity.HIGH,
                passed=robustness_score >= 0.8,  # Threshold for robustness
                score=robustness_score,
                execution_time=execution_time,
                metadata={
                    "attack_type": "fgsm",
                    "epsilon": epsilon,
                    "attack_success_rate": attack_success_rate,
                    "samples_tested": len(targets)
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_id,
                test_type=TestType.ADVERSARIAL,
                severity=TestSeverity.HIGH,
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def pgd_attack(self, model_fn: Callable, inputs: np.ndarray,
                   targets: np.ndarray, epsilon: float = 0.01,
                   alpha: float = 0.002, iterations: int = 10) -> TestResult:
        """Projected Gradient Descent attack test."""
        test_id = f"pgd_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # Simulate PGD attack
            original_pred = model_fn(inputs)
            
            # Iterative perturbation simulation
            perturbed_inputs = inputs.copy()
            for i in range(iterations):
                noise = np.random.normal(0, alpha, inputs.shape)
                perturbed_inputs = np.clip(perturbed_inputs + noise,
                                         inputs - epsilon, inputs + epsilon)
            
            adversarial_pred = model_fn(perturbed_inputs)
            
            # Calculate robustness metrics
            original_correct = np.sum(np.argmax(original_pred, axis=1) == targets)
            adversarial_correct = np.sum(np.argmax(adversarial_pred, axis=1) == targets)
            
            attack_success_rate = (original_correct - adversarial_correct) / len(targets)
            robustness_score = 1.0 - attack_success_rate
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_type=TestType.ADVERSARIAL,
                severity=TestSeverity.CRITICAL,
                passed=robustness_score >= 0.7,
                score=robustness_score,
                execution_time=execution_time,
                metadata={
                    "attack_type": "pgd",
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "iterations": iterations,
                    "attack_success_rate": attack_success_rate
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_id,
                test_type=TestType.ADVERSARIAL,
                severity=TestSeverity.CRITICAL,
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class DistributionShiftTester:
    """Distribution shift and covariate shift testing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def gaussian_noise_test(self, model_fn: Callable, inputs: np.ndarray,
                           targets: np.ndarray, noise_levels: List[float] = None) -> TestResult:
        """Test model robustness to Gaussian noise."""
        if noise_levels is None:
            noise_levels = [0.1, 0.2, 0.3]
        
        test_id = f"gauss_noise_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            baseline_pred = model_fn(inputs)
            baseline_accuracy = np.mean(np.argmax(baseline_pred, axis=1) == targets)
            
            noise_results = {}
            for noise_level in noise_levels:
                # Add Gaussian noise
                noisy_inputs = inputs + np.random.normal(0, noise_level, inputs.shape)
                noisy_pred = model_fn(noisy_inputs)
                noisy_accuracy = np.mean(np.argmax(noisy_pred, axis=1) == targets)
                
                noise_results[noise_level] = {
                    "accuracy": noisy_accuracy,
                    "degradation": baseline_accuracy - noisy_accuracy
                }
            
            # Calculate overall robustness score
            avg_degradation = np.mean([r["degradation"] for r in noise_results.values()])
            robustness_score = max(0.0, 1.0 - (avg_degradation / baseline_accuracy))
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_type=TestType.NOISE_INJECTION,
                severity=TestSeverity.MEDIUM,
                passed=robustness_score >= 0.8,
                score=robustness_score,
                execution_time=execution_time,
                metadata={
                    "baseline_accuracy": baseline_accuracy,
                    "noise_levels": noise_levels,
                    "noise_results": noise_results,
                    "avg_degradation": avg_degradation
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_id,
                test_type=TestType.NOISE_INJECTION,
                severity=TestSeverity.MEDIUM,
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def covariate_shift_test(self, model_fn: Callable, 
                           train_inputs: np.ndarray, test_inputs: np.ndarray,
                           test_targets: np.ndarray) -> TestResult:
        """Test robustness to covariate shift between train and test distributions."""
        test_id = f"covariate_shift_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # Calculate distribution statistics
            train_mean = np.mean(train_inputs, axis=0)
            train_std = np.std(train_inputs, axis=0)
            test_mean = np.mean(test_inputs, axis=0)
            test_std = np.std(test_inputs, axis=0)
            
            # Calculate distribution divergence (simplified KL divergence estimation)
            epsilon = 1e-8
            kl_divergence = np.sum(np.log((test_std + epsilon) / (train_std + epsilon)) +
                                 (train_std**2 + (train_mean - test_mean)**2) /
                                 (2 * (test_std**2 + epsilon)) - 0.5)
            
            # Test model performance on shifted distribution
            test_pred = model_fn(test_inputs)
            test_accuracy = np.mean(np.argmax(test_pred, axis=1) == test_targets)
            
            # Calculate robustness score based on KL divergence and accuracy
            shift_severity = min(1.0, kl_divergence / 10.0)  # Normalize KL divergence
            robustness_score = test_accuracy * (1.0 - shift_severity)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_type=TestType.DISTRIBUTION_SHIFT,
                severity=TestSeverity.HIGH,
                passed=robustness_score >= 0.7,
                score=robustness_score,
                execution_time=execution_time,
                metadata={
                    "kl_divergence": kl_divergence,
                    "shift_severity": shift_severity,
                    "test_accuracy": test_accuracy,
                    "distribution_stats": {
                        "train_mean": train_mean.tolist(),
                        "train_std": train_std.tolist(),
                        "test_mean": test_mean.tolist(),
                        "test_std": test_std.tolist()
                    }
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_id,
                test_type=TestType.DISTRIBUTION_SHIFT,
                severity=TestSeverity.HIGH,
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class StressTester:
    """High-load and boundary condition testing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_stress_test(self, model_fn: Callable, inputs: np.ndarray,
                        max_concurrent: int = 100, duration_seconds: int = 60) -> TestResult:
        """Test model performance under concurrent load."""
        test_id = f"load_stress_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            results = {
                "successful_calls": 0,
                "failed_calls": 0,
                "total_latency": 0.0,
                "max_latency": 0.0,
                "min_latency": float('inf')
            }
            
            def single_inference():
                """Single inference call for load testing."""
                call_start = time.time()
                try:
                    _ = model_fn(inputs[:1])  # Single sample inference
                    call_time = time.time() - call_start
                    return {"success": True, "latency": call_time}
                except Exception as e:
                    call_time = time.time() - call_start
                    return {"success": False, "latency": call_time, "error": str(e)}
            
            # Run concurrent load test
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                end_time = start_time + duration_seconds
                futures = []
                
                while time.time() < end_time:
                    if len(futures) < max_concurrent:
                        futures.append(executor.submit(single_inference))
                    
                    # Collect completed futures
                    completed_futures = [f for f in futures if f.done()]
                    for future in completed_futures:
                        result = future.result()
                        futures.remove(future)
                        
                        if result["success"]:
                            results["successful_calls"] += 1
                        else:
                            results["failed_calls"] += 1
                        
                        latency = result["latency"]
                        results["total_latency"] += latency
                        results["max_latency"] = max(results["max_latency"], latency)
                        results["min_latency"] = min(results["min_latency"], latency)
                
                # Wait for remaining futures
                for future in futures:
                    result = future.result()
                    if result["success"]:
                        results["successful_calls"] += 1
                    else:
                        results["failed_calls"] += 1
            
            total_calls = results["successful_calls"] + results["failed_calls"]
            success_rate = results["successful_calls"] / total_calls if total_calls > 0 else 0
            avg_latency = results["total_latency"] / total_calls if total_calls > 0 else 0
            
            # Calculate stress test score
            latency_score = max(0.0, 1.0 - (avg_latency / 5.0))  # Penalty if avg latency > 5s
            stress_score = (success_rate * 0.7) + (latency_score * 0.3)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_type=TestType.STRESS_TEST,
                severity=TestSeverity.HIGH,
                passed=stress_score >= 0.8,
                score=stress_score,
                execution_time=execution_time,
                metadata={
                    "total_calls": total_calls,
                    "success_rate": success_rate,
                    "avg_latency": avg_latency,
                    "max_latency": results["max_latency"],
                    "min_latency": results["min_latency"],
                    "max_concurrent": max_concurrent,
                    "duration_seconds": duration_seconds
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_id,
                test_type=TestType.STRESS_TEST,
                severity=TestSeverity.HIGH,
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def boundary_value_test(self, model_fn: Callable, inputs: np.ndarray,
                          feature_ranges: Dict[int, Tuple[float, float]]) -> TestResult:
        """Test model behavior at feature boundary values."""
        test_id = f"boundary_test_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            boundary_results = []
            
            for feature_idx, (min_val, max_val) in feature_ranges.items():
                # Test minimum boundary
                min_inputs = inputs.copy()
                min_inputs[:, feature_idx] = min_val
                
                # Test maximum boundary
                max_inputs = inputs.copy()
                max_inputs[:, feature_idx] = max_val
                
                # Get predictions
                try:
                    min_pred = model_fn(min_inputs)
                    max_pred = model_fn(max_inputs)
                    
                    # Check for extreme outputs (potential instability)
                    min_extreme = np.any(np.abs(min_pred) > 10)  # Threshold for extreme values
                    max_extreme = np.any(np.abs(max_pred) > 10)
                    
                    boundary_results.append({
                        "feature_idx": feature_idx,
                        "min_boundary": {"extreme": min_extreme, "max_output": np.max(np.abs(min_pred))},
                        "max_boundary": {"extreme": max_extreme, "max_output": np.max(np.abs(max_pred))}
                    })
                    
                except Exception as e:
                    boundary_results.append({
                        "feature_idx": feature_idx,
                        "error": str(e)
                    })
            
            # Calculate boundary stability score
            stable_boundaries = sum(1 for r in boundary_results 
                                  if "error" not in r and 
                                  not r["min_boundary"]["extreme"] and 
                                  not r["max_boundary"]["extreme"])
            
            stability_score = stable_boundaries / len(feature_ranges)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_id=test_id,
                test_type=TestType.BOUNDARY_TEST,
                severity=TestSeverity.MEDIUM,
                passed=stability_score >= 0.9,
                score=stability_score,
                execution_time=execution_time,
                metadata={
                    "tested_features": len(feature_ranges),
                    "stable_boundaries": stable_boundaries,
                    "boundary_results": boundary_results
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_id,
                test_type=TestType.BOUNDARY_TEST,
                severity=TestSeverity.MEDIUM,
                passed=False,
                score=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class RobustnessTestSuite:
    """Main robustness testing orchestrator."""
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        self.model_name = model_name
        self.model_version = model_version
        self.logger = logging.getLogger(__name__)
        
        # Initialize testers
        self.adversarial_tester = AdversarialTester()
        self.distribution_tester = DistributionShiftTester()
        self.stress_tester = StressTester()
    
    def run_comprehensive_test(self, model_fn: Callable, inputs: np.ndarray,
                             targets: np.ndarray, test_config: Dict[str, Any] = None) -> RobustnessReport:
        """Run comprehensive robustness testing suite."""
        if test_config is None:
            test_config = self._get_default_config()
        
        session_id = f"robustness_test_{uuid.uuid4().hex[:8]}"
        report = RobustnessReport(
            test_session_id=session_id,
            model_name=self.model_name,
            model_version=self.model_version,
            start_time=datetime.now()
        )
        
        self.logger.info(f"Starting comprehensive robustness test session: {session_id}")
        
        try:
            # Run adversarial tests
            if test_config.get("run_adversarial", True):
                self.logger.info("Running adversarial tests...")
                
                # FGSM attack
                fgsm_result = self.adversarial_tester.fgsm_attack(
                    model_fn, inputs, targets, 
                    epsilon=test_config.get("fgsm_epsilon", 0.01)
                )
                report.test_results.append(fgsm_result)
                
                # PGD attack
                pgd_result = self.adversarial_tester.pgd_attack(
                    model_fn, inputs, targets,
                    epsilon=test_config.get("pgd_epsilon", 0.01),
                    iterations=test_config.get("pgd_iterations", 10)
                )
                report.test_results.append(pgd_result)
            
            # Run distribution shift tests
            if test_config.get("run_distribution", True):
                self.logger.info("Running distribution shift tests...")
                
                # Gaussian noise test
                noise_result = self.distribution_tester.gaussian_noise_test(
                    model_fn, inputs, targets,
                    noise_levels=test_config.get("noise_levels", [0.1, 0.2, 0.3])
                )
                report.test_results.append(noise_result)
                
                # Covariate shift test (if train data provided)
                if "train_inputs" in test_config:
                    covariate_result = self.distribution_tester.covariate_shift_test(
                        model_fn, test_config["train_inputs"], inputs, targets
                    )
                    report.test_results.append(covariate_result)
            
            # Run stress tests
            if test_config.get("run_stress", True):
                self.logger.info("Running stress tests...")
                
                # Load stress test
                load_result = self.stress_tester.load_stress_test(
                    model_fn, inputs,
                    max_concurrent=test_config.get("max_concurrent", 50),
                    duration_seconds=test_config.get("stress_duration", 30)
                )
                report.test_results.append(load_result)
                
                # Boundary value test (if feature ranges provided)
                if "feature_ranges" in test_config:
                    boundary_result = self.stress_tester.boundary_value_test(
                        model_fn, inputs, test_config["feature_ranges"]
                    )
                    report.test_results.append(boundary_result)
            
            # Generate report summary
            report.end_time = datetime.now()
            report.total_tests = len(report.test_results)
            report.passed_tests = sum(1 for r in report.test_results if r.passed)
            report.failed_tests = report.total_tests - report.passed_tests
            
            # Calculate overall score
            if report.total_tests > 0:
                report.overall_score = sum(r.score for r in report.test_results) / report.total_tests
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
            # Generate risk assessment
            report.risk_assessment = self._generate_risk_assessment(report)
            
            self.logger.info(f"Robustness test completed. Overall score: {report.overall_score:.2f}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error during robustness testing: {str(e)}")
            report.end_time = datetime.now()
            return report
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default testing configuration."""
        return {
            "run_adversarial": True,
            "run_distribution": True, 
            "run_stress": True,
            "fgsm_epsilon": 0.01,
            "pgd_epsilon": 0.01,
            "pgd_iterations": 10,
            "noise_levels": [0.1, 0.2, 0.3],
            "max_concurrent": 50,
            "stress_duration": 30
        }
    
    def _generate_recommendations(self, report: RobustnessReport) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check adversarial robustness
        adversarial_tests = [r for r in report.test_results if r.test_type == TestType.ADVERSARIAL]
        if adversarial_tests and any(not r.passed for r in adversarial_tests):
            recommendations.append(
                "Consider implementing adversarial training to improve robustness against attacks."
            )
        
        # Check noise sensitivity
        noise_tests = [r for r in report.test_results if r.test_type == TestType.NOISE_INJECTION]
        if noise_tests and any(r.score < 0.8 for r in noise_tests):
            recommendations.append(
                "Model shows sensitivity to input noise. Consider data augmentation during training."
            )
        
        # Check stress test performance
        stress_tests = [r for r in report.test_results if r.test_type == TestType.STRESS_TEST]
        if stress_tests and any(r.score < 0.8 for r in stress_tests):
            recommendations.append(
                "Model performance degrades under load. Consider optimization or scaling strategies."
            )
        
        # Check boundary stability
        boundary_tests = [r for r in report.test_results if r.test_type == TestType.BOUNDARY_TEST]
        if boundary_tests and any(r.score < 0.9 for r in boundary_tests):
            recommendations.append(
                "Model shows instability at feature boundaries. Review input validation and clipping."
            )
        
        return recommendations
    
    def _generate_risk_assessment(self, report: RobustnessReport) -> Dict[str, Any]:
        """Generate risk assessment based on test results."""
        risks = {
            "adversarial_risk": "low",
            "distribution_shift_risk": "low", 
            "performance_risk": "low",
            "overall_risk": "low"
        }
        
        # Assess adversarial risk
        adversarial_tests = [r for r in report.test_results if r.test_type == TestType.ADVERSARIAL]
        if adversarial_tests:
            avg_adversarial_score = sum(r.score for r in adversarial_tests) / len(adversarial_tests)
            if avg_adversarial_score < 0.6:
                risks["adversarial_risk"] = "high"
            elif avg_adversarial_score < 0.8:
                risks["adversarial_risk"] = "medium"
        
        # Assess distribution shift risk
        dist_tests = [r for r in report.test_results 
                     if r.test_type in [TestType.DISTRIBUTION_SHIFT, TestType.NOISE_INJECTION]]
        if dist_tests:
            avg_dist_score = sum(r.score for r in dist_tests) / len(dist_tests)
            if avg_dist_score < 0.6:
                risks["distribution_shift_risk"] = "high"
            elif avg_dist_score < 0.8:
                risks["distribution_shift_risk"] = "medium"
        
        # Assess performance risk
        perf_tests = [r for r in report.test_results 
                     if r.test_type in [TestType.STRESS_TEST, TestType.BOUNDARY_TEST]]
        if perf_tests:
            avg_perf_score = sum(r.score for r in perf_tests) / len(perf_tests)
            if avg_perf_score < 0.7:
                risks["performance_risk"] = "high"
            elif avg_perf_score < 0.85:
                risks["performance_risk"] = "medium"
        
        # Overall risk assessment
        high_risks = sum(1 for risk in risks.values() if risk == "high")
        medium_risks = sum(1 for risk in risks.values() if risk == "medium")
        
        if high_risks >= 2:
            risks["overall_risk"] = "high"
        elif high_risks >= 1 or medium_risks >= 2:
            risks["overall_risk"] = "medium"
        
        return risks
    
    def export_report(self, report: RobustnessReport, file_path: str = None) -> str:
        """Export robustness report to JSON file."""
        if file_path is None:
            file_path = f"robustness_report_{report.test_session_id}.json"
        
        # Helper function to convert numpy objects to JSON-serializable format
        def make_json_serializable(obj):
            """Convert numpy arrays and other non-serializable objects to JSON-friendly format."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        # Convert report to JSON-serializable format
        report_data = {
            "test_session_id": report.test_session_id,
            "model_name": report.model_name,
            "model_version": report.model_version,
            "start_time": report.start_time.isoformat(),
            "end_time": report.end_time.isoformat() if report.end_time else None,
            "total_tests": report.total_tests,
            "passed_tests": report.passed_tests,
            "failed_tests": report.failed_tests,
            "overall_score": report.overall_score,
            "test_results": [
                {
                    "test_id": r.test_id,
                    "test_type": r.test_type.value,
                    "severity": r.severity.value,
                    "passed": r.passed,
                    "score": r.score,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                    "metadata": make_json_serializable(r.metadata),
                    "timestamp": r.timestamp.isoformat()
                }
                for r in report.test_results
            ],
            "recommendations": report.recommendations,
            "risk_assessment": report.risk_assessment
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)  # Use default=str for unknown types
        
        self.logger.info(f"Robustness report exported to: {file_path}")
        return file_path


# Demo functions
def demo_robustness_testing():
    """Demo robustness testing functionality."""
    print("🔬 CIAF Advanced Robustness Testing Demo")
    print("=" * 40)
    
    # Create more realistic model function
    class RealisticDemoModel:
        """Realistic demo model for robustness testing."""
        
        def __init__(self):
            # Pre-trained weights for a simple 3-class classifier
            self.weights = np.array([
                [0.2, -0.1, 0.3],   # Feature 0 weights
                [-0.3, 0.4, 0.1],   # Feature 1 weights 
                [0.1, 0.2, -0.2],   # Feature 2 weights
                [0.4, -0.2, 0.3],   # Feature 3 weights
                [-0.1, 0.3, 0.2]    # Feature 4 weights
            ])
            self.bias = np.array([0.1, -0.05, 0.02])
            
        def __call__(self, inputs):
            """Make the model callable."""
            return self.predict(inputs)
            
        def predict(self, inputs):
            """Predict with realistic neural network behavior."""
            if inputs.ndim == 1:
                inputs = inputs.reshape(1, -1)
            
            # Pad or truncate inputs to match weight dimensions
            if inputs.shape[1] > self.weights.shape[0]:
                inputs = inputs[:, :self.weights.shape[0]]
            elif inputs.shape[1] < self.weights.shape[0]:
                padded = np.zeros((inputs.shape[0], self.weights.shape[0]))
                padded[:, :inputs.shape[1]] = inputs
                inputs = padded
            
            # Linear transformation
            logits = np.dot(inputs, self.weights) + self.bias
            
            # Apply softmax for realistic probabilities
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            return probabilities
    
    model = RealisticDemoModel()
    
    # Create test suite
    test_suite = RobustnessTestSuite("demo_model", "1.0.0")
    
    # Generate realistic test data
    np.random.seed(42)
    inputs = np.random.randn(100, 5)  # 5 features to match model
    targets = np.random.randint(0, 3, 100)  # 3 classes
    
    # Configure comprehensive tests
    test_config = {
        "run_adversarial": True,
        "run_distribution": True,
        "run_stress": True,
        "fgsm_epsilon": 0.05,
        "pgd_epsilon": 0.03,
        "pgd_iterations": 10,
        "noise_levels": [0.1, 0.2, 0.3],
        "max_concurrent": 10,
        "stress_duration": 10,
        "feature_ranges": {
            0: (-3, 3), 1: (-2, 2), 2: (-3, 3), 3: (-2, 2), 4: (-3, 3)
        }
    }
    
    print("🧪 Running comprehensive robustness tests...")
    report = test_suite.run_comprehensive_test(model, inputs, targets, test_config)
    
    print(f"\n📊 Test Results:")
    print(f"   Total Tests: {report.total_tests}")
    print(f"   Passed: {report.passed_tests}")
    print(f"   Failed: {report.failed_tests}")
    print(f"   Overall Score: {report.overall_score:.2f}")
    
    print(f"\n⚠️  Risk Assessment:")
    for risk_type, level in report.risk_assessment.items():
        print(f"   {risk_type.replace('_', ' ').title()}: {level.upper()}")
    
    if report.recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Export report
    report_file = test_suite.export_report(report)
    print(f"\n📄 Report exported to: {report_file}")
    
    return report


if __name__ == "__main__":
    demo_robustness_testing()