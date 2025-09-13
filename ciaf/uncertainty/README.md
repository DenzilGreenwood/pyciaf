# CIAF Uncertainty Quantification

The uncertainty package provides comprehensive uncertainty quantification capabilities for AI models, supporting regulatory compliance with transparency and reliability requirements.

## Overview

The uncertainty quantification system enables AI models to communicate their confidence and reliability:

- **Aleatoric Uncertainty** — Data-dependent uncertainty from noise in observations
- **Epistemic Uncertainty** — Model uncertainty from lack of knowledge or training data
- **Multiple Methods** — Monte Carlo Dropout, Bootstrap Aggregation, Bayesian approaches
- **Regulatory Compliance** — EU AI Act, NIST AI RMF, ISO 27001 compliance support
- **Calibration Support** — Uncertainty calibration for improved reliability
- **Integration Ready** — Seamless integration with CIAF framework components

## Components

### UncertaintyEstimate (Data Class)

Container for comprehensive uncertainty information.

**Key Attributes:**
- **prediction** — The model's primary prediction
- **confidence** — Overall confidence score (0-1)
- **aleatoric_uncertainty** — Data-dependent uncertainty
- **epistemic_uncertainty** — Model knowledge uncertainty
- **total_uncertainty** — Combined uncertainty estimate
- **confidence_interval** — Statistical confidence bounds
- **method** — Uncertainty quantification method used
- **explanation** — Human-readable explanation

### CIAFUncertaintyQuantifier

Primary class for uncertainty quantification with multiple methods.

**Key Features:**
- **Multiple Methods** — Support for various uncertainty quantification approaches
- **Automatic Method Selection** — Choose optimal method based on model type
- **Calibration Support** — Improve uncertainty estimates through calibration
- **Compliance Reporting** — Generate regulatory compliance metadata

**Usage Example:**
```python
from ciaf.uncertainty import CIAFUncertaintyQuantifier, UncertaintyMethod
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create and train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train = np.random.randn(1000, 10)
y_train = np.random.randint(0, 2, 1000)
model.fit(X_train, y_train)

# Create uncertainty quantifier
quantifier = CIAFUncertaintyQuantifier(
    model=model,
    method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
    n_samples=100,
    confidence_level=0.95
)

# Fit quantifier (optional calibration)
X_cal = np.random.randn(200, 10)
y_cal = np.random.randint(0, 2, 200)
quantifier.fit(X_cal, y_cal)

# Make prediction with uncertainty
X_test = np.random.randn(1, 10)
uncertainty_estimate = quantifier.predict_with_uncertainty(X_test)

print(f"Prediction: {uncertainty_estimate.prediction}")
print(f"Confidence: {uncertainty_estimate.confidence:.3f}")
print(f"Total uncertainty: {uncertainty_estimate.total_uncertainty:.3f}")
print(f"Aleatoric uncertainty: {uncertainty_estimate.aleatoric_uncertainty:.3f}")
print(f"Epistemic uncertainty: {uncertainty_estimate.epistemic_uncertainty:.3f}")
print(f"Confidence interval: {uncertainty_estimate.confidence_interval}")
print(f"Method: {uncertainty_estimate.method.value}")
print(f"Explanation: {uncertainty_estimate.explanation}")
```

**Output Example:**
```
Prediction: 0.75
Confidence: 0.850
Total uncertainty: 0.150
Aleatoric uncertainty: 0.090
Epistemic uncertainty: 0.060
Confidence interval: (0.62, 0.88)
Method: monte_carlo_dropout
Explanation: Monte Carlo Dropout with 100 samples
```

### CIAFUncertaintyManager

Global manager for uncertainty quantification across multiple models.

**Usage Example:**
```python
from ciaf.uncertainty import uncertainty_manager, UncertaintyMethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

# Register multiple models with uncertainty quantification
classification_model = LogisticRegression()
regression_model = GradientBoostingRegressor()

# Train models (assume X_train, y_train available)
classification_model.fit(X_train_class, y_train_class)
regression_model.fit(X_train_reg, y_train_reg)

# Register with uncertainty manager
class_quantifier = uncertainty_manager.register_quantifier(
    model_id="medical_classifier",
    model=classification_model,
    method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
    n_samples=150
)

reg_quantifier = uncertainty_manager.register_quantifier(
    model_id="severity_predictor",
    model=regression_model,
    method=UncertaintyMethod.BOOTSTRAP_AGGREGATION,
    n_samples=100
)

# Get predictions with uncertainty
X_new = np.array([[0.5, -1.2, 0.8, 0.1, -0.3, 0.7, -0.1, 0.4, -0.6, 0.2]])

# Classification with uncertainty
class_estimate = uncertainty_manager.predict_with_uncertainty(
    "medical_classifier", X_new
)

# Regression with uncertainty  
reg_estimate = uncertainty_manager.predict_with_uncertainty(
    "severity_predictor", X_new
)

# Get compliance metadata
class_metadata = uncertainty_manager.get_uncertainty_metadata("medical_classifier")
print(f"Compliance frameworks: {class_metadata['compliance_frameworks']}")
```

## Uncertainty Quantification Methods

### Monte Carlo Dropout

Estimates uncertainty by running multiple forward passes with dropout enabled.

**Usage Example:**
```python
from ciaf.uncertainty import create_monte_carlo_quantifier

# Create Monte Carlo quantifier
mc_quantifier = create_monte_carlo_quantifier(
    model=neural_network_model,
    n_samples=200  # More samples = better estimates but slower
)

# Make prediction with uncertainty
uncertainty_estimate = mc_quantifier.predict_with_uncertainty(X_test)

print(f"Monte Carlo uncertainty: {uncertainty_estimate.total_uncertainty:.3f}")
print(f"Confidence interval: {uncertainty_estimate.confidence_interval}")
```

**Best For:**
- Neural networks with dropout layers
- Deep learning models
- When model architecture allows stochastic inference

### Bootstrap Aggregation

Uses input perturbation and prediction variance to estimate uncertainty.

**Usage Example:**
```python
from ciaf.uncertainty import create_bootstrap_quantifier

# Create bootstrap quantifier
bootstrap_quantifier = create_bootstrap_quantifier(
    model=ensemble_model,
    n_samples=100
)

# Predict with uncertainty
uncertainty_estimate = bootstrap_quantifier.predict_with_uncertainty(X_test)

print(f"Bootstrap uncertainty: {uncertainty_estimate.total_uncertainty:.3f}")
print(f"Aleatoric component: {uncertainty_estimate.aleatoric_uncertainty:.3f}")
print(f"Epistemic component: {uncertainty_estimate.epistemic_uncertainty:.3f}")
```

**Best For:**
- Ensemble models
- Tree-based models (Random Forest, Gradient Boosting)
- Models without built-in uncertainty estimation

### Prediction Intervals

Uses model confidence and prediction variance for uncertainty estimation.

**Usage Example:**
```python
from ciaf.uncertainty import CIAFUncertaintyQuantifier, UncertaintyMethod

# Create prediction interval quantifier
pi_quantifier = CIAFUncertaintyQuantifier(
    model=linear_model,
    method=UncertaintyMethod.PREDICTION_INTERVALS,
    confidence_level=0.90  # 90% confidence intervals
)

# Get uncertainty
uncertainty_estimate = pi_quantifier.predict_with_uncertainty(X_test)

print(f"90% confidence interval: {uncertainty_estimate.confidence_interval}")
print(f"Prediction: {uncertainty_estimate.prediction}")
```

**Best For:**
- Linear models
- Models with built-in confidence measures
- Quick uncertainty estimation with minimal computational overhead

### Automatic Method Selection

Automatically choose the best uncertainty method based on model type.

**Usage Example:**
```python
from ciaf.uncertainty import create_auto_quantifier

# Automatic method selection
auto_quantifier = create_auto_quantifier(your_model)

# The quantifier automatically chooses:
# - Monte Carlo Dropout for models with predict_proba
# - Prediction Intervals for regression models
# - Bootstrap for ensemble models

uncertainty_estimate = auto_quantifier.predict_with_uncertainty(X_test)
print(f"Auto-selected method: {uncertainty_estimate.method.value}")
```

## Advanced Features

### Uncertainty Calibration

Improve uncertainty estimates through calibration on held-out data.

**Usage Example:**
```python
from ciaf.uncertainty import CIAFUncertaintyQuantifier
import numpy as np

# Create quantifier
quantifier = CIAFUncertaintyQuantifier(model, UncertaintyMethod.MONTE_CARLO_DROPOUT)

# Prepare calibration data (separate from training/test)
X_calibration = np.random.randn(500, 10)
y_calibration = np.random.randint(0, 2, 500)

# Calibrate uncertainty estimates
calibration_metrics = quantifier.calibrate(X_calibration, y_calibration)

print(f"Calibration score: {calibration_metrics['calibration_score']:.3f}")
print(f"Average confidence: {calibration_metrics['average_confidence']:.3f}")
print(f"Average error: {calibration_metrics['average_error']:.3f}")

# Now predictions will use calibrated uncertainty
calibrated_estimate = quantifier.predict_with_uncertainty(X_test)
print(f"Calibrated uncertainty: {calibrated_estimate.total_uncertainty:.3f}")
```

### Medical AI Uncertainty Quantification

Healthcare-specific uncertainty quantification with regulatory compliance.

**Usage Example:**
```python
from ciaf.uncertainty import CIAFUncertaintyQuantifier, UncertaintyMethod
from sklearn.ensemble import RandomForestClassifier

# Medical diagnostic model
medical_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5  # Conservative for medical applications
)

# Train on medical data
medical_X = np.array([
    [fever, cough, chest_pain, age, white_blood_count],
    # ... more patient features
])
medical_y = np.array([0, 1, 1, 0])  # 0=healthy, 1=pneumonia

medical_model.fit(medical_X, medical_y)

# Create medical uncertainty quantifier
medical_quantifier = CIAFUncertaintyQuantifier(
    model=medical_model,
    method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
    n_samples=200,  # Higher samples for medical applications
    confidence_level=0.99  # Higher confidence for patient safety
)

# Medical prediction with uncertainty
patient_features = np.array([[38.5, 1, 1, 65, 12000]])  # fever, cough, chest pain, age, WBC
medical_estimate = medical_quantifier.predict_with_uncertainty(patient_features)

print(f"Pneumonia probability: {medical_estimate.prediction:.3f}")
print(f"Medical confidence: {medical_estimate.confidence:.3f}")
print(f"Clinical uncertainty: {medical_estimate.total_uncertainty:.3f}")
print(f"99% confidence interval: {medical_estimate.confidence_interval}")

# Medical decision support
if medical_estimate.confidence < 0.8:
    print("⚠️  LOW CONFIDENCE: Recommend additional diagnostic tests")
elif medical_estimate.total_uncertainty > 0.3:
    print("⚠️  HIGH UNCERTAINTY: Consider specialist consultation")
else:
    print("✅ CONFIDENT DIAGNOSIS: Proceed with recommended treatment")

# Generate medical compliance report
medical_compliance = {
    "patient_safety": {
        "confidence_threshold": 0.8,
        "uncertainty_threshold": 0.3,
        "confidence_level": medical_quantifier.confidence_level,
        "samples_used": medical_quantifier.n_samples
    },
    "regulatory_compliance": {
        "fda_guidance": "AI/ML-based medical devices - uncertainty quantification",
        "eu_mdr": "Medical Device Regulation - risk management",
        "iso_14155": "Clinical investigation - uncertainty reporting"
    },
    "clinical_decision_support": {
        "uncertainty_method": medical_estimate.method.value,
        "explanation": medical_estimate.explanation,
        "recommendation": "High confidence diagnosis suitable for clinical use"
    }
}
```

### Financial Risk Uncertainty

Financial applications with risk quantification and regulatory compliance.

**Usage Example:**
```python
from ciaf.uncertainty import CIAFUncertaintyQuantifier, UncertaintyMethod
from sklearn.ensemble import GradientBoostingRegressor

# Financial risk model
risk_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8
)

# Train on financial features
financial_X = np.array([
    [credit_score, income, debt_ratio, employment_years, loan_amount],
    # ... more financial features
])
financial_y = np.array([0.02, 0.15, 0.08, 0.01])  # Default probabilities

risk_model.fit(financial_X, financial_y)

# Create financial uncertainty quantifier
financial_quantifier = CIAFUncertaintyQuantifier(
    model=risk_model,
    method=UncertaintyMethod.BOOTSTRAP_AGGREGATION,
    n_samples=150,
    confidence_level=0.95
)

# Risk assessment with uncertainty
applicant_features = np.array([[720, 75000, 0.3, 5, 250000]])
risk_estimate = financial_quantifier.predict_with_uncertainty(applicant_features)

print(f"Default risk: {risk_estimate.prediction:.4f}")
print(f"Risk confidence: {risk_estimate.confidence:.3f}")
print(f"Risk uncertainty: {risk_estimate.total_uncertainty:.4f}")
print(f"95% risk interval: {risk_estimate.confidence_interval}")

# Financial decision framework
risk_threshold = 0.05
uncertainty_threshold = 0.02

if risk_estimate.prediction > risk_threshold:
    if risk_estimate.total_uncertainty > uncertainty_threshold:
        decision = "DEFER - High risk with high uncertainty"
        action = "Request additional financial information"
    else:
        decision = "DECLINE - High risk with low uncertainty"
        action = "Decline loan application"
elif risk_estimate.total_uncertainty > uncertainty_threshold:
    decision = "MANUAL REVIEW - Low risk but high uncertainty"
    action = "Human underwriter review required"
else:
    decision = "APPROVE - Low risk with low uncertainty"
    action = "Approve loan application"

print(f"Decision: {decision}")
print(f"Action: {action}")

# Financial regulatory compliance
financial_compliance = {
    "risk_management": {
        "basel_iii": "Credit risk uncertainty quantification",
        "dodd_frank": "Model risk management requirements",
        "sr_11_7": "Supervisory guidance on model risk management"
    },
    "uncertainty_governance": {
        "model_validation": "Independent validation of uncertainty estimates",
        "back_testing": "Historical validation of uncertainty predictions",
        "stress_testing": "Uncertainty under adverse scenarios"
    },
    "decision_framework": {
        "risk_threshold": risk_threshold,
        "uncertainty_threshold": uncertainty_threshold,
        "decision": decision,
        "action": action
    }
}
```

### Multi-Model Uncertainty Ensemble

Combine uncertainty from multiple models for robust decision making.

**Usage Example:**
```python
from ciaf.uncertainty import uncertainty_manager
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Create ensemble of models
models = {
    "random_forest": RandomForestClassifier(n_estimators=100),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100),
    "svm": SVC(probability=True)
}

# Train all models
for name, model in models.items():
    model.fit(X_train, y_train)
    
    # Register with uncertainty manager
    uncertainty_manager.register_quantifier(
        model_id=name,
        model=model,
        method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
        n_samples=100
    )

# Get predictions from all models
X_new = np.array([[0.5, -1.2, 0.8, 0.1, -0.3]])
ensemble_estimates = {}

for model_name in models.keys():
    estimate = uncertainty_manager.predict_with_uncertainty(model_name, X_new)
    ensemble_estimates[model_name] = estimate

# Combine uncertainties using ensemble statistics
predictions = [est.prediction for est in ensemble_estimates.values()]
confidences = [est.confidence for est in ensemble_estimates.values()]
uncertainties = [est.total_uncertainty for est in ensemble_estimates.values()]

# Ensemble statistics
ensemble_prediction = np.mean(predictions)
ensemble_confidence = np.mean(confidences)
ensemble_uncertainty = np.sqrt(np.mean([u**2 for u in uncertainties]))  # RMS uncertainty
prediction_disagreement = np.std(predictions)  # Model disagreement

print(f"Ensemble prediction: {ensemble_prediction:.3f}")
print(f"Ensemble confidence: {ensemble_confidence:.3f}")
print(f"Ensemble uncertainty: {ensemble_uncertainty:.3f}")
print(f"Model disagreement: {prediction_disagreement:.3f}")

# Enhanced decision making with ensemble uncertainty
total_ensemble_uncertainty = np.sqrt(ensemble_uncertainty**2 + prediction_disagreement**2)

if total_ensemble_uncertainty > 0.3:
    print("⚠️  HIGH ENSEMBLE UNCERTAINTY: Models disagree significantly")
    recommendation = "Seek additional data or expert consultation"
elif prediction_disagreement > 0.2:
    print("⚠️  MODEL DISAGREEMENT: Individual models provide different predictions")
    recommendation = "Review model assumptions and training data"
else:
    print("✅ CONSENSUS PREDICTION: Models agree with reasonable uncertainty")
    recommendation = "Proceed with ensemble prediction"

print(f"Recommendation: {recommendation}")

# Individual model details
for name, estimate in ensemble_estimates.items():
    print(f"{name}: prediction={estimate.prediction:.3f}, "
          f"confidence={estimate.confidence:.3f}, "
          f"uncertainty={estimate.total_uncertainty:.3f}")
```

## Regulatory Compliance Integration

### EU AI Act Compliance

Uncertainty quantification for EU AI Act compliance requirements.

**Usage Example:**
```python
from ciaf.uncertainty import CIAFUncertaintyQuantifier
from ciaf.compliance import ComplianceValidator

# High-risk AI system under EU AI Act
eu_ai_model = RandomForestClassifier(n_estimators=200)
eu_ai_model.fit(X_train, y_train)

# EU AI Act compliant uncertainty quantifier
eu_quantifier = CIAFUncertaintyQuantifier(
    model=eu_ai_model,
    method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
    n_samples=200,
    confidence_level=0.95
)

# EU AI Act compliance validation
compliance_validator = ComplianceValidator("eu_ai_system")

# Get uncertainty estimate
uncertainty_estimate = eu_quantifier.predict_with_uncertainty(X_test)

# Generate EU AI Act compliance report
eu_compliance_report = {
    "article_15_requirements": {
        "accuracy_assessment": "Uncertainty quantification implemented",
        "robustness_testing": "Monte Carlo sampling with 200 iterations",
        "uncertainty_disclosure": f"Total uncertainty: {uncertainty_estimate.total_uncertainty:.3f}"
    },
    "transparency_obligations": {
        "uncertainty_method": uncertainty_estimate.method.value,
        "confidence_level": eu_quantifier.confidence_level,
        "explanation": uncertainty_estimate.explanation
    },
    "risk_management": {
        "uncertainty_monitoring": "Continuous uncertainty tracking enabled",
        "threshold_management": "Uncertainty thresholds defined and monitored",
        "human_oversight": "High uncertainty triggers human review"
    },
    "technical_documentation": {
        "uncertainty_quantification_method": "Monte Carlo Dropout",
        "validation_methodology": "Calibration on held-out validation set",
        "performance_metrics": {
            "confidence": uncertainty_estimate.confidence,
            "total_uncertainty": uncertainty_estimate.total_uncertainty,
            "aleatoric_uncertainty": uncertainty_estimate.aleatoric_uncertainty,
            "epistemic_uncertainty": uncertainty_estimate.epistemic_uncertainty
        }
    }
}

print("EU AI Act Compliance Report:")
for section, details in eu_compliance_report.items():
    print(f"  {section}: {details}")
```

### NIST AI RMF Integration

Integration with NIST AI Risk Management Framework.

**Usage Example:**
```python
from ciaf.uncertainty import uncertainty_manager

# NIST AI RMF uncertainty integration
nist_compliance = {
    "govern_function": {
        "uncertainty_governance": "Established uncertainty quantification policies",
        "risk_tolerance": "Defined acceptable uncertainty thresholds",
        "accountability": "Clear roles for uncertainty management"
    },
    "map_function": {
        "uncertainty_context": "Uncertainty mapped to decision contexts",
        "stakeholder_impact": "Uncertainty impact on stakeholders assessed",
        "risk_categorization": "Uncertainty-based risk categories defined"
    },
    "measure_function": {
        "uncertainty_metrics": {
            "aleatoric_uncertainty": uncertainty_estimate.aleatoric_uncertainty,
            "epistemic_uncertainty": uncertainty_estimate.epistemic_uncertainty,
            "total_uncertainty": uncertainty_estimate.total_uncertainty,
            "confidence_level": uncertainty_estimate.confidence
        },
        "measurement_methodology": uncertainty_estimate.method.value,
        "validation_approach": "Calibration and cross-validation"
    },
    "manage_function": {
        "uncertainty_monitoring": "Continuous uncertainty tracking",
        "threshold_management": "Automated uncertainty threshold alerts",
        "mitigation_strategies": "Uncertainty-based decision frameworks"
    }
}

# Generate NIST compliance metadata
nist_metadata = uncertainty_manager.get_uncertainty_metadata("nist_model")
print("NIST AI RMF Compliance Metadata:")
for framework, description in nist_metadata.get("compliance_frameworks", {}).items():
    print(f"  {framework}: {description}")
```

## Performance Optimization

### Efficient Uncertainty Computation

Optimize uncertainty quantification for production environments.

**Usage Example:**
```python
from ciaf.uncertainty import CIAFUncertaintyQuantifier
import time

class OptimizedUncertaintyQuantifier(CIAFUncertaintyQuantifier):
    def __init__(self, model, method, n_samples=50, cache_size=1000):
        super().__init__(model, method, n_samples)
        self.prediction_cache = {}
        self.cache_size = cache_size
    
    def predict_with_uncertainty_cached(self, X):
        # Create cache key from input
        cache_key = hash(str(X.tobytes() if hasattr(X, 'tobytes') else str(X)))
        
        # Check cache first
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # Compute uncertainty
        result = self.predict_with_uncertainty(X)
        
        # Cache result (with size limit)
        if len(self.prediction_cache) < self.cache_size:
            self.prediction_cache[cache_key] = result
        
        return result

# Benchmark uncertainty methods
def benchmark_uncertainty_methods(model, X_test, methods):
    results = {}
    
    for method in methods:
        quantifier = CIAFUncertaintyQuantifier(model, method, n_samples=50)
        
        start_time = time.time()
        uncertainty_estimate = quantifier.predict_with_uncertainty(X_test)
        end_time = time.time()
        
        results[method.value] = {
            "computation_time": end_time - start_time,
            "uncertainty": uncertainty_estimate.total_uncertainty,
            "confidence": uncertainty_estimate.confidence
        }
    
    return results

# Run benchmark
benchmark_methods = [
    UncertaintyMethod.MONTE_CARLO_DROPOUT,
    UncertaintyMethod.BOOTSTRAP_AGGREGATION,
    UncertaintyMethod.PREDICTION_INTERVALS
]

benchmark_results = benchmark_uncertainty_methods(model, X_test, benchmark_methods)

print("Uncertainty Method Benchmark:")
for method, metrics in benchmark_results.items():
    print(f"  {method}:")
    print(f"    Time: {metrics['computation_time']:.4f}s")
    print(f"    Uncertainty: {metrics['uncertainty']:.3f}")
    print(f"    Confidence: {metrics['confidence']:.3f}")
```

## Best Practices

### 1. Method Selection Guidelines

```python
def select_uncertainty_method(model, use_case, performance_requirements):
    """Select optimal uncertainty quantification method."""
    
    # Performance-critical applications
    if performance_requirements == "real_time":
        return UncertaintyMethod.PREDICTION_INTERVALS
    
    # High-stakes applications (medical, financial)
    elif use_case in ["medical", "financial"]:
        if hasattr(model, "predict_proba"):
            return UncertaintyMethod.MONTE_CARLO_DROPOUT
        else:
            return UncertaintyMethod.BOOTSTRAP_AGGREGATION
    
    # General applications
    else:
        if hasattr(model, "predict_proba"):
            return UncertaintyMethod.MONTE_CARLO_DROPOUT
        else:
            return UncertaintyMethod.PREDICTION_INTERVALS

# Example usage
optimal_method = select_uncertainty_method(
    model=medical_model,
    use_case="medical",
    performance_requirements="high_accuracy"
)
```

### 2. Uncertainty Threshold Management

```python
def set_uncertainty_thresholds(domain, risk_tolerance):
    """Define uncertainty thresholds for different domains."""
    
    thresholds = {
        "medical": {
            "low_risk": {"uncertainty": 0.1, "confidence": 0.95},
            "medium_risk": {"uncertainty": 0.2, "confidence": 0.90},
            "high_risk": {"uncertainty": 0.3, "confidence": 0.85}
        },
        "financial": {
            "low_risk": {"uncertainty": 0.05, "confidence": 0.98},
            "medium_risk": {"uncertainty": 0.15, "confidence": 0.95},
            "high_risk": {"uncertainty": 0.25, "confidence": 0.90}
        },
        "general": {
            "low_risk": {"uncertainty": 0.2, "confidence": 0.80},
            "medium_risk": {"uncertainty": 0.4, "confidence": 0.70},
            "high_risk": {"uncertainty": 0.6, "confidence": 0.60}
        }
    }
    
    return thresholds.get(domain, thresholds["general"])[risk_tolerance]

# Example usage
medical_thresholds = set_uncertainty_thresholds("medical", "low_risk")
print(f"Medical uncertainty threshold: {medical_thresholds}")
```

### 3. Uncertainty Validation

```python
def validate_uncertainty_quality(quantifier, X_val, y_val):
    """Validate uncertainty quantification quality."""
    
    # Get uncertainty estimates for validation set
    estimates = []
    for i in range(len(X_val)):
        est = quantifier.predict_with_uncertainty(X_val[i:i+1])
        estimates.append(est)
    
    # Extract predictions and uncertainties
    predictions = [est.prediction for est in estimates]
    uncertainties = [est.total_uncertainty for est in estimates]
    confidences = [est.confidence for est in estimates]
    
    # Calculate validation metrics
    prediction_errors = [abs(pred - true) for pred, true in zip(predictions, y_val)]
    
    # Correlation between uncertainty and error
    uncertainty_error_corr = np.corrcoef(uncertainties, prediction_errors)[0, 1]
    
    # Calibration: high confidence should correlate with low error
    confidence_error_corr = np.corrcoef(confidences, prediction_errors)[0, 1]
    
    # Reliability: check if uncertainty intervals contain true values
    interval_coverage = 0
    for est, true_val in zip(estimates, y_val):
        lower, upper = est.confidence_interval
        if lower <= true_val <= upper:
            interval_coverage += 1
    interval_coverage = interval_coverage / len(estimates)
    
    return {
        "uncertainty_error_correlation": uncertainty_error_corr,
        "confidence_error_correlation": confidence_error_corr,
        "interval_coverage": interval_coverage,
        "expected_coverage": quantifier.confidence_level,
        "calibration_quality": abs(interval_coverage - quantifier.confidence_level)
    }

# Example usage
validation_metrics = validate_uncertainty_quality(quantifier, X_validation, y_validation)
print("Uncertainty Validation Metrics:")
for metric, value in validation_metrics.items():
    print(f"  {metric}: {value:.3f}")
```

## Contributing

When extending the uncertainty package:

1. **Scientific Rigor** — Ensure uncertainty methods are statistically sound
2. **Computational Efficiency** — Optimize for production environments
3. **Regulatory Compliance** — Support emerging AI governance frameworks
4. **Validation Framework** — Include uncertainty validation utilities
5. **Domain Expertise** — Consider domain-specific uncertainty requirements

## Dependencies

The uncertainty package depends on:
- `numpy` — Numerical operations for uncertainty calculations
- `dataclasses` — Data structures for uncertainty estimates
- `enum` — Enumeration of uncertainty methods and types
- `typing` — Type hints for better code clarity
- `warnings` — Warning management for robust error handling
- `datetime` — Timestamp generation for uncertainty tracking

---

*For integration examples and advanced patterns, see the [examples folder](../examples/) and [API documentation](../api/).*