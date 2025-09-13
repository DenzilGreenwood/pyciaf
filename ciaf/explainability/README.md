# CIAF Explainability Framework

The explainability package provides comprehensive explainable AI capabilities, supporting regulatory compliance with transparency requirements from EU AI Act, NIST AI RMF, and GDPR.

## Overview

The explainability system enables AI models to provide clear, interpretable explanations of their decisions:

- **Multiple Explanation Methods** — SHAP, LIME, gradient-based, and feature importance
- **Regulatory Compliance** — EU AI Act Article 13, NIST AI RMF, GDPR Article 22
- **Universal Model Support** — Works with tree-based, linear, neural network, and custom models
- **Feature Attribution** — Detailed feature-level impact analysis
- **Confidence Scoring** — Reliability measures for explanations themselves
- **Performance Optimization** — Efficient explanation generation for production systems

## Components

### CIAFExplainer

Primary class for generating model explanations with various methods.

**Key Features:**
- **Multiple Methods** — Support for SHAP, LIME, and feature importance
- **Automatic Method Selection** — Choose optimal method based on model type
- **Feature Attribution** — Detailed analysis of feature contributions
- **Confidence Scoring** — Reliability assessment for explanations

**Usage Example:**
```python
from ciaf.explainability import CIAFExplainer, ExplanationMethod
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create and train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train = np.random.randn(1000, 10)
y_train = np.random.randint(0, 2, 1000)
model.fit(X_train, y_train)

# Create explainer with feature names
feature_names = [f"feature_{i}" for i in range(10)]
explainer = CIAFExplainer(
    model=model,
    method=ExplanationMethod.SHAP_TREE,
    feature_names=feature_names
)

# Fit explainer on training data
explainer.fit(X_train)

# Generate explanation for a single prediction
X_test = np.random.randn(1, 10)
explanation = explainer.explain(X_test, max_features=5)

print(f"Explanation method: {explanation['method']}")
print(f"Explanation confidence: {explanation['explanation_confidence']}")
print("Top feature attributions:")
for attr in explanation['feature_attributions']:
    print(f"  {attr['feature_name']}: {attr['attribution_value']:.3f}")
```

**Output Example:**
```
Explanation method: SHAP
Explanation confidence: 0.95
Top feature attributions:
  feature_3: 0.142
  feature_7: -0.089
  feature_1: 0.067
  feature_9: -0.052
  feature_4: 0.031
```

### CIAFExplainabilityManager

Global manager for explainability across multiple models.

**Usage Example:**
```python
from ciaf.explainability import explainability_manager, ExplanationMethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Register multiple models with explainability
models = {
    "classifier": LogisticRegression(),
    "ensemble": GradientBoostingClassifier()
}

feature_names = [f"feature_{i}" for i in range(10)]

# Train and register models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    # Register with explainability manager
    explainer = explainability_manager.register_explainer(
        model_id=model_name,
        model=model,
        method=ExplanationMethod.SHAP_KERNEL,
        feature_names=feature_names
    )
    
    # Fit explainer
    explainability_manager.fit_explainer(model_name, X_train)

# Generate explanations for predictions
X_new = np.random.randn(1, 10)

for model_name, model in models.items():
    prediction = model.predict(X_new)[0]
    explanation = explainability_manager.explain_prediction(
        model_id=model_name,
        X=X_new,
        prediction=prediction,
        max_features=3
    )
    
    print(f"\n{model_name} explanation:")
    print(f"Prediction: {prediction}")
    print(f"Method: {explanation['method']}")
    print(f"Confidence: {explanation['explanation_confidence']}")

# Get compliance metadata
compliance_meta = explainability_manager.get_explainability_metadata("classifier")
print(f"Compliance frameworks: {compliance_meta['compliance_frameworks']}")
```

## Explanation Methods

### SHAP (SHapley Additive exPlanations)

State-of-the-art explanation method providing feature attribution values.

**Tree Explainer (for tree-based models):**
```python
from ciaf.explainability import create_shap_explainer
from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Create SHAP tree explainer
feature_names = ["age", "income", "credit_score", "debt_ratio", "employment_years"]
shap_explainer = create_shap_explainer(model, feature_names)
shap_explainer.fit(X_train)

# Generate explanation
explanation = shap_explainer.explain(X_test)

print("SHAP Attribution Values:")
for attr in explanation['feature_attributions']:
    print(f"  {attr['feature_name']}: {attr['attribution_value']:.3f}")

# SHAP values sum to (prediction - base_value)
print(f"Total attribution: {explanation['total_attribution']:.3f}")
```

**Linear Explainer (for linear models):**
```python
from sklearn.linear_model import LogisticRegression
from ciaf.explainability import CIAFExplainer, ExplanationMethod

# Linear model
linear_model = LogisticRegression()
linear_model.fit(X_train, y_train)

# SHAP linear explainer
linear_explainer = CIAFExplainer(
    model=linear_model,
    method=ExplanationMethod.SHAP_LINEAR,
    feature_names=feature_names
)
linear_explainer.fit(X_train)

# Fast linear explanations
explanation = linear_explainer.explain(X_test)
print(f"Linear SHAP explanation confidence: {explanation['explanation_confidence']}")
```

**Kernel Explainer (for any model):**
```python
from ciaf.explainability import CIAFExplainer, ExplanationMethod

# Works with any model (including black boxes)
any_model = YourCustomModel()  # Any model with predict method
any_model.fit(X_train, y_train)

# Kernel explainer (model-agnostic)
kernel_explainer = CIAFExplainer(
    model=any_model,
    method=ExplanationMethod.SHAP_KERNEL,
    feature_names=feature_names
)
kernel_explainer.fit(X_train)  # Uses subset for background distribution

# Generate model-agnostic explanation
explanation = kernel_explainer.explain(X_test)
print("Model-agnostic SHAP explanation generated")
```

### LIME (Local Interpretable Model-agnostic Explanations)

Local explanations around individual predictions.

**Tabular Data Explanations:**
```python
from ciaf.explainability import create_lime_explainer

# Create LIME explainer for tabular data
lime_explainer = create_lime_explainer(model, feature_names)
lime_explainer.fit(X_train)

# Generate local explanation
explanation = lime_explainer.explain(X_test, max_features=5)

print("LIME Local Explanation:")
print(f"Local prediction confidence: {explanation['explanation_confidence']:.3f}")
for attr in explanation['feature_attributions']:
    direction = "increases" if attr['attribution_value'] > 0 else "decreases"
    print(f"  {attr['feature_name']} {direction} prediction by {abs(attr['attribution_value']):.3f}")
```

**Text Data Explanations:**
```python
from ciaf.explainability import CIAFExplainer, ExplanationMethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Text classification pipeline
texts = ["This movie is great!", "Terrible film", "Amazing story"]
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(texts)
text_model = MultinomialNB()
text_model.fit(X_text, [1, 0, 1])

# LIME text explainer
text_explainer = CIAFExplainer(
    model=text_model,
    method=ExplanationMethod.LIME_TEXT
)

# Explain text prediction
test_text = "This is a wonderful movie"
# Note: You'll need to handle text preprocessing for LIME text explanations
```

### Feature Importance

Model-native feature importance explanations.

**Usage Example:**
```python
from ciaf.explainability import CIAFExplainer, ExplanationMethod
from sklearn.ensemble import RandomForestClassifier

# Tree-based model with built-in feature importance
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Feature importance explainer
importance_explainer = CIAFExplainer(
    model=model,
    method=ExplanationMethod.FEATURE_IMPORTANCE,
    feature_names=feature_names
)
importance_explainer.fit(X_train)

# Get global feature importance explanation
explanation = importance_explainer.explain(X_test)

print("Global Feature Importance:")
for attr in explanation['feature_attributions']:
    print(f"  {attr['feature_name']}: {attr['attribution_value']:.3f} "
          f"(rank {attr['importance_rank']})")
```

### Automatic Method Selection

Let CIAF choose the optimal explanation method.

**Usage Example:**
```python
from ciaf.explainability import create_auto_explainer

# Automatic explainer selection
auto_explainer = create_auto_explainer(model, feature_names)
auto_explainer.fit(X_train)

# CIAF automatically chose the best method for your model type
explanation = auto_explainer.explain(X_test)
print(f"Auto-selected method: {explanation['method']}")

# Method selection logic:
# - Tree models → SHAP Tree (if available) or Feature Importance
# - Linear models → SHAP Linear (if available) or Feature Importance  
# - Other models → SHAP Kernel (if available) or LIME Tabular
```

## Advanced Use Cases

### Medical AI Explainability

Healthcare AI with clinical decision support explanations.

**Usage Example:**
```python
from ciaf.explainability import CIAFExplainer, ExplanationMethod
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Medical diagnostic model
medical_features = [
    "age", "systolic_bp", "diastolic_bp", "cholesterol", 
    "glucose", "bmi", "smoking", "family_history"
]

# Train medical model
medical_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,  # Conservative for medical
    random_state=42
)

# Medical training data (synthetic)
medical_X = np.random.randn(1000, len(medical_features))
medical_y = np.random.randint(0, 2, 1000)  # 0=healthy, 1=disease
medical_model.fit(medical_X, medical_y)

# Medical explainer with high confidence requirements
medical_explainer = CIAFExplainer(
    model=medical_model,
    method=ExplanationMethod.SHAP_TREE,
    feature_names=medical_features
)
medical_explainer.fit(medical_X)

# Patient case explanation
patient_data = np.array([[65, 140, 90, 220, 110, 28.5, 1, 1]])  # 65yr old with risk factors
explanation = medical_explainer.explain(patient_data, max_features=8)

print("Medical AI Decision Explanation:")
print(f"Explanation confidence: {explanation['explanation_confidence']:.3f}")
print("\nKey diagnostic factors:")

for attr in explanation['feature_attributions']:
    impact = "INCREASES" if attr['attribution_value'] > 0 else "DECREASES"
    clinical_interpretation = interpret_medical_factor(
        attr['feature_name'], 
        attr['attribution_value']
    )
    
    print(f"  {attr['feature_name']}: {impact} risk by {abs(attr['attribution_value']):.3f}")
    print(f"    Clinical note: {clinical_interpretation}")

def interpret_medical_factor(feature_name, attribution_value):
    """Provide clinical interpretation of feature attributions."""
    interpretations = {
        "age": "Advanced age increases cardiovascular risk",
        "systolic_bp": "Elevated systolic pressure indicates hypertension",
        "cholesterol": "High cholesterol contributes to atherosclerosis",
        "glucose": "Elevated glucose suggests diabetes risk",
        "bmi": "Higher BMI indicates obesity-related risk",
        "smoking": "Smoking significantly increases cardiovascular risk",
        "family_history": "Genetic predisposition increases risk"
    }
    
    base_interpretation = interpretations.get(feature_name, "Clinical significance noted")
    
    if attribution_value > 0:
        return f"{base_interpretation} (risk factor)"
    else:
        return f"Protective factor: {base_interpretation.lower()}"

# Medical compliance reporting
medical_compliance = {
    "clinical_decision_support": {
        "explanation_method": explanation['method'],
        "confidence_level": explanation['explanation_confidence'],
        "feature_count": len(explanation['feature_attributions']),
        "total_attribution": explanation.get('total_attribution', 'N/A')
    },
    "regulatory_compliance": {
        "fda_guidance": "AI/ML medical devices - explainability requirements",
        "eu_mdr": "Medical Device Regulation - transparency obligations",
        "hipaa": "Patient right to understand automated decisions"
    },
    "clinical_validation": {
        "explanation_reviewed": True,
        "clinical_coherence": "Explanations align with medical knowledge",
        "actionable_insights": "Clear diagnostic reasoning provided"
    }
}

print(f"\nMedical compliance: {medical_compliance}")
```

### Financial Risk Explainability

Financial services with regulatory explainability requirements.

**Usage Example:**
```python
from ciaf.explainability import explainability_manager
from sklearn.ensemble import GradientBoostingClassifier

# Financial risk model
financial_features = [
    "credit_score", "annual_income", "debt_to_income", 
    "employment_years", "loan_amount", "property_value",
    "payment_history", "credit_utilization"
]

# Risk assessment model
risk_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# Financial training data
financial_X = np.random.randn(2000, len(financial_features))
financial_y = np.random.randint(0, 2, 2000)  # 0=approve, 1=decline
risk_model.fit(financial_X, financial_y)

# Register financial explainer
financial_explainer = explainability_manager.register_explainer(
    model_id="credit_risk_model",
    model=risk_model,
    method=ExplanationMethod.SHAP_TREE,
    feature_names=financial_features
)

explainability_manager.fit_explainer("credit_risk_model", financial_X)

# Credit application explanation
applicant_data = np.array([[720, 85000, 0.25, 8, 300000, 450000, 0.95, 0.15]])
risk_prediction = risk_model.predict_proba(applicant_data)[0][1]  # Risk probability

explanation = explainability_manager.explain_prediction(
    model_id="credit_risk_model",
    X=applicant_data,
    prediction=risk_prediction,
    max_features=6
)

print("Credit Risk Assessment Explanation:")
print(f"Risk probability: {risk_prediction:.3f}")
print(f"Explanation confidence: {explanation['explanation_confidence']:.3f}")
print("\nRisk factors analysis:")

for attr in explanation['feature_attributions']:
    financial_impact = interpret_financial_factor(
        attr['feature_name'], 
        attr['attribution_value']
    )
    
    direction = "INCREASES" if attr['attribution_value'] > 0 else "DECREASES"
    print(f"  {attr['feature_name']}: {direction} risk by {abs(attr['attribution_value']):.3f}")
    print(f"    Impact: {financial_impact}")

def interpret_financial_factor(feature_name, attribution_value):
    """Provide financial interpretation of feature attributions."""
    if attribution_value > 0:
        # Risk-increasing factors
        risk_interpretations = {
            "debt_to_income": "High debt-to-income ratio indicates repayment stress",
            "credit_utilization": "High credit utilization suggests financial strain",
            "loan_amount": "Large loan amount increases exposure risk",
            "employment_years": "Limited employment history reduces stability"
        }
        return risk_interpretations.get(feature_name, "Factor increases credit risk")
    else:
        # Risk-decreasing factors
        protective_interpretations = {
            "credit_score": "High credit score demonstrates creditworthiness",
            "annual_income": "Higher income improves repayment capacity",
            "property_value": "Valuable collateral reduces lender risk",
            "payment_history": "Good payment history shows reliability"
        }
        return protective_interpretations.get(feature_name, "Factor reduces credit risk")

# Financial regulatory compliance
financial_compliance = {
    "regulatory_requirements": {
        "fair_credit_reporting_act": "Explainable automated credit decisions",
        "equal_credit_opportunity_act": "Non-discriminatory lending practices",
        "dodd_frank": "Risk assessment transparency requirements",
        "basel_iii": "Credit risk model validation and documentation"
    },
    "explainability_standards": {
        "method": explanation['method'],
        "confidence": explanation['explanation_confidence'],
        "feature_transparency": "All decision factors disclosed",
        "audit_trail": "Complete explanation audit trail maintained"
    },
    "consumer_protection": {
        "adverse_action_notice": "Specific reasons for credit denial provided",
        "appeal_process": "Customers can request explanation review",
        "bias_monitoring": "Regular bias assessment of explanations"
    }
}

print(f"\nFinancial regulatory compliance: {financial_compliance}")
```

### Multi-Model Explanation Ensemble

Combine explanations from multiple models for robust decision making.

**Usage Example:**
```python
from ciaf.explainability import explainability_manager
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Ensemble of different model types
models = {
    "random_forest": RandomForestClassifier(n_estimators=100),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100),
    "logistic_regression": LogisticRegression()
}

feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]

# Train and register all models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    # Register with appropriate explanation method
    if "forest" in model_name or "boosting" in model_name:
        method = ExplanationMethod.SHAP_TREE
    else:
        method = ExplanationMethod.SHAP_LINEAR
    
    explainer = explainability_manager.register_explainer(
        model_id=model_name,
        model=model,
        method=method,
        feature_names=feature_names
    )
    
    explainability_manager.fit_explainer(model_name, X_train)

# Generate ensemble explanations
X_new = np.random.randn(1, 5)
ensemble_explanations = {}

for model_name, model in models.items():
    prediction = model.predict_proba(X_new)[0][1]
    explanation = explainability_manager.explain_prediction(
        model_id=model_name,
        X=X_new,
        prediction=prediction,
        max_features=5
    )
    ensemble_explanations[model_name] = explanation

# Aggregate explanations across models
feature_consensus = {}
for model_name, explanation in ensemble_explanations.items():
    for attr in explanation['feature_attributions']:
        feature_name = attr['feature_name']
        if feature_name not in feature_consensus:
            feature_consensus[feature_name] = []
        feature_consensus[feature_name].append(attr['attribution_value'])

# Calculate consensus explanations
print("Ensemble Explanation Consensus:")
for feature_name, attributions in feature_consensus.items():
    mean_attribution = np.mean(attributions)
    std_attribution = np.std(attributions)
    consensus_strength = 1.0 - (std_attribution / (abs(mean_attribution) + 1e-6))
    
    print(f"  {feature_name}:")
    print(f"    Mean attribution: {mean_attribution:.3f}")
    print(f"    Std deviation: {std_attribution:.3f}")
    print(f"    Consensus strength: {consensus_strength:.3f}")
    
    if consensus_strength > 0.8:
        print(f"    ✅ STRONG CONSENSUS")
    elif consensus_strength > 0.5:
        print(f"    ⚠️  MODERATE CONSENSUS")
    else:
        print(f"    ❌ LOW CONSENSUS - Models disagree")

# Individual model details
print("\nIndividual Model Explanations:")
for model_name, explanation in ensemble_explanations.items():
    print(f"\n{model_name.upper()}:")
    print(f"  Method: {explanation['method']}")
    print(f"  Confidence: {explanation['explanation_confidence']:.3f}")
    print(f"  Prediction: {explanation['prediction']:.3f}")
```

## Regulatory Compliance Integration

### EU AI Act Compliance

Explainability for EU AI Act Article 13 transparency requirements.

**Usage Example:**
```python
from ciaf.explainability import explainability_manager
from ciaf.compliance import ComplianceValidator

# High-risk AI system under EU AI Act
eu_ai_model = RandomForestClassifier(n_estimators=200)
eu_ai_model.fit(X_train, y_train)

# EU AI Act compliant explainer
eu_explainer = explainability_manager.register_explainer(
    model_id="eu_high_risk_ai",
    model=eu_ai_model,
    method=ExplanationMethod.SHAP_TREE,
    feature_names=feature_names
)

explainability_manager.fit_explainer("eu_high_risk_ai", X_train)

# Generate explanation with EU AI Act compliance
prediction = eu_ai_model.predict(X_test)[0]
explanation = explainability_manager.explain_prediction(
    model_id="eu_high_risk_ai",
    X=X_test,
    prediction=prediction
)

# EU AI Act compliance report
eu_compliance_report = {
    "article_13_transparency": {
        "explanation_provided": True,
        "method_documented": explanation['method'],
        "confidence_disclosed": explanation['explanation_confidence'],
        "feature_attributions": len(explanation['feature_attributions'])
    },
    "technical_documentation": {
        "explainability_method": explanation['method'],
        "explanation_accuracy": explanation['explanation_confidence'],
        "validation_methodology": "Cross-validation and expert review",
        "limitations_documented": "Explanations represent model reasoning, not causal relationships"
    },
    "transparency_obligations": {
        "human_readable": "Feature attributions provided in plain language",
        "appropriate_depth": "Explanation detail appropriate for decision stakes",
        "user_understanding": "Explanations designed for user comprehension"
    },
    "risk_management": {
        "explanation_monitoring": "Regular validation of explanation quality",
        "bias_detection": "Systematic review for biased explanations",
        "human_oversight": "Human review of high-stakes explanations"
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
# NIST AI RMF explainability integration
nist_compliance = {
    "govern_function": {
        "explainability_policies": "Organizational policies for AI explainability",
        "transparency_governance": "Clear governance for explanation requirements",
        "stakeholder_engagement": "Regular review with affected stakeholders"
    },
    "map_function": {
        "explanation_context": "Explanations mapped to decision contexts",
        "user_needs": "Explanation format tailored to user requirements",
        "risk_categorization": "Explanation depth based on risk level"
    },
    "measure_function": {
        "explanation_metrics": {
            "confidence": explanation['explanation_confidence'],
            "coverage": len(explanation['feature_attributions']),
            "consistency": "Validated across multiple predictions"
        },
        "validation_approach": "Systematic validation of explanation quality",
        "performance_monitoring": "Continuous monitoring of explanation effectiveness"
    },
    "manage_function": {
        "explanation_quality": "Active management of explanation accuracy",
        "user_feedback": "Collection and analysis of user feedback",
        "continuous_improvement": "Regular updates to explanation methods"
    }
}

# Generate NIST compliance metadata
nist_metadata = explainability_manager.get_explainability_metadata("eu_high_risk_ai")
print("NIST AI RMF Compliance Metadata:")
for framework, description in nist_metadata.get("compliance_frameworks", {}).items():
    print(f"  {framework}: {description}")
```

## Performance Optimization

### Efficient Explanation Generation

Optimize explainability for production environments.

**Usage Example:**
```python
from ciaf.explainability import CIAFExplainer
import time

class OptimizedExplainer(CIAFExplainer):
    def __init__(self, model, method, feature_names, cache_size=1000):
        super().__init__(model, method, feature_names)
        self.explanation_cache = {}
        self.cache_size = cache_size
    
    def explain_with_cache(self, X, max_features=10):
        # Create cache key
        cache_key = hash(str(X.tobytes() if hasattr(X, 'tobytes') else str(X)))
        
        # Check cache first
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        # Generate explanation
        explanation = self.explain(X, max_features)
        
        # Cache result (with size limit)
        if len(self.explanation_cache) < self.cache_size:
            self.explanation_cache[cache_key] = explanation
        
        return explanation

# Benchmark explanation methods
def benchmark_explanation_methods(model, X_test, methods):
    results = {}
    
    for method in methods:
        explainer = CIAFExplainer(model, method, feature_names)
        explainer.fit(X_train)
        
        start_time = time.time()
        explanation = explainer.explain(X_test)
        end_time = time.time()
        
        results[method] = {
            "computation_time": end_time - start_time,
            "confidence": explanation['explanation_confidence'],
            "feature_count": len(explanation['feature_attributions'])
        }
    
    return results

# Run benchmark
benchmark_methods = [
    ExplanationMethod.SHAP_TREE,
    ExplanationMethod.FEATURE_IMPORTANCE,
    ExplanationMethod.SHAP_KERNEL
]

benchmark_results = benchmark_explanation_methods(model, X_test, benchmark_methods)

print("Explanation Method Benchmark:")
for method, metrics in benchmark_results.items():
    print(f"  {method}:")
    print(f"    Time: {metrics['computation_time']:.4f}s")
    print(f"    Confidence: {metrics['confidence']:.3f}")
    print(f"    Features: {metrics['feature_count']}")
```

### Batch Explanation Processing

Handle multiple explanations efficiently.

**Usage Example:**
```python
def batch_explain(explainer, X_batch, max_features=5):
    """Generate explanations for multiple predictions efficiently."""
    explanations = []
    
    # Process in batches for memory efficiency
    batch_size = 100
    for i in range(0, len(X_batch), batch_size):
        batch = X_batch[i:i + batch_size]
        
        batch_explanations = []
        for j, X_single in enumerate(batch):
            if i + j % 10 == 0:
                print(f"Processing explanation {i + j + 1}/{len(X_batch)}")
            
            explanation = explainer.explain(X_single.reshape(1, -1), max_features)
            batch_explanations.append(explanation)
        
        explanations.extend(batch_explanations)
    
    return explanations

# Process large batch of explanations
large_X_test = np.random.randn(1000, 10)
batch_explanations = batch_explain(explainer, large_X_test)

print(f"Generated {len(batch_explanations)} explanations")

# Aggregate explanation statistics
confidence_scores = [exp['explanation_confidence'] for exp in batch_explanations]
feature_counts = [len(exp['feature_attributions']) for exp in batch_explanations]

print(f"Average confidence: {np.mean(confidence_scores):.3f}")
print(f"Average features explained: {np.mean(feature_counts):.1f}")
```

## Best Practices

### 1. Method Selection Guidelines

```python
def select_explanation_method(model, use_case, performance_requirements):
    """Select optimal explanation method based on context."""
    
    # High-stakes applications require highest confidence
    if use_case in ["medical", "legal", "financial"]:
        if hasattr(model, "feature_importances_"):
            return ExplanationMethod.SHAP_TREE
        elif hasattr(model, "coef_"):
            return ExplanationMethod.SHAP_LINEAR
        else:
            return ExplanationMethod.SHAP_KERNEL
    
    # Performance-critical applications
    elif performance_requirements == "real_time":
        if hasattr(model, "feature_importances_"):
            return ExplanationMethod.FEATURE_IMPORTANCE
        else:
            return ExplanationMethod.SHAP_LINEAR
    
    # General applications
    else:
        return ExplanationMethod.SHAP_KERNEL
```

### 2. Explanation Validation

```python
def validate_explanation_quality(explainer, X_test, y_test):
    """Validate explanation quality and consistency."""
    
    explanations = []
    for i in range(min(100, len(X_test))):  # Sample for validation
        explanation = explainer.explain(X_test[i:i+1])
        explanations.append(explanation)
    
    # Check explanation consistency
    confidence_scores = [exp['explanation_confidence'] for exp in explanations]
    avg_confidence = np.mean(confidence_scores)
    
    # Check feature attribution consistency
    feature_importance_variance = {}
    for explanation in explanations:
        for attr in explanation['feature_attributions']:
            feature_name = attr['feature_name']
            if feature_name not in feature_importance_variance:
                feature_importance_variance[feature_name] = []
            feature_importance_variance[feature_name].append(attr['attribution_value'])
    
    # Calculate consistency metrics
    consistency_scores = {}
    for feature, values in feature_importance_variance.items():
        consistency_scores[feature] = 1.0 - (np.std(values) / (np.mean(np.abs(values)) + 1e-6))
    
    return {
        "average_confidence": avg_confidence,
        "explanation_count": len(explanations),
        "feature_consistency": consistency_scores,
        "overall_consistency": np.mean(list(consistency_scores.values()))
    }
```

### 3. User-Friendly Explanations

```python
def format_explanation_for_user(explanation, domain="general"):
    """Format technical explanations for end users."""
    
    formatted = {
        "summary": f"Prediction confidence: {explanation['explanation_confidence']:.1%}",
        "key_factors": [],
        "technical_details": explanation
    }
    
    # Convert technical attributions to user-friendly language
    for attr in explanation['feature_attributions'][:3]:  # Top 3 factors
        feature_name = attr['feature_name']
        attribution = attr['attribution_value']
        
        if domain == "medical":
            impact = "increases diagnosis likelihood" if attribution > 0 else "decreases diagnosis likelihood"
            explanation_text = f"{feature_name.replace('_', ' ').title()} {impact}"
        elif domain == "financial":
            impact = "increases approval chances" if attribution < 0 else "decreases approval chances"
            explanation_text = f"{feature_name.replace('_', ' ').title()} {impact}"
        else:
            impact = "supports" if attribution > 0 else "opposes"
            explanation_text = f"{feature_name.replace('_', ' ').title()} {impact} the prediction"
        
        formatted["key_factors"].append({
            "factor": feature_name.replace('_', ' ').title(),
            "impact": explanation_text,
            "strength": abs(attribution)
        })
    
    return formatted

# Example usage
user_explanation = format_explanation_for_user(explanation, domain="medical")
print("User-Friendly Explanation:")
print(user_explanation["summary"])
print("Key factors:")
for factor in user_explanation["key_factors"]:
    print(f"  • {factor['impact']}")
```

## Contributing

When extending the explainability package:

1. **Scientific Rigor** — Ensure explanation methods are theoretically sound
2. **Regulatory Compliance** — Support emerging explainability requirements
3. **Performance Optimization** — Consider production deployment needs
4. **User Experience** — Make explanations accessible to non-technical users
5. **Validation Framework** — Include explanation quality assessment tools

## Dependencies

The explainability package depends on:
- `numpy` — Numerical operations for feature attributions
- `typing` — Type hints for better code clarity
- `datetime` — Timestamp generation for explanations
- `json` — Serialization for explanation data
- `warnings` — Warning management for optional dependencies
- `shap` — (Optional) SHAP explanations
- `lime` — (Optional) LIME explanations

---

*For integration examples and advanced patterns, see the [examples folder](../examples/) and [API documentation](../api/).*