# CIAF Model Wrappers

The wrappers package provides drop-in solutions for integrating existing ML models with CIAF's comprehensive audit and compliance framework.

## Overview

The wrappers system enables seamless integration of existing machine learning models with CIAF capabilities:

- **Drop-in Integration** — Minimal code changes to existing ML workflows
- **Automatic Audit Trails** — Transparent generation of provenance and receipts
- **Compliance Integration** — Built-in regulatory compliance (HIPAA, GDPR, EU AI Act)
- **Enhanced Features** — Explainability, uncertainty quantification, metadata tags
- **Multi-Framework Support** — Works with scikit-learn, PyTorch, TensorFlow, and custom models
- **Performance Optimization** — Efficient processing with lazy loading and caching

## Components

### CIAFModelWrapper (`model_wrapper.py`)

The primary wrapper class that transforms any ML model into a CIAF-compliant system.

**Key Features:**
- **Universal Compatibility** — Works with any model that has `fit()` and `predict()` methods
- **Automatic Provenance** — Creates provenance capsules for all training data
- **Training Snapshots** — Generates cryptographic training records
- **Inference Receipts** — Creates verifiable audit trails for all predictions
- **Enhanced AI Features** — Optional SHAP/LIME explanations, uncertainty quantification
- **Compliance Modes** — Pre-configured settings for healthcare, financial, and general use cases

## Basic Usage

### Simple Model Wrapping

```python
from ciaf.wrappers import CIAFModelWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Wrap an existing scikit-learn model
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
wrapped_model = CIAFModelWrapper(
    model=base_model,
    model_name="MedicalDiagnosticModel",
    compliance_mode="healthcare"  # Pre-configured for HIPAA compliance
)

# Prepare training data in CIAF format
training_data = [
    {
        "content": "Patient symptoms: fever, cough, fatigue",
        "metadata": {
            "id": "patient_001",
            "source": "hospital_alpha",
            "consent_status": "obtained",
            "target": "pneumonia"  # For supervised learning
        }
    },
    {
        "content": "Patient symptoms: chest pain, shortness of breath",
        "metadata": {
            "id": "patient_002",
            "source": "clinic_beta", 
            "consent_status": "obtained",
            "target": "cardiac"
        }
    }
]

# Train with automatic CIAF integration
training_snapshot = wrapped_model.train(
    dataset_id="medical_training_v1",
    training_data=training_data,
    master_password="secure_medical_password",
    model_version="1.0.0"
)

print(f"Training snapshot: {training_snapshot.snapshot_id}")

# Make predictions with audit trails
prediction, receipt = wrapped_model.predict(
    query="Patient presents with persistent cough and fever"
)

print(f"Prediction: {prediction}")
print(f"Receipt: {receipt.receipt_hash}")

# Verify inference integrity
verification = wrapped_model.verify(receipt)
print(f"Verification: {verification}")
```

### Enhanced Features Configuration

```python
# Enable all advanced features
enhanced_wrapper = CIAFModelWrapper(
    model=RandomForestClassifier(),
    model_name="EnhancedDiagnosticModel",
    compliance_mode="healthcare",
    enable_explainability=True,    # SHAP/LIME explanations
    enable_uncertainty=True,       # Uncertainty quantification
    enable_metadata_tags=True,     # CIAF metadata tags
    enable_preprocessing=True,     # Automatic vectorization
    enable_chaining=True          # Receipt chaining
)

# Training automatically includes enhanced features
training_snapshot = enhanced_wrapper.train(
    dataset_id="enhanced_medical_v1",
    training_data=medical_training_data,
    master_password="enhanced_password",
    training_params={
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5
    }
)

# Predictions include explainability and uncertainty
prediction, receipt = enhanced_wrapper.predict(
    query="Complex medical case with multiple symptoms"
)

# Access enhanced information
if hasattr(receipt, 'enhanced_info'):
    print(f"Explainability: {receipt.enhanced_info['explainability']}")
    print(f"Uncertainty: {receipt.enhanced_info['uncertainty']}")
    print(f"Metadata tag: {receipt.enhanced_info['metadata_tag']}")
```

## Advanced Use Cases

### Healthcare Compliance Workflow

Complete healthcare AI deployment with full regulatory compliance:

```python
from ciaf.wrappers import CIAFModelWrapper
from ciaf.compliance import ComplianceValidator, BiasValidator
from sklearn.ensemble import GradientBoostingClassifier

# Create HIPAA-compliant medical AI system
medical_model = GradientBoostingClassifier(n_estimators=100)
wrapped_medical = CIAFModelWrapper(
    model=medical_model,
    model_name="PneumoniaDetectionSystem",
    compliance_mode="healthcare",
    enable_explainability=True,
    enable_uncertainty=True
)

# Prepare anonymized medical training data
medical_training = [
    {
        "content": "Chest X-ray findings: bilateral infiltrates, consolidation",
        "metadata": {
            "id": "case_001",
            "source": "radiology_dept_alpha",
            "consent_status": "explicit_written_consent",
            "consent_date": "2025-09-01",
            "patient_age_group": "45-65",
            "target": "pneumonia_positive",
            "phi_removed": True,
            "anonymization_method": "k_anonymity_5"
        }
    },
    {
        "content": "Chest X-ray findings: clear lung fields, normal cardiac silhouette",
        "metadata": {
            "id": "case_002",
            "source": "radiology_dept_beta",
            "consent_status": "explicit_written_consent",
            "consent_date": "2025-09-01",
            "patient_age_group": "25-45",
            "target": "normal",
            "phi_removed": True,
            "anonymization_method": "k_anonymity_5"
        }
    }
]

# Train with HIPAA compliance
training_snapshot = wrapped_medical.train(
    dataset_id="pneumonia_training_hipaa_v1",
    training_data=medical_training,
    master_password="hipaa_compliant_password",
    training_params={
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "min_samples_leaf": 10  # Prevent overfitting on small patient groups
    },
    model_version="1.0.0_hipaa"
)

# Validate compliance
compliance_validator = ComplianceValidator("pneumonia_detection")
compliance_report = compliance_validator.validate_training_snapshot(
    snapshot=training_snapshot,
    frameworks=["HIPAA", "FDA_21CFR11"]
)

# Check for bias
bias_validator = BiasValidator("medical_bias_check")
bias_report = bias_validator.assess_training_data_bias(
    provenance_hashes=training_snapshot.provenance_capsule_hashes,
    protected_attributes=["age_group", "gender", "ethnicity"]
)

print(f"HIPAA Compliance: {compliance_report['hipaa_compliant']}")
print(f"Bias detected: {bias_report['bias_detected']}")

# Clinical inference with full audit trail
clinical_query = "Patient chest X-ray shows bilateral lower lobe opacities"
diagnosis, receipt = wrapped_medical.predict(
    query=clinical_query,
    model_version="1.0.0_hipaa"
)

# Access clinical decision support information
if hasattr(receipt, 'enhanced_info'):
    explainability = receipt.enhanced_info['explainability']
    uncertainty = receipt.enhanced_info['uncertainty']
    
    print(f"Diagnosis: {diagnosis}")
    print(f"Confidence: {explainability['confidence']}")
    print(f"Key features: {explainability['top_features']}")
    print(f"Uncertainty: {uncertainty['total_uncertainty']}")
    print(f"EU AI Act compliant: {explainability['eu_ai_act_compliant']}")

# Generate clinical audit report
audit_report = {
    "clinical_system": "PneumoniaDetectionSystem",
    "diagnosis_receipt": receipt.receipt_hash,
    "training_snapshot": training_snapshot.snapshot_id,
    "compliance_validation": compliance_report,
    "bias_assessment": bias_report,
    "clinical_user": "radiologist_alice",
    "timestamp": receipt.timestamp,
    "hipaa_audit_trail": "complete"
}

print(f"Clinical audit report generated: {audit_report}")
```

### Financial Services Compliance

Financial AI system with SOX and regulatory compliance:

```python
from ciaf.wrappers import CIAFModelWrapper
from sklearn.linear_model import LogisticRegression

# Create financial compliance model
financial_model = LogisticRegression(max_iter=1000)
wrapped_financial = CIAFModelWrapper(
    model=financial_model,
    model_name="FraudDetectionSystem",
    compliance_mode="financial",
    enable_explainability=True,
    enable_uncertainty=True,
    enable_chaining=True  # Important for financial audit trails
)

# Prepare financial training data
financial_training = [
    {
        "content": "Transaction: $500 grocery purchase, normal pattern",
        "metadata": {
            "id": "txn_001",
            "source": "transaction_processor_alpha",
            "compliance_frameworks": ["SOX", "GDPR"],
            "data_classification": "financial_data",
            "target": "legitimate",
            "risk_score": "low",
            "geographical_region": "US_domestic"
        }
    },
    {
        "content": "Transaction: $50000 international wire, unusual pattern",
        "metadata": {
            "id": "txn_002",
            "source": "wire_transfer_system",
            "compliance_frameworks": ["SOX", "AML", "OFAC"],
            "data_classification": "high_value_financial",
            "target": "suspicious",
            "risk_score": "high",
            "geographical_region": "international"
        }
    }
]

# Train financial model
financial_snapshot = wrapped_financial.train(
    dataset_id="fraud_detection_v1",
    training_data=financial_training,
    master_password="financial_regulatory_password",
    training_params={
        "C": 1.0,
        "solver": "liblinear",
        "penalty": "l2",
        "class_weight": "balanced"  # Handle imbalanced fraud data
    },
    model_version="1.0.0_sox"
)

# Real-time fraud detection with audit
suspicious_transaction = "Large cash withdrawal $9500 at ATM, unusual location"
fraud_score, receipt = wrapped_financial.predict(
    query=suspicious_transaction,
    model_version="1.0.0_sox"
)

# Financial regulatory reporting
regulatory_report = {
    "system": "FraudDetectionSystem",
    "transaction_analysis": fraud_score,
    "audit_receipt": receipt.receipt_hash,
    "training_provenance": financial_snapshot.snapshot_id,
    "sox_compliance": True,
    "explainable_decision": receipt.enhanced_info.get('explainability', {}),
    "confidence_metrics": receipt.enhanced_info.get('uncertainty', {}),
    "regulatory_timestamp": receipt.timestamp
}
```

### Multi-Model Integration

Integrating multiple models with shared CIAF infrastructure:

```python
from ciaf.wrappers import CIAFModelWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

class MultiModelMedicalSystem:
    def __init__(self):
        # Create multiple specialized models
        self.diagnostic_model = CIAFModelWrapper(
            model=RandomForestClassifier(n_estimators=100),
            model_name="DiagnosticClassifier",
            compliance_mode="healthcare",
            enable_explainability=True
        )
        
        self.severity_model = CIAFModelWrapper(
            model=LinearRegression(),
            model_name="SeverityPredictor", 
            compliance_mode="healthcare",
            enable_uncertainty=True
        )
        
        self.treatment_model = CIAFModelWrapper(
            model=SVC(probability=True),
            model_name="TreatmentRecommender",
            compliance_mode="healthcare",
            enable_explainability=True,
            enable_uncertainty=True
        )
    
    def train_all_models(self, diagnostic_data, severity_data, treatment_data):
        # Train diagnostic model
        diagnostic_snapshot = self.diagnostic_model.train(
            dataset_id="diagnostic_training_v1",
            training_data=diagnostic_data,
            master_password="medical_password_diagnostic",
            model_version="1.0.0"
        )
        
        # Train severity model
        severity_snapshot = self.severity_model.train(
            dataset_id="severity_training_v1",
            training_data=severity_data,
            master_password="medical_password_severity",
            model_version="1.0.0"
        )
        
        # Train treatment model
        treatment_snapshot = self.treatment_model.train(
            dataset_id="treatment_training_v1",
            training_data=treatment_data,
            master_password="medical_password_treatment",
            model_version="1.0.0"
        )
        
        return {
            "diagnostic": diagnostic_snapshot,
            "severity": severity_snapshot,
            "treatment": treatment_snapshot
        }
    
    def comprehensive_analysis(self, patient_case):
        # Run multi-model analysis with full audit trail
        diagnostic_result, diagnostic_receipt = self.diagnostic_model.predict(
            query=patient_case
        )
        
        severity_result, severity_receipt = self.severity_model.predict(
            query=patient_case
        )
        
        treatment_result, treatment_receipt = self.treatment_model.predict(
            query=patient_case
        )
        
        # Create comprehensive analysis report
        analysis_report = {
            "patient_case": patient_case,
            "diagnostic_prediction": diagnostic_result,
            "severity_prediction": severity_result,
            "treatment_recommendation": treatment_result,
            "audit_receipts": {
                "diagnostic": diagnostic_receipt.receipt_hash,
                "severity": severity_receipt.receipt_hash,
                "treatment": treatment_receipt.receipt_hash
            },
            "confidence_scores": {
                "diagnostic": diagnostic_receipt.enhanced_info.get('uncertainty', {}).get('confidence_level', 'N/A'),
                "severity": severity_receipt.enhanced_info.get('uncertainty', {}).get('confidence_level', 'N/A'),
                "treatment": treatment_receipt.enhanced_info.get('uncertainty', {}).get('confidence_level', 'N/A')
            },
            "explainability": {
                "diagnostic_features": diagnostic_receipt.enhanced_info.get('explainability', {}).get('top_features', []),
                "treatment_factors": treatment_receipt.enhanced_info.get('explainability', {}).get('top_features', [])
            },
            "regulatory_compliance": "HIPAA_compliant",
            "timestamp": diagnostic_receipt.timestamp
        }
        
        return analysis_report

# Example usage
medical_system = MultiModelMedicalSystem()

# Prepare training data for each model
diagnostic_training = [
    {"content": "Symptoms: fever, cough", "metadata": {"id": "d1", "target": "pneumonia"}},
    {"content": "Symptoms: chest pain", "metadata": {"id": "d2", "target": "cardiac"}}
]

severity_training = [
    {"content": "Pneumonia case, stable vitals", "metadata": {"id": "s1", "target": 3.2}},
    {"content": "Cardiac event, elevated troponin", "metadata": {"id": "s2", "target": 7.8}}
]

treatment_training = [
    {"content": "Mild pneumonia", "metadata": {"id": "t1", "target": "antibiotics"}},
    {"content": "Acute MI", "metadata": {"id": "t2", "target": "emergency_intervention"}}
]

# Train all models
snapshots = medical_system.train_all_models(
    diagnostic_training, severity_training, treatment_training
)

# Perform comprehensive analysis
patient_case = "65-year-old patient with chest pain, elevated troponin, ECG changes"
analysis = medical_system.comprehensive_analysis(patient_case)

print(f"Comprehensive analysis: {analysis}")
```

## Performance Optimization

### Efficient Batch Processing

Handle large-scale model deployment efficiently:

```python
from ciaf.wrappers import CIAFModelWrapper
from concurrent.futures import ThreadPoolExecutor
import time

class OptimizedCIAFDeployment:
    def __init__(self, model, model_name, batch_size=1000):
        self.wrapper = CIAFModelWrapper(
            model=model,
            model_name=model_name,
            compliance_mode="general",
            enable_preprocessing=True,
            enable_chaining=False  # Disable for better batch performance
        )
        self.batch_size = batch_size
        self.prediction_cache = {}
    
    def batch_train(self, large_dataset, master_password):
        # Process training data in batches for memory efficiency
        print(f"Training on {len(large_dataset)} samples in batches of {self.batch_size}")
        
        start_time = time.time()
        
        # Split into batches
        batches = [
            large_dataset[i:i + self.batch_size]
            for i in range(0, len(large_dataset), self.batch_size)
        ]
        
        # Train on all data at once (CIAF handles memory efficiently)
        training_snapshot = self.wrapper.train(
            dataset_id=f"large_dataset_{len(large_dataset)}",
            training_data=large_dataset,
            master_password=master_password,
            model_version="1.0.0_optimized"
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Throughput: {len(large_dataset) / training_time:.2f} samples/second")
        
        return training_snapshot
    
    def batch_predict(self, queries, use_cache=True):
        # Efficient batch prediction with optional caching
        predictions = []
        receipts = []
        
        start_time = time.time()
        
        for i, query in enumerate(queries):
            if i % 100 == 0:
                print(f"Processing query {i + 1}/{len(queries)}")
            
            # Check cache first
            cache_key = hash(str(query))
            if use_cache and cache_key in self.prediction_cache:
                pred, receipt = self.prediction_cache[cache_key]
            else:
                pred, receipt = self.wrapper.predict(query)
                if use_cache:
                    self.prediction_cache[cache_key] = (pred, receipt)
            
            predictions.append(pred)
            receipts.append(receipt)
        
        prediction_time = time.time() - start_time
        print(f"Batch prediction completed in {prediction_time:.2f} seconds")
        print(f"Throughput: {len(queries) / prediction_time:.2f} predictions/second")
        
        return predictions, receipts
    
    def parallel_predict(self, queries, max_workers=4):
        # Parallel prediction processing
        def predict_single(query):
            return self.wrapper.predict(query)
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(predict_single, queries))
        
        parallel_time = time.time() - start_time
        predictions = [r[0] for r in results]
        receipts = [r[1] for r in results]
        
        print(f"Parallel prediction completed in {parallel_time:.2f} seconds")
        print(f"Throughput: {len(queries) / parallel_time:.2f} predictions/second")
        
        return predictions, receipts

# Example usage
from sklearn.ensemble import RandomForestClassifier

optimized_deployment = OptimizedCIAFDeployment(
    model=RandomForestClassifier(n_estimators=50),
    model_name="OptimizedModel",
    batch_size=1000
)

# Generate large synthetic dataset
large_training_data = [
    {
        "content": f"Sample training data {i}",
        "metadata": {"id": f"sample_{i}", "target": i % 2}
    }
    for i in range(10000)
]

# Efficient training
snapshot = optimized_deployment.batch_train(
    large_training_data, 
    "optimized_password"
)

# Efficient prediction
test_queries = [f"Test query {i}" for i in range(1000)]
predictions, receipts = optimized_deployment.batch_predict(test_queries)

print(f"Processed {len(predictions)} predictions with full CIAF audit trails")
```

## Integration Patterns

### Framework Integration

```python
# Integration with popular ML frameworks
from ciaf.wrappers import CIAFModelWrapper

# Scikit-learn integration
from sklearn.ensemble import RandomForestClassifier
sklearn_wrapper = CIAFModelWrapper(
    model=RandomForestClassifier(),
    model_name="SklearnModel",
    auto_configure=True
)

# PyTorch integration (if available)
try:
    import torch
    import torch.nn as nn
    
    class SimplePyTorchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
        
        def fit(self, X, y):
            # Custom training logic
            pass
        
        def predict(self, X):
            with torch.no_grad():
                return self.forward(torch.tensor(X, dtype=torch.float32))
    
    pytorch_wrapper = CIAFModelWrapper(
        model=SimplePyTorchModel(),
        model_name="PyTorchModel",
        auto_configure=True
    )
except ImportError:
    print("PyTorch not available")

# TensorFlow/Keras integration (if available)
try:
    from tensorflow import keras
    
    class CIAFKerasModel:
        def __init__(self):
            self.model = keras.Sequential([
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            self.model.compile(optimizer='adam', loss='binary_crossentropy')
        
        def fit(self, X, y):
            self.model.fit(X, y, epochs=10, verbose=0)
        
        def predict(self, X):
            return self.model.predict(X)
    
    keras_wrapper = CIAFModelWrapper(
        model=CIAFKerasModel(),
        model_name="KerasModel",
        auto_configure=True
    )
except ImportError:
    print("TensorFlow not available")
```

### Custom Model Integration

```python
# Integration with custom models
class CustomAIModel:
    def __init__(self, model_params):
        self.params = model_params
        self.trained = False
    
    def fit(self, X, y):
        # Custom training logic
        print(f"Training custom model with {len(X)} samples")
        self.trained = True
    
    def predict(self, X):
        if not self.trained:
            raise RuntimeError("Model not trained")
        # Custom prediction logic
        return f"Custom prediction for: {X}"

# Wrap custom model
custom_wrapper = CIAFModelWrapper(
    model=CustomAIModel({"param1": "value1"}),
    model_name="CustomAISystem",
    compliance_mode="general",
    enable_explainability=True
)

# Use normally
custom_training_data = [
    {"content": "Custom data 1", "metadata": {"id": "c1", "target": "A"}},
    {"content": "Custom data 2", "metadata": {"id": "c2", "target": "B"}}
]

custom_snapshot = custom_wrapper.train(
    dataset_id="custom_dataset",
    training_data=custom_training_data,
    master_password="custom_password"
)

custom_prediction, custom_receipt = custom_wrapper.predict("Test custom query")
```

## Best Practices

### 1. Model Selection and Configuration

```python
# Choose appropriate wrapper configuration
def configure_wrapper_for_use_case(model, use_case):
    if use_case == "healthcare":
        return CIAFModelWrapper(
            model=model,
            model_name=f"Medical_{type(model).__name__}",
            compliance_mode="healthcare",
            enable_explainability=True,  # Required for medical decisions
            enable_uncertainty=True,     # Critical for patient safety
            enable_chaining=True,        # Full audit trail
            enable_metadata_tags=True    # Regulatory tagging
        )
    elif use_case == "finance":
        return CIAFModelWrapper(
            model=model,
            model_name=f"Financial_{type(model).__name__}",
            compliance_mode="financial",
            enable_explainability=True,  # Required for regulatory reporting
            enable_chaining=True,        # Audit trail for transactions
            enable_uncertainty=False,    # May not be required
            enable_metadata_tags=True    # Regulatory compliance
        )
    else:
        return CIAFModelWrapper(
            model=model,
            model_name=f"General_{type(model).__name__}",
            compliance_mode="general",
            enable_explainability=False,
            enable_uncertainty=False,
            enable_chaining=False,
            enable_metadata_tags=False
        )
```

### 2. Data Preparation

```python
# Standardized data preparation
def prepare_training_data(raw_data, data_source, compliance_mode):
    formatted_data = []
    
    for i, item in enumerate(raw_data):
        # Standard metadata
        metadata = {
            "id": f"{data_source}_{i:06d}",
            "source": data_source,
            "timestamp": datetime.now().isoformat(),
            "data_quality": "validated"
        }
        
        # Compliance-specific metadata
        if compliance_mode == "healthcare":
            metadata.update({
                "consent_status": "explicit_written_consent",
                "phi_removed": True,
                "hipaa_compliant": True
            })
        elif compliance_mode == "financial":
            metadata.update({
                "pii_handling": "anonymized",
                "sox_compliant": True,
                "aml_checked": True
            })
        
        # Add target if available
        if "target" in item:
            metadata["target"] = item["target"]
        
        formatted_data.append({
            "content": item["content"],
            "metadata": metadata
        })
    
    return formatted_data
```

### 3. Error Handling

```python
# Robust error handling
class RobustCIAFWrapper:
    def __init__(self, model, model_name, **kwargs):
        try:
            self.wrapper = CIAFModelWrapper(model, model_name, **kwargs)
        except Exception as e:
            print(f"Failed to initialize wrapper: {e}")
            raise
    
    def safe_train(self, dataset_id, training_data, master_password, **kwargs):
        try:
            return self.wrapper.train(dataset_id, training_data, master_password, **kwargs)
        except ValueError as e:
            print(f"Training data validation error: {e}")
            raise
        except RuntimeError as e:
            print(f"Training runtime error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected training error: {e}")
            raise
    
    def safe_predict(self, query, **kwargs):
        try:
            return self.wrapper.predict(query, **kwargs)
        except RuntimeError as e:
            print(f"Prediction error: {e}")
            # Return fallback prediction with receipt
            fallback_prediction = f"Fallback response for: {query}"
            fallback_receipt = self._create_fallback_receipt(query, fallback_prediction)
            return fallback_prediction, fallback_receipt
        except Exception as e:
            print(f"Unexpected prediction error: {e}")
            raise
    
    def _create_fallback_receipt(self, query, prediction):
        # Create minimal receipt for fallback scenarios
        from ciaf.inference import InferenceReceipt
        return InferenceReceipt(
            query=str(query),
            ai_output=str(prediction),
            model_version="fallback",
            training_snapshot_id="fallback",
            training_snapshot_merkle_root="fallback"
        )
```

## Contributing

When extending the wrappers package:

1. **Universal Compatibility** — Ensure wrappers work with diverse model types
2. **Performance Optimization** — Minimize overhead for production systems
3. **Compliance Focus** — Maintain regulatory compliance features
4. **Error Resilience** — Handle edge cases gracefully
5. **Documentation** — Provide clear integration examples

## Dependencies

The wrappers package depends on:
- `ciaf.api` — Core CIAF framework integration
- `ciaf.inference` — Inference receipt generation
- `ciaf.provenance` — Training provenance and snapshots
- `ciaf.preprocessing` — Optional enhanced preprocessing (if available)
- `ciaf.explainability` — Optional explainability features (if available)
- `ciaf.uncertainty` — Optional uncertainty quantification (if available)
- `ciaf.metadata_tags` — Optional metadata tagging (if available)
- `numpy` — Numerical operations
- `typing` — Type hints for better code clarity

---

*For integration examples and advanced patterns, see the [examples folder](../examples/) and [API documentation](../api/).*