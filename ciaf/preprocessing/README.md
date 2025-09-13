# CIAF Preprocessing Module

The `preprocessing` module provides comprehensive data preprocessing and vectorization capabilities for the Cognitive Insight Audit Framework (CIAF). This module ensures real ML model training with proper feature extraction for text, numerical, and mixed data types while maintaining CIAF's compliance and auditability standards.

## 📋 Overview

This module bridges the gap between raw data and ML models by providing automatic data type detection, feature extraction, and model adaptation capabilities. It supports both text and numerical data processing with seamless integration into CIAF's auditing and compliance ecosystem.

## 🏗️ Architecture

```
preprocessing/
├── __init__.py              # Core preprocessing classes and utilities
└── __pycache__/            # Compiled Python files
```

## 🔧 Core Components

### CIAFPreprocessor (Abstract Base Class)
Base class for all CIAF preprocessing components with standardized interface:
- `fit()` - Fit preprocessor on training data
- `transform()` - Transform data to numerical format  
- `fit_transform()` - Combined fit and transform operation

### TextVectorizer
Advanced text preprocessing with TF-IDF and Count Vectorization:

```python
from ciaf.preprocessing import TextVectorizer

# Initialize with TF-IDF (default)
vectorizer = TextVectorizer(
    method="tfidf",
    max_features=1000,
    ngram_range=(1, 2),
    stop_words="english"
)

# Fit and transform training data
training_data = [
    {"content": "This is positive feedback", "metadata": {"target": 1}},
    {"content": "This is negative feedback", "metadata": {"target": 0}}
]
X_train = vectorizer.fit_transform(training_data)

# Transform new data
new_text = "This is a new document"
X_new = vectorizer.transform(new_text)

# Get feature names for explainability
features = vectorizer.get_feature_names()
```

### NumericalPreprocessor
Standardized numerical data preprocessing with scaling and normalization:

```python
from ciaf.preprocessing import NumericalPreprocessor

# Initialize with normalization
preprocessor = NumericalPreprocessor(normalize=True)

# Process numerical training data
numerical_data = [
    {"content": [1.5, 2.3, 0.8], "metadata": {"target": 1}},
    {"content": [0.2, 1.8, 2.1], "metadata": {"target": 0}}
]
X_processed = preprocessor.fit_transform(numerical_data)

# Transform new numerical data
new_data = [1.0, 2.0, 1.5]
X_new = preprocessor.transform(new_data)
```

### MixedDataPreprocessor
Intelligent preprocessing for mixed text and numerical data with auto-detection:

```python
from ciaf.preprocessing import MixedDataPreprocessor

# Auto-detecting preprocessor
preprocessor = MixedDataPreprocessor(
    text_method="tfidf",
    normalize_numerical=True,
    auto_detect=True
)

# Mixed training data
mixed_data = [
    {"content": "Positive feedback text", "metadata": {"target": 1}},
    {"content": [1.5, 2.3], "metadata": {"target": 0}}
]

# Automatically detects and processes appropriately
X_processed = preprocessor.fit_transform(mixed_data)

# Get feature information
feature_info = preprocessor.get_feature_info()
print(f"Data type: {feature_info['type']}")
print(f"Features: {feature_info['features']}")
```

### CIAFModelAdapter
Complete ML model integration with automatic preprocessing and CIAF compliance:

```python
from ciaf.preprocessing import CIAFModelAdapter
from sklearn.linear_model import LogisticRegression

# Create model with automatic preprocessing
model = LogisticRegression()
adapter = CIAFModelAdapter(
    model=model,
    auto_preprocess=True
)

# Training data in CIAF format
training_data = [
    {"content": "Great product quality", "metadata": {"target": 1}},
    {"content": "Poor customer service", "metadata": {"target": 0}},
    {"content": "Excellent value for money", "metadata": {"target": 1}}
]

# Fit with automatic preprocessing
adapter.fit(training_data)

# Make predictions
prediction = adapter.predict("Amazing experience!")
print(f"Prediction: {prediction}")

# Get preprocessing metadata
preprocessing_info = adapter.get_preprocessing_info()
```

## 🚀 Quick Start Examples

### Basic Text Classification Pipeline
```python
from ciaf.preprocessing import create_text_classifier_adapter
from sklearn.naive_bayes import MultinomialNB

# Create text classifier with TF-IDF
model = MultinomialNB()
adapter = create_text_classifier_adapter(model, method="tfidf")

# Training data
data = [
    {"content": "I love this product", "metadata": {"target": "positive"}},
    {"content": "This is terrible", "metadata": {"target": "negative"}},
    {"content": "Best purchase ever", "metadata": {"target": "positive"}}
]

# Train and predict
adapter.fit(data)
result = adapter.predict("This is amazing!")
```

### Numerical Regression Pipeline
```python
from ciaf.preprocessing import create_numerical_regressor_adapter
from sklearn.linear_model import LinearRegression

# Create numerical regressor
model = LinearRegression()
adapter = create_numerical_regressor_adapter(model, normalize=True)

# Numerical training data
data = [
    {"content": [1.0, 2.0, 3.0], "metadata": {"target": 10.5}},
    {"content": [2.0, 3.0, 4.0], "metadata": {"target": 15.2}},
    {"content": [0.5, 1.5, 2.5], "metadata": {"target": 8.1}}
]

# Train and predict
adapter.fit(data)
result = adapter.predict([1.5, 2.5, 3.5])
```

### Auto-Detection Pipeline
```python
from ciaf.preprocessing import create_auto_adapter
from sklearn.ensemble import RandomForestClassifier

# Create auto-detecting adapter
model = RandomForestClassifier()
adapter = create_auto_adapter(model)

# Mixed data - adapter will auto-detect type
mixed_data = [
    {"content": "Text data sample", "metadata": {"target": 1}},
    {"content": "Another text example", "metadata": {"target": 0}}
]

# Automatic preprocessing and training
adapter.fit(mixed_data)
prediction = adapter.predict("New text sample")
```

## 🔧 Utility Functions

### auto_preprocess_data()
Standalone preprocessing function for direct ML model integration:

```python
from ciaf.preprocessing import auto_preprocess_data

# Prepare data
X = ["positive text", "negative text", "neutral text"]
y = [1, 0, 1]

# Auto-preprocess
X_processed, y_processed = auto_preprocess_data(X, y)

# Use with any scikit-learn model
from sklearn.svm import SVC
model = SVC()
model.fit(X_processed, y_processed)
```

## 🔗 Integration with CIAF Ecosystem

### LCM Integration
```python
from ciaf.lcm import TrainingManager
from ciaf.preprocessing import CIAFModelAdapter

# Create training manager with preprocessing
training_manager = TrainingManager()

# Model with preprocessing
adapter = CIAFModelAdapter(model, auto_preprocess=True)

# Register with LCM for lifecycle management
training_manager.register_model("text_classifier", adapter)
```

### Compliance Integration
```python
from ciaf.compliance import AuditTrailGenerator
from ciaf.preprocessing import TextVectorizer

# Create auditable preprocessing pipeline
vectorizer = TextVectorizer()
audit_trail = AuditTrailGenerator()

# Log preprocessing operations
audit_trail.log_preprocessing_operation(
    operation="text_vectorization",
    parameters={"method": "tfidf", "max_features": 1000},
    input_hash="data_hash_123"
)
```

### Inference Integration
```python
from ciaf.inference import InferenceReceipt
from ciaf.preprocessing import CIAFModelAdapter

# Create auditable inference pipeline
adapter = CIAFModelAdapter(model, auto_preprocess=True)

# Generate inference receipt with preprocessing metadata
receipt = InferenceReceipt(
    model_id="preprocessed_model_v1",
    input_data="New text to classify",
    preprocessing_info=adapter.get_preprocessing_info()
)
```

## 🔒 Security and Compliance Features

### Data Privacy Protection
- **Secure Processing**: All preprocessing operations maintain data integrity
- **Memory Management**: Automatic cleanup of sensitive intermediate data
- **Audit Logging**: Complete preprocessing operation logging

### Regulatory Compliance
- **GDPR**: Data minimization through feature selection
- **NIST AI RMF**: Traceable preprocessing operations
- **EU AI Act**: Transparent feature extraction methods

### Cryptographic Integration
```python
from ciaf.core import CryptoManager
from ciaf.preprocessing import TextVectorizer

# Secure preprocessing with encryption
crypto_manager = CryptoManager()
vectorizer = TextVectorizer()

# Encrypt preprocessed features
features = vectorizer.fit_transform(data)
encrypted_features = crypto_manager.encrypt_data(features.tobytes())
```

## 📊 Performance and Scalability

### Optimization Features
- **Lazy Loading**: Memory-efficient processing for large datasets
- **Batch Processing**: Efficient vectorization for multiple documents
- **Feature Caching**: Reuse of fitted preprocessors across sessions

### Scalability Patterns
```python
# Memory-efficient processing for large datasets
def process_large_dataset(data_stream, batch_size=1000):
    adapter = CIAFModelAdapter(model, auto_preprocess=True)
    
    for batch in data_stream.batches(batch_size):
        processed_batch = adapter.preprocessor.transform(batch)
        yield processed_batch
```

## 🧪 Testing and Validation

### Unit Testing
```python
# Test preprocessing pipeline
def test_text_vectorizer():
    vectorizer = TextVectorizer()
    data = [{"content": "test text", "metadata": {"target": 1}}]
    
    # Test fit and transform
    result = vectorizer.fit_transform(data)
    assert result.shape[0] == len(data)
    
    # Test feature names
    features = vectorizer.get_feature_names()
    assert len(features) > 0
```

### Integration Testing
```python
# Test full preprocessing pipeline
def test_ciaf_model_adapter():
    from sklearn.dummy import DummyClassifier
    
    model = DummyClassifier()
    adapter = CIAFModelAdapter(model, auto_preprocess=True)
    
    data = [
        {"content": "positive", "metadata": {"target": 1}},
        {"content": "negative", "metadata": {"target": 0}}
    ]
    
    adapter.fit(data)
    prediction = adapter.predict("test")
    assert prediction is not None
```

## 🔮 Advanced Usage

### Custom Preprocessor Development
```python
from ciaf.preprocessing import CIAFPreprocessor
import numpy as np

class CustomPreprocessor(CIAFPreprocessor):
    def __init__(self, custom_param=None):
        self.custom_param = custom_param
        self.is_fitted = False
    
    def fit(self, data):
        # Custom fitting logic
        self.is_fitted = True
        return self
    
    def transform(self, data):
        # Custom transformation logic
        return np.array(data)
    
    def fit_transform(self, data):
        return self.fit(data).transform(data)
```

### Pipeline Composition
```python
# Compose multiple preprocessing steps
class CompositePreprocessor(CIAFPreprocessor):
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors
    
    def fit(self, data):
        for preprocessor in self.preprocessors:
            data = preprocessor.fit_transform(data)
        return self
    
    def transform(self, data):
        for preprocessor in self.preprocessors:
            data = preprocessor.transform(data)
        return data
```

## 🤝 Contributing

When contributing to the preprocessing module:

1. **Maintain Interface Compatibility**: All preprocessors must inherit from `CIAFPreprocessor`
2. **Add Comprehensive Tests**: Include unit and integration tests
3. **Document Feature Extraction**: Provide explainability for new preprocessing methods
4. **Consider Privacy**: Implement data protection measures for sensitive data
5. **Validate Performance**: Benchmark against existing methods

## 📚 Related Documentation

- [CIAF Core Framework](../api/README.md) - Main framework integration
- [LCM System](../lcm/README.md) - Lifecycle management integration  
- [Compliance Engine](../compliance/README.md) - Regulatory compliance features
- [Model Wrappers](../wrappers/README.md) - Enhanced model integration
- [Inference System](../inference/README.md) - Auditable prediction pipelines

---

The preprocessing module ensures that all data transformation operations within CIAF maintain the highest standards of auditability, compliance, and performance while providing seamless integration with popular ML frameworks and libraries.