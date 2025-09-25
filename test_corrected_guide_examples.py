#!/usr/bin/env python3
"""
Test the corrected MODEL_BUILDING_GUIDE examples to ensure they work
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_quick_start_example():
    """Test the updated quick start example from the guide."""
    print("🧪 Testing Quick Start Example from Corrected Guide")
    print("=" * 60)
    
    try:
        from ciaf import CIAFFramework
        from ciaf.lcm import LCMModelManager, LCMTrainingManager, LCMDatasetManager
        from ciaf.wrappers import EnhancedCIAFModelWrapper
        from ciaf.preprocessing import validate_ciaf_dataset  # NOW AVAILABLE
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 1. Initialize production framework
        framework = CIAFFramework("production_ml_project")
        model_manager = LCMModelManager()
        training_manager = LCMTrainingManager()
        dataset_manager = LCMDatasetManager()
        print("✅ Framework initialized")

        # 2. Create sample training data in CIAF format
        training_data = [
            {"content": "Excellent product with great quality", "metadata": {"target": 1}},
            {"content": "Poor service and low quality", "metadata": {"target": 0}},
            {"content": "Amazing experience, highly recommend", "metadata": {"target": 1}},
            {"content": "Disappointing purchase, not as described", "metadata": {"target": 0}},
            {"content": "Outstanding value for money", "metadata": {"target": 1}},
            {"content": "Terrible quality, waste of money", "metadata": {"target": 0}},
            {"content": "Perfect product, exceeded expectations", "metadata": {"target": 1}},
            {"content": "Poor build quality, broke quickly", "metadata": {"target": 0}},
            {"content": "Fantastic service and delivery", "metadata": {"target": 1}},
            {"content": "Unsatisfactory performance overall", "metadata": {"target": 0}},
        ]

        # 3. Validate data quality (NOW WORKING!)
        print("🔍 Validating data quality...")
        validation_result = validate_ciaf_dataset(training_data, min_samples=8, require_targets=True)

        if not validation_result.is_valid:
            raise ValueError(f"Data quality issues: {validation_result.errors}")

        quality_score = validation_result.metrics.get('quality_score', 0)
        print(f"✅ Data quality score: {quality_score}/100")

        # 4. Create production dataset anchor
        dataset_anchor = dataset_manager.simulate_dataset_anchor(
            dataset_id='production_dataset_v1',
            dataset_path='training_data.json',
            split_type='train'
        )
        print(f"✅ Dataset anchor created: {dataset_anchor.dataset_id}")

        # 5. Preprocess and train model
        texts = [item['content'] for item in training_data]
        labels = [item['metadata']['target'] for item in training_data]

        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X_vectorized = vectorizer.fit_transform(texts)
        X_dense = X_vectorized.toarray()

        # Create and train model
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }

        rf_model = RandomForestClassifier(**model_params)
        rf_model.fit(X_dense, labels)
        print("✅ Model trained successfully")

        # 6. Create CIAF wrapper
        ciaf_model = EnhancedCIAFModelWrapper(
            model=rf_model,
            model_name="production_classifier_v1",
            compliance_mode="enterprise"
        )

        # Store preprocessing for inference
        ciaf_model._vectorizer = vectorizer
        ciaf_model.is_trained = True
        print("✅ CIAF wrapper created")

        # 7. Create model anchor
        model_anchor = model_manager.create_model_anchor(
            model_id='production_classifier_v1',
            model_params=model_params
        )
        print(f"✅ Model anchor created: {model_anchor.anchor_id}")

        # 8. Create training session
        from ciaf.lcm import DatasetSplit
        training_session = training_manager.create_training_session(
            session_id='production_training_001',
            model_anchor=model_anchor,
            datasets_root_anchor=dataset_anchor.anchor_id,
            training_config=model_params,
            data_splits={DatasetSplit.TRAIN: dataset_anchor.anchor_id}
        )
        print(f"✅ Training session created: {training_session.session_id}")

        # 9. Test inference
        test_text = "This product is fantastic and works perfectly!"
        text_vectorized = vectorizer.transform([test_text])
        prediction = rf_model.predict(text_vectorized.toarray())[0]
        confidence = rf_model.predict_proba(text_vectorized.toarray())[0].max()

        print("\n🎉 Quick Start Example: COMPLETE SUCCESS!")
        print(f"📊 Dataset: {dataset_anchor.dataset_id}")
        print(f"🔗 Model: {model_anchor.anchor_id}")
        print(f"🏋️ Training: {training_session.session_id}")
        print(f"🔮 Test prediction: {prediction} (confidence: {confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick start example failed: {e}")
        return False

def test_enterprise_classifier_example():
    """Test the enterprise classifier example from the guide."""
    print("\n🧪 Testing Enterprise Classifier Example")
    print("=" * 60)
    
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from ciaf.preprocessing import DataQualityValidator, validate_ciaf_dataset
        from ciaf.wrappers import EnhancedCIAFModelWrapper
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Production-grade classifier with comprehensive data validation
        class ProductionClassifier:
            def __init__(self):
                # Initialize data quality validator (NOW WORKING)
                self.validator = DataQualityValidator(
                    min_samples=8,  # Reduced for demo
                    max_missing_ratio=0.2,
                    check_duplicates=True,
                    check_outliers=False  # Skip for text data
                )
                
                # Enterprise model configuration
                self.model_params = {
                    'n_estimators': 50,  # Reduced for demo
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
                
                self.base_model = GradientBoostingClassifier(**self.model_params)
                
            def train_with_validation(self, training_data):
                """Train model with comprehensive data quality validation."""
                
                # 1. Validate data quality before training
                print("🔍 Validating data quality...")
                validation_result = validate_ciaf_dataset(
                    training_data, 
                    min_samples=len(training_data)//2,  # At least half of data
                    require_targets=True
                )
                
                if not validation_result.is_valid:
                    raise ValueError(f"Data quality validation failed: {validation_result.errors}")
                
                if validation_result.warnings:
                    print("⚠️ Data quality warnings:")
                    for warning in validation_result.warnings:
                        print(f"  - {warning}")
                
                quality_score = validation_result.metrics.get('quality_score', 0)
                print(f"✅ Data quality score: {quality_score}/100")
                
                # 2. Extract and preprocess data
                texts = [item['content'] for item in training_data]
                labels = [item['metadata']['target'] for item in training_data]
                
                # 3. Vectorize text data
                vectorizer = TfidfVectorizer(
                    max_features=100,  # Reduced for demo
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                X_vectorized = vectorizer.fit_transform(texts)
                X_dense = X_vectorized.toarray()
                
                # 4. Train the model
                self.base_model.fit(X_dense, labels)
                
                # 5. Create CIAF wrapper
                ciaf_model = EnhancedCIAFModelWrapper(
                    model=self.base_model,
                    model_name="enterprise_classifier_v1",
                    compliance_mode="enterprise"
                )
                
                # Store preprocessing components for inference
                ciaf_model._vectorizer = vectorizer
                ciaf_model._quality_score = quality_score
                ciaf_model.is_trained = True
                
                print("✅ Model training completed with data validation")
                return ciaf_model
            
            def predict(self, ciaf_model, text_input):
                """Make prediction using the trained model."""
                if not hasattr(ciaf_model, '_vectorizer'):
                    raise ValueError("Model not properly trained or vectorizer missing")
                
                # Vectorize input text
                input_vectorized = ciaf_model._vectorizer.transform([text_input])
                input_dense = input_vectorized.toarray()
                
                # Make prediction
                prediction = ciaf_model.model.predict(input_dense)[0]
                confidence = ciaf_model.model.predict_proba(input_dense)[0].max()
                
                return prediction, confidence

        # Usage Example
        classifier = ProductionClassifier()

        # Training data in CIAF format
        training_data = [
            {"content": "Excellent product, highly recommended", "metadata": {"target": 1}},
            {"content": "Poor quality, not worth the money", "metadata": {"target": 0}},
            {"content": "Outstanding service and fast delivery", "metadata": {"target": 1}},
            {"content": "Disappointing experience, poor support", "metadata": {"target": 0}},
            {"content": "Amazing value for the price", "metadata": {"target": 1}},
            {"content": "Terrible build quality, broke immediately", "metadata": {"target": 0}},
            {"content": "Perfect product, exceeds expectations", "metadata": {"target": 1}},
            {"content": "Waste of money, completely useless", "metadata": {"target": 0}},
        ]

        # Train with validation
        ciaf_classifier = classifier.train_with_validation(training_data)

        # Make predictions
        prediction, confidence = classifier.predict(ciaf_classifier, "This product is amazing!")
        print(f"✅ Prediction: {prediction}, Confidence: {confidence:.2f}")
        
        print("\n🎉 Enterprise Classifier Example: COMPLETE SUCCESS!")
        return True
        
    except Exception as e:
        print(f"❌ Enterprise classifier example failed: {e}")
        return False

def main():
    """Run all guide tests."""
    print("🔬 Testing MODEL_BUILDING_GUIDE_V1_1_0_CORRECTED Examples")
    print("=" * 80)
    
    results = []
    
    # Test quick start example
    results.append(test_quick_start_example())
    
    # Test enterprise classifier example
    results.append(test_enterprise_classifier_example())
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    success_count = sum(results)
    total_tests = len(results)
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print(f"✅ {success_count}/{total_tests} examples working correctly")
        print("\n🏆 MODEL_BUILDING_GUIDE_V1_1_0_CORRECTED.md: FULLY VERIFIED")
        return True
    else:
        print(f"❌ {total_tests - success_count} tests failed")
        print(f"✅ {success_count}/{total_tests} examples working")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)