"""
Risk Assessment and Audit Example for CIAF

This example demonstrates how to use CIAF's risk assessment capabilities
to support compliance with EU AI Act Article 9 and similar risk management requirements.

Created: 2025-09-12
Author: Denzil James Greenwood
Version: 1.0.0
"""

from ciaf import CIAFFramework
from ciaf.compliance import (
    RiskAssessmentEngine, 
    BiasValidator, 
    UncertaintyQuantifier,
    ComplianceFramework
)
import numpy as np
from typing import Dict, List, Any

def demonstrate_risk_assessment():
    """
    Comprehensive risk assessment example using CIAF.
    
    This example shows how CIAF supports risk management requirements
    through provenance tracking, bias detection, and uncertainty quantification.
    """
    print("🛡️ CIAF Risk Assessment and Audit Example")
    print("=" * 50)
    
    # Initialize framework
    framework = CIAFFramework("HighRisk_AI_System")
    
    # Step 1: Create dataset anchor with risk metadata
    print("\n📊 Step 1: Dataset Risk Assessment")
    dataset_anchor = framework.create_dataset_anchor(
        dataset_id="medical_imaging_data",
        dataset_metadata={
            "source": "multi_hospital_consortium",
            "type": "medical_images",
            "risk_level": "high",
            "patient_population": "adult_radiology",
            "collection_period": "2020-2024",
            "bias_mitigation": {
                "demographic_balance": True,
                "hospital_diversity": True,
                "annotation_quality": "double_blind_review"
            },
            "known_limitations": [
                "Underrepresentation of rare conditions",
                "Geographic bias toward urban hospitals",
                "Age skew toward 30-70 years"
            ]
        },
        master_password="secure_medical_anchor_2024"
    )
    
    # Step 2: Create high-risk data items with consent tracking
    print("\n🏥 Step 2: Consent and Provenance Tracking")
    risk_data_items = [
        {
            "content": "chest_xray_001.dcm",
            "metadata": {
                "patient_id": "ANON_001",
                "consent_status": "explicit_research_consent",
                "data_sensitivity": "high",
                "diagnosis": "pneumonia",
                "demographics": {"age_group": "40-50", "gender": "F"},
                "institution": "hospital_a"
            }
        },
        {
            "content": "chest_xray_002.dcm", 
            "metadata": {
                "patient_id": "ANON_002",
                "consent_status": "explicit_research_consent",
                "data_sensitivity": "high",
                "diagnosis": "normal",
                "demographics": {"age_group": "60-70", "gender": "M"},
                "institution": "hospital_b"
            }
        }
    ]
    
    capsules = framework.create_provenance_capsules("medical_imaging_data", risk_data_items)
    
    # Step 3: Risk-aware model anchor creation
    print("\n🤖 Step 3: Model Risk Profile")
    model_anchor = framework.create_model_anchor(
        model_name="pneumonia_detection_ai",
        model_parameters={
            "architecture": "convolutional_neural_network",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "regularization": "dropout_0.5",
            "risk_profile": {
                "intended_use": "diagnostic_assistance",
                "risk_category": "high_risk_medical_device",
                "critical_performance_metrics": ["sensitivity", "specificity", "ppv", "npv"],
                "safety_thresholds": {
                    "min_sensitivity": 0.95,  # Critical for medical diagnosis
                    "min_specificity": 0.90,
                    "max_false_negative_rate": 0.05
                }
            }
        },
        model_architecture={
            "input_shape": [224, 224, 1],
            "layers": [
                {"type": "conv2d", "filters": 32, "kernel_size": 3},
                {"type": "maxpool", "pool_size": 2},
                {"type": "conv2d", "filters": 64, "kernel_size": 3},
                {"type": "maxpool", "pool_size": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dropout", "rate": 0.5},
                {"type": "dense", "units": 1, "activation": "sigmoid"}
            ]
        },
        authorized_datasets=["medical_imaging_data"],
        master_password="secure_model_anchor_2024"
    )
    
    # Step 4: Training with risk monitoring
    print("\n🎯 Step 4: Risk-Monitored Training")
    training_snapshot = framework.train_model(
        model_name="pneumonia_detection_ai",
        capsules=capsules,
        maa=model_anchor,
        training_params={
            "epochs": 100,
            "validation_split": 0.2,
            "early_stopping": True,
            "risk_monitoring": {
                "track_bias_metrics": True,
                "uncertainty_estimation": True,
                "performance_thresholds": model_anchor["model_parameters"]["risk_profile"]["safety_thresholds"]
            }
        },
        model_version="v1.0_risk_assessed"
    )
    
    # Step 5: Comprehensive risk assessment
    print("\n⚠️ Step 5: Post-Training Risk Assessment")
    
    # Simulate model predictions for risk assessment
    test_predictions = np.random.rand(100)  # Simulated probabilities
    test_labels = np.random.randint(0, 2, 100)  # Simulated ground truth
    protected_attributes = {
        "age_group": np.random.choice(["30-40", "40-50", "50-60", "60-70"], 100),
        "gender": np.random.choice(["M", "F"], 100),
        "institution": np.random.choice(["hospital_a", "hospital_b", "hospital_c"], 100)
    }
    
    # Bias assessment
    bias_validator = BiasValidator()
    bias_results = bias_validator.validate_predictions(
        predictions=test_predictions,
        protected_attributes=protected_attributes
    )
    
    # Uncertainty quantification
    uncertainty_quantifier = UncertaintyQuantifier()
    uncertainty_metrics = uncertainty_quantifier.calculate_metrics(
        predictions=test_predictions,
        labels=test_labels,
        method="entropy"
    )
    
    # Overall risk assessment
    risk_engine = RiskAssessmentEngine()
    risk_assessment = risk_engine.comprehensive_assessment(
        model_metadata=model_anchor,
        training_snapshot=training_snapshot,
        bias_results=bias_results,
        uncertainty_metrics=uncertainty_metrics,
        compliance_framework=ComplianceFramework.EU_AI_ACT
    )
    
    # Step 6: Risk-aware inference with audit trail
    print("\n🔍 Step 6: Risk-Monitored Inference")
    
    inference_receipt = framework.perform_inference_with_audit(
        model_name="pneumonia_detection_ai",
        query="new_chest_xray.dcm",
        ai_output={
            "prediction": "pneumonia_detected",
            "confidence": 0.87,
            "uncertainty_score": 0.15,
            "risk_factors": ["low_image_quality", "rare_presentation"]
        },
        training_snapshot=training_snapshot,
        user_id="radiologist_dr_smith",
        risk_metadata={
            "clinical_context": "emergency_department",
            "oversight_required": True,
            "decision_support_only": True
        }
    )
    
    # Step 7: Generate compliance documentation
    print("\n📋 Step 7: Risk Documentation Generation")
    
    # Generate risk management documentation
    risk_documentation = {
        "risk_assessment_summary": risk_assessment,
        "bias_analysis": bias_results,
        "uncertainty_analysis": uncertainty_metrics,
        "training_provenance": training_snapshot,
        "inference_audit_trail": inference_receipt,
        "compliance_status": {
            "eu_ai_act_article_9": "compliant_with_evidence",
            "risk_management_measures": "implemented",
            "bias_mitigation": "active_monitoring",
            "human_oversight": "required_and_logged"
        }
    }
    
    print(f"✅ Risk Assessment Complete!")
    print(f"   - Bias metrics calculated: {len(bias_results)} demographic groups")
    print(f"   - Uncertainty quantification: {uncertainty_metrics['mean_uncertainty']:.3f}")
    print(f"   - Risk level: {risk_assessment['overall_risk_level']}")
    print(f"   - Compliance status: {risk_documentation['compliance_status']['eu_ai_act_article_9']}")
    
    # Step 8: Audit trail verification
    print("\n🔐 Step 8: Audit Trail Verification")
    
    audit_trail = framework.get_complete_audit_trail("pneumonia_detection_ai")
    integrity_verified = framework.validate_training_integrity(training_snapshot)
    
    print(f"✅ Audit Trail Verification:")
    print(f"   - Training integrity: {'VERIFIED' if integrity_verified else 'FAILED'}")
    print(f"   - Total audit records: {audit_trail['verification']['total_audit_records']}")
    print(f"   - Inference receipts: {audit_trail['inference_chain']['total_receipts']}")
    print(f"   - Risk assessments: {len([r for r in audit_trail.get('risk_assessments', [])])}")
    
    return {
        "risk_assessment": risk_assessment,
        "audit_trail": audit_trail,
        "compliance_documentation": risk_documentation,
        "integrity_verified": integrity_verified
    }

if __name__ == "__main__":
    # Run the risk assessment demonstration
    results = demonstrate_risk_assessment()
    
    print("\n" + "=" * 50)
    print("🎯 Risk Assessment Example Complete!")
    print("This demonstrates CIAF's capability to support:")
    print("  • EU AI Act Article 9 (Risk Management)")
    print("  • Article 10 (Data Governance)")
    print("  • Article 12 (Record Keeping)")
    print("  • Article 15 (Accuracy, Robustness, Cybersecurity)")
    print("  • NIST AI RMF risk management functions")
    print("  • Healthcare data protection requirements")
    print("\nFor production use, integrate with your risk management")
    print("processes and ensure human oversight workflows.")