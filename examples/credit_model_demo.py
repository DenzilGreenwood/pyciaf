#!/usr/bin/env python3
"""
CIAF Credit Model Demo with GDPR Model Wrapper

Demonstrates the ultimate GDPR model wrapper for a credit approval AI model.
This demo simulates credit application processing with comprehensive GDPR compliance,
multi-framework regulatory adherence, and verifiable audit trails.

Features demonstrated:
- GDPR-compliant credit scoring model
- Multi-framework compliance (GDPR, NIST-AI-RMF, ISO/IEC-42001)
- Enhanced explainability for credit decisions
- Advanced uncertainty quantification
- Comprehensive audit trails and receipts
- Privacy-preserving inference
"""

import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path to import ciaf
sys.path.insert(0, str(Path(__file__).parent.parent))

from ciaf import CIAFFramework
from ciaf.wrappers import create_ultimate_gdpr_wrapper


class CreditScoringModel:
    """
    Mock credit scoring model for demonstration purposes.

    This simulates a real ML model that would be used for credit approval decisions.
    In practice, this could be XGBoost, Random Forest, Neural Network, etc.
    """

    def __init__(self):
        self.is_fitted = False
        self.feature_names = ["income", "credit_score", "debt_to_income", "loan_amount"]
        self.model_type = "credit_classifier"

    def fit(self, X, y):
        """Simulate model training."""
        self.is_fitted = True
        self.training_samples = len(X) if hasattr(X, "__len__") else 100
        print(f"   📊 Training on {self.training_samples} credit applications")
        return self

    def predict(self, X):
        """Simulate credit scoring predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Simulate credit approval logic
        if isinstance(X, dict):
            # Single application
            credit_score = X.get("credit_score", 600)
            debt_to_income = X.get("debt_to_income", 0.5)
            income = X.get("income", 50000)

            # Simple scoring logic
            if credit_score >= 720 and debt_to_income <= 0.3:
                return "APPROVED"
            elif credit_score >= 650 and debt_to_income <= 0.4 and income >= 50000:
                return "APPROVED"
            elif credit_score >= 600:
                return "REVIEW"
            else:
                return "DENIED"
        else:
            # Batch processing
            return [
                "APPROVED" if np.random.random() > 0.3 else "DENIED"
                for _ in range(len(X))
            ]

    def predict_proba(self, X):
        """Simulate probability predictions for uncertainty quantification."""
        if isinstance(X, dict):
            decision = self.predict(X)
            if decision == "APPROVED":
                return {"approved": 0.85, "denied": 0.15}
            elif decision == "REVIEW":
                return {"approved": 0.45, "denied": 0.55}
            else:
                return {"approved": 0.15, "denied": 0.85}
        else:
            return [
                {"approved": np.random.random(), "denied": np.random.random()}
                for _ in range(len(X))
            ]


def create_sample_credit_data():
    """Generate sample credit application data."""
    return [
        {
            "content": "credit_app_001",
            "metadata": {
                "id": "1",
                "application_id": "APP001",
                "income": 75000,
                "credit_score": 720,
                "debt_to_income": 0.25,
                "loan_amount": 250000,
                "decision": "approve",
            },
        },
        {
            "content": "credit_app_002",
            "metadata": {
                "id": "2",
                "application_id": "APP002",
                "income": 45000,
                "credit_score": 650,
                "debt_to_income": 0.40,
                "loan_amount": 180000,
                "decision": "review",
            },
        },
    ]


def main():
    """Main demo function."""
    print("=" * 70)
    print(" CIAF Credit Model Demo - Ultimate GDPR Model Wrapper")
    print("=" * 70)

    try:
        # Initialize CIAF Framework
        print("🚀 Initializing CIAF Framework...")
        framework = CIAFFramework("CreditAI_Demo")

        # Create and configure the credit scoring model
        print("🤖 Creating credit scoring model...")
        credit_model = CreditScoringModel()

        # Create GDPR-compliant wrapper with comprehensive features
        print("🛡️  Creating ultimate GDPR wrapper for credit model...")
        gdpr_wrapper = create_ultimate_gdpr_wrapper(
            model=credit_model,
            model_name="gdpr_credit_scorer",
            compliance_level="strict",  # Maximum compliance for financial services
            performance_mode="optimized",  # High performance for real-time decisions
            enable_all_features=True,
            # GDPR-specific configuration for financial services
            lawful_basis="legitimate_interests",
            purpose_of_processing="automated_credit_decision_making",
            dpo_contact="dpo@creditbank.example",
            dsr_endpoint="https://creditbank.example/data-subject-requests",
            retention_days=2555,  # 7 years for financial records
            # Enable all regulatory frameworks for financial compliance
            enable_nist_ai_rmf=True,
            enable_iso_iec_42001=True,
            enable_hipaa=False,  # Not applicable for credit scoring
        )

        print(f"✅ GDPR wrapper created: {type(gdpr_wrapper).__name__}")
        print(
            f"   🏛️  Compliance frameworks: {len(gdpr_wrapper.regulatory_frameworks)}"
        )
        print(f"   🎯 Model type: {gdpr_wrapper.model_type}")

        # Create dataset anchor
        print("📊 Creating dataset anchor...")
        framework.create_dataset_anchor(
            dataset_id="credit_applications_2024",
            dataset_metadata={
                "source": "loan_origination_system",
                "type": "credit_applications",
                "gdpr_lawful_basis": "legitimate_interests",
                "data_retention_days": 2555,
            },
            master_password="secure_credit_anchor",
        )

        # Create provenance capsules
        print("📦 Creating provenance capsules...")
        credit_data = create_sample_credit_data()
        capsules = framework.create_provenance_capsules(
            "credit_applications_2024", credit_data
        )
        print(f"   ✅ Created {len(capsules)} GDPR-compliant capsules")

        # Train model with GDPR compliance
        print("🎓 Training GDPR-compliant credit model...")

        # Prepare training data from capsules
        training_data = []
        training_labels = []
        for capsule in capsules:
            if hasattr(capsule, "metadata") and capsule.metadata:
                app_data = {
                    "income": capsule.metadata.get("income", 50000),
                    "credit_score": capsule.metadata.get("credit_score", 650),
                    "debt_to_income": capsule.metadata.get("debt_to_income", 0.35),
                    "loan_amount": capsule.metadata.get("loan_amount", 200000),
                }
                training_data.append(app_data)
                training_labels.append(capsule.metadata.get("decision", "review"))

        # Use GDPR wrapper for compliant training
        training_receipt = gdpr_wrapper.train_gdpr(
            training_data=training_data,
            training_labels=training_labels,
            user_id="ml_engineer",
            training_purpose="credit_risk_assessment",
            data_source="loan_origination_system",
        )

        print("✅ Model trained with GDPR compliance")
        print(
            f"   📋 Training receipt: {training_receipt.get('receipt_id', 'Generated')}"
        )
        print(
            f"   🔒 Privacy measures: {len(training_receipt.get('privacy_measures', []))} applied"
        )

        # Create model anchor with GDPR metadata
        print("⚓ Creating GDPR-compliant model anchor...")
        model_anchor = framework.create_model_anchor(
            model_name="gdpr_credit_model",
            model_parameters={
                "algorithm": "gdpr_compliant_credit_classifier",
                "compliance_frameworks": gdpr_wrapper.regulatory_frameworks,
                "privacy_measures": [
                    "data_minimization",
                    "purpose_limitation",
                    "storage_limitation",
                ],
            },
            model_architecture={
                "type": "gdpr_classifier",
                "features": 4,
                "gdpr_compliant": True,
                "explainable": True,
            },
            authorized_datasets=["credit_applications_2024"],
            master_password="secure_gdpr_model_password",
        )

        # Perform GDPR-compliant credit decisions
        print("💳 Making GDPR-compliant credit decisions...")

        # Test credit application
        test_app = {
            "application_id": "TEST001",
            "income": 60000,
            "credit_score": 700,
            "debt_to_income": 0.30,
            "loan_amount": 200000,
            "applicant_consent": True,  # GDPR consent
            "data_processing_purpose": "credit_assessment",
        }

        # Use GDPR wrapper for compliant inference
        print(f"📋 Processing application: {test_app['application_id']}")
        inference_receipt = gdpr_wrapper.predict_gdpr(
            query_data=test_app,
            user_id="loan_officer",
            purpose="credit_decision_making",
            include_explanations=True,  # Required for transparency
            include_uncertainty=True,  # Risk assessment
            data_subject_id=test_app["application_id"],
        )

        decision = inference_receipt.get("prediction", "REVIEW")
        confidence = inference_receipt.get("confidence", 0.5)
        explanation = inference_receipt.get("explanation", {})
        uncertainty = inference_receipt.get("uncertainty", {})

        print(f"✅ Credit Decision: {decision}")
        print(f"   📊 Confidence: {confidence:.2%}")
        print(
            f"   🔍 Explainability: {explanation.get('method', 'Enhanced SHAP/LIME')}"
        )
        print(f"   📈 Uncertainty: {uncertainty.get('total_uncertainty', 0.12):.3f}")
        print(f"   🧾 GDPR Receipt: {inference_receipt.get('receipt_id', 'Generated')}")

        # Demonstrate batch processing capability
        print("📦 Testing batch processing for multiple applications...")
        batch_applications = [
            {
                "application_id": "BATCH001",
                "income": 80000,
                "credit_score": 750,
                "debt_to_income": 0.25,
                "loan_amount": 300000,
            },
            {
                "application_id": "BATCH002",
                "income": 45000,
                "credit_score": 620,
                "debt_to_income": 0.45,
                "loan_amount": 150000,
            },
            {
                "application_id": "BATCH003",
                "income": 70000,
                "credit_score": 690,
                "debt_to_income": 0.35,
                "loan_amount": 220000,
            },
        ]

        batch_receipt = gdpr_wrapper.predict_batch_gdpr(
            batch_queries=batch_applications,
            user_id="batch_processor",
            purpose="bulk_credit_assessment",
            include_explanations=True,
        )

        print("✅ Batch processing completed")
        print(f"   📊 Processed: {len(batch_applications)} applications")
        print(f"   🧾 Batch receipt: {batch_receipt.get('batch_id', 'Generated')}")

        # Show individual results from batch
        batch_results = batch_receipt.get("individual_results", [])
        for i, result in enumerate(batch_results[:3]):  # Show first 3
            app_id = batch_applications[i]["application_id"]
            decision = result.get("prediction", "REVIEW")
            confidence = result.get("confidence", 0.5)
            print(f"   • {app_id}: {decision} (confidence: {confidence:.2%})")

        # Save comprehensive outputs
        outputs_dir = Path(__file__).parent / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save individual decision receipt
        receipt_file = outputs_dir / f"gdpr_credit_receipt_{timestamp}.json"

        # Save comprehensive GDPR-compliant receipt data
        individual_receipt_data = {
            "receipt_id": inference_receipt.get("receipt_id", f"GDPR_{timestamp}"),
            "timestamp": timestamp,
            "application_id": test_app["application_id"],
            "decision": decision,
            "confidence": confidence,
            "gdpr_compliance": {
                "lawful_basis": gdpr_wrapper.gdpr_manifest.lawful_basis,
                "purpose": gdpr_wrapper.gdpr_manifest.purpose_of_processing,
                "retention_days": gdpr_wrapper.gdpr_manifest.retention_days,
                "frameworks": gdpr_wrapper.regulatory_frameworks,
            },
            "explainability": explanation,
            "uncertainty": uncertainty,
            "privacy_measures": inference_receipt.get("privacy_measures", []),
            "query_data": test_app,
            "processing_details": {
                "model_type": gdpr_wrapper.model_type,
                "user_id": "loan_officer",
                "processing_purpose": "credit_decision_making",
            },
        }

        with open(receipt_file, "w") as f:
            json.dump(individual_receipt_data, f, indent=2)

        # Save batch processing results
        batch_file = outputs_dir / f"gdpr_batch_receipt_{timestamp}.json"
        batch_receipt_data = {
            "batch_id": batch_receipt.get("batch_id", f"BATCH_{timestamp}"),
            "timestamp": timestamp,
            "batch_size": len(batch_applications),
            "applications": batch_applications,
            "results": batch_results,
            "gdpr_compliance": individual_receipt_data["gdpr_compliance"],
            "processing_details": {
                "model_type": gdpr_wrapper.model_type,
                "user_id": "batch_processor",
                "processing_purpose": "bulk_credit_assessment",
            },
        }

        with open(batch_file, "w") as f:
            json.dump(batch_receipt_data, f, indent=2)

        # Export comprehensive compliance report
        compliance_file = outputs_dir / f"gdpr_compliance_report_{timestamp}.json"
        compliance_report = gdpr_wrapper.export_compliance_report(
            include_detailed_validations=True
        )

        with open(compliance_file, "w") as f:
            json.dump(compliance_report, f, indent=2)

        print(f"\n📁 Individual receipt saved to: {receipt_file}")
        print(f"📁 Batch receipt saved to: {batch_file}")
        print(f"📁 Compliance report saved to: {compliance_file}")
        print("\n🎉 GDPR Credit Model Demo Completed Successfully!")
        print("✅ Demonstrated:")
        print("   • GDPR-compliant credit scoring with ultimate wrapper")
        print(
            "   • Multi-framework regulatory compliance (GDPR, NIST-AI-RMF, ISO/IEC-42001)"
        )
        print("   • Enhanced explainability and uncertainty quantification")
        print("   • Comprehensive audit trails and receipts")
        print("   • Batch processing capabilities")
        print("   • Privacy-preserving inference with transparency")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
