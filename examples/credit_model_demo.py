#!/usr/bin/env python3
"""
CIAF Credit Model Demo

Demonstrates Lazy Capsule M        # Train model
        print("🤖 Training model...")
        training_snapshot = framework.train_model_with_audit(
            model_name="credit_model",
            capsules=capsules,
            training_params={"epochs": 100},
            model_version="1.0.0",
            user_id="ml_engineer"
        )
        print("✅ Model trained successfully") for a credit approval AI model.
This demo simulates credit application processing with verifiable audit trails.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path to import ciaf
sys.path.insert(0, str(Path(__file__).parent.parent))

from ciaf import CIAFFramework


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
                "decision": "approve"
            }
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
                "decision": "review"
            }
        }
    ]


def main():
    """Main demo function."""
    print("=" * 60)
    print(" CIAF Credit Model Demo - Lazy Capsule Materialization")
    print("=" * 60)

    try:
        # Initialize CIAF Framework
        print(" Initializing CIAF Framework...")
        framework = CIAFFramework("CreditAI_Demo")

        # Create dataset anchor
        print(" Creating dataset anchor...")
        framework.create_dataset_anchor(
            dataset_id="credit_applications_2024",
            dataset_metadata={
                "source": "loan_origination_system",
                "type": "credit_applications"
            },
            master_password="secure_credit_anchor"
        )

        # Create provenance capsules
        print(" Creating provenance capsules...")
        credit_data = create_sample_credit_data()
        capsules = framework.create_provenance_capsules(
            "credit_applications_2024", credit_data)
        print(f"   Created {len(capsules)} capsules")

        # Create model anchor
        print(" Creating model anchor...")
        model_anchor = framework.create_model_anchor(
            model_name="credit_model",
            model_parameters={"algorithm": "xgboost", "max_depth": 6},
            model_architecture={"type": "classifier", "features": 4},
            authorized_datasets=["credit_applications_2024"],
            master_password="secure_model_password"
        )

        # Train model
        print(" Training model...")
        training_snapshot = framework.train_model_with_audit(
            model_name="credit_model",
            capsules=capsules,
            training_params={"epochs": 100},
            model_version="1.0.0",
            user_id="ml_engineer"
        )
        print(" Model trained successfully")

        # Perform inference
        print(" Making credit decisions...")
        test_app = {
            "application_id": "TEST001",
            "income": 60000,
            "credit_score": 700,
            "debt_to_income": 0.30,
            "loan_amount": 200000
        }

        decision = "APPROVED" if test_app["credit_score"] >= 650 else "DENIED"
        query = f"Credit application: {json.dumps(test_app)}"
        ai_output = f"Decision: {decision}"

        receipt = framework.perform_inference_with_audit(
            model_name="credit_model",
            query=query,
            ai_output=ai_output,
            training_snapshot=training_snapshot,
            user_id="loan_officer"
        )

        print(f"   Decision: {decision}")
        print(f"   Receipt generated: {getattr(receipt, 'receipt_id', 'N/A')}")

        # Save outputs
        outputs_dir = Path(__file__).parent / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        receipt_file = outputs_dir / f"credit_receipt_{timestamp}.json"

        # Convert receipt to dictionary for JSON serialization
        receipt_data = {
            "receipt_id": getattr(receipt, 'receipt_id', 'N/A'),
            "timestamp": timestamp,
            "decision": decision,
            "query": query,
            "ai_output": ai_output
        }

        with open(receipt_file, 'w') as f:
            json.dump(receipt_data, f, indent=2)

        print(f"\n Receipt saved to: {receipt_file}")
        print("\n Credit Model Demo Completed Successfully!")
        print("Demonstrated: Verifiable credit decisions with audit trails")

        return True

    except Exception as e:
        print(f"\n Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
