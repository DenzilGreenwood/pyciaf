"""
CIAF Enterprise Features Demo - Simplified

This demonstrates the new enterprise compliance features without
the full CIAF framework integration to avoid circular import issues.

Created: 2025-09-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_human_oversight():
    """Demonstrate the Human Oversight Engine."""
    print("\n👁️ Human Oversight Engine Demo")
    print("=" * 30)
    
    try:
        from compliance.human_oversight import HumanOversightEngine, AlertType
        
        # Create oversight engine
        oversight = HumanOversightEngine("demo_model")
        
        # Configure monitoring
        oversight.configure_model_monitoring(
            model_name="demo_model",
            confidence_threshold=0.8,
            uncertainty_threshold=0.3,
            enable_alerts=[AlertType.LOW_CONFIDENCE, AlertType.HIGH_UNCERTAINTY]
        )
        
        print("✅ Oversight engine configured")
        
        # Simulate some inferences
        for i in range(5):
            confidence = np.random.uniform(0.5, 0.95)
            uncertainty = np.random.uniform(0.1, 0.5)
            
            result = oversight.evaluate_inference(
                model_name="demo_model",
                input_data={"inference_id": f"demo_{i}"},
                prediction={"confidence": confidence},
                metadata={"timestamp": datetime.now().isoformat()}
            )
            
            status = "⚠️ FLAGGED" if result["requires_review"] else "✅ PASSED"
            print(f"   Inference {i}: {status} (conf: {confidence:.2f})")
        
        # Get metrics
        metrics = oversight.get_oversight_metrics("demo_model")
        print(f"\n📊 Oversight Metrics:")
        print(f"   Total Inferences: {metrics['total_inferences']}")
        print(f"   Flagged: {metrics['flagged_inferences']}")
        print(f"   Alert Rate: {metrics.get('alert_rate', 0):.1%}")
        
    except ImportError as e:
        print(f"❌ Human Oversight not available: {e}")


def demo_robustness_testing():
    """Demonstrate the Robustness Testing Suite.""" 
    print("\n🔬 Robustness Testing Demo")
    print("=" * 30)
    
    try:
        from compliance.robustness_testing import RobustnessTestSuite
        
        # Mock model function
        def mock_model(inputs):
            if inputs.ndim == 1:
                inputs = inputs.reshape(1, -1)
            weights = np.random.random((min(inputs.shape[1], 5), 3))
            padded_inputs = np.zeros((inputs.shape[0], weights.shape[0]))
            padded_inputs[:, :min(inputs.shape[1], weights.shape[0])] = inputs[:, :weights.shape[0]]
            return np.random.softmax(np.dot(padded_inputs, weights), axis=1)
        
        # Create test suite
        suite = RobustnessTestSuite("demo_model", "1.0.0")
        
        # Generate test data
        np.random.seed(42)
        inputs = np.random.randn(50, 5)
        targets = np.random.randint(0, 3, 50)
        
        # Configure lightweight tests
        config = {
            "run_adversarial": True,
            "run_distribution": True,
            "run_stress": True,
            "fgsm_epsilon": 0.1,
            "max_concurrent": 5,
            "stress_duration": 5
        }
        
        print("🧪 Running robustness tests...")
        report = suite.run_comprehensive_test(mock_model, inputs, targets, config)
        
        print(f"✅ Testing completed!")
        print(f"   Total Tests: {report.total_tests}")
        print(f"   Passed: {report.passed_tests}")
        print(f"   Failed: {report.failed_tests}")
        print(f"   Overall Score: {report.overall_score:.2f}")
        
        if report.recommendations:
            print("💡 Recommendations:")
            for i, rec in enumerate(report.recommendations[:2], 1):
                print(f"   {i}. {rec}")
        
        # Export report
        report_file = suite.export_report(report)
        print(f"📄 Report saved: {report_file}")
        
    except ImportError as e:
        print(f"❌ Robustness Testing not available: {e}")
    except Exception as e:
        print(f"❌ Testing failed: {e}")


def demo_enterprise_config():
    """Demonstrate enterprise configuration templates."""
    print("\n⚙️ Enterprise Configuration Demo")
    print("=" * 30)
    
    try:
        from metadata_config import create_enterprise_config, create_development_config
        
        # Create enterprise config
        enterprise_config = create_enterprise_config()
        print("✅ Enterprise configuration created:")
        print(f"   Worker threads: {enterprise_config.get('worker_threads', 'N/A')}")
        print(f"   Queue size: {enterprise_config.get('deferred_queue_size', 'N/A')}")
        print(f"   Encryption: {enterprise_config.get('enable_encryption', False)}")
        
        # Create development config
        dev_config = create_development_config()
        print("\n✅ Development configuration created:")
        print(f"   Worker threads: {dev_config.get('worker_threads', 'N/A')}")
        print(f"   Debug mode: {dev_config.get('debug_mode', False)}")
        
        # Save configs
        config_file = "demo_enterprise_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                "enterprise": enterprise_config,
                "development": dev_config
            }, f, indent=2)
        
        print(f"📄 Configurations saved to: {config_file}")
        
    except ImportError as e:
        print(f"❌ Enterprise config not available: {e}")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")


def demo_compliance_reports():
    """Demonstrate compliance report generation."""
    print("\n📋 Compliance Reports Demo")
    print("=" * 30)
    
    try:
        # Mock compliance data
        compliance_data = {
            "model_name": "demo_model",
            "compliance_score": 87.5,
            "frameworks": {
                "EU_AI_ACT": {"score": 85, "status": "compliant"},
                "GDPR": {"score": 92, "status": "compliant"},
                "NIST_AI_RMF": {"score": 78, "status": "partial"}
            },
            "audit_events": 1540,
            "violations": 2,
            "last_assessment": datetime.now().isoformat()
        }
        
        # Generate report
        report_data = {
            "report_id": f"compliance_report_{int(time.time())}",
            "generated_at": datetime.now().isoformat(),
            "model_assessment": compliance_data,
            "summary": {
                "overall_status": "compliant",
                "critical_issues": 0,
                "recommendations": [
                    "Consider additional NIST AI RMF controls",
                    "Review data processing consent mechanisms"
                ]
            }
        }
        
        # Export report
        report_file = f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print("✅ Compliance report generated:")
        print(f"   Model: {compliance_data['model_name']}")
        print(f"   Score: {compliance_data['compliance_score']}%")
        print(f"   Status: {report_data['summary']['overall_status'].upper()}")
        print(f"📄 Report saved: {report_file}")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")


def main():
    """Main demonstration function."""
    print("🏢 CIAF Enterprise Features Demo")
    print("=" * 50)
    print("This demo showcases the new enterprise-grade compliance")
    print("and monitoring features added to the CIAF framework.")
    print()
    
    # Demo individual features
    demo_human_oversight()
    demo_robustness_testing()
    demo_enterprise_config()
    demo_compliance_reports()
    
    print("\n🎉 Enterprise Features Demo Complete!")
    print("=" * 50)
    print("\n📈 Summary of Enterprise Features:")
    print("✅ Human Oversight Engine - EU AI Act Article 14 compliance")
    print("✅ Advanced Robustness Testing - Adversarial & stress testing")
    print("✅ Enterprise Configuration Templates - Production-ready configs")
    print("✅ Enhanced Compliance Reports - PDF/CSV export capabilities")
    print("✅ Web Dashboard - Real-time monitoring (requires Flask)")
    
    print("\n💡 Next Steps:")
    print("1. Install optional dependencies: pip install flask flask-socketio plotly")
    print("2. Review generated reports and configurations")
    print("3. Integrate with your existing ML pipeline")
    print("4. Configure monitoring thresholds for your use case")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)