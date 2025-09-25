"""
CIAF Enterprise Integration Demo

This script demonstrates the complete enterprise-grade CIAF framework
with all advanced compliance and enterprise features integrated.

Created: 2025-09-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# CIAF Core imports
from ciaf import CIAFFramework, MetadataConfig, create_config_template

# Enterprise compliance features (with availability checks)
try:
    from ciaf.compliance.human_oversight import HumanOversightEngine, AlertType
    from ciaf.compliance.web_dashboard import create_dashboard
    from ciaf.compliance.robustness_testing import RobustnessTestSuite
    from ciaf.compliance.reports import ComplianceReportGenerator
    from ciaf.metadata_config import create_enterprise_config
    ENTERPRISE_FEATURES_AVAILABLE = True
except ImportError:
    ENTERPRISE_FEATURES_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnterpriseAISystem:
    """
    Comprehensive enterprise AI system with full CIAF integration.
    
    This demonstrates how to build a production-ready AI system with:
    - Comprehensive compliance monitoring
    - Human oversight integration
    - Advanced robustness testing
    - Web dashboard for monitoring
    - Enterprise configuration management
    """
    
    def __init__(self, system_name: str = "enterprise_ai_system"):
        self.system_name = system_name
        self.logger = logging.getLogger(f"{__name__}.{system_name}")
        
        # Initialize CIAF framework
        self.ciaf = CIAFFramework(
            anchoring_method="dataset_hash",
            compliance_framework="EU_AI_ACT"
        )
        
        # Enterprise configuration
        self.config = create_enterprise_config() if ENTERPRISE_FEATURES_AVAILABLE else {}
        
        # Initialize enterprise components
        self._initialize_enterprise_components()
    
    def _initialize_enterprise_components(self):
        """Initialize enterprise-grade components."""
        if not ENTERPRISE_FEATURES_AVAILABLE:
            self.logger.warning("Enterprise features not available. Install optional dependencies.")
            self.oversight_engine = None
            self.dashboard = None
            self.robustness_suite = None
            return
        
        # Human oversight engine
        self.oversight_engine = HumanOversightEngine(
            model_name=self.system_name,
            framework=self.ciaf
        )
        
        # Web dashboard
        self.dashboard = create_dashboard(
            ciaf_framework=self.ciaf,
            host="127.0.0.1",
            port=5000
        )
        
        # Robustness testing suite
        self.robustness_suite = RobustnessTestSuite(
            model_name=self.system_name,
            model_version="1.0.0"
        )
        
        # Compliance report generator
        self.report_generator = ComplianceReportGenerator(
            framework=self.ciaf
        )
        
        self.logger.info("✅ Enterprise components initialized successfully")
    
    def deploy_model(self, model_func, model_name: str, training_data: np.ndarray, 
                    test_data: np.ndarray = None):
        """
        Deploy a model with full enterprise compliance integration.
        
        Args:
            model_func: The model function to deploy
            model_name: Name of the model
            training_data: Training dataset
            test_data: Test dataset for validation
        """
        self.logger.info(f"🚀 Deploying model: {model_name}")
        
        try:
            # 1. Create model anchor in CIAF
            anchor_result = self.ciaf.create_model_anchor(
                model_name=model_name,
                model_version="1.0.0",
                training_data=training_data,
                hyperparameters={"model_type": "enterprise_model"}
            )
            
            self.logger.info(f"✅ Model anchor created: {anchor_result['anchor_id']}")
            
            # 2. Configure human oversight
            if self.oversight_engine:
                self.oversight_engine.configure_model_monitoring(
                    model_name=model_name,
                    confidence_threshold=0.8,
                    uncertainty_threshold=0.3,
                    enable_alerts=[AlertType.LOW_CONFIDENCE, AlertType.HIGH_UNCERTAINTY]
                )
                self.logger.info("✅ Human oversight configured")
            
            # 3. Run comprehensive robustness testing
            if self.robustness_suite and test_data is not None:
                self.logger.info("🔬 Running comprehensive robustness tests...")
                
                # Mock targets for demonstration
                test_targets = np.random.randint(0, 3, len(test_data))
                
                test_config = {
                    "run_adversarial": True,
                    "run_distribution": True,
                    "run_stress": True,
                    "fgsm_epsilon": 0.01,
                    "max_concurrent": 20,
                    "stress_duration": 15
                }
                
                robustness_report = self.robustness_suite.run_comprehensive_test(
                    model_func, test_data, test_targets, test_config
                )
                
                self.logger.info(f"✅ Robustness testing completed. Score: {robustness_report.overall_score:.2f}")
                
                # Export robustness report
                report_file = self.robustness_suite.export_report(robustness_report)
                self.logger.info(f"📄 Robustness report saved: {report_file}")
            
            # 4. Generate compliance report
            if self.report_generator:
                self.logger.info("📋 Generating compliance report...")
                
                compliance_report = self.report_generator.generate_comprehensive_report(
                    model_name=model_name
                )
                
                # Export as PDF (if available)
                try:
                    pdf_path = f"compliance_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    self.report_generator._generate_pdf_report(compliance_report, pdf_path)
                    self.logger.info(f"✅ PDF compliance report generated: {pdf_path}")
                except Exception as e:
                    self.logger.warning(f"PDF generation failed: {e}")
            
            # 5. Start monitoring
            self._start_monitoring(model_name, model_func)
            
            self.logger.info(f"🎉 Model {model_name} deployed successfully with enterprise features!")
            
            return {
                "status": "success",
                "anchor_id": anchor_result["anchor_id"],
                "monitoring_active": self.oversight_engine is not None,
                "dashboard_url": "http://127.0.0.1:5000" if self.dashboard else None,
                "robustness_score": robustness_report.overall_score if 'robustness_report' in locals() else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Model deployment failed: {str(e)}")
            raise
    
    def _start_monitoring(self, model_name: str, model_func):
        """Start continuous monitoring for the deployed model."""
        if not self.oversight_engine:
            return
        
        self.logger.info(f"👁️ Starting monitoring for {model_name}")
        
        # Simulate some inferences with oversight
        try:
            # Mock inference data
            mock_inputs = np.random.randn(5, 10)
            
            for i, input_data in enumerate(mock_inputs):
                # Run inference
                prediction = model_func(input_data.reshape(1, -1))
                
                # Calculate mock confidence and uncertainty
                confidence = np.random.uniform(0.6, 0.95)
                uncertainty = np.random.uniform(0.1, 0.4)
                
                # Check with oversight engine
                oversight_result = self.oversight_engine.evaluate_inference(
                    model_name=model_name,
                    input_data={"input_shape": input_data.shape},
                    prediction={"confidence": confidence},
                    metadata={"inference_id": f"demo_{i}"}
                )
                
                if oversight_result["requires_review"]:
                    self.logger.info(f"⚠️ Inference {i} flagged for human review")
                
                time.sleep(0.1)  # Brief pause between inferences
            
            # Get oversight metrics
            metrics = self.oversight_engine.get_oversight_metrics(model_name)
            self.logger.info(f"📊 Oversight metrics: {metrics['total_inferences']} inferences, "
                           f"{metrics['flagged_inferences']} flagged")
            
        except Exception as e:
            self.logger.warning(f"Monitoring simulation failed: {e}")
    
    def start_dashboard(self):
        """Start the web dashboard for monitoring."""
        if not self.dashboard:
            self.logger.warning("Dashboard not available. Install Flask dependencies.")
            return
        
        self.logger.info("🌐 Starting CIAF Enterprise Dashboard...")
        self.logger.info("   Visit http://127.0.0.1:5000 to view the dashboard")
        
        try:
            self.dashboard.run(debug=False)
        except KeyboardInterrupt:
            self.logger.info("👋 Dashboard stopped")
    
    def generate_enterprise_report(self) -> dict:
        """Generate comprehensive enterprise compliance report."""
        if not ENTERPRISE_FEATURES_AVAILABLE:
            return {"error": "Enterprise features not available"}
        
        self.logger.info("📋 Generating enterprise compliance report...")
        
        report = {
            "system_name": self.system_name,
            "timestamp": datetime.now().isoformat(),
            "compliance_status": "active",
            "features_enabled": {
                "human_oversight": self.oversight_engine is not None,
                "web_dashboard": self.dashboard is not None,
                "robustness_testing": self.robustness_suite is not None
            }
        }
        
        # Add oversight metrics if available
        if self.oversight_engine:
            try:
                all_models = self.ciaf.model_anchors.keys() if hasattr(self.ciaf, 'model_anchors') else []
                oversight_summary = {}
                
                for model_name in all_models:
                    metrics = self.oversight_engine.get_oversight_metrics(model_name)
                    oversight_summary[model_name] = metrics
                
                report["oversight_summary"] = oversight_summary
            except Exception as e:
                report["oversight_error"] = str(e)
        
        # Add system health metrics
        report["system_health"] = {
            "ciaf_active": True,
            "compliance_frameworks": ["EU_AI_ACT", "GDPR", "NIST_AI_RMF"],
            "last_robustness_test": datetime.now().isoformat(),
            "monitoring_status": "active"
        }
        
        return report


def mock_model_function(inputs: np.ndarray) -> np.ndarray:
    """
    Mock model function for demonstration.
    
    In a real system, this would be your actual ML model.
    """
    # Simple linear transformation for demo
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    
    # Simulate model output (3 classes)
    weights = np.array([[0.1, 0.3, 0.6], [0.2, 0.4, 0.4], [0.3, 0.3, 0.4]])
    if inputs.shape[1] >= 3:
        output = np.dot(inputs[:, :3], weights)
    else:
        # Pad inputs if needed
        padded_inputs = np.zeros((inputs.shape[0], 3))
        padded_inputs[:, :inputs.shape[1]] = inputs[:, :min(inputs.shape[1], 3)]
        output = np.dot(padded_inputs, weights)
    
    # Apply softmax for probabilities
    exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
    return exp_output / np.sum(exp_output, axis=1, keepdims=True)


def main():
    """
    Main demonstration of enterprise CIAF integration.
    """
    print("🏢 CIAF Enterprise Integration Demo")
    print("=" * 50)
    
    if not ENTERPRISE_FEATURES_AVAILABLE:
        print("⚠️  Enterprise features require additional dependencies:")
        print("   pip install flask flask-socketio plotly weasyprint")
        print("   Some features will be limited in this demo.")
        print()
    
    # Initialize enterprise AI system
    enterprise_system = EnterpriseAISystem("credit_risk_model")
    
    # Generate mock training and test data
    np.random.seed(42)
    training_data = np.random.randn(1000, 10)
    test_data = np.random.randn(200, 10)
    
    # Deploy model with full enterprise integration
    print("\n🚀 Deploying Model with Enterprise Integration...")
    deployment_result = enterprise_system.deploy_model(
        model_func=mock_model_function,
        model_name="credit_risk_model",
        training_data=training_data,
        test_data=test_data
    )
    
    print(f"✅ Deployment Result: {deployment_result}")
    
    # Generate enterprise report
    print("\n📋 Generating Enterprise Compliance Report...")
    enterprise_report = enterprise_system.generate_enterprise_report()
    
    # Save report to file
    report_file = f"enterprise_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(enterprise_report, f, indent=2)
    
    print(f"📄 Enterprise report saved to: {report_file}")
    
    # Display summary
    print("\n📊 Enterprise System Summary:")
    print(f"   System Name: {enterprise_system.system_name}")
    print(f"   Enterprise Features: {'✅ Available' if ENTERPRISE_FEATURES_AVAILABLE else '❌ Limited'}")
    if deployment_result.get("dashboard_url"):
        print(f"   Dashboard URL: {deployment_result['dashboard_url']}")
    if deployment_result.get("robustness_score"):
        print(f"   Robustness Score: {deployment_result['robustness_score']:.2f}")
    
    print("\n🎯 Enterprise Integration Complete!")
    print("\nNext Steps:")
    print("1. Visit the dashboard for real-time monitoring")
    print("2. Review generated compliance reports")
    print("3. Configure additional oversight rules")
    print("4. Set up automated testing schedules")
    
    # Optionally start dashboard
    start_dashboard = input("\n🌐 Start web dashboard? (y/N): ").lower().strip()
    if start_dashboard == 'y' and ENTERPRISE_FEATURES_AVAILABLE:
        print("\n🌐 Starting dashboard... (Press Ctrl+C to stop)")
        try:
            enterprise_system.start_dashboard()
        except KeyboardInterrupt:
            print("\n👋 Demo complete!")
    else:
        print("\n👋 Demo complete!")


if __name__ == "__main__":
    main()