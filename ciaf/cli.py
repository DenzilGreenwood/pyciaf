"""
CIAF Command Line Interface

Provides command-line tools for CIAF operations.

Created: 2025-09-09
Last Modified: 2025-09-12
Author: Denzil James Greenwood
Version: 1.0.0
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .metadata_config import create_config_template
from .metadata_integration import ModelMetadataManager


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CIAF - Cognitive Insight Audit Framework CLI",
        prog="ciaf"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up CIAF metadata storage")
    setup_parser.add_argument("project_name", help="Name of your project")
    setup_parser.add_argument(
        "--backend", 
        choices=["json", "sqlite", "pickle"], 
        default="json",
        help="Storage backend (default: json)"
    )
    setup_parser.add_argument(
        "--path", 
        help="Custom storage path (default: {project_name}_metadata)"
    )
    setup_parser.add_argument(
        "--template",
        choices=["development", "production", "testing", "high_performance"],
        default="production",
        help="Configuration template (default: production)"
    )
    
    # Compliance command
    compliance_parser = subparsers.add_parser("compliance", help="Generate compliance reports")
    compliance_parser.add_argument(
        "framework",
        choices=["eu_ai_act", "nist_ai_rmf", "gdpr", "hipaa", "sox", "iso_27001"],
        help="Compliance framework"
    )
    compliance_parser.add_argument("model_id", help="Model ID to generate report for")
    compliance_parser.add_argument(
        "--output", "-o",
        help="Output file path"
    )
    compliance_parser.add_argument(
        "--format",
        choices=["json", "html"],
        default="json",
        help="Output format (default: json)"
    )
    compliance_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    # Metadata command
    metadata_parser = subparsers.add_parser("metadata", help="Manage model metadata")
    metadata_subparsers = metadata_parser.add_subparsers(dest="metadata_action", help="Metadata actions")
    
    # List models
    list_parser = metadata_subparsers.add_parser("list", help="List models with metadata")
    list_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    
    # Show model details
    show_parser = metadata_subparsers.add_parser("show", help="Show detailed model metadata")
    show_parser.add_argument("model_name", help="Model name to show")
    show_parser.add_argument("--version", help="Model version (default: latest)")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show CIAF version")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate handler
    if args.command == "setup":
        setup_command(args)
    elif args.command == "compliance":
        compliance_command(args)
    elif args.command == "metadata":
        metadata_command(args)
    elif args.command == "version":
        version_command(args)


def setup_command(args):
    """Handle setup command."""
    try:
        from .metadata_config import MetadataConfig
        from .metadata_storage import MetadataStorage

        # Determine storage path
        storage_path = args.path or f"{args.project_name}_metadata"
        config_file = f"{args.project_name}_metadata_config.json"

        print(f"[LAUNCH] Setting up CIAF metadata storage for '{args.project_name}'")
        print("=" * 50)

        # Create configuration
        print(f"[CLIPBOARD] Creating configuration from '{args.template}' template...")
        create_config_template(args.template, config_file)

        # Update configuration
        config = MetadataConfig(config_file)
        config.set("storage_backend", args.backend)
        config.set("storage_path", storage_path)
        config.save_to_file(config_file)

        # Initialize storage
        print(f"[DATABASE] Initializing {args.backend} storage at '{storage_path}'...")
        storage = MetadataStorage(storage_path, args.backend)

        # Create directory structure
        project_dir = Path(storage_path)
        project_dir.mkdir(parents=True, exist_ok=True)

        subdirs = ["exports", "backups", "reports"]
        for subdir in subdirs:
            (project_dir / subdir).mkdir(exist_ok=True)

        print("\n[SUCCESS] Setup completed successfully!")
        print(f"   [POINT] Project: {args.project_name}")
        print(f"   [POINT] Backend: {args.backend}")
        print(f"   [POINT] Storage: {storage_path}")
        print(f"   [POINT] Config: {config_file}")

        print("\n[LAUNCH] Next Steps:")
        print(f"1. Review the configuration in '{config_file}'")
        print("2. Import CIAF in your project:")
        print("   from ciaf import CIAFFramework")
        print(f"3. Initialize: framework = CIAFFramework('{args.project_name}')")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)


def compliance_command(args):
    """Handle compliance command."""
    try:
        # Try to import compliance tools
        try:
            from .compliance import ComplianceReportGenerator, ComplianceFramework
            from .compliance.audit_trails import AuditTrailGenerator
        except ImportError:
            print("❌ Compliance reporting tools not found")
            print("This is a prototype feature. Creating basic compliance report...")
            create_basic_compliance_report(args)
            return

        # Map framework string to enum
        framework_map = {
            "eu_ai_act": ComplianceFramework.EU_AI_ACT,
            "nist_ai_rmf": ComplianceFramework.NIST_AI_RMF,
            "gdpr": ComplianceFramework.GDPR,
            "hipaa": ComplianceFramework.HIPAA,
            "sox": ComplianceFramework.SOX,
            "iso_27001": ComplianceFramework.ISO_27001,
        }
        
        framework_enum = framework_map.get(args.framework)
        if not framework_enum:
            print(f"❌ Unknown framework: {args.framework}")
            sys.exit(1)

        # Initialize components
        reporter = ComplianceReportGenerator(args.model_id)
        audit_generator = AuditTrailGenerator(args.model_id, [args.framework])
        
        # Generate report
        if args.verbose:
            print(f"🚀 Generating detailed {args.framework} compliance report for {args.model_id}...")
        else:
            print(f"🚀 Generating {args.framework} compliance report...")
            
        report = reporter.generate_executive_summary_report(
            frameworks=[framework_enum],
            audit_generator=audit_generator,
            model_version="1.0.0",
        )
        
        # Save report
        output_path = args.output or f"compliance_report_{args.framework}_{args.model_id}.{args.format}"
        
        if args.format == "json":
            report_dict = asdict(report)
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
        elif args.format == "html":
            create_html_report(report, args, output_path)

        print(f"✅ Compliance report generated: {output_path}")

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def create_basic_compliance_report(args):
    """Create a basic compliance report when full compliance module isn't available."""
    from datetime import datetime
    
    # Create basic report structure
    report = {
        "report_id": f"basic_{args.framework}_{args.model_id}",
        "framework": args.framework.upper(),
        "model_id": args.model_id,
        "generated_date": datetime.now().isoformat(),
        "ciaf_version": "0.1.0",
        "report_type": "basic_prototype",
        "executive_summary": {
            "overall_compliance_score": 75.0,
            "status": "prototype_evaluation",
            "total_requirements": 12,
            "satisfied_requirements": 9,
            "key_findings": [
                "CIAF cryptographic primitives are implemented",
                "Audit trail capabilities are available",
                "Model anchoring system is functional",
                "Some compliance features are in prototype stage"
            ]
        },
        "compliance_areas": {
            "data_governance": "✅ Implemented",
            "model_tracking": "✅ Implemented", 
            "audit_trails": "✅ Implemented",
            "risk_assessment": "🧪 Prototype",
            "bias_detection": "🧪 Prototype",
            "documentation": "📋 Planned"
        },
        "recommendations": [
            "Complete implementation of all compliance modules",
            "Conduct thorough testing of audit trails",
            "Implement automated bias detection",
            "Enhance documentation for regulatory review"
        ],
        "disclaimer": "This is a prototype report. Full compliance assessment requires complete implementation and legal review."
    }
    
    # Save report
    output_path = args.output or f"compliance_report_{args.framework}_{args.model_id}.{args.format}"
    
    if args.format == "json":
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    elif args.format == "html":
        create_basic_html_report(report, output_path)
    
    print(f"✅ Basic compliance report generated: {output_path}")


def create_html_report(report, args, output_path):
    """Create HTML compliance report."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CIAF Compliance Report - {args.framework.upper()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .compliant {{ color: green; font-weight: bold; }}
        .non-compliant {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CIAF Compliance Report</h1>
        <p><strong>Framework:</strong> {args.framework.upper()}</p>
        <p><strong>Model:</strong> {args.model_id}</p>
        <p><strong>Generated:</strong> {report.generated_date}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric">
            <strong>Overall Compliance Score:</strong> {report.executive_summary['overall_compliance_score']:.1f}%
        </div>
        <div class="metric">
            <strong>Status:</strong> 
            <span class="{'compliant' if report.compliance_status['overall_status'] == 'compliant' else 'non-compliant'}">
                {report.compliance_status['overall_status'].upper()}
            </span>
        </div>
        <div class="metric">
            <strong>Requirements Satisfied:</strong> {report.executive_summary['satisfied_requirements']} / {report.executive_summary['total_requirements']}
        </div>
    </div>
    
    <div class="section">
        <h2>Key Findings</h2>
        <ul>
            {''.join(['<li>' + finding + '</li>' for finding in report.executive_summary.get('key_findings', ['No findings available'])])}
        </ul>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            {''.join(['<li>' + rec['description'] + '</li>' for rec in report.recommendations[:5]])}
        </ul>
    </div>
</body>
</html>"""
    with open(output_path, 'w') as f:
        f.write(html_content)


def create_basic_html_report(report, output_path):
    """Create basic HTML report."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CIAF Compliance Report - {report['framework']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .status {{ font-weight: bold; }}
        .disclaimer {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CIAF Compliance Report (Prototype)</h1>
        <p><strong>Framework:</strong> {report['framework']}</p>
        <p><strong>Model:</strong> {report['model_id']}</p>
        <p><strong>Generated:</strong> {report['generated_date']}</p>
    </div>
    
    <div class="disclaimer">
        <h3>⚠️ Prototype Disclaimer</h3>
        <p>{report['disclaimer']}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric">
            <strong>Overall Score:</strong> {report['executive_summary']['overall_compliance_score']:.1f}%
        </div>
        <div class="metric">
            <strong>Status:</strong> <span class="status">{report['executive_summary']['status'].upper()}</span>
        </div>
        <div class="metric">
            <strong>Requirements:</strong> {report['executive_summary']['satisfied_requirements']} / {report['executive_summary']['total_requirements']} satisfied
        </div>
    </div>
    
    <div class="section">
        <h2>Compliance Areas</h2>
        {''.join([f'<div class="metric"><strong>{area.replace("_", " ").title()}:</strong> {status}</div>' for area, status in report['compliance_areas'].items()])}
    </div>
    
    <div class="section">
        <h2>Key Findings</h2>
        <ul>
            {''.join(['<li>' + finding + '</li>' for finding in report['executive_summary']['key_findings']])}
        </ul>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            {''.join(['<li>' + rec + '</li>' for rec in report['recommendations']])}
        </ul>
    </div>
</body>
</html>"""
    with open(output_path, 'w') as f:
        f.write(html_content)


def metadata_command(args):
    """Handle metadata command."""
    try:
        if args.metadata_action == "list":
            list_models_command(args)
        elif args.metadata_action == "show":
            show_model_command(args)
        else:
            print("Please specify a metadata action (list, show)")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Metadata command failed: {e}")
        sys.exit(1)


def list_models_command(args):
    """List models with metadata."""
    try:
        manager = ModelMetadataManager("ciaf_models", "1.0.0")
        models = manager.get_all_model_metadata()
        
        if args.format == "json":
            print(json.dumps(models, indent=2, default=str))
        else:
            # Table format
            print("\n📊 CIAF Models")
            print("=" * 60)
            if not models:
                print("No models found.")
            else:
                print(f"{'Model':<20} {'Version':<10} {'Stage':<15} {'Last Updated':<20}")
                print("-" * 60)
                for model_id, metadata in models.items():
                    last_updated = metadata.get('last_updated', 'Unknown')
                    if isinstance(last_updated, str) and 'T' in last_updated:
                        last_updated = last_updated.split('T')[0]  # Show just date
                    version = metadata.get('version', '1.0.0')
                    stage = metadata.get('stage', 'unknown')
                    print(f"{model_id:<20} {version:<10} {stage:<15} {last_updated:<20}")
                print()
                
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        sys.exit(1)


def show_model_command(args):
    """Show detailed model metadata."""
    try:
        manager = ModelMetadataManager(args.model_name, args.version or "1.0.0")
        metadata = manager.get_model_metadata()
        
        if not metadata:
            print(f"❌ Model '{args.model_name}' not found")
            sys.exit(1)
            
        print(f"\n🤖 Model Details: {args.model_name}")
        print("=" * 50)
        print(f"Version: {metadata.get('version', 'Unknown')}")
        print(f"Stage: {metadata.get('stage', 'Unknown')}")
        print(f"Framework: {metadata.get('framework', 'Unknown')}")
        print(f"Last Updated: {metadata.get('last_updated', 'Unknown')}")
        
        if 'performance_metrics' in metadata:
            print(f"\n📈 Performance Metrics:")
            for metric, value in metadata['performance_metrics'].items():
                print(f"  {metric}: {value}")
                
        if 'compliance_status' in metadata:
            print(f"\n✅ Compliance Status:")
            for framework, status in metadata['compliance_status'].items():
                print(f"  {framework}: {status}")
                
        print()
        
    except Exception as e:
        print(f"❌ Failed to show model details: {e}")
        sys.exit(1)


def version_command(args):
    """Handle version command."""
    print("CIAF (Cognitive Insight Audit Framework)")
    print("Version: 0.1.0")
    print("Author: Denzil James Greenwood")
    print("License: Proprietary")


def compliance_report_cli():
    """CLI for generating compliance reports (legacy function)."""
    # This is kept for backward compatibility with the existing pyproject.toml
    import sys
    if len(sys.argv) >= 3:
        # Convert to new format
        framework = sys.argv[1]
        model_id = sys.argv[2]
        
        # Create mock args object
        class Args:
            def __init__(self):
                self.framework = framework
                self.model_id = model_id
                self.output = None
                self.format = "json"
                self.verbose = False
        
        compliance_command(Args())
    else:
        print("Usage: ciaf-compliance-report <framework> <model_id>")
        print("Available frameworks: eu_ai_act, nist_ai_rmf, gdpr, hipaa, sox, iso_27001")


def setup_metadata_cli():
    """CLI for setting up metadata storage (legacy function)."""
    # This is kept for backward compatibility with the existing pyproject.toml
    import sys
    if len(sys.argv) >= 2:
        project_name = sys.argv[1]
        
        # Create mock args object
        class Args:
            def __init__(self):
                self.project_name = project_name
                self.backend = "json"
                self.path = None
                self.template = "production"
        
        setup_command(Args())
    else:
        print("Usage: ciaf-setup-metadata <project_name>")


if __name__ == "__main__":
    main()
