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


def compliance_report_cli():
    """CLI for generating compliance reports."""
    parser = argparse.ArgumentParser(
        description="Generate CIAF compliance reports", prog="ciaf-compliance-report"
    )

    parser.add_argument(
        "framework",
        choices=["eu_ai_act", "nist_ai_rmf", "gdpr", "hipaa", "sox", "iso_27001"],
        help="Compliance framework to generate report for",
    )

    parser.add_argument("model_id", help="Model ID to generate report for")

    parser.add_argument(
        "-o", "--output", help="Output file path (default: compliance_report.json)"
    )

    parser.add_argument(
        "-s",
        "--storage",
        default="ciaf_metadata",
        help="Metadata storage path (default: ciaf_metadata)",
    )

    parser.add_argument(
        "--format",
        choices=["json", "html", "pdf"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args()

    try:
        # Try to import compliance tools
        try:
            from .compliance import ComplianceReportGenerator, ComplianceFramework
            from .compliance.audit_trails import AuditTrailGenerator
        except ImportError:
            print("❌ Compliance reporting tools not found")
            print(
                "Make sure the compliance module is available or install ciaf[full]"
            )
            sys.exit(1)

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
        print(f"🚀 Generating {args.framework} compliance report for {args.model_id}...")
        report = reporter.generate_executive_summary_report(
            frameworks=[framework_enum],
            audit_generator=audit_generator,
            model_version="1.0.0",  # Default version
        )
        
        # Save report
        output_path = args.output or f"compliance_report_{args.framework}_{args.model_id}.json"
        
        if args.format == "json":
            report_dict = asdict(report)
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
        elif args.format == "html":
            # Generate basic HTML report
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
        else:
            print(f"❌ Format '{args.format}' not supported. Available: json, html")
            sys.exit(1)

        print(f"✅ Compliance report generated: {output_path}")

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def setup_metadata_cli():
    """CLI for setting up metadata storage."""
    parser = argparse.ArgumentParser(
        description="Set up CIAF metadata storage for a project",
        prog="ciaf-setup-metadata",
    )

    parser.add_argument("project_name", help="Name of your project")

    parser.add_argument(
        "-b",
        "--backend",
        choices=["json", "sqlite", "pickle"],
        default="json",
        help="Storage backend (default: json)",
    )

    parser.add_argument(
        "-p", "--path", help="Custom storage path (default: {project_name}_metadata)"
    )

    parser.add_argument(
        "-t",
        "--template",
        choices=["development", "production", "testing", "high_performance"],
        default="production",
        help="Configuration template (default: production)",
    )

    args = parser.parse_args()

    try:
        from .metadata_config import MetadataConfig
        from .metadata_storage import MetadataStorage

        # Determine storage path
        storage_path = args.path or f"{args.project_name}_metadata"
        config_file = f"{args.project_name}_metadata_config.json"

        print(f"🚀 Setting up CIAF metadata storage for '{args.project_name}'")
        print("=" * 50)

        # Create configuration
        print(f"📋 Creating configuration from '{args.template}' template...")
        create_config_template(args.template, config_file)

        # Update configuration
        config = MetadataConfig(config_file)
        config.set("storage_backend", args.backend)
        config.set("storage_path", storage_path)
        config.save_to_file(config_file)

        # Initialize storage
        print(f"🗄️ Initializing {args.backend} storage at '{storage_path}'...")
        storage = MetadataStorage(storage_path, args.backend)

        # Create directory structure
        project_dir = Path(storage_path)
        project_dir.mkdir(parents=True, exist_ok=True)

        subdirs = ["exports", "backups", "reports"]
        for subdir in subdirs:
            (project_dir / subdir).mkdir(exist_ok=True)

        print("\n✅ Setup completed successfully!")
        print(f"   🔸 Project: {args.project_name}")
        print(f"   🔸 Backend: {args.backend}")
        print(f"   🔸 Storage: {storage_path}")
        print(f"   🔸 Config: {config_file}")

        print("\n🚀 Next Steps:")
        print(f"1. Review the configuration in '{config_file}'")
        print("2. Import CIAF in your project:")
        print("   from ciaf import ModelMetadataManager")
        print(
            f"3. Initialize: manager = ModelMetadataManager('{args.project_name}', '1.0.0')"
        )

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # This allows the module to be run directly for testing
    if len(sys.argv) > 1 and sys.argv[1] == "compliance":
        sys.argv = sys.argv[1:]  # Remove the "compliance" argument
        compliance_report_cli()
    elif len(sys.argv) > 1 and sys.argv[1] == "setup":
        sys.argv = sys.argv[1:]  # Remove the "setup" argument
        setup_metadata_cli()
    else:
        print("CIAF CLI Tools")
        print("Available commands:")
        print("  ciaf-compliance-report - Generate compliance reports")
        print("  ciaf-setup-metadata - Set up metadata storage")
