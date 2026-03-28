"""
Production Infrastructure Changes Scenario

Demonstrates CIAF Agentic Execution Boundaries for production system changes
with change management, approval workflows, and comprehensive audit trails.

This scenario shows:
- Environment-based access control (dev, staging, production)
- Change approval workflow with ticket tracking
- Rollback capabilities with audit preservation
- Infrastructure-as-Code (IaC) agent execution boundaries
- Complete chain-of-custody for production changes

Created: 2026-03-28
Author: Denzil James Greenwood
"""

from ciaf.agents import (
    ActionRequest,
    EvidenceVault,
    IAMStore,
    Identity,
    PAMStore,
    Permission,
    PolicyEngine,
    PrincipalType,
    Resource,
    RoleDefinition,
    ToolExecutor,
    combine_and,
    same_environment_only,
    same_tenant_only,
)


def run_production_changes_scenario():
    """Run a complete production infrastructure changes scenario."""
    print("=" * 70)
    print("CIAF Agentic Execution Boundaries - Production Changes Scenario")
    print("=" * 70)
    print()

    # Setup
    print("Setting up components...")
    iam = IAMStore()
    pam = PAMStore()
    vault = EvidenceVault(signing_secret="infra-demo-secret-2026")
    policy = PolicyEngine(
        iam,
        pam,
        sensitive_actions={
            "deploy_production",
            "modify_firewall",
            "delete_resources",
            "scale_production",
        },
        compliance_frameworks=["ISO_27001", "SOC2"],
    )
    executor = ToolExecutor(policy, vault, pam)

    # Define roles
    print("Defining infrastructure roles...")

    dev_ops_role = RoleDefinition(
        name="devops_engineer",
        permissions=[
            Permission(
                action="read_config",
                resource_type="infrastructure",
                condition=same_tenant_only,
                description="Read infrastructure configuration",
            ),
            Permission(
                action="deploy_staging",
                resource_type="infrastructure",
                condition=combine_and(same_tenant_only, same_environment_only),
                description="Deploy to staging environment",
            ),
            Permission(
                action="monitor_metrics",
                resource_type="infrastructure",
                condition=same_tenant_only,
                description="Monitor system metrics",
            ),
        ],
    )

    prod_ops_role = RoleDefinition(
        name="production_operator",
        permissions=[
            Permission(
                action="deploy_production",
                resource_type="infrastructure",
                condition=combine_and(same_tenant_only, same_environment_only),
                description="Deploy to production (requires elevation)",
                requires_elevation=True,
            ),
            Permission(
                action="scale_production",
                resource_type="infrastructure",
                condition=combine_and(same_tenant_only, same_environment_only),
                description="Scale production resources (requires elevation)",
                requires_elevation=True,
            ),
        ],
        inherits_from={"devops_engineer"},
    )

    iam.add_role(dev_ops_role)
    iam.add_role(prod_ops_role)

    # Create infrastructure agent
    infra_agent = Identity(
        principal_id="agent-infra-automation-001",
        principal_type=PrincipalType.AGENT,
        display_name="Infrastructure Automation Agent",
        roles={"production_operator"},
        attributes={"team": "platform", "access_level": "production"},
        tenant_id="acme-platform",
        environment="production",
    )

    iam.add_identity(infra_agent)
    print(f"✓ Infrastructure agent created: {infra_agent.principal_id}")
    print()

    # Register tools
    def deploy_production_tool(deployment_id: str, version: str):
        return {
            "deployment_id": deployment_id,
            "version": version,
            "status": "deployed",
            "instances": 5,
            "deployed_at": "2026-03-28T15:30:00Z",
        }

    def scale_production_tool(resource_id: str, target_instances: int):
        return {
            "resource_id": resource_id,
            "previous_count": 5,
            "new_count": target_instances,
            "status": "scaling",
        }

    executor.register_tool("deploy_production", deploy_production_tool)
    executor.register_tool("scale_production", scale_production_tool)
    print("✓ Tools registered: deploy_production, scale_production")
    print()

    # Scenario 1: Deploy to production without approval (denied)
    print("--- Scenario 1: Deploy to production WITHOUT change approval ---")

    prod_resource = Resource(
        resource_id="prod-api-cluster",
        resource_type="infrastructure",
        owner_tenant="acme-platform",
        attributes={
            "environment": "production",
            "service": "api-gateway",
            "current_version": "v2.3.1",
        },
        sensitivity_level="critical",
    )

    request1 = ActionRequest(
        action="deploy_production",
        resource=prod_resource,
        params={"deployment_id": "deploy-2026-001", "version": "v2.4.0"},
        justification="Deploy new API version with performance improvements",
        requested_by=infra_agent,
        correlation_id="deploy-2026-001",
    )

    result1 = executor.execute_tool("deploy_production", request1)
    print(f"Decision: {'ALLOWED' if result1.allowed else 'DENIED'}")
    print(f"Reason: {result1.reason}")
    print()

    # Scenario 2: Request change approval and deploy
    print("--- Scenario 2: Get change approval, then deploy to production ---")

    # SRE manager approves the change
    grant = pam.create_grant(
        principal_id="agent-infra-automation-001",
        elevated_role="production_operator",
        duration_minutes=120,
        approved_by="sre-manager-bob-001",
        ticket_reference="CHG-2026-03-28-001",
        purpose="Deploy API v2.4.0 with approved change window (2026-03-28 15:00-17:00 UTC)",
        max_uses=3,  # Deploy + potential rollback + verification deploy
    )

    print(f"✓ Change approval grant: {grant.grant_id}")
    print(f"  Change ticket: {grant.ticket_reference}")
    print(f"  Approved by: {grant.approved_by}")
    print(f"  Valid for: 2 hours")
    print(f"  Max operations: {grant.max_uses}")
    print()

    request2 = ActionRequest(
        action="deploy_production",
        resource=prod_resource,
        params={"deployment_id": "deploy-2026-001", "version": "v2.4.0"},
        justification=f"Approved deployment per change ticket {grant.ticket_reference}",
        requested_by=infra_agent,
        correlation_id="deploy-2026-002",
    )

    result2 = executor.execute_tool("deploy_production", request2)
    print(f"Decision: {'ALLOWED' if result2.allowed else 'DENIED'}")
    print(f"Reason: {result2.reason}")
    print(f"Grant used: {result2.elevation_grant_id}")
    print(f"Policy obligations: {result2.policy_obligations}")
    print(f"Executed: {result2.executed}")
    if result2.executed:
        print(f"Output: {result2.result['tool_output']}")
    print()

    # Scenario 3: Auto-scale using same grant
    print("--- Scenario 3: Auto-scale production (using same change approval) ---")

    request3 = ActionRequest(
        action="scale_production",
        resource=prod_resource,
        params={"resource_id": "prod-api-cluster", "target_instances": 8},
        justification=f"Scale up for traffic spike - within change window {grant.ticket_reference}",
        requested_by=infra_agent,
        correlation_id="scale-2026-001",
    )

    result3 = executor.execute_tool("scale_production", request3)
    print(f"Decision: {'ALLOWED' if result3.allowed else 'DENIED'}")
    print(f"Reason: {result3.reason}")
    print(f"Grant used: {result3.elevation_grant_id}")
    print(f"Executed: {result3.executed}")
    if result3.executed:
        print(f"Output: {result3.result['tool_output']}")
    print()

    # Check grant usage
    updated_grant = pam.get_grant(grant.grant_id)
    print(f"Grant usage: {updated_grant.used_count}/{updated_grant.max_uses}")
    print()

    # Verify evidence
    print("--- Cryptographic Evidence Chain ---")
    chain_valid = vault.verify_chain()
    all_receipts = vault.get_all_receipts()

    print(f"Receipt chain integrity: {'✓ VALID' if chain_valid else '✗ INVALID'}")
    print(f"Total operations recorded: {len(all_receipts)}")
    print()

    # Show change audit trail
    print("--- Change Management Audit Trail ---")
    approved_changes = [r for r in all_receipts if r.elevation_grant_id]
    print(f"Operations with change approval: {len(approved_changes)}")

    for i, receipt in enumerate(approved_changes, 1):
        print(f"\n  Change {i}:")
        print(f"    Action: {receipt.action}")
        print(f"    Resource: {receipt.resource_id}")
        print(f"    Correlation: {receipt.correlation_id}")
        print(f"    Timestamp: {receipt.timestamp}")
        print(f"    Receipt hash: {receipt.get_receipt_hash()[:32]}...")

    print()

    # Summary
    print("=" * 70)
    print("Scenario Summary")
    print("=" * 70)
    print(f"✓ Total infrastructure operations: {len(all_receipts)}")
    print(f"✓ Approved operations: {len(approved_changes)}")
    print(f"✓ Denied operations: {len([r for r in all_receipts if not r.decision])}")
    print(f"✓ Change tickets referenced: {len(set(r.elevation_grant_id for r in approved_changes))}")
    print(f"✓ Chain integrity: {chain_valid}")
    print()

    # Critical operations logging
    critical_ops = [
        r
        for r in all_receipts
        if "heightened_logging" in r.policy_obligations
    ]
    print(f"✓ Operations with heightened logging: {len(critical_ops)}")
    print()

    print("Production infrastructure changes scenario complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_production_changes_scenario()
