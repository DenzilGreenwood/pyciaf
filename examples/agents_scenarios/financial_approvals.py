"""
Financial Payment Approvals Scenario

Demonstrates CIAF Agentic Execution Boundaries for financial transactions
with SOX compliance and dual-control requirements.

This scenario shows:
- Multi-level approval workflows
- Monetary threshold-based authorization
- Just-in-time privilege elevation
- SOX compliance obligations (two-person review, heightened logging)
- Complete audit trail for regulatory compliance

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
    same_tenant_only,
)


def threshold_check(max_amount: float):
    """Create a condition checking payment amount threshold."""
    from ciaf.agents.core.types import Identity, Resource

    def condition(identity: Identity, resource: Resource) -> bool:
        amount = resource.attributes_dict.get("amount", 0)
        return amount <= max_amount

    return condition


def run_financial_approvals_scenario():
    """Run a complete financial payment approvals scenario."""
    print("=" * 70)
    print("CIAF Agentic Execution Boundaries - Financial Approvals Scenario")
    print("=" * 70)
    print()

    # Setup
    print("Setting up components...")
    iam = IAMStore()
    pam = PAMStore()
    vault = EvidenceVault(signing_secret="finance-demo-secret-2026")
    policy = PolicyEngine(
        iam,
        pam,
        sensitive_actions={"approve_payment", "wire_transfer"},
        compliance_frameworks=["SOX", "EU_AI_Act"],
    )
    executor = ToolExecutor(policy, vault, pam)

    # Define tiered approval roles
    print("Defining roles with tiered approval limits...")

    junior_approver = RoleDefinition(
        name="junior_approver",
        permissions=[
            Permission(
                action="read_payment",
                resource_type="payment",
                condition=same_tenant_only,
                description="Read payment requests",
            ),
            Permission(
                action="approve_payment",
                resource_type="payment",
                condition=combine_and(same_tenant_only, threshold_check(10000)),
                description="Approve payments up to $10,000",
                requires_elevation=False,  # Standing permission
            ),
        ],
    )

    senior_approver = RoleDefinition(
        name="senior_approver",
        permissions=[
            Permission(
                action="approve_payment",
                resource_type="payment",
                condition=combine_and(same_tenant_only, threshold_check(50000)),
                description="Approve payments up to $50,000 (requires elevation)",
                requires_elevation=True,
            ),
        ],
        inherits_from={"junior_approver"},
    )

    iam.add_role(junior_approver)
    iam.add_role(senior_approver)

    # Create payment agent
    payment_agent = Identity(
        principal_id="agent-payment-001",
        principal_type=PrincipalType.AGENT,
        display_name="Payment Approval Agent",
        roles={"senior_approver"},
        attributes={"department": "finance", "clearance": "financial_data"},
        tenant_id="acme-corp",
        environment="production",
    )

    iam.add_identity(payment_agent)
    print(f"✓ Payment agent created: {payment_agent.principal_id}")
    print()

    # Register tools
    def approve_payment_tool(payment_id: str, amount: float):
        return {
            "payment_id": payment_id,
            "amount": amount,
            "status": "approved",
            "approved_at": "2026-03-28T10:00:00Z",
        }

    executor.register_tool("approve_payment", approve_payment_tool)
    print("✓ Tool registered: approve_payment")
    print()

    # Scenario 1: Small payment (under $10K, allowed with standing permission)
    print("--- Scenario 1: Approve $5,000 payment (within standing authority) ---")

    small_payment = Resource(
        resource_id="payment-001",
        resource_type="payment",
        owner_tenant="acme-corp",
        attributes={"amount": 5000, "vendor": "SupplyCo", "invoice": "INV-2026-001"},
        sensitivity_level="standard",
    )

    request1 = ActionRequest(
        action="approve_payment",
        resource=small_payment,
        params={"payment_id": "payment-001", "amount": 5000},
        justification="Routine supplier payment per contract",
        requested_by=payment_agent,
        correlation_id="fin-2026-001",
    )

    result1 = executor.execute_tool("approve_payment", request1)
    print(f"Decision: {'ALLOWED' if result1.allowed else 'DENIED'}")
    print(f"Reason: {result1.reason}")
    print(f"Executed: {result1.executed}")
    if result1.executed:
        print(f"Output: {result1.result['tool_output']}")
    print()

    # Scenario 2: Large payment without elevation (denied)
    print("--- Scenario 2: Approve $35,000 payment WITHOUT elevation ---")

    large_payment = Resource(
        resource_id="payment-002",
        resource_type="payment",
        owner_tenant="acme-corp",
        attributes={
            "amount": 35000,
            "vendor": "MegaVendor",
            "invoice": "INV-2026-002",
            "contract": "CON-2026-ANNUAL",
        },
        sensitivity_level="sensitive",
    )

    request2 = ActionRequest(
        action="approve_payment",
        resource=large_payment,
        params={"payment_id": "payment-002", "amount": 35000},
        justification="Annual licensing fee",
        requested_by=payment_agent,
        correlation_id="fin-2026-002",
    )

    result2 = executor.execute_tool("approve_payment", request2)
    print(f"Decision: {'ALLOWED' if result2.allowed else 'DENIED'}")
    print(f"Reason: {result2.reason}")
    print()

    # Scenario 3: Grant elevation and approve large payment
    print("--- Scenario 3: Get approval grant, then approve $35,000 payment ---")

    # CFO approves elevation
    grant = pam.create_grant(
        principal_id="agent-payment-001",
        elevated_role="senior_approver",
        duration_minutes=60,
        approved_by="cfo-alice-001",
        ticket_reference="TICKET-FIN-2026-Q1-BATCH",
        purpose="Approve Q1 vendor payments batch (under $50K each)",
        max_uses=10,  # Can approve up to 10 payments
    )

    print(f"✓ Elevation grant: {grant.grant_id}")
    print(f"  Approved by: {grant.approved_by}")
    print(f"  Purpose: {grant.purpose}")
    print(f"  Max uses: {grant.max_uses}")
    print()

    request3 = ActionRequest(
        action="approve_payment",
        resource=large_payment,
        params={"payment_id": "payment-002", "amount": 35000},
        justification="Annual licensing fee - approved per ticket TICKET-FIN-2026-Q1-BATCH",
        requested_by=payment_agent,
        correlation_id="fin-2026-003",
    )

    result3 = executor.execute_tool("approve_payment", request3)
    print(f"Decision: {'ALLOWED' if result3.allowed else 'DENIED'}")
    print(f"Reason: {result3.reason}")
    print(f"Grant used: {result3.elevation_grant_id}")
    print(f"Policy obligations: {result3.policy_obligations}")
    print(f"Executed: {result3.executed}")
    if result3.executed:
        print(f"Output: {result3.result['tool_output']}")
    print()

    # Verify evidence
    print("--- Evidence Chain Verification ---")
    chain_valid = vault.verify_chain()
    all_receipts = vault.get_all_receipts()

    print(f"Receipt chain integrity: {'✓ VALID' if chain_valid else '✗ INVALID'}")
    print(f"Total receipts: {len(all_receipts)}")
    print()

    # Show SOX compliance obligations
    print("--- SOX Compliance Obligations ---")
    sox_receipts = [r for r in all_receipts if "sox_controls" in r.policy_obligations]
    print(f"Actions with SOX controls: {len(sox_receipts)}")

    two_person_receipts = [
        r for r in all_receipts if "two_person_review" in r.policy_obligations
    ]
    print(f"Actions requiring two-person review: {len(two_person_receipts)}")
    print()

    # Summary
    print("=" * 70)
    print("Scenario Summary")
    print("=" * 70)
    print(f"✓ Total payment requests: {len(all_receipts)}")
    print(f"✓ Approved: {len([r for r in all_receipts if r.decision])}")
    print(f"✓ Denied: {len([r for r in all_receipts if not r.decision])}")
    print(f"✓ With elevation: {len([r for r in all_receipts if r.elevation_grant_id])}")
    print(f"✓ Chain integrity: {chain_valid}")
    print()
    print("Financial approvals scenario complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_financial_approvals_scenario()
