"""
Healthcare Claims Processing Scenario

Demonstrates CIAF Agentic Execution Boundaries in a healthcare setting
with HIPAA compliance requirements and patient data protection.

This scenario shows:
- Agent identity with role-based access control
- Attribute-based access (same tenant only)
- Privilege elevation for sensitive operations
- Complete cryptographic audit trail
- HIPAA compliance obligations

Created: 2026-03-28
Author: Denzil James Greenwood
"""

from datetime import datetime, timedelta, timezone

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
    same_tenant_only,
)


def run_healthcare_claims_scenario():
    """Run a complete healthcare claims processing scenario."""
    print("=" * 70)
    print("CIAF Agentic Execution Boundaries - Healthcare Claims Scenario")
    print("=" * 70)
    print()

    # ===== Setup Components =====
    print("Setting up IAM, PAM, and Evidence components...")
    iam = IAMStore()
    pam = PAMStore()
    vault = EvidenceVault(signing_secret="healthcare-demo-secret-2026")
    policy = PolicyEngine(
        iam,
        pam,
        sensitive_actions={"approve_claim", "access_phi"},
        compliance_frameworks=["HIPAA", "EU_AI_Act"],
    )
    executor = ToolExecutor(policy, vault, pam)

    # ===== Define Roles =====
    print("Defining roles: claims_analyst, claims_approver...")

    analyst_role = RoleDefinition(
        name="claims_analyst",
        permissions=[
            Permission(
                action="read_claim",
                resource_type="insurance_claim",
                condition=same_tenant_only,
                description="Read claims from same tenant",
            ),
            Permission(
                action="read_record",
                resource_type="patient_record",
                condition=same_tenant_only,
                description="Read patient records from same tenant",
            ),
            Permission(
                action="update_claim",
                resource_type="insurance_claim",
                condition=same_tenant_only,
                description="Update claims from same tenant",
            ),
        ],
    )

    approver_role = RoleDefinition(
        name="claims_approver",
        permissions=[
            Permission(
                action="approve_claim",
                resource_type="insurance_claim",
                condition=same_tenant_only,
                description="Approve claims (requires elevation)",
                requires_elevation=True,
            ),
        ],
        inherits_from={"claims_analyst"},
    )

    iam.add_role(analyst_role)
    iam.add_role(approver_role)

    # ===== Create Agent Identity =====
    print("Creating AI agent identity: agent-claims-001...")

    claims_agent = Identity(
        principal_id="agent-claims-001",
        principal_type=PrincipalType.AGENT,
        display_name="Claims Analysis Agent",
        roles={"claims_analyst", "claims_approver"},
        attributes={"tenant": "acme-health", "clearance": "phi_access"},
        tenant_id="acme-health",
        environment="production",
    )

    iam.add_identity(claims_agent)
    print(f"✓ Agent fingerprint: {claims_agent.get_fingerprint()[:16]}...")
    print()

    # ===== Register Mock Tools =====
    def read_claim_tool(claim_id: str):
        return {"claim_id": claim_id, "status": "pending", "amount": 1500}

    def update_claim_tool(claim_id: str, status: str):
        return {"claim_id": claim_id, "updated_status": status, "success": True}

    def approve_claim_tool(claim_id: str, amount: float):
        return {
            "claim_id": claim_id,
            "amount": amount,
            "approved": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    executor.register_tool("read_claim", read_claim_tool)
    executor.register_tool("update_claim", update_claim_tool)
    executor.register_tool("approve_claim", approve_claim_tool)
    print("✓ Tools registered: read_claim, update_claim, approve_claim")
    print()

    # ===== Scenario 1: Standard Claims Analysis (Allowed) =====
    print("--- Scenario 1: Agent reads a claim (standard IAM permission) ---")

    claim_resource = Resource(
        resource_id="claim-12345",
        resource_type="insurance_claim",
        owner_tenant="acme-health",
        attributes={"patient_id": "P-9876", "status": "pending"},
        sensitivity_level="sensitive",
    )

    read_request = ActionRequest(
        action="read_claim",
        resource=claim_resource,
        params={"claim_id": "claim-12345"},
        justification="Analyzing claim for approval",
        requested_by=claims_agent,
        correlation_id="corr-2026-001",
    )

    result1 = executor.execute_tool("read_claim", read_request)
    print(f"Decision: {'ALLOWED' if result1.allowed else 'DENIED'}")
    print(f"Reason: {result1.reason}")
    print(f"Executed: {result1.executed}")
    print(f"Tool output: {result1.result['tool_output']}")
    print(f"Receipt ID: {result1.result['receipt_id']}")
    print()

    # ===== Scenario 2: Approve Claim Without Elevation (Denied) =====
    print("--- Scenario 2: Agent tries to approve claim WITHOUT elevation ---")

    approve_request_no_grant = ActionRequest(
        action="approve_claim",
        resource=claim_resource,
        params={"claim_id": "claim-12345", "amount": 1500},
        justification="Claim meets approval criteria",
        requested_by=claims_agent,
        correlation_id="corr-2026-002",
    )

    result2 = executor.execute_tool("approve_claim", approve_request_no_grant)
    print(f"Decision: {'ALLOWED' if result2.allowed else 'DENIED'}")
    print(f"Reason: {result2.reason}")
    print()

    # ===== Scenario 3: Grant Elevation and Approve (Allowed) =====
    print("--- Scenario 3: Grant elevation, then approve claim ---")

    # Manager approves elevation
    grant = pam.create_grant(
        principal_id="agent-claims-001",
        elevated_role="claims_approver",
        duration_minutes=30,
        approved_by="manager-health-001",
        ticket_reference="TICKET-2026-APPROVE-001",
        purpose="Approve batch of validated claims under $2000",
    )

    print(f"✓ Elevation grant created: {grant.grant_id}")
    print(f"  Valid until: {grant.valid_until}")
    print(f"  Approved by: {grant.approved_by}")
    print()

    # Now try to approve with the grant active
    approve_request_with_grant = ActionRequest(
        action="approve_claim",
        resource=claim_resource,
        params={"claim_id": "claim-12345", "amount": 1500},
        justification="Claim validated and approved per ticket TICKET-2026-APPROVE-001",
        requested_by=claims_agent,
        correlation_id="corr-2026-003",
    )

    result3 = executor.execute_tool("approve_claim", approve_request_with_grant)
    print(f"Decision: {'ALLOWED' if result3.allowed else 'DENIED'}")
    print(f"Reason: {result3.reason}")
    print(f"Executed: {result3.executed}")
    print(f"Grant used: {result3.elevation_grant_id}")
    print(f"Policy obligations: {result3.policy_obligations}")
    print(f"Tool output: {result3.result['tool_output']}")
    print()

    # ===== Verify Evidence Chain =====
    print("--- Cryptographic Evidence Verification ---")
    chain_valid = vault.verify_chain()
    print(f"Receipt chain integrity: {'✓ VALID' if chain_valid else '✗ INVALID'}")

    all_receipts = vault.get_all_receipts()
    print(f"Total receipts recorded: {len(all_receipts)}")

    # Show receipt details
    for i, receipt in enumerate(all_receipts, 1):
        print(f"\nReceipt {i}:")
        print(f"  ID: {receipt.receipt_id}")
        print(f"  Action: {receipt.action}")
        print(f"  Decision: {'ALLOWED' if receipt.decision else 'DENIED'}")
        print(f"  Elevation: {receipt.elevation_grant_id or 'None'}")
        print(f"  Signature: {receipt.signature[:32]}...")
        print(f"  Hash: {receipt.get_receipt_hash()[:32]}...")

    print()

    # ===== Summary =====
    print("=" * 70)
    print("Scenario Summary")
    print("=" * 70)
    print(f"✓ IAM identities: {len(iam.list_identities())}")
    print(f"✓ IAM roles: {len(iam.list_roles())}")
    print(f"✓ Elevation grants issued: {len(pam.list_all_grants())}")
    print(f"✓ Cryptographic receipts: {len(all_receipts)}")
    print(f"✓ Chain integrity: {chain_valid}")
    print()

    denied_receipts = vault.get_denied_receipts()
    print(f"Actions allowed: {len(all_receipts) - len(denied_receipts)}")
    print(f"Actions denied: {len(denied_receipts)}")
    print()

    print("Healthcare claims scenario complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_healthcare_claims_scenario()
