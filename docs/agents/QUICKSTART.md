# CIAF Agentic Execution Boundaries - Quick Start

Get started with CIAF Agentic Execution Boundaries in 5 minutes.

## Installation

```bash
# Install CIAF (includes agents module)
pip install -e .
```

## Verify Installation

```python
from ciaf import AGENTS_AVAILABLE
print(f"Agents module available: {AGENTS_AVAILABLE}")
```

## 5-Minute Tutorial

### Step 1: Setup Components

```python
from ciaf.agents import IAMStore, PAMStore, PolicyEngine, EvidenceVault, ToolExecutor

# Initialize all components
iam = IAMStore()
pam = PAMStore()
vault = EvidenceVault(signing_secret="demo-secret-key")
policy = PolicyEngine(iam, pam, sensitive_actions={"delete_data"})
executor = ToolExecutor(policy, vault, pam)
```

### Step 2: Define a Role

```python
from ciaf.agents import RoleDefinition, Permission, same_tenant_only

# Create a role with permissions
analyst_role = RoleDefinition(
    name="data_analyst",
    permissions=[
        Permission(
            action="read_data",
            resource_type="dataset",
            condition=same_tenant_only,
            description="Read datasets from same tenant"
        ),
        Permission(
            action="export_report",
            resource_type="report",
            condition=same_tenant_only,
            description="Export reports from same tenant"
        ),
    ]
)

iam.add_role(analyst_role)
```

### Step 3: Create an Agent Identity

```python
from ciaf.agents import Identity, PrincipalType

agent = Identity(
    principal_id="agent-demo-001",
    principal_type=PrincipalType.AGENT,
    display_name="Demo Analysis Agent",
    roles={"data_analyst"},
    attributes={"tenant": "demo-tenant", "team": "analytics"},
    tenant_id="demo-tenant",
    environment="production"
)

iam.add_identity(agent)
```

### Step 4: Register a Tool

```python
def read_data_tool(dataset_id: str):
    """Simulated data reading tool."""
    return {
        "dataset_id": dataset_id,
        "records": 1000,
        "status": "success"
    }

executor.register_tool("read_data", read_data_tool)
```

### Step 5: Execute an Action

```python
from ciaf.agents import ActionRequest, Resource

# Create a resource
dataset = Resource(
    resource_id="dataset-2026-q1",
    resource_type="dataset",
    owner_tenant="demo-tenant",
    attributes={"size": "10GB", "classification": "confidential"}
)

# Create an action request
request = ActionRequest(
    action="read_data",
    resource=dataset,
    params={"dataset_id": "dataset-2026-q1"},
    justification="Q1 analysis for quarterly report",
    requested_by=agent,
    correlation_id="analysis-2026-q1-001"
)

# Execute with full authorization and audit
result = executor.execute_tool("read_data", request)

# Check result
if result.allowed and result.executed:
    print("✓ Action allowed and executed")
    print(f"Output: {result.result['tool_output']}")
    print(f"Receipt ID: {result.result['receipt_id']}")
else:
    print(f"✗ Action denied: {result.reason}")
```

### Step 6: Verify Evidence Chain

```python
# Verify cryptographic integrity
chain_valid = vault.verify_chain()
print(f"Evidence chain valid: {chain_valid}")

# Get all receipts
receipts = vault.get_all_receipts()
print(f"Total actions recorded: {len(receipts)}")

# Examine a receipt
receipt = receipts[0]
print(f"Receipt hash: {receipt.get_receipt_hash()[:32]}...")
print(f"Signature: {receipt.signature[:32]}...")
```

## Adding Privilege Elevation (PAM)

For sensitive operations requiring approval:

```python
# Mark action as sensitive
policy.add_sensitive_action("delete_data")

# Attempt without elevation (will be denied)
delete_request = ActionRequest(
    action="delete_data",
    resource=dataset,
    params={"dataset_id": "dataset-2026-q1"},
    justification="Remove outdated data",
    requested_by=agent
)

result = executor.execute_tool("delete_data", delete_request)
print(f"Without elevation: {result.allowed}")  # False

# Grant elevation
grant = pam.create_grant(
    principal_id="agent-demo-001",
    elevated_role="data_admin",
    duration_minutes=30,
    approved_by="manager-alice",
    ticket_reference="TICKET-2026-DELETE-001",
    purpose="Remove Q1 test data after migration"
)

print(f"✓ Elevation granted: {grant.grant_id}")

# Now attempt with elevation (will succeed if role has permission)
# Note: You'd need to add a role with delete_data permission first
```

## Complete Example

See `examples/agents_scenarios/healthcare_claims.py` for a complete, runnable example with:

- Multi-role authorization
- Privilege elevation workflows
- Compliance obligations
- Complete evidence trails

Run it:
```bash
python examples/agents_scenarios/healthcare_claims.py
```

## Next Steps

1. **Read the Developer Guide**: `docs/agents/DEVELOPER_GUIDE.md`
2. **Run Example Scenarios**: `python examples/agents_scenarios/run_all.py`
3. **Integrate with Your System**: Use the patterns in the examples

## Common Patterns

### Multi-Tenant Isolation

```python
# All permissions use tenant checks
Permission(
    action="read",
    resource_type="data",
    condition=same_tenant_only
)

# Identities have tenant binding
Identity(
    principal_id="agent-customer-a",
    tenant_id="customer-a",
    ...
)

# Resources specify tenant ownership
Resource(
    resource_id="data-123",
    owner_tenant="customer-a",
    ...
)
```

### Time-Bound Elevation

```python
# Short-lived grant for high-risk operations
grant = pam.create_grant(
    principal_id="agent-001",
    elevated_role="admin",
    duration_minutes=15,  # Only valid for 15 minutes
    approved_by="security-officer",
    ticket_reference="URGENT-TICKET-001",
    purpose="Emergency security patching"
)
```

### Compliance Tracking

```python
# Enable compliance frameworks
policy = PolicyEngine(
    iam,
    pam,
    compliance_frameworks=["HIPAA", "SOX", "GDPR"]
)

# Check obligations after execution
result = executor.execute(request)
if "hipaa_audit_logging" in result.policy_obligations:
    send_to_hipaa_log(result)
```

## Troubleshooting

### "Identity not found"
Make sure you've registered the identity with `iam.add_identity(agent)`

### "Action requires privilege elevation"
Create a PAM grant using `pam.create_grant(...)`

### "Permission denied"
Check that:
1. The role is registered: `iam.add_role(role)`
2. The identity has the role: `agent.roles = {"role_name"}`
3. The permission exists in the role
4. The ABAC condition passes (if any)

## Help & Resources

- **Documentation**: `docs/agents/`
- **Examples**: `examples/agents_scenarios/`
- **Module README**: `ciaf/agents/README.md`
- **API Reference**: See docstrings in code

---

**Ready to dive deeper?** Check out the [Developer Guide](DEVELOPER_GUIDE.md) for comprehensive integration patterns.
