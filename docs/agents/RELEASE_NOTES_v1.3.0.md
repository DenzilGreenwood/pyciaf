# CIAF v1.3.0 Release Summary

## Release: Agentic Execution Boundaries

**Version:** 1.3.0
**Release Date:** 2026-03-28
**Type:** Major Feature Addition

## Overview

CIAF v1.3.0 introduces **Agentic Execution Boundaries** - a comprehensive zero-trust framework for controlling autonomous AI agent actions with cryptographic provenance and audit trails.

## What's New

### Agentic Execution Boundaries Module (`ciaf.agents`)

A complete IAM/PAM system specifically designed for AI agents:

#### Core Components

1. **Identity Management (IAM)**
   - Agent identity with cryptographic fingerprints
   - Role-based access control (RBAC)
   - Attribute-based access control (ABAC)
   - Tenant and environment isolation
   - Role inheritance and permission aggregation

2. **Privileged Access Management (PAM)**
   - Just-in-time (JIT) privilege elevation
   - Time-bound access grants
   - Approval workflow tracking
   - Usage limits and grant revocation
   - Ticket-based change management

3. **Policy Evaluation Engine**
   - Combined RBAC + ABAC evaluation
   - Sensitive action detection
   - Compliance framework integration
   - Policy obligation enforcement
   - Contextual authorization decisions

4. **Evidence Vault**
   - Cryptographic receipt generation
   - HMAC-SHA256 signatures
   - Hash-chained audit trails
   - Tamper-evident evidence preservation
   - Complete chain-of-custody tracking

5. **Tool Execution Mediation**
   - Controlled tool invocation
   - Parameter validation
   - Authorization enforcement
   - Automatic evidence recording
   - Grant usage tracking

#### Key Features

- ✅ **Zero-trust execution** - Every action requires authorization
- ✅ **Cryptographic proof** - HMAC-signed receipts with hash chaining
- ✅ **Privilege elevation** - JIT grants with approval workflows
- ✅ **Multi-tenant isolation** - Tenant-based access controls
- ✅ **Compliance integration** - Maps to HIPAA, SOX, ISO 27001, EU AI Act
- ✅ **Complete auditability** - Tamper-evident audit trails
- ✅ **Policy obligations** - Enforces two-person review, heightened logging, etc.

## File Structure

```
ciaf/agents/
├── README.md                   # Module documentation
├── __init__.py                 # Public API exports
├── core/                       # Core types and interfaces
│   ├── types.py               # Identity, Resource, ActionRequest, etc.
│   └── interfaces.py          # Protocol-based interfaces
├── iam/                        # Identity and Access Management
│   ├── store.py               # IAM store implementation
│   └── policy_conditions.py   # ABAC condition functions
├── pam/                        # Privileged Access Management
│   └── store.py               # PAM store implementation
├── policy/                     # Policy evaluation
│   └── engine.py              # Policy engine
├── evidence/                   # Evidence and receipts
│   └── vault.py               # Evidence vault
└── execution/                  # Tool execution
    └── executor.py            # Tool executor

docs/agents/
└── DEVELOPER_GUIDE.md          # Comprehensive developer guide

examples/agents_scenarios/
├── healthcare_claims.py        # HIPAA compliance scenario
├── financial_approvals.py     # SOX compliance scenario
├── production_changes.py      # Change management scenario
└── run_all.py                 # Run all scenarios
```

## Usage Examples

### Basic IAM Setup

```python
from ciaf.agents import IAMStore, Identity, Permission, RoleDefinition, same_tenant_only

iam = IAMStore()

# Define role
analyst_role = RoleDefinition(
    name="data_analyst",
    permissions=[
        Permission("read_record", "patient_record", same_tenant_only),
    ]
)
iam.add_role(analyst_role)

# Create agent identity
agent = Identity(
    principal_id="agent-001",
    principal_type="agent",
    roles={"data_analyst"},
    attributes={"tenant": "acme-health"}
)
iam.add_identity(agent)
```

### Full Execution with Evidence

```python
from ciaf.agents import (
    IAMStore, PAMStore, PolicyEngine, EvidenceVault, ToolExecutor,
    ActionRequest, Resource
)

# Initialize components
iam = IAMStore()
pam = PAMStore()
vault = EvidenceVault(signing_secret="your-secret")
policy = PolicyEngine(iam, pam, sensitive_actions={"approve_payment"})
executor = ToolExecutor(policy, vault, pam)

# Execute action
request = ActionRequest(
    action="approve_payment",
    resource=Resource(
        resource_id="payment-123",
        resource_type="payment",
        owner_tenant="acme-corp",
        attributes={"amount": 50000}
    ),
    params={"amount": 50000},
    justification="Vendor invoice approval",
    requested_by=agent
)

result = executor.execute(request)
print(f"Allowed: {result.allowed}, Reason: {result.reason}")

# Verify evidence chain
assert vault.verify_chain()
```

## Example Scenarios

Three complete scenarios are provided:

1. **Healthcare Claims Processing** (`examples/agents_scenarios/healthcare_claims.py`)
   - HIPAA compliance patterns
   - PHI access controls
   - Multi-level approval workflows

2. **Financial Payment Approvals** (`examples/agents_scenarios/financial_approvals.py`)
   - SOX compliance controls
   - Monetary threshold-based authorization
   - Two-person review requirements

3. **Production Infrastructure Changes** (`examples/agents_scenarios/production_changes.py`)
   - Change management workflows
   - Environment-based access control
   - Ticket-based approval

Run all scenarios:
```bash
python examples/agents_scenarios/run_all.py
```

## Integration with CIAF Core

The agents module seamlessly integrates with existing CIAF components:

- **Cryptographic Primitives**: Uses `ciaf.core.crypto` for hashing
- **Audit Trails**: Extends `ciaf.compliance.audit_trails` patterns
- **Compliance**: Maps to same regulatory frameworks
- **Evidence Storage**: Compatible with `ciaf.vault` for persistence

## API Updates

### New Imports

```python
from ciaf import (
    # Core types
    Identity,
    PrincipalType,
    Resource,
    ActionRequest,
    Permission,
    RoleDefinition,
    ExecutionResult,
    ElevationGrant,
    ActionReceipt,

    # Components
    IAMStore,
    PAMStore,
    PolicyEngine,
    EvidenceVault,
    ToolExecutor,

    # Policy conditions
    same_tenant_only,
    same_environment_only,
    sensitivity_level_check,
)
```

### Feature Flag

```python
from ciaf import AGENTS_AVAILABLE

if AGENTS_AVAILABLE:
    # Use agents module
    from ciaf.agents import IAMStore
```

## Documentation

- **Module README**: `ciaf/agents/README.md`
- **Developer Guide**: `docs/agents/DEVELOPER_GUIDE.md`
- **Example Scenarios**: `examples/agents_scenarios/`
- **API Reference**: See docstrings in each module

## Compliance Mappings

The agents module maps to:

- **EU AI Act** - Risk management, documentation, human oversight
- **NIST AI RMF** - Govern, Map, Measure, Manage
- **GDPR** - Data minimization, purpose limitation, audit trails
- **HIPAA** - PHI access controls, audit logging, minimum necessary
- **SOX** - Financial controls, two-person review, audit trails
- **ISO/IEC 27001** - Access control, audit trails, incident response

## Testing

All agents components include comprehensive docstrings and can be tested:

```python
# Unit test IAM
def test_iam_permissions():
    iam = IAMStore()
    # ... setup roles and identities
    assert iam.has_permission("agent-001", "read", "data")

# Integration test full flow
def test_full_execution():
    # ... setup components
    result = executor.execute_tool("test_tool", request)
    assert result.allowed
    assert vault.verify_chain()
```

## Breaking Changes

None. This is a purely additive release. All existing CIAF functionality remains unchanged.

## Upgrade Guide

No changes required for existing CIAF users. To use the new agents module:

1. Import the agents components as shown above
2. Review the example scenarios
3. Consult the Developer Guide for integration patterns

## Performance

The agents module has minimal overhead:

- IAM permission lookups: O(1) with caching
- Policy evaluation: O(n) where n = number of permissions
- Evidence recording: O(1) per action
- Chain verification: O(n) where n = number of receipts

## Future Roadmap

Potential future enhancements:

- Persistent storage backends for IAM/PAM/Evidence
- Advanced analytics on agent behavior
- Machine learning-based anomaly detection
- Integration with external identity providers (OIDC, SAML)
- Distributed evidence ledger with blockchain backend

## Credits

**Module Design & Implementation**: Following CIAF's architectural patterns
**Compliance Mapping**: Based on CIAF's regulatory framework
**Cryptographic Approach**: Using CIAF's core primitives

## License

Same as CIAF - Business Source License 1.1 (BUSL-1.1)

---

**Questions or Issues?**
See `docs/agents/DEVELOPER_GUIDE.md` or open an issue on GitHub.

**Version**: 1.3.0
**Release Date**: 2026-03-28
**Author**: Denzil James Greenwood
