# CIAF Agentic Execution Boundaries

**Zero-trust execution boundaries for autonomous AI agents with cryptographic provenance**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)]()
[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-blue.svg)]()
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()

## The Problem

Current AI agents operate with over-privileged "God-mode" access, performing actions without verification, approval, or audit trails. This creates catastrophic risk in production systems:

- ❌ **Unauthorized data exfiltration** - No access controls
- ❌ **Destructive operations** - No approval workflows
- ❌ **Zero accountability** - No audit trails
- ❌ **Compliance violations** - No regulatory alignment

## The Solution

**CIAF Agentic Execution Boundaries** provides a zero-trust framework ensuring agents only execute authorized actions with verifiable cryptographic proof. Every action is authenticated, authorized, mediated, and audited—with tamper-evident evidence chains proving what happened and who approved it.

## Why Agentic Actors Need Independent IAM/PAM

### The Fundamental Difference

**Agents are not just service accounts.** Traditional enterprise IAM/PAM was designed for three types of principals:

1. **Humans** - Make deliberate, bounded decisions
2. **Service Accounts** - Execute narrow, pre-defined functions
3. **Applications** - Run deterministic workflows

**Autonomous agents break these assumptions.** An agent can:
- Interpret goals and choose actions dynamically
- Chain tools in novel sequences
- Adapt behavior based on context
- Operate continuously at machine speed
- Discover and exploit permission combinations

This makes agents closer to **semi-autonomous operators** than background processes.

### What Enterprise IAM Cannot Answer for Agents

Traditional IAM can answer:
- ✅ What account is this?
- ✅ What roles does it have?
- ✅ What systems can it reach?

But agentic control also needs to answer:
- ❓ **What objective is currently active?**
- ❓ **What task scope is authorized?**
- ❓ **What data classes are in bounds?**
- ❓ **Has the action sequence drifted from approved purpose?**
- ❓ **Is the agent attempting emergent privilege aggregation?**

These questions require our **Policy Plane + Privilege Plane + Execution Plane + Evidence Plane** model—much more than ordinary RBAC alone.

### The Rouge Agent Amplification Problem

**If an agent is plugged directly into existing IAM/PAM without an independent mediation layer**, it can:

- 🔴 **Reuse trusted APIs faster than humans** - Execute thousands of actions per minute
- 🔴 **Enumerate accessible resources at scale** - Discover all permissions systematically
- 🔴 **Chain low-risk permissions into high-impact outcomes** - Combine benign actions into dangerous sequences
- 🔴 **Act continuously without fatigue** - Operate 24/7 without oversight gaps
- 🔴 **Exploit overbroad legacy permissions** - Find and use permissions humans rarely exercise

**The risk is not just "the agent has access."**

**The risk is: the agent can operationalize that access more aggressively and creatively than the control model anticipated.**

### Why Human PAM Logic Doesn't Transfer

Traditional human PAM assumes:

```
1. A person requests elevation
2. There is a ticket
3. There is an approver
4. The person uses access for a short task window
5. Trust is based on identity and intent
```

**Agentic PAM requires additional constraints:**

```
✓ Purpose binding - What specific task is authorized
✓ Task binding - What sequence of actions is allowed
✓ Tool binding - What specific tools can be used
✓ Data-scope binding - What data can be accessed
✓ Maximum action count - Hard limits on operations
✓ Mandatory runtime mediation - Every action is wrapped
✓ Immediate revocation on anomaly - Automatic cutoff
✓ Cryptographic proof - Tamper-evident evidence
```

This is why our JIT grants include `ticket_reference`, `purpose`, `scope`, `max_uses`, and `valid_until` fields. These are not optional niceties—they are **essential runtime constraints**.

### The Architectural Principle

**Enterprise IAM/PAM should remain the root authority, but agentic actors should operate through an independent agentic IAM/PAM control layer that translates enterprise permissions into tightly mediated, task-scoped, evidence-backed execution rights.**

This gives us a clean model:

```
┌─────────────────────────────────────────────────────────────┐
│              Enterprise IAM/PAM                             │
│         (Source of Organizational Authority)                │
└────────────────────┬────────────────────────────────────────┘
                     │ Delegates to
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              Agentic IAM/PAM (This Module)                  │
│         Runtime Control Envelope for Autonomous Actors      │
│  • Identity resolution    • Policy evaluation               │
│  • Privilege verification • Mediated execution              │
│  • Task-scoped grants     • Context awareness               │
└────────────────────┬────────────────────────────────────────┘
                     │ Enforces through
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              CIAF Evidence Layer                            │
│         Proof That Controls Were Actually Enforced          │
│  • Cryptographic receipts • Hash-chained audit trails       │
│  • Tamper-evident vault   • Independent verification        │
└─────────────────────────────────────────────────────────────┘
```

### An Analogy

A company may trust an employee badge system for building access. That **does not** mean you let an autonomous robot roam the building with a master badge and no movement boundaries.

Instead, the robot would need:
- A unique machine identity
- Zone restrictions
- Mission scope
- Supervisor approval for restricted areas
- Live mediation and monitoring
- Full telemetry and forensic evidence

**That is essentially independent IAM/PAM for agentic actors.**

### The Core Principle

> **No agent should ever call a privileged tool, API, or system directly using inherited enterprise permissions alone.**

Every action should require:

1. ✅ Verified agent identity
2. ✅ Policy check against task context
3. ✅ Privilege check for sensitive actions
4. ✅ Mediated execution wrapper
5. ✅ Cryptographic receipt

This is the difference between:
- ❌ "An AI connected to IAM"
- ✅ "An agent operating inside an enforceable execution boundary"

### When This Matters Most

The independent layer becomes essential for:

- 🏥 **Healthcare agents** - PHI access, HIPAA compliance
- 💰 **Payment/finance agents** - Financial transactions, SOX controls
- 🔧 **Admin/DevOps agents** - Infrastructure changes, production access
- 📊 **Data export agents** - Sensitive data extraction, GDPR compliance
- ☁️ **Agents with cloud control-plane access** - Resource provisioning, deletion
- 🤖 **Multi-agent systems** - Where one agent can trigger another

In these cases, the combination of **autonomy + privilege + scale** creates a control problem that normal IAM was not designed to solve.

### Design Principles for Agentic IAM/PAM

This module enforces seven non-negotiable principles:

#### 1. Separate Principal Type for Agents
Never let agents masquerade as generic service accounts. Use `PrincipalType.AGENT` explicitly.

#### 2. No Raw Standing Privilege
Agents receive scoped, expiring capabilities—not broad persistent permissions. Use conditions and `requires_elevation=True`.

#### 3. Purpose-Bound PAM Grants
Elevation must include:
- `reason` - Why is this needed?
- `ticket_reference` - What change ticket approves this?
- `approved_by` - Who authorized it?
- `scope` - What boundaries apply?
- `valid_until` - When does it expire?
- `max_uses` - How many times can it be used?

#### 4. Mediated Execution Only
Tools are wrapped through `ToolExecutor`. No direct unmanaged calls.

#### 5. Context-Aware Policy
Decisions factor in: tenant, environment, data classification, task type, approval state, and risk score.

#### 6. Evidence by Default
Every allow, deny, elevation, and override produces a signed receipt. No silent actions.

#### 7. Independent Verification
Security and audit teams can verify controls without trusting the agent framework itself. Evidence chain verification is cryptographically independent.

### The Strategic Position

**This is not just a security feature. It is a governance architecture.**

Most agent frameworks focus on orchestration and tool use. We focus on **independently enforceable access governance** for autonomous actors.

Our position:

> **Human IAM/PAM was designed for people and static services. Agentic actors require a separate control layer because they can interpret goals, chain actions, and exploit inherited privileges autonomously. Enterprise IAM should define the outer authority, but agentic IAM/PAM must enforce task-scoped access, mediated execution, and tamper-evident proof at runtime.**

### Integration with Enterprise Systems

We are **not** replacing enterprise IAM. We are **constraining** how agents consume enterprise permissions.

**Recommended architecture:**

1. **Enterprise IAM** defines who/what the agent represents organizationally
2. **Agentic IAM/PAM** (this module) enforces what the agent can do **right now** for **this specific task**
3. **CIAF Evidence** proves both systems worked correctly together

This creates **defense in depth** where:
- Enterprise IAM handles identity and broad authorization
- Agentic IAM/PAM handles runtime mediation and task scoping
- CIAF Evidence provides cryptographic proof of compliance

### Why This Matters for CIAF

CIAF provides cryptographic provenance for:
- ✅ Training data lineage
- ✅ Model parameter fingerprints
- ✅ Inference receipts
- ✅ Audit trail integrity

**Agentic Execution Boundaries extends this to:**
- ✅ **Agent action authorization**
- ✅ **Privilege elevation approval**
- ✅ **Tool invocation mediation**
- ✅ **Complete chain-of-custody for agent operations**

This completes CIAF's vision: **cryptographic provenance from data collection through model training through autonomous agent execution**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Action Request                         │
│  (Agent, Action, Resource, Parameters, Justification)       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                    Identity Resolution                      │
│  • Authenticate agent principal                             │
│  • Load roles and attributes                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                    Policy Evaluation                        │
│  • Check standing IAM permissions                           │
│  • Evaluate boundary conditions (ABAC)                      │
│  • Determine if elevation required                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  Privilege Verification                     │
│  • Find active elevation grant (if required)                │
│  • Validate grant scope and expiry                          │
│  • Check approver and ticket reference                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                   Mediated Execution                        │
│  • Invoke guarded tool wrapper                              │
│  • Apply runtime controls                                   │
│  • Capture execution result                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                   Evidence Recording                        │
│  • Generate signed receipt                                  │
│  • Chain to prior receipt                                   │
│  • Store in tamper-evident vault                            │
└─────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Execution Boundaries

An agentic execution boundary defines the complete control envelope for an AI agent:

```
Agent Boundary = Identity + Authorization + Context + Elevation Control + Mediation + Auditability
```

### Control Planes

1. **Identity Plane** - Who is the agent?
   - Unique principal IDs
   - Workload credentials
   - Role assignments
   - Tenant and environment binding

2. **Policy Plane** - What can the agent do?
   - Role-based access control (RBAC)
   - Attribute-based access control (ABAC)
   - Contextual conditions
   - Resource scoping

3. **Privilege Plane** - When can the agent elevate?
   - Just-in-time (JIT) privilege grants
   - Time-bound elevation
   - Approval workflows
   - Purpose binding

4. **Execution Plane** - How are actions mediated?
   - Tool wrappers
   - Schema validation
   - Interruptibility controls
   - Output filtering

5. **Evidence Plane** - How is it proven?
   - Cryptographic receipts
   - Chain-of-custody
   - Tamper detection
   - Audit trail preservation

## Quick Start

### Basic IAM Setup

```python
from ciaf.agents import (
    Identity,
    Permission,
    RoleDefinition,
    IAMStore,
    same_tenant_only,
)

# Create the IAM store
iam = IAMStore()

# Define a role with specific permissions
analyst_role = RoleDefinition(
    name="data_analyst",
    permissions=[
        Permission("read_record", "patient_record", same_tenant_only),
        Permission("export_report", "report", same_tenant_only),
    ]
)
iam.add_role(analyst_role)

# Create an agent identity with that role
agent = Identity(
    principal_id="agent-claims-001",
    principal_type="agent",
    display_name="Claims Analysis Agent",
    roles={"data_analyst"},
    attributes={"tenant": "acme-health"}
)
iam.add_identity(agent)
```

### Full Execution with Evidence

```python
from ciaf.agents import (
    Identity,
    Resource,
    ActionRequest,
    IAMStore,
    PAMStore,
    PolicyEngine,
    EvidenceVault,
    ToolExecutor,
)

# Initialize all components
iam = IAMStore()
pam = PAMStore()
vault = EvidenceVault(signing_secret="your-secret-key-here")
policy = PolicyEngine(iam, pam, sensitive_actions={"approve_payment"})
executor = ToolExecutor(policy, vault, pam)

# Setup role and identity (see above)...

# Create an action request
payment_request = ActionRequest(
    action="approve_payment",
    resource=Resource(
        resource_id="payment-123",
        resource_type="payment",
        owner_tenant="acme-health",
        attributes={"amount": 50000}
    ),
    params={"amount": 50000},
    justification="Approve vendor invoice INV-2024-001",
    requested_by=payment_agent
)

# Execute with full IAM/PAM/Evidence controls
result = executor.execute(payment_request)
print(f"Action allowed: {result.allowed}")
print(f"Reason: {result.reason}")

# Verify the complete cryptographic evidence chain
is_valid = vault.verify_chain()
print(f"Evidence chain valid: {is_valid}")
```

## Module Structure

```
ciaf/agents/
├── core/           # Core types (Identity, Resource, Request, Receipt)
│   ├── types.py
│   └── interfaces.py
├── iam/            # Identity and access management
│   ├── store.py
│   └── policy_conditions.py
├── pam/            # Privileged access management
│   └── store.py
├── policy/         # Policy evaluation engine
│   └── engine.py
├── evidence/       # Evidence vault and receipt chain
│   └── vault.py
└── execution/      # Tool executor with mediation
    └── executor.py
```

## Example Scenarios

Complete working examples are provided in `examples/agents_scenarios/`:

### 1. Healthcare Claims Processing
```bash
python examples/agents_scenarios/healthcare_claims.py
```

Demonstrates:
- HIPAA compliance patterns
- PHI access controls
- Multi-level approval workflows
- Audit trail requirements

### 2. Financial Payment Approvals
```bash
python examples/agents_scenarios/financial_approvals.py
```

Demonstrates:
- SOX compliance controls
- Monetary threshold-based authorization
- Two-person review requirements
- JIT privilege elevation

### 3. Production Infrastructure Changes
```bash
python examples/agents_scenarios/production_changes.py
```

Demonstrates:
- Change management workflows
- Environment-based access control
- Ticket-based approval
- Complete chain-of-custody

### Run All Scenarios
```bash
python examples/agents_scenarios/run_all.py
```

## Cryptographic Evidence & Receipts

Every action generates a tamper-evident receipt with complete chain-of-custody. Receipts are cryptographically signed and chained to prove what happened and prevent unauthorized modification.

### Example Receipt

```json
{
  "receipt_id": "rcpt-2024-001-abc123",
  "timestamp": "2024-03-18T14:32:45.123Z",
  "principal_id": "agent-payment-001",
  "principal_type": "agent",
  "action": "approve_payment",
  "resource_id": "payment-123",
  "resource_type": "payment",
  "correlation_id": "corr-2024-001",
  "decision": true,
  "reason": "Allowed by IAM and runtime boundary policy",
  "elevation_grant_id": "grant-2024-xyz789",
  "approved_by": "manager-hr-001",
  "params_hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "policy_obligations": ["two_person_review", "heightened_logging"],
  "prior_receipt_hash": "sha256:a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
  "signature": "hmac-sha256:8d74e34d2c5b6f8a9e1c3d5f7b2a4c6e8f0a1b3d5f7a9b1c3e5d7f9a1b3c5"
}
```

### Verify Receipts

```python
# Check a single receipt
is_valid = vault.verify_receipt(receipt)

# Verify entire chain (detects tampering)
chain_valid = vault.verify_chain()

# Get all receipts for an agent
receipts = vault.get_receipts_by_principal("agent-payment-001")

# Get denied actions
denied = vault.get_denied_receipts()
```

## Integration with CIAF Core

This module integrates seamlessly with CIAF's core components:

- **Cryptographic Primitives**: Uses `ciaf.core.crypto` for hashing and signing
- **Audit Trails**: Extends `ciaf.compliance.audit_trails` patterns
- **Compliance Frameworks**: Maps to EU AI Act, NIST AI RMF, GDPR, HIPAA, SOX, ISO 27001
- **Evidence Storage**: Can integrate with `ciaf.vault` for persistent storage

## Configuration

### Chain Verification
Regularly verify receipt chain integrity to detect any modifications or gaps:

```python
# Run periodic verification
if not vault.verify_chain():
    alert("Evidence chain integrity compromised!")
```

### Audit Review
Implement automated detection of anomalous patterns:

```python
# Monitor denied access attempts
denied_receipts = vault.get_denied_receipts()
if len(denied_receipts) > threshold:
    alert(f"High number of denied operations: {len(denied_receipts)}")
```

## API Reference

See individual module docstrings for complete API documentation:

- `ciaf.agents.core` - Core types and interfaces
- `ciaf.agents.iam` - Identity and access management
- `ciaf.agents.pam` - Privileged access management
- `ciaf.agents.policy` - Policy evaluation engine
- `ciaf.agents.evidence` - Evidence vault and receipts
- `ciaf.agents.execution` - Tool executor

## Contributing

This module follows CIAF's contribution guidelines. See the main project README for details.

## License

This module is part of CIAF and is licensed under **Business Source License 1.1 (BUSL-1.1)**.

See the main project LICENSE file for full details.

---

**Part of the Cognitive Insight Audit Framework (CIAF)**
Version: 1.0.0
Created: 2026-03-28
Author: Denzil James Greenwood
