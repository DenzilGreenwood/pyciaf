# CIAF Web - Browser-Based AI Usage Governance

**Version**: 1.0.0
**Status**: Production-Ready Core
**Created**: 2026-03-24

## Overview

CIAF Web is the **browser-based capture and policy enforcement layer** that records organizational AI usage as verifiable governance evidence.

### The Key Difference

Most tools today try to answer: **"Can we block bad usage?"**

CIAF Web also answers:
- **"Can we prove what happened?"**
- **"Can we reconstruct the event?"**
- **"Can we verify whether policy was followed?"**
- **"Can we tie the output back to a governed record?"**

This is **governance**, not surveillance. **Evidence**, not just logs.

## Core Principles

1. **Not surveillance → Governance**
   - Minimum necessary capture by default
   - Privacy-preserving hashing
   - Consent-based content storage

2. **Not logs → Verifiable evidence**
   - Cryptographic receipts
   - Hash chains for tamper detection
   - Ed25519 digital signatures

3. **Not monitoring → Provable governance**
   - Policy enforcement with evidence
   - Incident reconstruction
   - Compliance auditing

## Architecture

```
ciaf/web/
├── __init__.py              # Main exports
├── events.py                # Event models (WebAIEvent, EventType, etc.)
├── detectors.py             # AI tool detection (15+ tools, shadow AI)
├── classifiers.py           # Content sensitivity classification
├── policy.py                # Policy engine (allow/warn/redact/block)
├── receipts.py              # Cryptographic receipt generation
├── vault_adapter.py         # Persistent storage integration
├── redaction.py             # Privacy-preserving redaction
└── collectors.py            # Browser integration layer
```

## Four Modes

### 1. Discovery Mode
**Find AI usage across the organization**

```python
from ciaf.web import WebAIVaultAdapter

vault = WebAIVaultAdapter()

# Get shadow AI usage
shadow_ai = vault.get_shadow_ai_events(org_id="acme-corp", days=7)

print(f"Found {len(shadow_ai)} shadow AI events:")
for event in shadow_ai:
    print(f"  - {event.tool_name} used by {event.user_id}")
```

**Answers**:
- Are employees using unapproved AI tools?
- Which AI tools are most popular?
- Which departments are using AI?
- What categories of work involve AI?

### 2. Policy Mode
**Evaluate and react to AI usage**

```python
from ciaf.web import PolicyEngine, WebAIEvent, EventType

# Create policy engine with default rules
engine = PolicyEngine()

# Evaluate event
event = WebAIEvent.create(
    event_type=EventType.PROMPT_SUBMIT,
    org_id="acme-corp",
    user_id="analyst-42",
    session_id="sess-123",
    tool_name="ChatGPT",
    tool_approved=False,  # Shadow AI!
)

result = engine.evaluate(event)

if result.is_blocked():
    print(f"BLOCKED: {result.reason}")
elif result.needs_warning():
    print(f"WARNING: {result.reason}")
else:
    print("ALLOWED")
```

**Policy decisions**:
- **ALLOW**: Permit the action
- **WARN**: Allow but notify user/admin
- **REDACT**: Remove sensitive parts
- **BLOCK**: Prevent the action
- **ESCALATE**: Route for manual review

### 3. Evidence Mode
**Create cryptographic receipts**

```python
from ciaf.web import generate_receipt, WebAIEvent, EventType

# Create event
event = WebAIEvent.create(
    event_type=EventType.PROMPT_SUBMIT,
    org_id="acme-corp",
    user_id="analyst-42",
    session_id="sess-123",
    tool_name="ChatGPT",
)

# Generate cryptographic receipt
receipt = generate_receipt(event, signer_id="ciaf-web-prod")

print(f"Receipt ID: {receipt.receipt_id}")
print(f"Event hash: {receipt.event_hash}")
print(f"Receipt hash: {receipt.receipt_hash}")
print(f"Signature: {receipt.signature[:20]}...")
print(f"Chained: {receipt.is_chained()}")
```

**Receipts enable**:
- Incident reconstruction
- Compliance auditing
- Legal defensibility
- Tamper detection

### 4. Provenance Mode
**Track AI-generated outputs**

```python
from ciaf.web import EventCollector, CollectionConfig

# Configure collector
config = CollectionConfig(
    approved_tools={"ChatGPT Enterprise", "GitHub Copilot"},
    capture_raw_content=False,  # Hash only
    redact_sensitive=True,
    generate_receipts=True,
)

collector = EventCollector(config=config)

# Collect prompt submission
result = collector.collect_prompt_submit(
    org_id="acme-corp",
    user_id="analyst-42",
    session_id="sess-123",
    url="https://chat.openai.com",
    prompt="Analyze this financial data...",
)

if result.was_blocked():
    print("Action blocked by policy")
elif result.is_shadow_ai():
    print("Shadow AI detected!")
elif result.is_high_risk():
    print("High-risk content detected")
else:
    print(f"Event recorded: {result.event.event_id}")
    print(f"Receipt: {result.receipt.receipt_id}")
```

## Quick Start

### Basic Event Collection

```python
from ciaf.web import collect_ai_event, EventType

# Simple event collection
result = collect_ai_event(
    event_type=EventType.PROMPT_SUBMIT,
    org_id="acme-corp",
    user_id="analyst-42",
    session_id="sess-123",
    url="https://chat.openai.com",
    content="Draft an email about Q4 results",
    approved_tools={"ChatGPT Enterprise"},
)

# Check results
if result.is_shadow_ai():
    print("⚠️  Shadow AI detected!")

if result.is_high_risk():
    print("⚠️  High-risk content detected!")

if result.was_blocked():
    print("🛑 Action blocked by policy")
else:
    print("✓ Event recorded")
```

### AI Tool Detection

```python
from ciaf.web import detect_ai_tool

# Detect AI tool from URL
result = detect_ai_tool(
    url="https://chat.openai.com/c/abc123",
    approved_tools={"ChatGPT Enterprise"}
)

if result:
    print(f"Tool: {result.tool_name}")
    print(f"Category: {result.tool_category}")
    print(f"Approved: {result.tool_approved}")
    print(f"Shadow AI: {result.is_shadow_ai()}")
```

**Detected tools** (15+):
- LLM Chat: ChatGPT, Claude, Gemini, Copilot, Perplexity
- Code: GitHub Copilot, Cursor, Codeium
- Image: Midjourney, DALL-E, Stable Diffusion
- Document: Notion AI, Google Docs AI
- Enterprise: Azure OpenAI

### Content Classification

```python
from ciaf.web import classify_content

# Classify content for sensitivity
result = classify_content(
    "The patient's SSN is 123-45-6789 and credit card is 4111-1111-1111-1111"
)

print(f"Classification: {result.classification}")
print(f"Sensitivity score: {result.sensitivity_score:.2f}")
print(f"Should block: {result.should_block()}")
print(f"Should warn: {result.should_warn()}")

# Matched rules
for rule in result.matched_rules:
    print(f"  - {rule.name}: {rule.description}")
```

**Classifications**:
- PUBLIC (0.0-0.3)
- INTERNAL (0.3-0.6)
- CONFIDENTIAL (0.6-0.8)
- RESTRICTED (0.8-0.9)
- HIGHLY_RESTRICTED (0.9-1.0)

**Detects**:
- PII (SSN, credit cards, email, phone)
- Financial data (bank accounts, salaries)
- Health data (PHI/medical records)
- Technical secrets (API keys, passwords, private keys)
- Business confidential (trade secrets, strategy)

### Content Redaction

```python
from ciaf.web import redact_pii, redact_credentials, redact_content

# Redact PII
text = "My SSN is 123-45-6789 and email is john@example.com"
redacted = redact_pii(text)
print(redacted)  # "My SSN is [SSN-REDACTED] and email is [EMAIL-REDACTED]"

# Redact credentials
code = "api_key = 'sk-abc123456789'"
safe_code = redact_credentials(code)
print(safe_code)  # "api_key = '[API-KEY-REDACTED]'"

# Aggressive redaction
sensitive = "SSN: 123-45-6789, API key: sk-abc123, Card: 4111-1111-1111-1111"
safe = redact_content(sensitive, aggressive=True)
print(safe)  # All sensitive data redacted
```

### Policy Enforcement

```python
from ciaf.web import PolicyEngine, PolicyRule, PolicyCondition, PolicyDecision

# Create custom policy
custom_rule = PolicyRule(
    rule_id="block_financials_public_ai",
    name="Block Financial Data in Public AI",
    description="Block financial data when using unapproved AI tools",
    conditions=[
        PolicyCondition.SHADOW_AI_DETECTED,
        PolicyCondition.HIGH_SENSITIVITY,
    ],
    decision=PolicyDecision.BLOCK,
    priority=5,  # High priority
    reason_template="Financial data cannot be sent to {tool_name}",
)

# Create engine with custom rules
engine = PolicyEngine(rules=[custom_rule])

# Evaluate
result = engine.evaluate(event)

print(f"Decision: {result.decision}")
print(f"Reason: {result.reason}")
print(f"Rule: {result.matched_rule.name if result.matched_rule else 'None'}")
```

## Event Model

### WebAIEvent

The core event model captures WHO, WHAT, WHEN, WHERE, WHY, and EVIDENCE:

```python
from ciaf.web import WebAIEvent, EventType

event = WebAIEvent.create(
    # WHO
    org_id="acme-corp",
    user_id="analyst-42",
    session_id="sess-abc123",
    device_id="device-xyz",

    # WHAT
    event_type=EventType.PROMPT_SUBMIT,
    tool_name="ChatGPT",
    tool_domain="chat.openai.com",
    tool_approved=False,

    # WHERE
    page_url_hash="sha256_of_url",

    # WHY (classification)
    data_classification=DataClassification.CONFIDENTIAL,
    sensitivity_score=0.75,

    # POLICY
    policy_decision=PolicyDecision.WARN,
    policy_rule_id="warn_shadow_ai",

    # EVIDENCE
    prompt_hash="sha256_of_prompt",
    signature="ed25519_signature",
)

# Helper methods
event.is_shadow_ai()  # Check if unapproved tool
event.is_high_risk()  # Check sensitivity
event.was_blocked()   # Check if blocked
event.needs_review()  # Check if needs manual review
```

### Event Types

- `PROMPT_SUBMIT` - User submits prompt to AI
- `OUTPUT_RECEIVE` - AI returns output
- `FILE_UPLOAD` - User uploads file to AI
- `FILE_DOWNLOAD` - User downloads AI output
- `PASTE_CONTENT` - User pastes content
- `COPY_OUTPUT` - User copies AI output
- `PAGE_VISIT` - Visit to AI tool page
- `SESSION_START/END` - Session boundaries
- `POLICY_BLOCK/WARN/REDACT` - Policy actions
- `SHADOW_AI_DETECT` - Unapproved tool detected

## Storage and Retrieval

### Store Events

```python
from ciaf.web import WebAIVaultAdapter

vault = WebAIVaultAdapter()

# Store event
vault.store_event(event)

# Store receipt
vault.store_receipt(receipt)

# Store batch
vault.store_batch(event_batch)
```

### Search and Query

```python
# Search events
events = vault.search_events(
    org_id="acme-corp",
    user_id="analyst-42",
    tool_name="ChatGPT",
    event_type=EventType.PROMPT_SUBMIT,
    policy_decision=PolicyDecision.BLOCK,
    start_time="2026-03-01T00:00:00Z",
    end_time="2026-03-24T23:59:59Z",
    limit=100,
)

# Get shadow AI events
shadow_events = vault.get_shadow_ai_events(
    org_id="acme-corp",
    days=30,
)

# Get high-risk events
high_risk = vault.get_high_risk_events(
    org_id="acme-corp",
    threshold=0.8,
    days=7,
)

# Get user activity
user_events = vault.get_user_activity(
    user_id="analyst-42",
    days=90,
)
```

## Integration Patterns

### Browser Extension

```javascript
// Browser extension (JavaScript)
// Detect ChatGPT prompt submission
document.querySelector('form').addEventListener('submit', async (e) => {
    const prompt = document.querySelector('textarea').value;

    // Call CIAF Web backend
    await fetch('/api/events/collect', {
        method: 'POST',
        body: JSON.stringify({
            event_type: 'prompt_submit',
            org_id: 'acme-corp',
            user_id: getCurrentUserId(),
            session_id: getSessionId(),
            url: window.location.href,
            content: prompt,
        })
    });
});
```

### Python Backend

```python
from flask import Flask, request, jsonify
from ciaf.web import EventCollector, CollectionConfig

app = Flask(__name__)

# Configure collector
config = CollectionConfig(
    approved_tools={"ChatGPT Enterprise", "GitHub Copilot"},
    enforce_policy=True,
    generate_receipts=True,
)

collector = EventCollector(config=config)

@app.route('/api/events/collect', methods=['POST'])
def collect_event():
    data = request.json

    # Collect event
    result = collector.collect_prompt_submit(
        org_id=data['org_id'],
        user_id=data['user_id'],
        session_id=data['session_id'],
        url=data['url'],
        prompt=data['content'],
    )

    # Return policy decision
    return jsonify({
        'allowed': result.action_allowed,
        'blocked': result.was_blocked(),
        'shadow_ai': result.is_shadow_ai(),
        'high_risk': result.is_high_risk(),
        'event_id': result.event.event_id,
        'receipt_id': result.receipt.receipt_id if result.receipt else None,
        'policy_message': result.policy_result.reason if result.policy_result else None,
    })
```

## Organizational Questions Answered

### Approved Use Monitoring
- ✓ Are employees using approved enterprise AI tools?
- ✓ How often?
- ✓ For what categories of work?
- ✓ Are they staying inside policy?

### Shadow AI Detection
- ✓ Are employees using non-approved public AI tools?
- ✓ Which ones?
- ✓ How often?
- ✓ Which departments or workflows are involved?

### Sensitive Data Exposure
- ✓ Was internal or restricted data sent to public AI?
- ✓ What kind of data was involved?
- ✓ Was it blocked, warned, or allowed?
- ✓ Is there evidence for investigation?

### Output Governance
- ✓ Did AI-generated content enter business workflows?
- ✓ Was it labeled or watermarked?
- ✓ Can the artifact be tied back to an event record?

### Incident Response
- ✓ If a leak, error, or policy violation occurred, what exactly happened?
- ✓ Who did what?
- ✓ In what sequence?
- ✓ Under what policy state?

## What Data is Collected

### Stored by Default:
- ✓ User ID
- ✓ Org ID
- ✓ Device/session ID
- ✓ Tool/domain
- ✓ Event type
- ✓ Timestamp
- ✓ Content **hash** (not content itself)
- ✓ Content classification
- ✓ Policy result
- ✓ File metadata
- ✓ Upload/download indicators

### Only stored when configured:
- ⚠️ Raw content (when explicitly enabled)
- ⚠️ Redacted content (high-risk events)
- ⚠️ File contents (compliance requirements)

This is **minimum necessary capture** by design.

## Security and Privacy

### What's Protected:
✓ Privacy-preserving hashing by default
✓ Configurable content capture
✓ PII redaction
✓ Consent-based model
✓ Cryptographic receipts
✓ Tamper-evident hash chains

### What's NOT:
✗ Not keystroke logging
✗ Not full browsing surveillance
✗ Not screen monitoring
✗ Not indiscriminate employee spying

### Positioning:
✓ AI usage governance
✓ Policy enforcement
✓ Sensitive data protection
✓ Evidence for approved vs unapproved AI use
✓ Organizational accountability

## Dependencies

**Required**:
- Python 3.8+
- Standard library only (hashlib, json, re, dataclasses)

**Optional**:
- CIAF vault (for PostgreSQL storage)
- Flask/FastAPI (for web API)
- Browser extension framework

**No external dependencies for core functionality!**

## Testing

```bash
# Run tests (when implemented)
python -m pytest tests/test_web_*.py

# Test detection
python -c "from ciaf.web import detect_ai_tool; print(detect_ai_tool('https://chat.openai.com'))"

# Test classification
python -c "from ciaf.web import classify_content; print(classify_content('SSN: 123-45-6789'))"
```

## Commercial Positioning

**Problem**: Organizations struggle with:
- Shadow AI usage
- BYO AI behavior
- Policy enforcement gaps
- Lack of evidence for investigations
- Executive pressure to "do something about AI usage"

**Solution**: CIAF Web provides:
- **The organizational AI usage governance sensor and evidence layer**
- **Detection + Policy + Verifiable Evidence**

**One-line pitch**:
> "CIAF Web helps organizations detect, govern, and prove how employees use approved and unapproved AI tools across the web."

## Roadmap

### v1.0 (Current) ✅
- Core event model
- AI tool detection (15+ tools)
- Content classification
- Policy engine
- Cryptographic receipts
- Vault storage
- Redaction
- Collection API

### v1.1 (Next)
- Browser extension reference implementation
- REST API with authentication
- Real-time dashboard
- Batch analytics
- Compliance reports
- User notifications

### v1.2 (Future)
- Machine learning-based classification
- Behavioral anomaly detection
- Integration with SIEM systems
- Blockchain anchoring
- Zero-knowledge proofs

## License

See main CIAF LICENSE file.

---

**Version**: 1.0.0
**Last Updated**: 2026-03-24
**Author**: Denzil James Greenwood
**Contact**: founder@cognitiveinsight.ai
