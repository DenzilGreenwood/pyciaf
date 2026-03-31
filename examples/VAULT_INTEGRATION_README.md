# CIAF Vault Integration Examples

Complete demonstration suite for connecting Python applications to the CIAF Vault web platform.

## Overview

These examples demonstrate how to:
- ✅ Send AI lifecycle events to the vault
- ✅ Implement watermarking for AI-generated content
- ✅ Track web AI usage with governance
- ✅ Build autonomous agents with IAM/PAM
- ✅ Generate cryptographic receipts for auditability

## Prerequisites

### 1. Start CIAF Vault

```bash
cd ciaf_vault
npm install
npm run dev
```

The vault will be available at `http://localhost:3000`

### 2. Install Dependencies

```bash
cd pyciaf
pip install -e .
pip install requests
```

## Quick Start

Run all examples at once:

```bash
python examples/vault_integration_complete.py
```

Or set a custom vault URL:

```bash
export CIAF_VAULT_URL=https://your-vault.vercel.app
python examples/vault_integration_complete.py
```

## Examples Included

### 1. Credit Scoring Model 🏦

**File**: `vault_integration_complete.py` → `CreditScoringModel`

**Features**:
- Model training with epoch-level tracking
- Inference predictions with watermarking
- Dual-hash verification (before/after watermark)
- Cryptographic receipts for every prediction
- Real-time performance monitoring

**Use Case**: Financial services needing audit trails for credit decisions

### 2. ChatGPT-like Interface 💬

**File**: `vault_integration_complete.py` → `ChatGPTInterface`

**Features**:
- Conversational AI with watermarked responses
- Web AI event tracking (prompt submit/response)
- PII detection in prompts and responses
- Content sensitivity scoring
- Policy enforcement integration

**Use Case**: Enterprise chatbots requiring governance and compliance

### 3. Agentic AI System 🤖

**File**: `vault_integration_complete.py` → `AgenticAISystem`

**Features**:
- Agent registration with identity management
- Role-based access control (IAM)
- Privilege elevation with approval workflow (PAM)
- Cryptographic action receipts
- Full audit trail for all agent actions

**Use Case**: Autonomous AI agents requiring zero-trust execution boundaries

## API Usage

### Vault Client Basics

```python
from ciaf.vault_client import VaultClient

# Initialize client
vault = VaultClient(vault_url="http://localhost:3000")

# Send inference event
receipt = vault.send_inference_event(
    model_name="my-model",
    input_data={"feature1": 42},
    prediction={"class": "A", "confidence": 0.95}
)

# Register an agent
agent = vault.register_agent(
    principal_id="agent-001",
    display_name="Payment Processor",
    roles=["payment_approver"]
)

# Execute agent action
result = vault.execute_agent_action(
    principal_id="agent-001",
    action="approve_payment",
    resource_id="payment-123",
    resource_type="payment",
    justification="Approved invoice"
)
```

### Watermarking Integration

```python
from ciaf.watermarks import TextWatermark
from ciaf.vault_client import VaultClient
import hashlib

vault = VaultClient()

# Generate AI content
ai_response = "This is an AI-generated response."

# Hash before watermark
hash_before = hashlib.sha256(ai_response.encode()).hexdigest()

# Add watermark
watermark = TextWatermark(
    artifact_id="resp-001",
    artifact_type="text",
    model_id="chat-model-v1",
    tag_text="AI Generated | Verify at https://vault.example.com"
)

ai_response_with_wm = ai_response + "\n\n" + watermark.tag_text

# Hash after watermark
hash_after = hashlib.sha256(ai_response_with_wm.encode()).hexdigest()

# Send to vault with dual hashes
vault.send_web_event(
    user_id="user-123",
    domain="chat.example.com",
    tool_name="ChatAssistant",
    event_type="response_generated",
    content=ai_response,
    watermarked=True,
    hash_before=hash_before,
    hash_after=hash_after
)
```

## Vault Dashboard Pages

After running the examples, view results in the vault:

| Page | URL | What to See |
|------|-----|-------------|
| **Dashboard** | http://localhost:3000 | Live stats, event stream, charts |
| **Models** | http://localhost:3000/models | Credit model training & inference |
| **Web Events** | http://localhost:3000/web | ChatGPT interface usage |
| **Agents** | http://localhost:3000/agents | Agent actions, IAM/PAM decisions |
| **Compliance** | http://localhost:3000/compliance | Framework mapping, scores |
| **Receipts** | http://localhost:3000/receipts | Cryptographic audit receipts |

## Example Output

```
================================================================================
CIAF VAULT INTEGRATION - COMPLETE DEMONSTRATION
================================================================================
Vault URL: http://localhost:3000
Organization: demo-corp
User: demo-user
================================================================================

🔍 Checking vault connectivity...
✅ Vault is accessible

================================================================================
EXAMPLE 1: CREDIT SCORING MODEL
================================================================================
🏦 Initializing Credit Scoring Model: credit-scorer-v1
   Version: 1.2.0
   Connected to Vault: http://localhost:3000

📚 Training model for 5 epochs...
   Epoch 3/5: Loss=0.2632, Acc=0.8500
✅ Training complete!

💳 Processing credit application...
   Applicant: Alice Johnson
   Score: 742
   Decision: ✅ APPROVED
   Receipt: rcpt-3f7a9b2c1d...

================================================================================
EXAMPLE 2: CHATGPT-LIKE INTERFACE
================================================================================
💬 Initializing Chat Assistant: chat-assistant-v1
   Connected to Vault: http://localhost:3000

💬 User: Hello! How are you today?...
   🤖 Assistant: Hello! I'm your AI assistant. How can I help you today?...
   📝 Watermarked: Yes

================================================================================
EXAMPLE 3: AGENTIC AI SYSTEM
================================================================================
🤖 Initializing Agentic AI System
   Connected to Vault: http://localhost:3000

🆔 Registering agent: agent-payment-processor-001
   Name: Payment Processing Agent
   Roles: viewer, analyst
   ✅ Registered successfully
   Fingerprint: 8a4f3e2b5c6d...

⚡ Executing action: approve_payment
   Agent: agent-payment-processor-001
   Resource: payment/payment-12345
   Justification: Monthly vendor payment
   ❌ Action DENIED
   Reason: Agent lacks required role for action 'approve_payment'

🔐 Granting privilege elevation
   Agent: agent-payment-processor-001
   Elevated Role: payment_approver
   Duration: 30 minutes
   ✅ Elevation granted

================================================================================
DEMONSTRATION COMPLETE
================================================================================

📊 View real-time results at: http://localhost:3000

✨ All examples completed successfully!
```

## Customization

### Custom Vault URL

```python
# Environment variable
export CIAF_VAULT_URL=https://your-vault.vercel.app

# Or in Python
from ciaf.vault_client import VaultClient
vault = VaultClient(vault_url="https://your-vault.vercel.app")
```

### Custom Organization/User

```python
# In the examples file, modify:
DEMO_ORG = "your-org-id"
DEMO_USER = "your-user-id"
```

### Add Your Own Models

```python
from ciaf.vault_client import VaultClient

class YourCustomModel:
    def __init__(self):
        self.vault = VaultClient()
    
    def predict(self, input_data):
        # Your model logic here
        prediction = your_model.predict(input_data)
        
        # Send to vault
        self.vault.send_inference_event(
            model_name="your-model",
            input_data=input_data,
            prediction=prediction
        )
        
        return prediction
```

## Troubleshooting

### Vault Not Accessible

If you see "Warning: Vault may not be accessible":

1. Make sure the vault is running:
   ```bash
   cd ciaf_vault
   npm run dev
   ```

2. Check the URL is correct:
   ```bash
   curl http://localhost:3000/api/stats
   ```

3. The examples will still run, but events won't be tracked

### Import Errors

Make sure pyciaf is installed:

```bash
cd pyciaf
pip install -e .
```

### Missing Dependencies

Install requests:

```bash
pip install requests
```

## Production Deployment

For production use:

1. **Deploy CIAF Vault** to Vercel (see `ciaf_vault/DEPLOYMENT.md`)
2. **Update URLs** in your Python applications
3. **Add authentication** if needed (currently disabled for demos)
4. **Monitor events** via the vault dashboard

## Next Steps

1. ✅ Run the complete demonstration
2. ✅ View events in the vault dashboard
3. ✅ Customize for your use case
4. ✅ Deploy vault to production
5. ✅ Integrate with your AI systems

## Support

- **Documentation**: See `CIAF_VAULT_GAP_ANALYSIS.md` for architecture
- **Issues**: GitHub Issues for bug reports
- **Questions**: GitHub Discussions for help

---

**Created**: 2026-03-31
**Version**: 1.0.0
**License**: BUSL-1.1
