"""
Vault Integration Examples

Complete demonstration suite showing:
1. Credit Scoring Model with Watermarking
2. ChatGPT-like Interface with Web AI Governance
3. Agentic AI System with IAM/PAM

All examples send events to CIAF Vault for real-time monitoring.

Created: 2026-03-31
Author: Denzil James Greenwood
"""

import os
import sys
import time
import random
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# Add ciaf to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ciaf.vault_client import VaultClient
from ciaf.watermarks import (
    TextWatermark,
    create_watermark,
    verify_watermark,
    WatermarkTier
)
from ciaf.agents import (
    IAMStore,
    PAMStore,
    PolicyEngine,
    EvidenceVault,
    ToolExecutor,
    Identity,
    PrincipalType,
    Resource,
    ActionRequest
)


# =============================================================================
# Configuration
# =============================================================================

VAULT_URL = os.getenv("CIAF_VAULT_URL", "http://localhost:3000")
DEMO_ORG = "demo-corp"
DEMO_USER = "demo-user"


# =============================================================================
# Example 1: Credit Scoring Model with Watermarking
# =============================================================================

class CreditScoringModel:
    """
    Credit scoring model with CIAF watermarking and vault integration.
    
    Demonstrates:
    - Model inference with watermarking
    - Cryptographic receipts for predictions
    - Real-time monitoring via vault
    - Dual-hash verification
    """
    
    def __init__(self, vault_client: VaultClient, model_name: str = "credit-scorer-v1"):
        self.vault = vault_client
        self.model_name = model_name
        self.model_version = "1.2.0"
        
        print(f"🏦 Initializing Credit Scoring Model: {model_name}")
        print(f"   Version: {self.model_version}")
        print(f"   Connected to Vault: {vault_client.config.vault_url}")
    
    def train_model(self, epochs: int = 10):
        """Simulate model training with vault tracking."""
        print(f"\n📚 Training model for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            # Simulate training metrics
            loss = 1.0 / (epoch * 0.3 + 1)
            accuracy = 0.7 + (epoch / epochs) * 0.25
            
            # Send training event to vault
            response = self.vault.send_training_event(
                model_name=self.model_name,
                model_version=self.model_version,
                epoch=epoch,
                metrics={
                    'loss': round(loss, 4),
                    'accuracy': round(accuracy, 4),
                    'learning_rate': 0.001
                },
                org_id=DEMO_ORG,
                user_id=DEMO_USER
            )
            
            if epoch % 3 == 0:
                print(f"   Epoch {epoch}/{epochs}: Loss={loss:.4f}, Acc={accuracy:.4f}")
        
        print("✅ Training complete!")
    
    def predict(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make credit prediction with watermarking.
        
        Args:
            applicant: Applicant data (income, debt, credit history)
        
        Returns:
            Prediction with watermark and receipt
        """
        print(f"\n💳 Processing credit application...")
        print(f"   Applicant: {applicant.get('name', 'Anonymous')}")
        
        # Simulate prediction
        credit_score = self._calculate_score(applicant)
        approved = credit_score >= 650
        
        # Create prediction result
        prediction = {
            'applicant_id': applicant.get('id', 'unknown'),
            'credit_score': credit_score,
            'approved': approved,
            'confidence': 0.92,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Generate watermark
        watermark_text = self._create_watermark(prediction)
        
        # Hash prediction (before watermark)
        prediction_json = str(prediction)
        content_hash_before = hashlib.sha256(prediction_json.encode()).hexdigest()
        
        # Add watermark to response
        prediction['watermark'] = watermark_text
        
        # Hash prediction (after watermark)
        prediction_with_wm = str(prediction)
        content_hash_after = hashlib.sha256(prediction_with_wm.encode()).hexdigest()
        
        # Send inference event to vault
        start_time = time.time()
        response = self.vault.send_inference_event(
            model_name=self.model_name,
            model_version=self.model_version,
            input_data=applicant,
            prediction=prediction,
            confidence=0.92,
            latency_ms=(time.time() - start_time) * 1000,
            content_hash_before=content_hash_before,
            content_hash_after=content_hash_after,
            watermarked=True,
            org_id=DEMO_ORG,
            user_id=DEMO_USER
        )
        
        # Extract receipt
        receipt = response.get('data', {}).get('receipt')
        
        print(f"   Score: {credit_score}")
        print(f"   Decision: {'✅ APPROVED' if approved else '❌ DENIED'}")
        print(f"   Receipt: {receipt['receipt_id'] if receipt else 'N/A'}")
        
        return {
            **prediction,
            'receipt': receipt
        }
    
    def _calculate_score(self, applicant: Dict) -> int:
        """Simple credit scoring algorithm."""
        base_score = 600
        
        # Income factor
        income = applicant.get('annual_income', 50000)
        income_score = min(100, income / 1000)
        
        # Debt factor
        debt = applicant.get('total_debt', 10000)
        debt_ratio = debt / income if income > 0 else 1
        debt_score = max(-100, -debt_ratio * 200)
        
        # Credit history factor
        credit_history_months = applicant.get('credit_history_months', 12)
        history_score = min(100, credit_history_months / 2)
        
        # Random variation
        variation = random.randint(-20, 20)
        
        total_score = base_score + income_score + debt_score + history_score + variation
        return max(300, min(850, int(total_score)))
    
    def _create_watermark(self, prediction: Dict) -> str:
        """Create watermark for prediction."""
        watermark = TextWatermark(
            artifact_id=f"pred-{prediction['applicant_id']}",
            artifact_type="credit_prediction",
            model_id=self.model_name,
            tag_text=f"AI Generated | Credit Score: {prediction['credit_score']} | Verify at {VAULT_URL}",
            metadata={
                'score': prediction['credit_score'],
                'approved': prediction['approved']
            }
        )
        return watermark.tag_text


# =============================================================================
# Example 2: ChatGPT-like Interface with Web AI Governance
# =============================================================================

class ChatGPTInterface:
    """
    ChatGPT-like conversational AI with watermarking and governance.
    
    Demonstrates:
    - Web AI event tracking
    - Content watermarking
    - PII detection
    - Policy enforcement
    """
    
    def __init__(self, vault_client: VaultClient, model_name: str = "chat-assistant-v1"):
        self.vault = vault_client
        self.model_name = model_name
        self.conversation_history: List[Dict] = []
        
        print(f"💬 Initializing Chat Assistant: {model_name}")
        print(f"   Connected to Vault: {vault_client.config.vault_url}")
    
    def chat(self, user_prompt: str, user_id: str = DEMO_USER) -> str:
        """
        Process user prompt and generate response with watermarking.
        
        Args:
            user_prompt: User's input prompt
            user_id: User identifier
        
        Returns:
            AI-generated response with watermark
        """
        print(f"\n💬 User: {user_prompt[:80]}...")
        
        # Track prompt submission
        self.vault.send_web_event(
            user_id=user_id,
            domain="chat.demo.com",
            tool_name="ChatAssistant",
            event_type="prompt_submit",
            content=user_prompt,
            policy_decision="allow",
            sensitivity_score=self._calculate_sensitivity(user_prompt),
            pii_detected=self._detect_pii(user_prompt),
            org_id=DEMO_ORG
        )
        
        # Generate response (simulated)
        response_text = self._generate_response(user_prompt)
        
        # Create watermark
        watermark_text = f"\n\n---\n🤖 AI Generated Content | Model: {self.model_name} | Verify: {VAULT_URL}/verify"
        response_with_watermark = response_text + watermark_text
        
        # Hash content before and after watermark
        hash_before = hashlib.sha256(response_text.encode()).hexdigest()
        hash_after = hashlib.sha256(response_with_watermark.encode()).hexdigest()
        
        # Track response generation
        self.vault.send_web_event(
            user_id=user_id,
            domain="chat.demo.com",
            tool_name="ChatAssistant",
            event_type="response_generated",
            content=response_text,
            policy_decision="allow",
            sensitivity_score=self._calculate_sensitivity(response_text),
            pii_detected=False,
            org_id=DEMO_ORG,
            watermarked=True,
            hash_before=hash_before,
            hash_after=hash_after
        )
        
        # Store in history
        self.conversation_history.append({
            'role': 'user',
            'content': user_prompt,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': response_with_watermark,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'watermarked': True
        })
        
        print(f"   🤖 Assistant: {response_text[:80]}...")
        print(f"   📝 Watermarked: Yes")
        
        return response_with_watermark
    
    def _generate_response(self, prompt: str) -> str:
        """Simulate AI response generation."""
        responses = {
            'hello': "Hello! I'm your AI assistant. How can I help you today?",
            'weather': "I don't have real-time weather data, but I can help you find weather resources.",
            'explain': "Let me explain that concept for you. [Detailed explanation would go here]",
            'code': "Here's a code example:\n```python\nprint('Hello, World!')\n```",
            'default': "That's an interesting question. Let me provide you with a thoughtful response based on my training."
        }
        
        prompt_lower = prompt.lower()
        for key, response in responses.items():
            if key in prompt_lower:
                return response
        
        return responses['default']
    
    def _calculate_sensitivity(self, text: str) -> float:
        """Calculate content sensitivity score."""
        sensitive_keywords = ['password', 'ssn', 'credit card', 'confidential', 'secret']
        sensitivity = 0.0
        
        text_lower = text.lower()
        for keyword in sensitive_keywords:
            if keyword in text_lower:
                sensitivity += 20.0
        
        return min(100.0, sensitivity)
    
    def _detect_pii(self, text: str) -> bool:
        """Simple PII detection."""
        import re
        # Check for email, SSN patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        
        return bool(re.search(email_pattern, text) or re.search(ssn_pattern, text))


# =============================================================================
# Example 3: Agentic AI System with IAM/PAM
# =============================================================================

class AgenticAISystem:
    """
    Autonomous AI agent system with full IAM/PAM enforcement.
    
    Demonstrates:
    - Agent registration
    - Role-based access control
    - Privilege elevation (PAM)
    - Action auditing with receipts
    """
    
    def __init__(self, vault_client: VaultClient):
        self.vault = vault_client
        self.agents: Dict[str, Dict] = {}
        
        print(f"🤖 Initializing Agentic AI System")
        print(f"   Connected to Vault: {vault_client.config.vault_url}")
    
    def register_agent(
        self,
        agent_id: str,
        display_name: str,
        roles: List[str],
        agent_type: str = "agent"
    ) -> Dict:
        """Register a new AI agent."""
        print(f"\n🆔 Registering agent: {agent_id}")
        print(f"   Name: {display_name}")
        print(f"   Roles: {', '.join(roles)}")
        
        response = self.vault.register_agent(
            principal_id=agent_id,
            display_name=display_name,
            principal_type=agent_type,
            roles=roles,
            attributes={
                'tenant_id': DEMO_ORG,
                'environment': 'demo',
                'version': '1.0.0'
            },
            created_by=DEMO_USER
        )
        
        if response.get('success'):
            agent_data = response['data']
            self.agents[agent_id] = agent_data
            print(f"   ✅ Registered successfully")
            print(f"   Fingerprint: {agent_data['fingerprint'][:16]}...")
            return agent_data
        else:
            print(f"   ❌ Registration failed: {response.get('error')}")
            return {}
    
    def execute_action(
        self,
        agent_id: str,
        action: str,
        resource_id: str,
        resource_type: str,
        justification: str,
        params: Dict = None
    ) -> Dict:
        """Execute an action with IAM/PAM enforcement."""
        print(f"\n⚡ Executing action: {action}")
        print(f"   Agent: {agent_id}")
        print(f"   Resource: {resource_type}/{resource_id}")
        print(f"   Justification: {justification}")
        
        response = self.vault.execute_agent_action(
            principal_id=agent_id,
            action=action,
            resource_id=resource_id,
            resource_type=resource_type,
            params=params or {},
            justification=justification,
            correlation_id=f"demo-{int(time.time())}"
        )
        
        if response.get('success'):
            result = response['data']
            if result['allowed']:
                print(f"   ✅ Action ALLOWED")
                print(f"   Receipt: {result.get('receipt', {}).get('receipt_id', 'N/A')[:20]}...")
            else:
                print(f"   ❌ Action DENIED")
                print(f"   Reason: {result['reason']}")
            
            return result
        else:
            print(f"   ❌ Execution failed: {response.get('error')}")
            return {}
    
    def grant_elevation(
        self,
        agent_id: str,
        elevated_role: str,
        approved_by: str,
        purpose: str,
        duration_minutes: int = 60
    ) -> Dict:
        """Grant temporary privilege elevation."""
        print(f"\n🔐 Granting privilege elevation")
        print(f"   Agent: {agent_id}")
        print(f"   Elevated Role: {elevated_role}")
        print(f"   Duration: {duration_minutes} minutes")
        
        valid_until = (datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)).isoformat()
        
        response = self.vault.grant_elevation(
            principal_id=agent_id,
            elevated_role=elevated_role,
            approved_by=approved_by,
            purpose=purpose,
            valid_until=valid_until,
            justification=f"Temporary elevation for {purpose}",
            max_uses=5
        )
        
        if response.get('success'):
            grant = response['data']
            print(f"   ✅ Elevation granted")
            print(f"   Grant ID: {grant['grant_id']}")
            return grant
        else:
            print(f"   ❌ Elevation failed: {response.get('error')}")
            return {}
    
    def demonstrate_workflow(self):
        """Demonstrate a complete agentic workflow."""
        print("\n" + "="*80)
        print("AGENTIC AI WORKFLOW DEMONSTRATION")
        print("="*80)
        
        # Register agents
        payment_agent = self.register_agent(
            agent_id="agent-payment-processor-001",
            display_name="Payment Processing Agent",
            roles=["viewer", "analyst"]
        )
        
        time.sleep(1)
        
        finance_agent = self.register_agent(
            agent_id="agent-finance-admin-001",
            display_name="Finance Admin Agent",
            roles=["payment_approver", "finance_admin"]
        )
        
        time.sleep(1)
        
        # Try action without privilege (should be denied)
        result1 = self.execute_action(
            agent_id="agent-payment-processor-001",
            action="approve_payment",
            resource_id="payment-12345",
            resource_type="payment",
            justification="Monthly vendor payment",
            params={"amount": 5000, "vendor": "ACME Corp"}
        )
        
        time.sleep(1)
        
        # Grant elevation
        if not result1.get('allowed'):
            grant = self.grant_elevation(
                agent_id="agent-payment-processor-001",
                elevated_role="payment_approver",
                approved_by="admin-user",
                purpose="Emergency payment approval",
                duration_minutes=30
            )
            
            time.sleep(1)
            
            # Retry with elevation
            result2 = self.execute_action(
                agent_id="agent-payment-processor-001",
                action="approve_payment",
                resource_id="payment-12345",
                resource_type="payment",
                justification="Monthly vendor payment (with elevation)",
                params={"amount": 5000, "vendor": "ACME Corp"}
            )
        
        time.sleep(1)
        
        # Action with proper role (should succeed)
        result3 = self.execute_action(
            agent_id="agent-finance-admin-001",
            action="approve_payment",
            resource_id="payment-67890",
            resource_type="payment",
            justification="Quarterly bonus payment",
            params={"amount": 10000, "vendor": "Payroll Services"}
        )


# =============================================================================
# Main Demo Runner
# =============================================================================

def run_all_examples():
    """Run all vault integration examples."""
    print("\n" + "="*80)
    print("CIAF VAULT INTEGRATION - COMPLETE DEMONSTRATION")
    print("="*80)
    print(f"Vault URL: {VAULT_URL}")
    print(f"Organization: {DEMO_ORG}")
    print(f"User: {DEMO_USER}")
    print("="*80)
    
    # Initialize vault client
    vault = VaultClient(vault_url=VAULT_URL)
    
    # Check vault connectivity
    print("\n🔍 Checking vault connectivity...")
    if vault.health_check():
        print("✅ Vault is accessible")
    else:
        print("❌ Warning: Vault may not be accessible")
        print("   Make sure CIAF Vault is running at", VAULT_URL)
        print("   You can still run the examples, but events won't be tracked.")
    
    # ======================
    # Example 1: Credit Model
    # ======================
    print("\n" + "="*80)
    print("EXAMPLE 1: CREDIT SCORING MODEL")
    print("="*80)
    
    credit_model = CreditScoringModel(vault)
    credit_model.train_model(epochs=5)
    
    # Make predictions
    applicants = [
        {
            'id': 'app-001',
            'name': 'Alice Johnson',
            'annual_income': 75000,
            'total_debt': 15000,
            'credit_history_months': 48
        },
        {
            'id': 'app-002',
            'name': 'Bob Smith',
            'annual_income': 45000,
            'total_debt': 35000,
            'credit_history_months': 24
        },
        {
            'id': 'app-003',
            'name': 'Carol Williams',
            'annual_income': 95000,
            'total_debt': 8000,
            'credit_history_months': 60
        }
    ]
    
    for applicant in applicants:
        credit_model.predict(applicant)
        time.sleep(0.5)
    
    # ======================
    # Example 2: Chat Interface
    # ======================
    print("\n" + "="*80)
    print("EXAMPLE 2: CHATGPT-LIKE INTERFACE")
    print("="*80)
    
    chat = ChatGPTInterface(vault)
    
    prompts = [
        "Hello! How are you today?",
        "Can you explain machine learning?",
        "Write a Python function to calculate fibonacci numbers",
        "What's the weather like?"
    ]
    
    for prompt in prompts:
        chat.chat(prompt)
        time.sleep(0.5)
    
    # ======================
    # Example 3: Agentic AI
    # ======================
    print("\n" + "="*80)
    print("EXAMPLE 3: AGENTIC AI SYSTEM")
    print("="*80)
    
    agentic = AgenticAISystem(vault)
    agentic.demonstrate_workflow()
    
    # ======================
    # Summary
    # ======================
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print(f"\n📊 View real-time results at: {VAULT_URL}")
    print("\nPages to check:")
    print(f"  • Dashboard:     {VAULT_URL}")
    print(f"  • Models:        {VAULT_URL}/models")
    print(f"  • Web Events:    {VAULT_URL}/web")
    print(f"  • Agents:        {VAULT_URL}/agents")
    print(f"  • Compliance:    {VAULT_URL}/compliance")
    print(f"  • Receipts:      {VAULT_URL}/receipts")
    print("\n✨ All examples completed successfully!")


if __name__ == "__main__":
    run_all_examples()
