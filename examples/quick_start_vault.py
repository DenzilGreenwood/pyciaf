"""
CIAF Vault - Quick Start Demo

Simplified demonstration that can be run immediately.
Shows credit model, chat interface, and agentic AI in action.

Usage:
    python quick_start_vault.py
    
Or with custom vault URL:
    CIAF_VAULT_URL=https://your-vault.vercel.app python quick_start_vault.py

Created: 2026-03-31
"""

import os
import sys
import time
import warnings

# Suppress optional dependency warnings for cleaner demo output
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ciaf import VaultClient, VAULT_CLIENT_AVAILABLE
    
    if not VAULT_CLIENT_AVAILABLE or VaultClient is None:
        print("❌ VaultClient not available. Install requests: pip install requests")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to install pyciaf: cd pyciaf && pip install -e .")
    sys.exit(1)


def main():
    """Run quick start demo."""
    
    # Configuration
    vault_url = os.getenv("CIAF_VAULT_URL", "http://localhost:3000")
    
    print("\n" + "="*80)
    print("CIAF VAULT - QUICK START DEMO")
    print("="*80)
    print(f"Vault URL: {vault_url}")
    print("="*80)
    
    # Initialize client
    print("\n🔌 Connecting to vault...")
    vault = VaultClient(vault_url=vault_url)
    
    if vault.health_check():
        print("✅ Connected successfully!")
    else:
        print("⚠️  Warning: Vault not accessible")
        print("   Make sure CIAF Vault is running:")
        print("   cd ciaf_vault && npm run dev")
        print("\n   Continuing anyway (events won't be tracked)...")
    
    # =========================================================================
    # Demo 1: Credit Model
    # =========================================================================
    print("\n" + "="*80)
    print("DEMO 1: Credit Scoring Model")
    print("="*80)
    
    print("\n📚 Training model...")
    for epoch in [1, 2, 3]:
        response = vault.send_training_event(
            model_name="credit-model-v1",
            model_version="1.0.0",
            epoch=epoch,
            metrics={
                'loss': round(1.0 / (epoch + 1), 4),
                'accuracy': round(0.7 + (epoch * 0.08), 4)
            }
        )
        print(f"   Epoch {epoch}/3 tracked")
        time.sleep(0.3)
    
    print("\n💳 Making credit predictions...")
    applicants = [
        {"name": "Alice", "income": 75000, "debt": 15000, "score": 742},
        {"name": "Bob", "income": 45000, "debt": 35000, "score": 628},
        {"name": "Carol", "income": 95000, "debt": 8000, "score": 789}
    ]
    
    for applicant in applicants:
        response = vault.send_inference_event(
            model_name="credit-model-v1",
            model_version="1.0.0",
            input_data={"name": applicant["name"]},
            prediction={"score": applicant["score"], "approved": applicant["score"] >= 650},
            confidence=0.92
        )
        
        approved = "✅ APPROVED" if applicant["score"] >= 650 else "❌ DENIED"
        print(f"   {applicant['name']}: Score {applicant['score']} - {approved}")
        time.sleep(0.3)
    
    # =========================================================================
    # Demo 2: Chat Interface
    # =========================================================================
    print("\n" + "="*80)
    print("DEMO 2: ChatGPT-like Interface")
    print("="*80)
    
    conversations = [
        {"prompt": "Hello! How are you?", "response": "I'm doing great! How can I help?"},
        {"prompt": "What is machine learning?", "response": "Machine learning is a subset of AI..."},
        {"prompt": "Write a Python function", "response": "Here's a Python function:\ndef example():\n    pass"}
    ]
    
    for conv in conversations:
        # Track prompt
        vault.send_web_event(
            user_id="demo-user",
            domain="chat.demo.com",
            tool_name="ChatAssistant",
            event_type="prompt_submit",
            content=conv["prompt"],
            policy_decision="allow"
        )
        
        print(f"\n💬 User: {conv['prompt']}")
        time.sleep(0.2)
        
        # Track response
        vault.send_web_event(
            user_id="demo-user",
            domain="chat.demo.com",
            tool_name="ChatAssistant",
            event_type="response_generated",
            content=conv["response"],
            policy_decision="allow",
            watermarked=True
        )
        
        print(f"   🤖 AI: {conv['response'][:60]}...")
        time.sleep(0.3)
    
    # =========================================================================
    # Demo 3: Agentic AI
    # =========================================================================
    print("\n" + "="*80)
    print("DEMO 3: Agentic AI System")
    print("="*80)
    
    # Register agents
    print("\n🆔 Registering agents...")
    
    agent1 = vault.register_agent(
        principal_id="agent-payment-001",
        display_name="Payment Processor Agent",
        principal_type="agent",
        roles=["viewer", "analyst"]
    )
    
    if agent1.get('success'):
        print("   ✅ Payment Agent registered")
    
    time.sleep(0.5)
    
    agent2 = vault.register_agent(
        principal_id="agent-finance-admin-001",
        display_name="Finance Admin Agent",
        principal_type="agent",
        roles=["payment_approver", "finance_admin"]
    )
    
    if agent2.get('success'):
        print("   ✅ Finance Agent registered")
    
    time.sleep(0.5)
    
    # Execute actions
    print("\n⚡ Executing agent actions...")
    
    # Action without permission (will be denied)
    result1 = vault.execute_agent_action(
        principal_id="agent-payment-001",
        action="approve_payment",
        resource_id="payment-12345",
        resource_type="payment",
        justification="Monthly vendor payment"
    )
    
    if result1.get('success'):
        decision = result1['data']
        status = "✅ ALLOWED" if decision.get('allowed') else "❌ DENIED"
        print(f"   Payment Agent: approve_payment - {status}")
        if not decision.get('allowed'):
            print(f"      Reason: {decision.get('reason', 'Unknown')[:60]}...")
    
    time.sleep(0.5)
    
    # Action with proper permission (will succeed)
    result2 = vault.execute_agent_action(
        principal_id="agent-finance-admin-001",
        action="approve_payment",
        resource_id="payment-67890",
        resource_type="payment",
        justification="Quarterly bonus payment"
    )
    
    if result2.get('success'):
        decision = result2['data']
        status = "✅ ALLOWED" if decision.get('allowed') else "❌ DENIED"
        print(f"   Finance Agent: approve_payment - {status}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    
    print(f"\n📊 View results at: {vault_url}")
    print("\nPages to check:")
    print(f"  • Dashboard:  {vault_url}/")
    print(f"  • Models:     {vault_url}/models")
    print(f"  • Web Events: {vault_url}/web")
    print(f"  • Agents:     {vault_url}/agents")
    
    print("\n✨ Quick start complete!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
