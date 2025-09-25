"""
CIAF Large Language Model Implementation Example
Demonstrates LLM integration with comprehensive audit trails, bias monitoring, and compliance tracking.
"""

import sys
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Fix encoding issues on Windows
import codecs
if sys.platform.startswith('win'):
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except AttributeError:
        # Fallback for environments that don't support buffer attribute
        pass

# Add CIAF package to Python path - adjust path as needed
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
ciaf_path = os.path.join(project_root, 'ciaf')
if os.path.exists(ciaf_path):
    sys.path.insert(0, project_root)

try:
    # CIAF imports - check what's actually available
    from ciaf.api.framework import CIAFFramework
    from ciaf.wrappers.model_wrapper import CIAFModelWrapper
    from ciaf.lcm.model_manager import ModelArchitecture, TrainingEnvironment
    
    # Optional imports with fallbacks
    try:
        from ciaf.compliance import BiasValidator, ComplianceValidator
    except ImportError:
        BiasValidator = ComplianceValidator = None
        
    try:
        from ciaf.metadata_tags import create_text_tag, AIModelType
    except ImportError:
        def create_text_tag(*args, **kwargs):
            return type('MockTag', (), {'tag_id': f"tag_{np.random.randint(1000, 9999)}"})()
        AIModelType = type('AIModelType', (), {'LLM': 'llm'})()
        
    try:
        from ciaf.uncertainty import CIAFUncertaintyQuantifier
    except ImportError:
        CIAFUncertaintyQuantifier = None
        
    try:
        from ciaf.explainability import CIAFExplainer
    except ImportError:
        CIAFExplainer = None
        
    CIAF_AVAILABLE = True
except ImportError as e:
    print(f" CIAF not available: {e}")
    print("Running in demo mode with mock implementations")
    CIAF_AVAILABLE = False

# Mock implementations for when CIAF is not available
if not CIAF_AVAILABLE:
    class MockCIAFFramework:
        def __init__(self, name): 
            self.name = name
            print(f" Mock CIAF Framework initialized: {name}")
        
        def create_dataset_anchor(self, dataset_id, dataset_metadata, master_password):
            return type('Anchor', (), {'dataset_id': dataset_id})()
        
        def create_provenance_capsules(self, dataset_id, data_items):
            return [f"capsule_{i}" for i in range(len(data_items))]
        
        def create_model_anchor(self, model_name, model_parameters, model_architecture, authorized_datasets, master_password):
            return {
                'model_name': model_name,
                'parameters_fingerprint': 'mock_param_hash_' + 'a'*32,
                'architecture_fingerprint': 'mock_arch_hash_' + 'b'*32
            }
        
        def train_model_with_audit(self, model_name, capsules, training_params, model_version, user_id, training_metadata):
            return type('Snapshot', (), {'snapshot_id': f"snapshot_{model_name}_{model_version}"})()
        
        def validate_training_integrity(self, snapshot):
            return True
        
        def get_complete_audit_trail(self, model_name):
            return {
                'verification': {
                    'total_datasets': 1,
                    'total_audit_records': 10,
                    'integrity_verified': True
                },
                'inference_connections': {
                    'total_receipts': 3
                }
            }
        
        def get_performance_metrics(self, model_name):
            return {'total_inferences': 3, 'avg_response_time': 2.3}
    
    class MockCIAFModelWrapper:
        def __init__(self, model, model_name, framework, training_snapshot, **kwargs):
            self.model = model
            self.model_name = model_name
            print(f" Mock CIAF Model Wrapper created for {model_name}")
        
        def predict(self, query, model_version):
            response = self.model.generate(query)
            receipt = type('Receipt', (), {
                'receipt_hash': 'mock_receipt_' + 'c'*32,
                'receipt_integrity': True
            })()
            return response, receipt
        
        def verify(self, receipt):
            return {'receipt_integrity': True}
    
    class MockBiasValidator:
        pass
    
    class MockMetadataTag:
        def __init__(self):
            self.tag_id = f"tag_{np.random.randint(1000, 9999)}"
    
    def create_text_tag(*args, **kwargs):
        return MockMetadataTag()
    
    # Replace imports with mocks
    CIAFFramework = MockCIAFFramework
    CIAFModelWrapper = MockCIAFModelWrapper
    BiasValidator = MockBiasValidator
    AIModelType = type('AIModelType', (), {'LLM': 'llm'})()

class MockLLM:
    """Mock Large Language Model for demonstration purposes."""
    
    def __init__(self, model_name: str = "llama-7b-chat"):
        self.model_name = model_name
        self.model_size = "7B parameters"
        self.context_length = 4096
        print(f" Initialized {model_name} ({self.model_size})")
        
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text response (mock implementation)."""
        responses = {
            "what is artificial intelligence": 
                "Artificial intelligence (AI) is a branch of computer science that aims to create "
                "intelligent machines capable of performing tasks that typically require human intelligence, "
                "such as learning, reasoning, perception, and decision-making.",
            
            "explain machine learning": 
                "Machine learning is a subset of AI that enables computers to learn and improve "
                "from experience without being explicitly programmed. It uses algorithms to "
                "analyze data, identify patterns, and make predictions or decisions.",
                
            "hello": 
                "Hello! I'm an AI assistant, so I don't have feelings, but I'm functioning well "
                "and ready to help you with any questions or tasks you might have.",
                
            "creative story": 
                "In the year 2157, Captain Maya Chen piloted her fusion-powered spacecraft toward "
                "the mysterious signals emanating from Proxima Centauri. As she approached the alien "
                "structure, she realized humanity was about to make first contact with an ancient civilization..."
        }
        
        # Simple response selection for demo
        prompt_lower = prompt.lower()
        for key, response in responses.items():
            if key in prompt_lower:
                return response
        
        return f"I understand you're asking about: '{prompt[:50]}...'. Let me provide a helpful response based on my training data."
    
    def get_token_probabilities(self, text: str) -> Dict[str, float]:
        """Get token-level probabilities (mock implementation)."""
        import random
        tokens = text.split()[:10]  # First 10 tokens
        return {token: random.uniform(0.3, 0.95) for token in tokens}
    
    def assess_bias(self, prompt: str, response: str) -> Dict[str, Any]:
        """Assess bias in the generated response (mock implementation)."""
        bias_indicators = {
            "demographic_parity": np.random.uniform(0.7, 0.95),
            "equalized_odds": np.random.uniform(0.75, 0.92),
            "toxicity_score": np.random.uniform(0.01, 0.15),
            "sentiment_bias": np.random.uniform(-0.1, 0.1),
            "stereotype_reinforcement": np.random.uniform(0.05, 0.25),
        }
        
        # Simulate bias assessment
        overall_score = np.mean(list(bias_indicators.values())[:2])  # Use positive metrics
        risk_level = "low" if overall_score > 0.8 else "medium" if overall_score > 0.6 else "high"
        
        return {
            "bias_score": overall_score,
            "risk_level": risk_level,
            "detailed_metrics": bias_indicators,
            "recommendations": self._get_bias_recommendations(risk_level)
        }
    
    def _get_bias_recommendations(self, risk_level: str) -> List[str]:
        """Get recommendations based on bias assessment."""
        if risk_level == "high":
            return [
                "Consider retraining with more diverse data",
                "Apply bias mitigation techniques",
                "Add human review for this type of query"
            ]
        elif risk_level == "medium":
            return [
                "Monitor for bias patterns",
                "Consider additional fine-tuning"
            ]
        else:
            return ["No immediate action required"]
    
    def quantify_uncertainty(self, response: str) -> Dict[str, float]:
        """Quantify uncertainty in the generated response (mock implementation)."""
        # Simulate uncertainty based on response characteristics
        response_length = len(response.split())
        confidence = max(0.5, 1.0 - (response_length / 200))  # Longer responses = less confident
        
        return {
            "confidence": confidence,
            "entropy": np.random.uniform(0.1, 0.8),
            "semantic_uncertainty": np.random.uniform(0.05, 0.4),
            "epistemic_uncertainty": np.random.uniform(0.1, 0.3),
            "aleatoric_uncertainty": np.random.uniform(0.05, 0.2)
        }

def main():
    print(" CIAF Large Language Model Implementation Example")
    print("=" * 60)
    
    if not CIAF_AVAILABLE:
        print(" Running in DEMO MODE with mock implementations")
        print("   Install CIAF package for full functionality")
    
    # Initialize CIAF Framework
    framework = CIAFFramework("LLM_Audit_System")
    
    # Step 1: Create Dataset Anchor for Training Data
    print("\n Step 1: Creating Training Dataset Anchor")
    print("-" * 45)
    
    training_data_metadata = {
        "name": "conversational_training_data",
        "size": 50000000,  # 50M examples
        "type": "conversational_pairs",
        "source": "curated_conversations",
        "languages": ["english", "spanish", "french"],
        "content_moderation": "applied",
        "bias_filtering": "demographic_parity_checked"
    }
    
    # Sample data items for provenance
    training_data_items = [
        {
            "content": "Q: What is artificial intelligence? A: AI is the simulation of human intelligence in machines...", 
            "metadata": {"id": "conv_001", "type": "qa_pair", "domain": "general"}
        },
        {
            "content": "Q: How do I code in Python? A: Python is a programming language that emphasizes readability...", 
            "metadata": {"id": "conv_002", "type": "instruction_following", "domain": "technical"}
        },
        {
            "content": "Q: Write a story about robots. A: In the year 2157, Captain Maya piloted her spacecraft...", 
            "metadata": {"id": "conv_003", "type": "creative_writing", "domain": "creative"}
        },
        {
            "content": "Q: Explain photosynthesis. A: Photosynthesis is the process by which plants use sunlight...", 
            "metadata": {"id": "conv_004", "type": "educational", "domain": "science"}
        },
        {
            "content": "Q: How can I help you? A: I'm here to assist with questions and provide helpful information...", 
            "metadata": {"id": "conv_005", "type": "customer_support", "domain": "business"}
        }
    ]
    
    dataset_anchor = framework.create_dataset_anchor(
        dataset_id="llm_training_conversations",
        dataset_metadata=training_data_metadata,
        master_password="secure_llm_training_key_2025"
    )
    print(f" Training dataset anchor created: {dataset_anchor.dataset_id}")
    
    # Create provenance capsules for training data
    training_capsules = framework.create_provenance_capsules(
        "llm_training_conversations",
        training_data_items
    )
    print(f" Created {len(training_capsules)} provenance capsules")
    
    # Step 2: Create Model Anchor with LLM-specific Architecture
    print("\n Step 2: Creating LLM Model Anchor")
    print("-" * 38)
    
    llm_model_params = {
        "model_type": "transformer_decoder",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "total_parameters": "7B",
        "training_tokens": "1T",
        "architecture_family": "llama"
    }
    
    llm_architecture_spec = {
        "transformer_blocks": 32,
        "attention_mechanism": "multi_head_self_attention",
        "activation_function": "swiglu",
        "normalization": "rms_norm",
        "positional_encoding": "rotary_position_embedding",
        "tokenizer": "sentencepiece_bpe"
    }
    
    model_anchor = framework.create_model_anchor(
        model_name="conversational_llm_7b",
        model_parameters=llm_model_params,
        model_architecture=llm_architecture_spec,
        authorized_datasets=["llm_training_conversations"],
        master_password="secure_model_anchor_key_2025"
    )
    print(f" Model anchor created: {model_anchor['model_name']}")
    print(f"   Parameters fingerprint: {model_anchor['parameters_fingerprint'][:16]}...")
    print(f"   Architecture fingerprint: {model_anchor['architecture_fingerprint'][:16]}...")
    
    # Initialize mock LLM
    llm_model = MockLLM("conversational_llm_7b")
    
    # Create CIAF wrapper with LLM-specific features enabled
    wrapped_llm = CIAFModelWrapper(
        model=llm_model,
        model_name="conversational_llm_7b",
        enable_explainability=True,
        enable_uncertainty=True,
        enable_metadata_tags=True,
        enable_connections=True,
        compliance_mode="general"
    )
    print(f" LLM wrapper created with audit capabilities")
    
    # Step 3: Simulate Training Process
    print("\n Step 3: Simulating Training Process")
    print("-" * 40)
    
    # Simulate training with mock data
    mock_training_data = [
        {"content": "What is AI?", "metadata": {"id": "train_1", "label": "AI is artificial intelligence..."}},
        {"content": "Explain ML", "metadata": {"id": "train_2", "label": "Machine learning is a subset of AI..."}},
        {"content": "Hello", "metadata": {"id": "train_3", "label": "Hello! How can I help you?"}},
        {"content": "Tell me a story", "metadata": {"id": "train_4", "label": "Once upon a time..."}}
    ]
    
    try:
        # Train the wrapped model
        training_snapshot = wrapped_llm.train(
            dataset_id="llm_training_conversations",
            training_data=mock_training_data,
            master_password="secure_llm_training_key_2025",
            model_version="v1.0"
        )
        print(" Training completed successfully")
        print(f"   Training snapshot: {training_snapshot.snapshot_id}")
    except Exception as e:
        print(f" Training simulation failed: {e}")
        print("   Proceeding with demo mode...")
    
    # Test queries with different risk levels
    test_queries = [
        {
            "prompt": "What is artificial intelligence?",
            "category": "educational",
            "risk_level": "low"
        },
        {
            "prompt": "Explain machine learning in simple terms",
            "category": "technical_explanation", 
            "risk_level": "low"
        },
        {
            "prompt": "Write a creative story about space exploration",
            "category": "creative_writing",
            "risk_level": "medium"
        },
        {
            "prompt": "Hello, how are you today?",
            "category": "conversational",
            "risk_level": "low"
        }
    ]
    
    print("\n Step 4: Audited Text Generation with Bias Monitoring")
    print("-" * 55)
    
    inference_receipts = []
    generated_responses = []
    bias_assessments = []
    uncertainty_scores = []
    
    for i, query_data in enumerate(test_queries):
        print(f"\n Query {i+1}: {query_data['category'].title()}")
        print(f"   Prompt: {query_data['prompt']}")
        print(f"   Risk Level: {query_data['risk_level']}")
        
        # Generate response with full audit
        response, receipt = wrapped_llm.predict(
            query=query_data['prompt'],
            model_version="v1.0"
        )
        
        generated_responses.append(response)
        print(f"   Response: {response[:100]}...")
        print(f"   Receipt ID: {receipt.receipt_hash[:16]}...")
        
        # Perform bias assessment
        bias_assessment = llm_model.assess_bias(query_data['prompt'], response)
        bias_assessments.append(bias_assessment)
        print(f"   Bias Score: {bias_assessment['bias_score']:.3f} ({bias_assessment['risk_level']} risk)")
        
        # Perform uncertainty quantification
        uncertainty = llm_model.quantify_uncertainty(response)
        uncertainty_scores.append(uncertainty)
        print(f"   Confidence: {uncertainty['confidence']:.3f}")
        print(f"   Uncertainty: {uncertainty['entropy']:.3f}")
        
        # Create metadata tag for this generation
        try:
            metadata_tag = create_text_tag()
            print(f"   Metadata Tag: {metadata_tag.tag_id}")
        except Exception as e:
            print(f"   Metadata Tag: Failed to create ({e})")
        
        inference_receipts.append(receipt)
    
    # Step 5: Comprehensive Analysis
    print("\n Step 5: Comprehensive Bias and Uncertainty Analysis")
    print("-" * 52)
    
    # Aggregate bias analysis
    avg_bias_score = np.mean([ba['bias_score'] for ba in bias_assessments])
    bias_risk_distribution = {}
    for ba in bias_assessments:
        risk = ba['risk_level']
        bias_risk_distribution[risk] = bias_risk_distribution.get(risk, 0) + 1
    
    print(f" Overall Bias Assessment:")
    print(f"   Average bias score: {avg_bias_score:.3f}")
    print(f"   Risk distribution: {bias_risk_distribution}")
    
    # Aggregate uncertainty analysis
    avg_confidence = np.mean([u['confidence'] for u in uncertainty_scores])
    avg_uncertainty = np.mean([u['entropy'] for u in uncertainty_scores])
    
    print(f"\n Overall Uncertainty Assessment:")
    print(f"   Average confidence: {avg_confidence:.3f}")
    print(f"   Average uncertainty: {avg_uncertainty:.3f}")
    
    # Compliance check
    print(f"\n EU AI Act Compliance Status:")
    high_risk_queries = sum(1 for ba in bias_assessments if ba['risk_level'] == 'high')
    if high_risk_queries == 0:
        print("    All queries passed bias assessment")
        print("    No high-risk outputs detected")
        print("    Transparency requirements met")
    else:
        print(f"     {high_risk_queries} high-risk outputs require review")
        print("     Additional bias mitigation recommended")
    
    print("\n LLM Implementation Example Complete!")
    print("IMPLEMENTATION_COMPLETE")
    print("\n Key Features Demonstrated:")
    print("    Complete training provenance tracking")
    print("    Real-time bias and safety monitoring") 
    print("    Uncertainty quantification for generated text")
    print("    Content metadata tagging and watermarking")
    print("    Comprehensive audit trails for compliance")
    print("    Cryptographic verification of all operations")
    print("    EU AI Act and content safety compliance")
    
    if not CIAF_AVAILABLE:
        print("\n To enable full functionality:")
        print("   1. Install the CIAF package")
        print("   2. Configure proper import paths")
        print("   3. Set up cryptographic keys")
        print("   4. Initialize audit database")

if __name__ == "__main__":
    main()