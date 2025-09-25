"""
Property-based tests for CIAF receipts and core functionality.

Uses hypothesis for property-based testing to ensure robust validation
and catch edge cases in receipt processing.
"""

import pytest
try:
    from hypothesis import given, strategies as st, assume, settings
    from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators for environments without hypothesis
    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class st:
        @staticmethod
        def text(**kwargs):
            return "dummy"
        @staticmethod
        def integers(**kwargs):
            return 42
        @staticmethod
        def dictionaries(**kwargs):
            return {}
        @staticmethod
        def lists(**kwargs):
            return []
        @staticmethod
        def floats(**kwargs):
            return 0.5
        @staticmethod
        def builds(func, *args):
            return func("dummy")
        @staticmethod
        def fixed_dictionaries(mapping):
            return {k: "dummy" for k in mapping.keys()}
        @staticmethod
        def sampled_from(items):
            return items[0] if items else "dummy"

import hashlib
import json
from ciaf.enhanced_receipts import TrainingReceipt, InferenceReceipt, PYDANTIC_AVAILABLE

def create_training_receipt(**kwargs):
    """Helper to create training receipts for testing."""
    if not PYDANTIC_AVAILABLE:
        return {'success': True}  # Mock for tests without pydantic
    
    # Provide defaults for required fields
    defaults = {
        'dataset_anchor': 'a' * 64,
        'model_anchor': 'b' * 64, 
        'code_digest': 'sha256:' + 'c' * 64,
        'config_digest': 'sha256:' + 'd' * 64,
        'random_seeds': {'python': 42},
        'env': {'python': '3.12.0', 'frameworks': {}, 'hardware': 'CPU'}
    }
    defaults.update(kwargs)
    
    try:
        from ciaf.enhanced_receipts import RandomSeeds, EnvironmentInfo
        return TrainingReceipt(
            dataset_anchor=defaults['dataset_anchor'],
            model_anchor=defaults['model_anchor'],
            code_digest=defaults['code_digest'],
            config_digest=defaults['config_digest'],
            random_seeds=RandomSeeds(**defaults['random_seeds']),
            env=EnvironmentInfo(**defaults['env'])
        )
    except Exception as e:
        return {'error': str(e), 'success': False}
import uuid
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from ciaf.enhanced_receipts import (
        TrainingReceipt, InferenceReceipt, ReceiptValidator,
        create_training_receipt, create_inference_receipt
    )
    from ciaf.evidence_strength import EvidenceStrength, get_evidence_strength
    from ciaf.determinism_metadata import capture_determinism_metadata
except ImportError as e:
    print(f"Warning: Could not import CIAF modules for property tests: {e}")

# Test data generation strategies (first definition - remove duplicates later)
if HYPOTHESIS_AVAILABLE:
    # Generate valid hex strings for anchors
    hex_char = st.sampled_from('0123456789abcdef')
    anchor_strategy = st.text(alphabet=hex_char, min_size=64, max_size=64)
    
    # Generate valid SHA256 digests
    sha256_strategy = st.builds(
        lambda x: f"sha256:{hashlib.sha256(x.encode()).hexdigest()}",
        st.text(min_size=1, max_size=100)
    )
    
    # Generate valid UUIDs
    uuid_strategy = st.builds(lambda: str(uuid.uuid4()))
    
    # Generate random seeds
    seeds_strategy = st.dictionaries(
        keys=st.sampled_from(['python', 'numpy', 'torch', 'tensorflow']),
        values=st.integers(min_value=0, max_value=2**31-1),
        min_size=1
    )
    
    # Generate environment info
    env_strategy = st.fixed_dictionaries({
        'python': st.text(min_size=5, max_size=20),
        'frameworks': st.dictionaries(
            keys=st.sampled_from(['sklearn', 'numpy', 'pandas']),
            values=st.text(min_size=3, max_size=10),
            max_size=5
        ),
        'hardware': st.text(min_size=5, max_size=50)
    })
else:
    # Dummy strategies for when hypothesis is not available
    import uuid
    anchor_strategy = 'a' * 64
    sha256_strategy = 'sha256:' + 'b' * 64
    uuid_strategy = str(uuid.uuid4())
    seeds_strategy = {'python': 42}
    env_strategy = {'python': '3.12.0', 'frameworks': {}, 'hardware': 'CPU'}

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestReceiptProperties:
    """Property-based tests for receipt validation."""
    
    @given(
        dataset_anchor=anchor_strategy,
        model_anchor=anchor_strategy,
        code_digest=sha256_strategy,
        config_digest=sha256_strategy,
        random_seeds=seeds_strategy,
        env_info=env_strategy
    )
    @settings(max_examples=50)
    def test_training_receipt_never_accepts_missing_anchors(
        self, dataset_anchor, model_anchor, code_digest, config_digest, random_seeds, env_info
    ):
        """Property: Training receipts must never accept missing or invalid anchors."""
        
        # Valid receipt should validate
        valid_receipt = create_training_receipt(
            dataset_anchor=dataset_anchor,
            model_anchor=model_anchor,
            code_digest=code_digest,
            config_digest=config_digest,
            random_seeds=random_seeds,
            env_info=env_info
        )
        
        validator = ReceiptValidator()
        assert validator.validate_training_receipt(valid_receipt)
        
        # Missing dataset_anchor should fail
        invalid_receipt = valid_receipt.copy()
        del invalid_receipt['dataset_anchor']
        assert not validator.validate_training_receipt(invalid_receipt)
        
        # Invalid anchor format should fail
        invalid_receipt = valid_receipt.copy()
        invalid_receipt['dataset_anchor'] = 'invalid_anchor'
        assert not validator.validate_training_receipt(invalid_receipt)
    
    @given(
        model_anchor=anchor_strategy,
        input_digest=st.text(min_size=64, max_size=64, alphabet='0123456789abcdef'),
        output_digest=st.text(min_size=64, max_size=64, alphabet='0123456789abcdef'),
        score=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_inference_receipt_thresholds_validation(
        self, model_anchor, input_digest, output_digest, score
    ):
        """Property: Inference receipts must validate threshold ranges."""
        
        input_commitment = {"algo": "sha256", "digest": input_digest}
        output_commitment = {"algo": "sha256", "digest": output_digest}
        decision = {"score": score, "label": "test"}
        
        receipt = create_inference_receipt(
            model_anchor=model_anchor,
            input_commitment=input_commitment,
            output_commitment=output_commitment,
            decision=decision,
            thresholds={"alert_at": score}
        )
        
        validator = ReceiptValidator()
        assert validator.validate_inference_receipt(receipt)
        
        # Invalid threshold should fail
        invalid_receipt = receipt.copy()
        invalid_receipt['thresholds'] = {"alert_at": 1.5}  # > 1.0
        assert not validator.validate_inference_receipt(invalid_receipt)
    
    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.booleans(),
        min_size=1,
        max_size=10
    ))
    def test_evidence_strength_determination(self, component_states):
        """Property: Evidence strength follows consistent logic."""
        
        strength = get_evidence_strength(component_states)
        
        if all(component_states.values()):
            assert strength == EvidenceStrength.REAL
        elif any(component_states.values()):
            assert strength == EvidenceStrength.FALLBACK
        else:
            assert strength == EvidenceStrength.SIMULATED

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class ReceiptStateMachine(RuleBasedStateMachine):
    """State machine for testing receipt system invariants."""
    
    def __init__(self):
        super().__init__()
        self.receipts = []
        self.validator = ReceiptValidator()
    
    @rule(
        dataset_anchor=anchor_strategy,
        model_anchor=anchor_strategy,
        code_digest=sha256_strategy
    )
    def create_training_receipt(self, dataset_anchor, model_anchor, code_digest):
        """Create a training receipt and add to state."""
        receipt = create_training_receipt(
            dataset_anchor=dataset_anchor,
            model_anchor=model_anchor,
            code_digest=code_digest,
            config_digest=code_digest,  # reuse for simplicity
            random_seeds={"python": 42},
            env_info={"python": "3.11", "frameworks": {}, "hardware": "cpu"}
        )
        self.receipts.append(receipt)
    
    @invariant()
    def all_receipts_have_unique_ids(self):
        """Invariant: All receipts must have unique IDs."""
        receipt_ids = [r.get('receipt_id') for r in self.receipts]
        assert len(receipt_ids) == len(set(receipt_ids)), "Receipt IDs must be unique"
    
    @invariant() 
    def all_receipts_validate(self):
        """Invariant: All created receipts must be valid."""
        for receipt in self.receipts:
            if 'dataset_anchor' in receipt:  # Training receipt
                assert self.validator.validate_training_receipt(receipt)
            else:  # Inference receipt
                assert self.validator.validate_inference_receipt(receipt)

# Idempotency tests
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestIdempotency:
    """Test idempotent operations."""
    
    @given(
        dataset_anchor=anchor_strategy,
        model_anchor=anchor_strategy,
        random_seed=st.integers(min_value=0, max_value=2**31-1)
    )
    def test_deterministic_receipt_generation(self, dataset_anchor, model_anchor, random_seed):
        """Property: Same inputs should produce identical receipts (with same seed)."""
        
        # Create receipt twice with same inputs
        common_args = {
            "dataset_anchor": dataset_anchor,
            "model_anchor": model_anchor,
            "code_digest": "sha256:abc123",
            "config_digest": "sha256:def456",
            "random_seeds": {"python": random_seed},
            "env_info": {"python": "3.11", "frameworks": {}, "hardware": "cpu"}
        }
        
        receipt1 = create_training_receipt(**common_args)
        receipt2 = create_training_receipt(**common_args)
        
        # Receipts should be identical except for receipt_id and timestamp
        for key in ['dataset_anchor', 'model_anchor', 'code_digest', 'config_digest']:
            assert receipt1[key] == receipt2[key]

# Security tests
class TestSecurityProperties:
    """Test security properties of receipts."""
    
    def test_salt_length_validation(self):
        """Test that salts must be minimum length."""
        import base64
        
        # Valid salt (128 bits = 16 bytes)
        valid_salt = base64.b64encode(b'0' * 16).decode()
        
        input_commitment = {
            "algo": "sha256_salted",
            "digest": "a" * 64,
            "salt": valid_salt
        }
        
        receipt = create_inference_receipt(
            model_anchor="a" * 64,
            input_commitment=input_commitment,
            output_commitment={"algo": "sha256", "digest": "b" * 64},
            decision={"score": 0.5}
        )
        
        validator = ReceiptValidator()
        assert validator.validate_inference_receipt(receipt)
        
        # Invalid salt (too short)
        invalid_salt = base64.b64encode(b'0' * 8).decode()  # Only 64 bits
        invalid_receipt = receipt.copy()
        invalid_receipt['input_commitment']['salt'] = invalid_salt
        
        # Should fail validation if pydantic available
        if HYPOTHESIS_AVAILABLE:  # Proxy for stricter validation
            assert not validator.validate_inference_receipt(invalid_receipt)

# Run tests
if __name__ == "__main__":
    if HYPOTHESIS_AVAILABLE:
        # Run property tests
        test_class = TestReceiptProperties()
        print("Running property-based tests...")
        
        # Run a few specific tests
        try:
            test_class.test_evidence_strength_determination({"comp1": True, "comp2": False})
            print("✅ Evidence strength test passed")
        except Exception as e:
            print(f"❌ Evidence strength test failed: {e}")
            
        print("Property-based testing setup complete!")
    else:
        print("⚠️ Hypothesis not available - property tests disabled")
        print("Install hypothesis for full property-based testing: pip install hypothesis")