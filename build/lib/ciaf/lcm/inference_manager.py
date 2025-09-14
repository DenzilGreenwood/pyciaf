"""
CIAF LCM Inference Manager

Enhanced inference management with proper commitments, chaining, and batch processing.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from ..core import sha256_hash, MerkleTree, secure_random_bytes
from ..inference import InferenceReceipt
from .policy import LCMPolicy, get_default_policy, CommitmentType, DomainType

if TYPE_CHECKING:
    from .model_manager import LCMModelAnchor
    from .deployment_manager import LCMDeploymentAnchor


@dataclass
class LCMInferenceCommitment:
    """Enhanced inference commitment with privacy protection."""
    commitment_type: CommitmentType
    commitment_value: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class LCMInferenceReceipt:
    """Enhanced inference receipt for LCM."""
    
    def __init__(
        self,
        receipt_id: str,
        model_anchor_ref: str,
        deployment_anchor_ref: str,
        request_id: str,
        query: str,
        ai_output: str,
        input_commitment: LCMInferenceCommitment,
        output_commitment: LCMInferenceCommitment,
        explanation_digests: List[str] = None,
        prev_chain_digest: str = None,
        policy: LCMPolicy = None
    ):
        """
        Initialize LCM inference receipt.
        
        Args:
            receipt_id: Unique receipt identifier
            model_anchor_ref: Reference to model anchor
            deployment_anchor_ref: Reference to deployment anchor
            request_id: Request identifier
            query: Input query/prompt
            ai_output: AI model output
            input_commitment: Input commitment for privacy
            output_commitment: Output commitment for privacy
            explanation_digests: Optional explanation digests (e.g., SHAP)
            prev_chain_digest: Previous chain digest for chaining
            policy: LCM policy
        """
        self.receipt_id = receipt_id
        self.model_anchor_ref = model_anchor_ref
        self.deployment_anchor_ref = deployment_anchor_ref
        self.request_id = request_id
        self.query = query
        self.ai_output = ai_output
        self.input_commitment = input_commitment
        self.output_commitment = output_commitment
        self.explanation_digests = explanation_digests or []
        self.prev_chain_digest = prev_chain_digest
        self.policy = policy or get_default_policy()
        self.timestamp = datetime.now().isoformat()
        
        # Compute receipt digest
        self.receipt_digest = self._compute_receipt_digest()
        
        # Compute chain digest (for chaining)
        self.chain_digest = self._compute_chain_digest()
        
        self.anchor_id = f"r_{self.receipt_digest[:8]}..."
        
        print(f"🧾 LCM Inference Receipt '{self.receipt_id}' created: {self.anchor_id}")
    
    def _compute_receipt_digest(self) -> str:
        """Compute receipt digest."""
        receipt_data = {
            "receipt_id": self.receipt_id,
            "model_anchor_ref": self.model_anchor_ref,
            "deployment_anchor_ref": self.deployment_anchor_ref,
            "request_id": self.request_id,
            "input_commitment": self.input_commitment.commitment_value,
            "output_commitment": self.output_commitment.commitment_value,
            "explanation_digests": self.explanation_digests,
            "timestamp": self.timestamp
        }
        canonical_json = json.dumps(receipt_data, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_json.encode('utf-8'))
    
    def _compute_chain_digest(self) -> str:
        """Compute chain digest for linking."""
        if self.prev_chain_digest:
            chain_data = f"{self.prev_chain_digest}||{self.receipt_digest}"
        else:
            chain_data = self.receipt_digest
        
        return sha256_hash(chain_data.encode('utf-8'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.anchor_id,
            "receipt_id": self.receipt_id,
            "model_anchor_ref": self.model_anchor_ref,
            "deployment_anchor_ref": self.deployment_anchor_ref,
            "request_id": self.request_id,
            "input_c": self.input_commitment.commitment_value,
            "output_c": self.output_commitment.commitment_value,
            "explanation_digests": self.explanation_digests,
            "timestamp": self.timestamp,
            "receipt_digest": self.receipt_digest,
            "chain_digest": self.chain_digest,
            "anchor": self.anchor_id
        }


class LCMInferenceChain:
    """Enhanced inference chain for LCM."""
    
    def __init__(self, chain_id: str, policy: LCMPolicy = None):
        """Initialize inference chain."""
        self.chain_id = chain_id
        self.policy = policy or get_default_policy()
        self.receipts: List[LCMInferenceReceipt] = []
        self.current_chain_digest = "genesis"
    
    def add_receipt(
        self,
        receipt_id: str,
        model_anchor_ref: str,
        deployment_anchor_ref: str,
        request_id: str,
        query: str,
        ai_output: str,
        explanation_digests: List[str] = None
    ) -> LCMInferenceReceipt:
        """Add receipt to the chain."""
        # Create input and output commitments
        input_commitment = self._create_commitment(query)
        output_commitment = self._create_commitment(ai_output)
        
        # Determine previous chain digest
        prev_chain_digest = self.current_chain_digest if self.current_chain_digest != "genesis" else None
        
        # Create receipt
        receipt = LCMInferenceReceipt(
            receipt_id=receipt_id,
            model_anchor_ref=model_anchor_ref,
            deployment_anchor_ref=deployment_anchor_ref,
            request_id=request_id,
            query=query,
            ai_output=ai_output,
            input_commitment=input_commitment,
            output_commitment=output_commitment,
            explanation_digests=explanation_digests,
            prev_chain_digest=prev_chain_digest,
            policy=self.policy
        )
        
        # Add to chain
        self.receipts.append(receipt)
        self.current_chain_digest = receipt.chain_digest
        
        return receipt
    
    def _create_commitment(self, data: str) -> LCMInferenceCommitment:
        """Create commitment for data according to policy."""
        if self.policy.commitments == CommitmentType.PLAINTEXT:
            return LCMInferenceCommitment(
                commitment_type=CommitmentType.PLAINTEXT,
                commitment_value=data
            )
        elif self.policy.commitments == CommitmentType.SALTED:
            # Simple salt-based commitment
            salt = secure_random_bytes(16)
            commitment_value = sha256_hash(salt + data.encode('utf-8'))[:16] + "..."
            return LCMInferenceCommitment(
                commitment_type=CommitmentType.SALTED,
                commitment_value=commitment_value,
                metadata={"salted": True}
            )
        elif self.policy.commitments == CommitmentType.HMAC_SHA256:
            # HMAC-based commitment (simplified for demo)
            import hmac
            key = secure_random_bytes(32)
            commitment_value = hmac.new(key, data.encode('utf-8'), 'sha256').hexdigest()[:16] + "..."
            return LCMInferenceCommitment(
                commitment_type=CommitmentType.HMAC_SHA256,
                commitment_value=commitment_value,
                metadata={"hmac": True}
            )
    
    def get_final_chain_digest(self) -> str:
        """Get final chain digest."""
        return self.current_chain_digest
    
    def verify_chain_integrity(self) -> bool:
        """Verify chain integrity."""
        if not self.receipts:
            return True
        
        prev_digest = "genesis"
        for receipt in self.receipts:
            expected_prev = prev_digest if prev_digest != "genesis" else None
            if receipt.prev_chain_digest != expected_prev:
                return False
            prev_digest = receipt.chain_digest
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "chain_id": self.chain_id,
            "mode": "linked",
            "final_chain_digest": self.get_final_chain_digest(),
            "receipts": [receipt.to_dict() for receipt in self.receipts],
            "chain_valid": self.verify_chain_integrity()
        }


class LCMInferenceManager:
    """Enhanced inference manager for LCM."""
    
    def __init__(self, policy: LCMPolicy = None):
        """Initialize LCM inference manager."""
        self.policy = policy or get_default_policy()
        self.inference_chains: Dict[str, LCMInferenceChain] = {}
        self.batch_windows: Dict[str, str] = {}  # window_id -> batch_root
    
    def create_inference_chain(self, chain_id: str) -> LCMInferenceChain:
        """Create new inference chain."""
        chain = LCMInferenceChain(chain_id, self.policy)
        self.inference_chains[chain_id] = chain
        return chain
    
    def get_inference_chain(self, chain_id: str) -> Optional[LCMInferenceChain]:
        """Get inference chain by ID."""
        return self.inference_chains.get(chain_id)
    
    def perform_inference_with_audit(
        self,
        chain_id: str,
        receipt_id: str,
        model_anchor_ref: str,
        deployment_anchor_ref: str,
        request_id: str,
        query: str,
        ai_output: str,
        explanation_digests: List[str] = None
    ) -> LCMInferenceReceipt:
        """Perform inference with complete audit trail."""
        # Get or create chain
        chain = self.get_inference_chain(chain_id)
        if not chain:
            chain = self.create_inference_chain(chain_id)
        
        # Add receipt to chain
        receipt = chain.add_receipt(
            receipt_id=receipt_id,
            model_anchor_ref=model_anchor_ref,
            deployment_anchor_ref=deployment_anchor_ref,
            request_id=request_id,
            query=query,
            ai_output=ai_output,
            explanation_digests=explanation_digests
        )
        
        return receipt
    
    def create_inference_batch_root(
        self,
        window_id: str,
        chain_ids: List[str]
    ) -> str:
        """
        Create inference batch root for a time window.
        
        Args:
            window_id: Time window identifier
            chain_ids: List of chain IDs to include in batch
            
        Returns:
            Batch root hash
        """
        # Collect all receipt digests from specified chains
        receipt_digests = []
        
        for chain_id in chain_ids:
            chain = self.get_inference_chain(chain_id)
            if chain:
                for receipt in chain.receipts:
                    receipt_digests.append(receipt.receipt_digest)
        
        if not receipt_digests:
            return "empty_batch"
        
        # Compute Merkle root
        merkle_tree = MerkleTree(receipt_digests)
        batch_root = merkle_tree.get_root()
        
        # Store batch root
        self.batch_windows[window_id] = batch_root
        
        print(f"🌳 Inference batch root created for window {window_id}: {batch_root[:8]}...{batch_root[-8:]}")
        
        return batch_root
    
    def get_batch_root(self, window_id: str) -> Optional[str]:
        """Get batch root for window."""
        return self.batch_windows.get(window_id)
    
    def format_inference_summary(self, chain_id: str, window_id: str = None) -> str:
        """Format inference summary for pretty printing."""
        chain = self.get_inference_chain(chain_id)
        if not chain:
            return f"Inference chain {chain_id} not found"
        
        lines = []
        for i, receipt in enumerate(chain.receipts[:3]):  # Show first 3
            input_preview = receipt.query[:30] + "..." if len(receipt.query) > 30 else receipt.query
            lines.append(f"  ▸ r{i+1} input_c={receipt.input_commitment.commitment_type.value}, output_c={receipt.output_commitment.commitment_type.value}")
            lines.append(f"    receipt: {receipt.anchor_id}  chain_digest: c_{receipt.chain_digest[:8]}...")
        
        if len(chain.receipts) > 3:
            lines.append(f"  ... and {len(chain.receipts) - 3} more receipts")
        
        if window_id:
            batch_root = self.get_batch_root(window_id)
            if batch_root:
                lines.append(f"  🌳 inference_batch_root (window={window_id}): ib_{batch_root[:8]}...{batch_root[-8:]}")
        
        return "\n".join(lines)
    
    def simulate_inference_receipt(
        self,
        inference_id: str,
        model_anchor: 'LCMModelAnchor',
        deployment_anchor: 'LCMDeploymentAnchor',
        query: str = "What is AI?"
    ) -> LCMInferenceReceipt:
        """
        Simulate inference receipt for demonstration.
        
        Args:
            inference_id: Inference identifier
            model_anchor: Model anchor
            deployment_anchor: Deployment anchor
            query: Input query
            
        Returns:
            LCMInferenceReceipt instance
        """
        print(f"🔮 Creating inference receipt: {inference_id}")
        
        # Create mock response
        mock_response = "AI is a field of computer science focused on creating intelligent machines."
        
        # Create input commitment
        input_commitment = LCMInferenceCommitment(
            commitment_type=CommitmentType.SALTED,
            commitment_value=sha256_hash(("salt" + query).encode('utf-8'))[:16] + "...",
            metadata={"salted": True}
        )
        
        # Create output commitment
        output_commitment = LCMInferenceCommitment(
            commitment_type=CommitmentType.SALTED,
            commitment_value=sha256_hash(("salt" + mock_response).encode('utf-8'))[:16] + "...",
            metadata={"salted": True}
        )
        
        # Create inference receipt
        receipt = self.create_inference_receipt(
            inference_id=inference_id,
            model_anchor=model_anchor,
            deployment_anchor=deployment_anchor,
            query=query,
            response=mock_response,
            input_commitment=input_commitment,
            output_commitment=output_commitment,
            inference_type="text_generation"
        )
        
        print(f"✅ Inference receipt created: {receipt.anchor_id}")
        return receipt
