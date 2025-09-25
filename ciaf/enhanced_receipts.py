"""
Enhanced Receipt Schemas with Strict Validation

Implements pydantic models for training and inference receipts with
comprehensive validation and the suggested schema improvements.
"""

try:
    from pydantic import BaseModel, Field, field_validator
    from pydantic import ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for environments without pydantic
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    def Field(**kwargs):
        return kwargs.get('default')
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    class PydanticValidationError(Exception):
        pass

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from enum import Enum
import uuid
import json

class EvidenceStrength(str, Enum):
    """Evidence strength levels."""
    REAL = "real"
    SIMULATED = "simulated"
    FALLBACK = "fallback"

class OversightStatus(str, Enum):
    """Oversight review status."""
    AUTO_APPROVED = "auto_approved"
    QUEUED = "queued" 
    REVIEWED = "reviewed"
    ESCALATED = "escalated"

class CommitmentAlgorithm(str, Enum):
    """Supported commitment algorithms."""
    SHA256 = "sha256"
    SHA256_SALTED = "sha256_salted"
    HMAC_SHA256 = "hmac_sha256"

class BaseReceipt(BaseModel if PYDANTIC_AVAILABLE else object):
    """Base receipt with common fields."""
    receipt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evidence_strength: EvidenceStrength = EvidenceStrength.REAL
    committed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    if PYDANTIC_AVAILABLE:
        @field_validator('receipt_id')
        @classmethod
        def validate_receipt_id(cls, v):
            try:
                uuid.UUID(v)
                return v
            except ValueError:
                raise ValueError('receipt_id must be a valid UUID')
        
        @field_validator('committed_at')
        @classmethod
        def validate_timestamp(cls, v):
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
                return v
            except ValueError:
                raise ValueError('committed_at must be RFC3339 timestamp')

class RandomSeeds(BaseModel if PYDANTIC_AVAILABLE else object):
    """Random seeds for reproducibility."""
    python: Optional[int] = None
    numpy: Optional[int] = None
    torch: Optional[int] = None
    tensorflow: Optional[int] = None
    
class EnvironmentInfo(BaseModel if PYDANTIC_AVAILABLE else object):
    """Environment information for reproducibility."""
    python: str
    frameworks: Dict[str, str] = Field(default_factory=dict)
    hardware: str
    git_commit: Optional[str] = None

class Commitment(BaseModel if PYDANTIC_AVAILABLE else object):
    """Cryptographic commitment structure."""
    algo: CommitmentAlgorithm
    digest: str
    salt: Optional[str] = None  # base64 encoded
    
    if PYDANTIC_AVAILABLE:
        @field_validator('salt')
        @classmethod
        def validate_salt_length(cls, v):
            if v is not None:
                import base64
                try:
                    salt_bytes = base64.b64decode(v)
                    if len(salt_bytes) < 16:  # 128 bits minimum
                        raise ValueError('Salt must be at least 128 bits (16 bytes)')
                except Exception:
                    raise ValueError('Salt must be valid base64')
            return v

class OversightDecision(BaseModel if PYDANTIC_AVAILABLE else object):
    """Oversight decision record."""
    status: OversightStatus
    actor_id: Optional[str] = None
    decision: Optional[str] = None
    rationale: Optional[str] = None
    competence_tags: List[str] = Field(default_factory=list)
    dual_control_required: bool = False
    secondary_reviewer: Optional[str] = None

class TrainingReceipt(BaseReceipt):
    """Enhanced training receipt schema."""
    model_config = {'protected_namespaces': ()}  # Allow model_anchor field
    
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_anchor: str = Field(..., min_length=64, max_length=64)  # hex string
    model_anchor: str = Field(..., min_length=64, max_length=64)   # hex string
    code_digest: str = Field(..., pattern=r'^sha256:[a-f0-9]{64}$')
    config_digest: str = Field(..., pattern=r'^sha256:[a-f0-9]{64}$')
    random_seeds: RandomSeeds
    env: EnvironmentInfo
    metrics: Dict[str, Any] = Field(default_factory=dict)
    merkle_path: List[str] = Field(default_factory=list)
    fallback_reasons: List[Dict[str, Any]] = Field(default_factory=list)
    
    if PYDANTIC_AVAILABLE:
        @field_validator('dataset_anchor', 'model_anchor')
        @classmethod
        def validate_anchor_format(cls, v):
            if not all(c in '0123456789abcdef' for c in v.lower()):
                raise ValueError('Anchor must be hex string')
            return v.lower()
        
        @field_validator('merkle_path')
        @classmethod
        def validate_merkle_path(cls, v):
            for hash_val in v:
                if len(hash_val) != 64 or not all(c in '0123456789abcdef' for c in hash_val.lower()):
                    raise ValueError('Merkle path must contain valid SHA256 hashes')
            return v

class InferenceReceipt(BaseReceipt):
    """Enhanced inference receipt schema."""
    model_config = {'protected_namespaces': ()}  # Allow model_anchor field
    
    model_anchor: str = Field(..., min_length=64, max_length=64)
    input_commitment: Commitment
    output_commitment: Commitment  
    thresholds: Dict[str, float] = Field(default_factory=dict)
    decision: Dict[str, Any] = Field(default_factory=dict)
    oversight: OversightDecision = Field(default_factory=OversightDecision)
    merkle_path: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    fallback_reasons: List[Dict[str, Any]] = Field(default_factory=list)
    
    if PYDANTIC_AVAILABLE:
        @field_validator('thresholds')
        @classmethod
        def validate_thresholds(cls, v):
            for key, val in v.items():
                if not 0 <= val <= 1:
                    raise ValueError(f'Threshold {key} must be between 0 and 1')
            return v

class ReceiptValidator:
    """Validates receipts against schema and business rules."""
    
    def __init__(self):
        self.validation_errors = []
    
    def validate_training_receipt(self, receipt_data: Dict[str, Any]) -> bool:
        """Validate training receipt data."""
        if not PYDANTIC_AVAILABLE:
            return self._basic_validation(receipt_data)
        
        try:
            TrainingReceipt(**receipt_data)
            return True
        except PydanticValidationError as e:
            self.validation_errors.append(str(e))
            return False
    
    def validate_inference_receipt(self, receipt_data: Dict[str, Any]) -> bool:
        """Validate inference receipt data."""
        if not PYDANTIC_AVAILABLE:
            return self._basic_validation(receipt_data)
        
        try:
            InferenceReceipt(**receipt_data)
            
            # Additional business rule validation
            oversight = receipt_data.get('oversight', {})
            if oversight.get('status') == 'reviewed' and not oversight.get('rationale'):
                raise ValueError('Reviewed oversight decisions must include rationale')
            
            return True
        except (PydanticValidationError, ValueError) as e:
            self.validation_errors.append(str(e))
            return False
    
    def _basic_validation(self, receipt_data: Dict[str, Any]) -> bool:
        """Basic validation for environments without pydantic."""
        required_fields = ['receipt_id', 'evidence_strength', 'committed_at']
        
        for field in required_fields:
            if field not in receipt_data:
                self.validation_errors.append(f'Missing required field: {field}')
                return False
        
        return True
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.validation_errors.copy()

def create_training_receipt(
    dataset_anchor: str,
    model_anchor: str,
    code_digest: str,
    config_digest: str,
    random_seeds: Dict[str, int],
    env_info: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """Create a validated training receipt."""
    
    receipt_data = {
        "dataset_anchor": dataset_anchor,
        "model_anchor": model_anchor,
        "code_digest": code_digest,
        "config_digest": config_digest,
        "random_seeds": random_seeds,
        "env": env_info,
        **kwargs
    }
    
    if PYDANTIC_AVAILABLE:
        receipt = TrainingReceipt(**receipt_data)
        return receipt.dict()
    else:
        # Basic receipt without validation
        receipt_data.update({
            "receipt_id": str(uuid.uuid4()),
            "committed_at": datetime.utcnow().isoformat(),
            "evidence_strength": "real"
        })
        return receipt_data

def create_inference_receipt(
    model_anchor: str,
    input_commitment: Dict[str, Any],
    output_commitment: Dict[str, Any],
    decision: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """Create a validated inference receipt."""
    
    receipt_data = {
        "model_anchor": model_anchor,
        "input_commitment": input_commitment,
        "output_commitment": output_commitment,
        "decision": decision,
        **kwargs
    }
    
    if PYDANTIC_AVAILABLE:
        receipt = InferenceReceipt(**receipt_data)
        return receipt.dict()
    else:
        # Basic receipt without validation
        receipt_data.update({
            "receipt_id": str(uuid.uuid4()),
            "committed_at": datetime.utcnow().isoformat(),
            "evidence_strength": "real",
            "oversight": {"status": "auto_approved"}
        })
        return receipt_data

# Export main validator
receipt_validator = ReceiptValidator()