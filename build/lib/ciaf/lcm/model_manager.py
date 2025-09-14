"""
CIAF LCM Model Manager

Enhanced model management with comprehensive metadata including params_root, 
arch_root, hp_digest, env_digest, trainer_commit, and authorized datasets.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import platform
import sys
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..core import sha256_hash, derive_model_anchor, derive_master_anchor, secure_random_bytes, SALT_LENGTH, to_hex
from .policy import LCMPolicy, get_default_policy, CommitmentType, DomainType
from .dataset_manager import LCMDatasetAnchor, DatasetSplit


@dataclass
class ModelArchitecture:
    """Model architecture definition."""
    type: str  # e.g., "MLP", "Transformer", "CNN"
    layers: List[Dict[str, Any]]
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    total_params: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "layers": self.layers,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "total_params": self.total_params
        }


@dataclass
class TrainingEnvironment:
    """Training environment metadata."""
    python_version: str
    framework: str
    framework_version: str
    cuda_version: Optional[str] = None
    os_info: str = ""
    hardware: str = ""
    dependencies: Dict[str, str] = None
    
    def __post_init__(self):
        """Auto-populate environment info."""
        if not self.python_version:
            self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if not self.os_info:
            self.os_info = f"{platform.system()} {platform.release()}"
        if self.dependencies is None:
            self.dependencies = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "python_version": self.python_version,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "cuda_version": self.cuda_version,
            "os_info": self.os_info,
            "hardware": self.hardware,
            "dependencies": self.dependencies
        }


class LCMModelAnchor:
    """Enhanced model anchor for LCM with comprehensive metadata."""
    
    def __init__(
        self,
        model_name: str,
        version: str,
        architecture: ModelArchitecture,
        hyperparameters: Dict[str, Any],
        environment: TrainingEnvironment,
        authorized_datasets: List[str],
        trainer_commit: str = "unknown",
        master_password: str = None,
        policy: LCMPolicy = None,
        salt: bytes = None
    ):
        """
        Initialize LCM model anchor.
        
        Args:
            model_name: Name of the model
            version: Model version
            architecture: Model architecture definition
            hyperparameters: Training hyperparameters
            environment: Training environment metadata
            authorized_datasets: List of authorized dataset IDs
            trainer_commit: Git commit hash of training code
            master_password: Master password for anchor derivation
            policy: LCM policy (uses default if None)
            salt: Salt for anchor derivation (generates if None)
        """
        self.model_name = model_name
        self.version = version
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        self.environment = environment
        self.authorized_datasets = authorized_datasets
        self.trainer_commit = trainer_commit
        self.policy = policy or get_default_policy()
        
        # Generate or use provided salt and password
        password = master_password or model_name
        if salt is not None:
            self.master_salt = salt
        else:
            self.master_salt = secure_random_bytes(SALT_LENGTH)
        
        # Derive anchors
        self.master_anchor = derive_master_anchor(password, self.master_salt)
        
        # Compute various digests
        self.params_root = self._compute_params_root()
        self.arch_root = self._compute_arch_root()
        self.hp_digest = self._compute_hp_digest()
        self.env_digest = self._compute_env_digest()
        
        # Compute model hash and derive model anchor
        self.model_hash = self._compute_model_hash()
        self.model_anchor = derive_model_anchor(self.master_anchor, self.model_hash)
        
        # Generate anchor ID
        self.anchor_id = f"m_{to_hex(self.model_anchor)[:8]}..."
        
        print(f"LCM Model Anchor '{self.model_name}' v{self.version} initialized with anchor: {self.anchor_id}")
    
    def _compute_params_root(self) -> str:
        """Compute parameters root hash (layer-wise)."""
        # For demonstration, compute hash of all parameters
        # In real implementation, this would be layer-wise parameter hashes
        params_data = {
            "hyperparameters": self.hyperparameters,
            "total_params": self.architecture.total_params,
            "architecture_type": self.architecture.type
        }
        canonical_json = json.dumps(params_data, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_json.encode('utf-8'))
    
    def _compute_arch_root(self) -> str:
        """Compute architecture root hash."""
        arch_data = self.architecture.to_dict()
        canonical_json = json.dumps(arch_data, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_json.encode('utf-8'))
    
    def _compute_hp_digest(self) -> str:
        """Compute hyperparameters digest."""
        canonical_json = json.dumps(self.hyperparameters, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_json.encode('utf-8'))
    
    def _compute_env_digest(self) -> str:
        """Compute environment digest."""
        env_data = self.environment.to_dict()
        canonical_json = json.dumps(env_data, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_json.encode('utf-8'))
    
    def _compute_model_hash(self) -> str:
        """Compute comprehensive model hash."""
        model_data = {
            "model_name": self.model_name,
            "version": self.version,
            "params_root": self.params_root,
            "arch_root": self.arch_root,
            "hp_digest": self.hp_digest,
            "env_digest": self.env_digest,
            "trainer_commit": self.trainer_commit,
            "authorized_datasets": sorted(self.authorized_datasets)
        }
        canonical_json = json.dumps(model_data, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_json.encode('utf-8'))
    
    def create_commitment(self, data: Any, commitment_type: CommitmentType = None) -> str:
        """Create commitment for data according to policy."""
        commitment_type = commitment_type or self.policy.commitments
        
        if commitment_type == CommitmentType.PLAINTEXT:
            return str(data)
        elif commitment_type == CommitmentType.SALTED:
            # Simple salt-based commitment
            salt = secure_random_bytes(16)
            data_str = json.dumps(data, sort_keys=True) if not isinstance(data, str) else data
            return sha256_hash((salt + data_str.encode('utf-8')))[:16] + "..."
        elif commitment_type == CommitmentType.HMAC_SHA256:
            # HMAC-based commitment using model anchor
            import hmac
            data_str = json.dumps(data, sort_keys=True) if not isinstance(data, str) else data
            return hmac.new(self.model_anchor, data_str.encode('utf-8'), 'sha256').hexdigest()[:16] + "..."
        else:
            raise ValueError(f"Unknown commitment type: {commitment_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.model_name,
            "version": self.version,
            "params_root": self.params_root[:8] + "...",
            "arch_root": self.arch_root[:8] + "...",
            "hp_digest": self.hp_digest[:8] + "...",
            "env_digest": self.env_digest[:8] + "...",
            "trainer_commit": self.trainer_commit,
            "authorized_dataset_refs": [f"ref_{ds}" for ds in self.authorized_datasets],
            "anchor": self.anchor_id,
            "commitment": self.policy.commitments.value,
            "model_hash": self.model_hash
        }


class LCMModelManager:
    """Enhanced model manager for LCM."""
    
    def __init__(self, policy: LCMPolicy = None):
        """Initialize LCM model manager."""
        self.policy = policy or get_default_policy()
        self.model_anchors: Dict[str, LCMModelAnchor] = {}
    
    def create_model_anchor(
        self,
        model_name: str,
        version: str,
        architecture: ModelArchitecture,
        hyperparameters: Dict[str, Any],
        environment: TrainingEnvironment = None,
        authorized_datasets: List[str] = None,
        trainer_commit: str = None,
        master_password: str = None
    ) -> LCMModelAnchor:
        """
        Create enhanced model anchor with comprehensive metadata.
        
        Args:
            model_name: Name of the model
            version: Model version
            architecture: Model architecture definition
            hyperparameters: Training hyperparameters
            environment: Training environment metadata
            authorized_datasets: List of authorized dataset IDs
            trainer_commit: Git commit hash of training code
            master_password: Master password for anchor derivation
            
        Returns:
            LCMModelAnchor instance
        """
        print(f"🎯 Creating enhanced LCM model anchor for: {model_name} v{version}")
        
        # Auto-detect environment if not provided
        if environment is None:
            environment = TrainingEnvironment(
                python_version="",  # Will be auto-populated
                framework="CIAF-Simulator",
                framework_version="1.0.0",
                os_info="",  # Will be auto-populated
                hardware="CPU"
            )
        
        # Auto-detect git commit if not provided
        if trainer_commit is None:
            trainer_commit = self._get_git_commit()
        
        # Default authorized datasets
        if authorized_datasets is None:
            authorized_datasets = []
        
        # Create model anchor
        anchor = LCMModelAnchor(
            model_name=model_name,
            version=version,
            architecture=architecture,
            hyperparameters=hyperparameters,
            environment=environment,
            authorized_datasets=authorized_datasets,
            trainer_commit=trainer_commit,
            master_password=master_password,
            policy=self.policy
        )
        
        # Store anchor
        model_key = f"{model_name}_{version}"
        self.model_anchors[model_key] = anchor
        
        print(f"✅ Model anchor created:")
        print(f"   🎯 Model: {model_name} v{version}")
        print(f"   🔐 Params root: {anchor.params_root[:16]}...")
        print(f"   🏗️ Arch root: {anchor.arch_root[:16]}...")
        print(f"   📊 HP digest: {anchor.hp_digest[:16]}...")
        print(f"   💻 Env digest: {anchor.env_digest[:16]}...")
        print(f"   🔗 Trainer commit: {trainer_commit}")
        print(f"   📋 Authorized datasets: {len(authorized_datasets)}")
        
        return anchor
    
    def _get_git_commit(self) -> str:
        """Attempt to get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                cwd='.',
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Fallback to timestamp-based commit
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"auto_{timestamp}"
    
    def get_model_anchor(self, model_name: str, version: str = None) -> Optional[LCMModelAnchor]:
        """Get model anchor by name and optional version."""
        if version:
            model_key = f"{model_name}_{version}"
            return self.model_anchors.get(model_key)
        
        # Find latest version if no version specified
        matching_keys = [k for k in self.model_anchors.keys() if k.startswith(f"{model_name}_")]
        if matching_keys:
            # Return the last one (assuming lexicographic ordering corresponds to version ordering)
            latest_key = sorted(matching_keys)[-1]
            return self.model_anchors[latest_key]
        
        return None
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.model_anchors.keys())
    
    def format_model_summary(self, model_name: str, version: str) -> str:
        """Format model summary for pretty printing."""
        anchor = self.get_model_anchor(model_name, version)
        if not anchor:
            return f"Model {model_name} v{version} not found"
        
        lines = [
            f"  model: {anchor.model_name}  version: {anchor.version}",
            f"  params_root: {anchor.params_root[:8]}...  arch_root: {anchor.arch_root[:8]}...",
            f"  hp_digest: {anchor.hp_digest[:8]}...    env_digest: {anchor.env_digest[:8]}...   trainer_commit: {anchor.trainer_commit}",
            f"  🔗 authorized_datasets: {anchor.authorized_datasets}",
            f"  ✅ model_anchor: {anchor.anchor_id}  (commitment={anchor.policy.commitments.value})"
        ]
        
        return "\n".join(lines)
    
    def simulate_model_anchor(
        self,
        model_id: str,
        model_params: Dict[str, Any],
        model_name: str = None
    ) -> LCMModelAnchor:
        """
        Simulate model anchor creation for demonstration.
        
        Args:
            model_id: Model identifier
            model_params: Model parameters dictionary
            model_name: Optional model name
            
        Returns:
            LCMModelAnchor instance
        """
        print(f"🤖 Creating model anchor: {model_id}")
        
        model_name = model_name or model_id
        
        # Create mock training environment
        training_env = TrainingEnvironment(
            python_version="3.8.10",
            framework="pytorch",
            framework_version="1.9.0",
            cuda_version="11.2",
            hardware="Tesla V100"
        )
        
        # Create mock architecture
        model_arch = ModelArchitecture(
            type="feedforward",
            layers=[{"type": "dense", "units": 64}, {"type": "dense", "units": 32}],
            input_dim=10,
            output_dim=1,
            total_params=sum(
                len(layer.get('weights', [])) + len(layer.get('bias', [])) 
                for layer in model_params.values() 
                if isinstance(layer, dict)
            )
        )
        
        # Create model anchor
        anchor = self.create_model_anchor(
            model_name=model_name,
            version="1.0.0",
            architecture=model_arch,
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
            environment=training_env,
            authorized_datasets=[f"{model_id}_dataset@v1"],
            master_password="demo_password"
        )
        
        print(f"✅ Model anchor created: {anchor.anchor_id}")
        print(f"   🎯 Model: {anchor.model_name} v{anchor.version}")
        print(f"   🔐 Params root: {anchor.params_root[:16]}...")
        print(f"   🏗️ Arch root: {anchor.arch_root[:16]}...")
        print(f"   📊 HP digest: {anchor.hp_digest[:16]}...")
        print(f"   💻 Env digest: {anchor.env_digest[:16]}...")
        print(f"   🔗 Trainer commit: {anchor.trainer_commit}")
        print(f"   📋 Authorized dataset family: {model_id}_dataset@v1 (splits: train, val, test)")
        
        return anchor
