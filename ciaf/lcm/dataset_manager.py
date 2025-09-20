"""
CIAF LCM Dataset Manager

Enhanced dataset management supporting train/validation/test splits with proper anchoring.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from ..core import sha256_hash, MerkleTree, derive_dataset_anchor, derive_master_anchor, secure_random_bytes, SALT_LENGTH, to_hex
from .policy import LCMPolicy, get_default_policy, CommitmentType, DomainType


def compute_split_assignment_digest(record_ids: List[str], salt: Optional[bytes] = None) -> str:
    """
    Compute split assignment digest for audit reproducibility.
    
    Args:
        record_ids: List of stable record IDs or pre-hashed row digests
        salt: Optional salt for privacy-preserving commitment
        
    Returns:
        SHA-256 digest of the sorted record assignment
    """
    # Sort record IDs for deterministic output
    items = sorted(record_ids)
    
    if salt:
        # Privacy-preserving variant: hash each record ID with salt
        items = [sha256_hash(salt + rid.encode("utf-8")) for rid in items]
    
    # Option 1: Flat digest (simple approach)
    concat = "".join(items).encode("utf-8")
    return sha256_hash(concat)
    
    # Option 2: Merkle root (stronger proof, uncomment if preferred)
    # return MerkleTree(items).get_root()


class DatasetSplit(Enum):
    """Dataset split types."""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
    FULL = "full"


@dataclass
class DatasetMetadata:
    """Enhanced dataset metadata for LCM."""
    name: str
    owner: str
    license: str
    schema_digest: str
    sampling_rules: Dict[str, Any]
    version: str
    content_root: str  # Merkle root or rolling hash
    
    # Privacy and compliance
    contains_pii: bool = False
    privacy_level: str = "public"
    compliance_frameworks: List[str] = None
    
    # Additional metadata
    creation_date: str = None
    description: str = ""
    tags: List[str] = None
    
    # RNG reproducibility
    rng_seed: Optional[int] = None
    rng_source: Optional[str] = None  # e.g., "numpy", "torch", "random"
    
    # Stratification
    stratify_by: Optional[List[str]] = None  # columns used for stratification
    
    # Split assignment commitment
    split_assignment_digest: Optional[str] = None  # SHA-256 over sorted record IDs or Merkle root
    
    def __post_init__(self):
        """Initialize default values."""
        if self.compliance_frameworks is None:
            self.compliance_frameworks = []
        if self.tags is None:
            self.tags = []
        if self.creation_date is None:
            self.creation_date = datetime.now().isoformat()


class LCMDatasetAnchor:
    """Enhanced dataset anchor for LCM with split support."""
    
    def __init__(
        self,
        dataset_id: str,
        split: DatasetSplit,
        metadata: DatasetMetadata,
        master_password: str,
        policy: LCMPolicy = None,
        salt: bytes = None
    ):
        """
        Initialize LCM dataset anchor.
        
        Args:
            dataset_id: Unique identifier for the dataset
            split: Dataset split type (train/val/test)
            metadata: Dataset metadata
            master_password: Master password for anchor derivation
            policy: LCM policy (uses default if None)
            salt: Salt for anchor derivation (generates if None)
        """
        self.dataset_id = dataset_id
        self.split = split
        self.metadata = metadata
        self.policy = policy or get_default_policy()
        
        # Generate or use provided salt
        if salt is not None:
            self.master_salt = salt
        else:
            self.master_salt = secure_random_bytes(SALT_LENGTH)
        
        # Derive anchors
        self.master_anchor = derive_master_anchor(master_password, self.master_salt)
        
        # Compute dataset hash including split information
        self.dataset_hash = self._compute_dataset_hash()
        self.dataset_anchor = derive_dataset_anchor(self.master_anchor, self.dataset_hash)
        
        # Sample tracking
        self.sample_hashes: List[str] = []
        self.total_samples = 0
        
        # Generate anchor ID with split prefix
        split_prefix = {
            DatasetSplit.TRAIN: "t_",
            DatasetSplit.VALIDATION: "v_", 
            DatasetSplit.TEST: "x_",
            DatasetSplit.FULL: "f_"
        }[split]
        
        self.anchor_id = f"{split_prefix}{to_hex(self.dataset_anchor)[:8]}..."
        
        print(f"LCM Dataset Anchor '{self.dataset_id}' ({self.split.value}) initialized with anchor: {self.anchor_id}")
    
    def _compute_dataset_hash(self) -> str:
        """Compute canonical hash of dataset metadata including split and RNG info."""
        # Create canonical representation including RNG reproducibility fields
        hash_data = {
            "dataset_id": self.dataset_id,
            "split": self.split.value,
            "name": self.metadata.name,
            "owner": self.metadata.owner,
            "license": self.metadata.license,
            "schema_digest": self.metadata.schema_digest,
            "sampling_rules": self.metadata.sampling_rules,
            "version": self.metadata.version,
            "content_root": self.metadata.content_root,
            "contains_pii": self.metadata.contains_pii,
            "privacy_level": self.metadata.privacy_level,
            "compliance_frameworks": sorted(self.metadata.compliance_frameworks),
            "tags": sorted(self.metadata.tags),
            "rng_seed": self.metadata.rng_seed,
            "rng_source": self.metadata.rng_source,
            "stratify_by": self.metadata.stratify_by,
            "split_assignment_digest": self.metadata.split_assignment_digest
        }
        
        canonical_json = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_json.encode('utf-8'))
    
    def add_sample_hash(self, sample_hash: str) -> None:
        """Add a sample hash to the dataset."""
        if sample_hash not in self.sample_hashes:
            self.sample_hashes.append(sample_hash)
            self.total_samples = len(self.sample_hashes)
    
    def get_merkle_root(self) -> Optional[str]:
        """Get Merkle root of all sample hashes."""
        if not self.sample_hashes:
            return None
        
        merkle_tree = MerkleTree(self.sample_hashes)
        return merkle_tree.get_root()
    
    def create_commitment(self, data: Any, commitment_type: CommitmentType = None) -> str:
        """Create commitment for data according to policy."""
        commitment_type = commitment_type or self.policy.commitments
        
        if commitment_type == CommitmentType.PLAINTEXT:
            return str(data)
        elif commitment_type == CommitmentType.SALTED:
            # Simple salt-based commitment (for demo purposes)
            salt = secure_random_bytes(16)
            data_str = json.dumps(data, sort_keys=True) if not isinstance(data, str) else data
            return sha256_hash((salt + data_str.encode('utf-8')))[:16] + "..."
        elif commitment_type == CommitmentType.HMAC_SHA256:
            # HMAC-based commitment using dataset anchor
            import hmac
            data_str = json.dumps(data, sort_keys=True) if not isinstance(data, str) else data
            return hmac.new(self.dataset_anchor, data_str.encode('utf-8'), 'sha256').hexdigest()[:16] + "..."
        else:
            raise ValueError(f"Unknown commitment type: {commitment_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": f"{self.metadata.name}_{self.split.value}",
            "anchor": self.anchor_id,
            "commitment": self.policy.commitments.value,
            "split": self.split.value,
            "dataset_hash": self.dataset_hash,
            "total_samples": self.total_samples,
            "merkle_root": self.get_merkle_root(),
            "metadata": {
                "owner": self.metadata.owner,
                "version": self.metadata.version,
                "contains_pii": self.metadata.contains_pii,
                "privacy_level": self.metadata.privacy_level,
                "compliance_frameworks": self.metadata.compliance_frameworks
            }
        }


class LCMDatasetManager:
    """
    Enhanced dataset manager supporting train/validation/test splits.
    """
    
    def __init__(self, policy: LCMPolicy = None):
        """Initialize LCM dataset manager."""
        self.policy = policy or get_default_policy()
        self.dataset_anchors: Dict[str, Dict[DatasetSplit, LCMDatasetAnchor]] = {}
    
    def create_dataset_splits(
        self,
        dataset_id: str,
        metadata: DatasetMetadata,
        master_password: str,
        splits: List[DatasetSplit] = None
    ) -> Dict[DatasetSplit, LCMDatasetAnchor]:
        """
        Create dataset anchors for train/validation/test splits.
        
        Args:
            dataset_id: Unique identifier for the dataset
            metadata: Dataset metadata
            master_password: Master password for anchor derivation
            splits: List of splits to create (defaults to train/val/test)
            
        Returns:
            Dictionary mapping splits to their anchors
        """
        if splits is None:
            splits = [DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST]
        
        print(f"🗃️ Creating dataset anchors for {dataset_id} with splits: {[s.value for s in splits]}")
        
        dataset_splits = {}
        
        base_seed = 42  # Default seed for reproducible splits
        
        for split in splits:
            # Generate deterministic seed for this split
            split_seed = base_seed + hash(split.value) % 10000
            
            # Generate mock record IDs for this split (simulate actual data assignment)
            num_samples = {"train": 1000, "val": 200, "test": 200}.get(split.value, 100)
            record_ids = [f"{dataset_id}_{split.value}_record_{i}" for i in range(num_samples)]
            
            # Compute split assignment digest
            assignment_digest = compute_split_assignment_digest(record_ids)
            
            # Create split-specific metadata with RNG info
            split_metadata = DatasetMetadata(
                name=f"{metadata.name}_{split.value}",
                owner=metadata.owner,
                license=metadata.license,
                schema_digest=metadata.schema_digest,
                sampling_rules=metadata.sampling_rules,
                version=metadata.version,
                content_root=f"{metadata.content_root}_{split.value}",
                contains_pii=metadata.contains_pii,
                privacy_level=metadata.privacy_level,
                compliance_frameworks=metadata.compliance_frameworks,
                description=f"{metadata.description} ({split.value} split)",
                tags=metadata.tags + [f"split:{split.value}"],
                rng_seed=split_seed,
                rng_source="random",  # Using Python's random module
                stratify_by=metadata.stratify_by,
                split_assignment_digest=assignment_digest
            )
            
            # Create anchor for this split
            anchor = LCMDatasetAnchor(
                dataset_id=f"{dataset_id}_{split.value}",
                split=split,
                metadata=split_metadata,
                master_password=master_password,
                policy=self.policy
            )
            
            dataset_splits[split] = anchor
        
        self.dataset_anchors[dataset_id] = dataset_splits
        
        print(f"✅ Created {len(splits)} dataset anchors for {dataset_id}")
        return dataset_splits
    
    def get_datasets_root_anchor(self, dataset_id: str) -> str:
        """
        Compute datasets root anchor from all splits.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Merkle root of all split anchors
        """
        if dataset_id not in self.dataset_anchors:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        splits = self.dataset_anchors[dataset_id]
        
        # Collect all split anchor hashes
        split_hashes = []
        for split in [DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST]:
            if split in splits:
                split_hashes.append(splits[split].dataset_hash)
        
        if not split_hashes:
            raise ValueError(f"No splits found for dataset {dataset_id}")
        
        # Compute Merkle root
        merkle_tree = MerkleTree(split_hashes)
        return merkle_tree.get_root()
    
    def compute_enhanced_split_map_digest(self, dataset_id: str) -> str:
        """
        Compute enhanced split map digest including assignment digests.
        
        This provides stronger audit guarantees by including actual membership
        information, not just that splits existed.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Enhanced split map digest
        """
        if dataset_id not in self.dataset_anchors:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        splits = self.dataset_anchors[dataset_id]
        
        split_map = {}
        for split, anchor in splits.items():
            split_map[split.value] = {
                "anchor": anchor.dataset_hash,  # existing: proves split exists
                "assignment": anchor.metadata.split_assignment_digest  # new: proves membership
            }
        
        # Compute digest: H(canon(split_map))
        canonical_map = json.dumps(split_map, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_map.encode('utf-8'))
    
    def get_dataset_anchor(self, dataset_id: str, split: DatasetSplit) -> Optional[LCMDatasetAnchor]:
        """Get dataset anchor for specific split."""
        if dataset_id in self.dataset_anchors and split in self.dataset_anchors[dataset_id]:
            return self.dataset_anchors[dataset_id][split]
        return None
    
    def get_all_splits(self, dataset_id: str) -> Dict[DatasetSplit, LCMDatasetAnchor]:
        """Get all splits for a dataset."""
        return self.dataset_anchors.get(dataset_id, {})
    
    def add_samples_to_split(
        self,
        dataset_id: str,
        split: DatasetSplit,
        samples: List[Dict[str, Any]]
    ) -> None:
        """Add samples to a specific dataset split."""
        anchor = self.get_dataset_anchor(dataset_id, split)
        if not anchor:
            raise ValueError(f"Dataset anchor not found: {dataset_id}_{split.value}")
        
        for sample in samples:
            # Create sample hash
            sample_data = json.dumps(sample, sort_keys=True)
            sample_hash = sha256_hash(sample_data.encode('utf-8'))
            anchor.add_sample_hash(sample_hash)
        
        print(f"✅ Added {len(samples)} samples to {dataset_id}_{split.value}")
    
    def format_datasets_summary(self, dataset_id: str) -> str:
        """Format dataset splits summary for pretty printing."""
        if dataset_id not in self.dataset_anchors:
            return f"Dataset {dataset_id} not found"
        
        splits = self.dataset_anchors[dataset_id]
        lines = []
        
        for split in [DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST]:
            if split in splits:
                anchor = splits[split]
                commitment = anchor.create_commitment(anchor.metadata.name)
                lines.append(f"  ✅ {split.value:<5}: {anchor.metadata.name:<20} (anchor= {anchor.anchor_id}, commitment={commitment})")
        
        datasets_root = self.get_datasets_root_anchor(dataset_id)
        lines.append(f"  🌳 datasets_root_anchor: {datasets_root[:4]}...{datasets_root[-4:]}")
        
        return "\n".join(lines)
    
    def simulate_dataset_anchor(
        self,
        dataset_id: str,
        dataset_path: str,
        split_type: str = "train"
    ) -> LCMDatasetAnchor:
        """
        Simulate dataset anchor creation for demonstration.
        
        Args:
            dataset_id: Dataset identifier
            dataset_path: Path to dataset file
            split_type: Type of split (train, validation, test)
            
        Returns:
            LCMDatasetAnchor instance
        """
        print(f"📊 Creating dataset anchor: {dataset_id} ({split_type})")
        
        # Convert string to enum
        if split_type == "train":
            split = DatasetSplit.TRAIN
        elif split_type == "validation":
            split = DatasetSplit.VALIDATION
        elif split_type == "test":
            split = DatasetSplit.TEST
        else:
            split = DatasetSplit.TRAIN
        
        # Create mock dataset metadata
        metadata = DatasetMetadata(
            name=dataset_id,
            owner="ciaf_user",
            license="Example License",
            schema_digest="",  # Will be computed
            sampling_rules={},
            version="1.0.0",
            content_root="",  # Will be computed
            description=f"Mock dataset for {split_type}"
        )
        
        # Create dataset splits if not exists
        if dataset_id not in self.dataset_anchors:
            self.create_dataset_splits(
                dataset_id=dataset_id,
                metadata=metadata,
                master_password="demo_password"
            )
        
        # Return the requested split
        anchor = self.get_dataset_anchor(dataset_id, split)
        print(f"✅ Dataset anchor created: {anchor.anchor_id}")
        return anchor
