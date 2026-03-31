"""
CIAF LCM Dataset Family and Split Management

Enhanced dataset management that properly represents one dataset with multiple splits
rather than treating splits as separate datasets.

Created: 2025-09-09
Last Modified: 2026-03-30
Author: Denzil James Greenwood
Version: 2.0.0 - Converted to Pydantic models
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from ..core import sha256_hash, MerkleTree
from .policy import LCMPolicy, get_default_policy


def compute_split_assignment_digest(
    record_ids: List[str], salt: Optional[bytes] = None
) -> str:
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


class DatasetFamilyMetadata(BaseModel):
    """Metadata for the complete dataset family."""

    name: str = Field(..., min_length=1, description="Dataset family name")
    version: str = Field(..., min_length=1, description="Dataset version")
    owner: str = Field(..., min_length=1, description="Dataset owner")
    license: str = Field(..., min_length=1, description="Dataset license")
    description: str = Field("", description="Dataset description")
    creation_date: Optional[str] = Field(None, description="Creation timestamp")
    tags: List[str] = Field(default_factory=list, description="Dataset tags")

    # Privacy and compliance
    contains_pii: bool = Field(False, description="Contains PII")
    privacy_level: str = Field("public", description="Privacy level")
    compliance_frameworks: List[str] = Field(
        default_factory=list, description="Compliance frameworks"
    )

    @model_validator(mode="after")
    def set_creation_date(self) -> "DatasetFamilyMetadata":
        """Set creation date if not provided."""
        if self.creation_date is None:
            self.creation_date = datetime.now().isoformat()
        return self


class SplitMetadata(BaseModel):
    """Metadata for a specific dataset split."""

    dataset_id: str = Field(..., min_length=1, description="Dataset identifier")
    split_name: str = Field(..., min_length=1, description="Split name")
    split_selection_rules: Dict[str, Any] = Field(
        default_factory=dict, description="Split selection rules"
    )
    split_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Split statistics"
    )
    sample_count: int = Field(0, ge=0, description="Number of samples")

    # RNG reproducibility
    rng_seed: Optional[int] = Field(None, description="RNG seed")
    rng_source: Optional[str] = Field(
        None, description="RNG source"
    )  # e.g., "numpy", "torch", "random"

    # Stratification
    stratify_by: Optional[List[str]] = Field(None, description="Stratification columns")

    # Split assignment commitment
    split_assignment_digest: Optional[str] = Field(
        None, description="SHA-256 over sorted record IDs or Merkle root"
    )


class LCMDatasetFamilyAnchor:
    """Dataset family anchor representing the complete dataset."""

    def __init__(
        self,
        family_metadata: DatasetFamilyMetadata,
        dataset_content_root: str,
        policy: LCMPolicy = None,
    ):
        """Initialize dataset family anchor."""
        self.family_metadata = family_metadata
        self.dataset_content_root = dataset_content_root
        self.policy = policy or get_default_policy()

        # Compute family anchor
        self.family_anchor = self._compute_family_anchor()

        print(
            f"🗃️ Dataset Family '{self.family_metadata.name}' v{self.family_metadata.version} initialized"
        )
        print(f"   📦 dataset_family_anchor: {self.anchor_id}")

    def _compute_family_anchor(self) -> str:
        """Compute dataset family anchor."""
        # Create canonical metadata
        meta_dict = {
            "name": self.family_metadata.name,
            "version": self.family_metadata.version,
            "owner": self.family_metadata.owner,
            "license": self.family_metadata.license,
            "description": self.family_metadata.description,
            "contains_pii": self.family_metadata.contains_pii,
            "privacy_level": self.family_metadata.privacy_level,
        }

        # Canonicalize metadata
        canonical_meta = json.dumps(meta_dict, sort_keys=True, separators=(",", ":"))

        # Compute family anchor: H("CIAF|dataset|family" || canon(meta) || content_root)
        domain = "CIAF|dataset|family"
        payload = (
            domain.encode("utf-8")
            + canonical_meta.encode("utf-8")
            + self.dataset_content_root.encode("utf-8")
        )

        return sha256_hash(payload)

    @property
    def anchor_id(self) -> str:
        """Get anchor ID."""
        return f"d_f_{self.family_anchor[:8]}..."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.family_metadata.name,
            "version": self.family_metadata.version,
            "family_anchor": self.anchor_id,
            "content_root": self.dataset_content_root[:8] + "...",
            "owner": self.family_metadata.owner,
            "license": self.family_metadata.license,
        }


class LCMDatasetSplitAnchor:
    """Dataset split anchor for a specific split of the family."""

    def __init__(
        self,
        split_metadata: SplitMetadata,
        split_content_root: str,
        policy: LCMPolicy = None,
    ):
        """Initialize dataset split anchor."""
        self.split_metadata = split_metadata
        self.split_content_root = split_content_root
        self.policy = policy or get_default_policy()
        self.sample_hashes: List[str] = []

        # Compute split anchor
        self.split_anchor = self._compute_split_anchor()

        print(
            f"   🗂️ Split '{self.split_metadata.split_name}' initialized: {self.anchor_id}"
        )

    def _compute_split_anchor(self) -> str:
        """Compute dataset split anchor."""
        # Create canonical split metadata including RNG and assignment info
        meta_dict = {
            "dataset_id": self.split_metadata.dataset_id,
            "split": self.split_metadata.split_name,
            "rules": self.split_metadata.split_selection_rules,
            "stats": self.split_metadata.split_stats,
            "rng_seed": self.split_metadata.rng_seed,
            "rng_source": self.split_metadata.rng_source,
            "stratify_by": self.split_metadata.stratify_by,
            "split_assignment_digest": self.split_metadata.split_assignment_digest,
        }

        # Canonicalize metadata
        canonical_meta = json.dumps(meta_dict, sort_keys=True, separators=(",", ":"))

        # Compute split anchor: H("CIAF|dataset|split" || canon(meta) || split_content_root)
        domain = "CIAF|dataset|split"
        payload = (
            domain.encode("utf-8")
            + canonical_meta.encode("utf-8")
            + self.split_content_root.encode("utf-8")
        )

        return sha256_hash(payload)

    @property
    def anchor_id(self) -> str:
        """Get anchor ID."""
        return f"d_s_{self.split_metadata.split_name}_{self.split_anchor[:8]}..."

    def add_sample_hash(self, sample_hash: str) -> None:
        """Add sample hash to this split."""
        self.sample_hashes.append(sample_hash)
        self.split_metadata.sample_count = len(self.sample_hashes)

    def get_merkle_root(self) -> Optional[str]:
        """Get Merkle root of all sample hashes in this split."""
        if not self.sample_hashes:
            return None

        merkle_tree = MerkleTree(self.sample_hashes)
        return merkle_tree.get_root()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.split_metadata.dataset_id,
            "split": self.split_metadata.split_name,
            "anchor": self.anchor_id,
            "sample_count": self.split_metadata.sample_count,
            "merkle_root": self.get_merkle_root(),
        }


class LCMDatasetFamilyManager:
    """Enhanced dataset family manager for proper dataset/split representation."""

    def __init__(self, policy: LCMPolicy = None):
        """Initialize dataset family manager."""
        self.policy = policy or get_default_policy()
        self.dataset_families: Dict[str, LCMDatasetFamilyAnchor] = {}
        self.split_anchors: Dict[str, Dict[DatasetSplit, LCMDatasetSplitAnchor]] = {}

    def create_dataset_family(
        self,
        dataset_id: str,
        family_metadata: DatasetFamilyMetadata,
        split_configs: Dict[DatasetSplit, Dict[str, Any]] = None,
    ) -> LCMDatasetFamilyAnchor:
        """
        Create dataset family with splits.

        Args:
            dataset_id: Dataset identifier
            family_metadata: Family metadata
            split_configs: Configuration for each split

        Returns:
            LCMDatasetFamilyAnchor instance
        """
        print(f"🗃️ Creating dataset family: {dataset_id}")

        # Default split configurations
        if split_configs is None:
            split_configs = {
                DatasetSplit.TRAIN: {"ratio": 0.7, "selection": "random"},
                DatasetSplit.VALIDATION: {"ratio": 0.15, "selection": "random"},
                DatasetSplit.TEST: {"ratio": 0.15, "selection": "random"},
            }

        # Simulate dataset content root (in practice, this would be computed from actual data)
        mock_content = f"{dataset_id}_{family_metadata.version}_content"
        dataset_content_root = sha256_hash(mock_content.encode("utf-8"))

        # Create family anchor
        family_anchor = LCMDatasetFamilyAnchor(
            family_metadata=family_metadata,
            dataset_content_root=dataset_content_root,
            policy=self.policy,
        )

        # Store family anchor
        self.dataset_families[dataset_id] = family_anchor

        # Create split anchors with RNG reproducibility
        split_anchors = {}
        base_seed = 42  # Default seed for reproducible splits

        for split, config in split_configs.items():
            # Generate deterministic seed for this split
            split_seed = base_seed + hash(split.value) % 10000

            # Generate mock record IDs for this split (simulate actual data assignment)
            if split == DatasetSplit.TRAIN:
                record_ids = [f"{dataset_id}_record_{i}" for i in range(100)]
            elif split == DatasetSplit.VALIDATION:
                record_ids = [
                    f"{dataset_id}_record_{i}" for i in range(100, 120)
                ]  # Different range
            else:  # TEST
                record_ids = [
                    f"{dataset_id}_record_{i}" for i in range(120, 140)
                ]  # Different range

            # Compute split assignment digest
            assignment_digest = compute_split_assignment_digest(record_ids)

            split_metadata = SplitMetadata(
                dataset_id=dataset_id,
                split_name=split.value,
                split_selection_rules=config,
                split_stats={"ratio": config.get("ratio", 0.0)},
                sample_count=0,
                rng_seed=split_seed,
                rng_source="random",  # Using Python's random module
                stratify_by=config.get("stratify_by"),
                split_assignment_digest=assignment_digest,
            )

            # Simulate split content root
            split_content = f"{dataset_id}_{split.value}_content"
            split_content_root = sha256_hash(split_content.encode("utf-8"))

            split_anchor = LCMDatasetSplitAnchor(
                split_metadata=split_metadata,
                split_content_root=split_content_root,
                policy=self.policy,
            )

            split_anchors[split] = split_anchor

            # Add mock samples with their hashes
            for record_id in record_ids:
                sample_hash = sha256_hash(f"{record_id}_data".encode("utf-8"))
                split_anchor.add_sample_hash(sample_hash)

        # Store split anchors
        self.split_anchors[dataset_id] = split_anchors

        # Print summary
        splits_list = [split.value for split in split_anchors.keys()]
        print(f"   🗂️ Splits created: {splits_list}")
        for split, anchor in split_anchors.items():
            print(f"      ▸ split_anchor({split.value}): {anchor.anchor_id}")

        print(f"   ✅ Dataset family created: {family_anchor.anchor_id}")

        return family_anchor

    def get_dataset_family(self, dataset_id: str) -> Optional[LCMDatasetFamilyAnchor]:
        """Get dataset family by ID."""
        return self.dataset_families.get(dataset_id)

    def get_split_anchor(
        self, dataset_id: str, split: DatasetSplit
    ) -> Optional[LCMDatasetSplitAnchor]:
        """Get split anchor by dataset ID and split."""
        if dataset_id not in self.split_anchors:
            return None
        return self.split_anchors[dataset_id].get(split)

    def get_all_splits(
        self, dataset_id: str
    ) -> Dict[DatasetSplit, LCMDatasetSplitAnchor]:
        """Get all split anchors for a dataset."""
        return self.split_anchors.get(dataset_id, {})

    def compute_split_map_digest(self, dataset_id: str) -> str:
        """
        Compute enhanced split map digest for training snapshot.

        This includes both split anchor hashes and assignment digests for stronger
        audit guarantees.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Split map digest
        """
        splits = self.get_all_splits(dataset_id)

        split_map = {}
        for split, anchor in splits.items():
            split_map[split.value] = {
                "anchor": anchor.split_anchor,  # existing: proves split exists
                "assignment": anchor.split_metadata.split_assignment_digest,  # new: proves membership
            }

        # Compute digest: H(canon(split_map))
        canonical_map = json.dumps(split_map, sort_keys=True, separators=(",", ":"))
        return sha256_hash(canonical_map.encode("utf-8"))

    def simulate_dataset_family(
        self, dataset_id: str, dataset_name: str = None, version: str = "v1"
    ) -> LCMDatasetFamilyAnchor:
        """
        Simulate dataset family creation for demonstration.

        Args:
            dataset_id: Dataset identifier
            dataset_name: Dataset name (defaults to dataset_id)
            version: Dataset version

        Returns:
            LCMDatasetFamilyAnchor instance
        """
        dataset_name = dataset_name or dataset_id

        # Create family metadata
        family_metadata = DatasetFamilyMetadata(
            name=dataset_name,
            version=version,
            owner="ciaf_user",
            license="Example License",
            description=f"Mock dataset family for {dataset_name}",
            privacy_level="public",
        )

        # Create dataset family with default splits
        return self.create_dataset_family(
            dataset_id=dataset_id, family_metadata=family_metadata
        )

    def format_family_summary(self, dataset_id: str) -> str:
        """Format dataset family summary for pretty printing."""
        family = self.get_dataset_family(dataset_id)
        if not family:
            return f"Dataset family {dataset_id} not found"

        splits = self.get_all_splits(dataset_id)

        lines = [
            f"📦 dataset: {family.family_metadata.name} {family.family_metadata.version}",
            f"🔐 dataset_family_anchor: {family.anchor_id}",
            f"🗂️ splits created: {[split.value for split in splits.keys()]}",
        ]

        for split, anchor in splits.items():
            lines.append(f"   ▸ split_anchor({split.value}): {anchor.anchor_id}")

        # Add authorization info
        splits_list = [split.value for split in splits.keys()]
        lines.append(
            f"✅ Authorized dataset: {family.family_metadata.name} {family.family_metadata.version} (splits: {', '.join(splits_list)})"
        )

        return "\n".join(lines)
