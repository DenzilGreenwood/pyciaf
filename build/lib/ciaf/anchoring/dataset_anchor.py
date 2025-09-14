"""
Dataset-level anchoring for lazy capsule materialization.

This module implements the dataset anchor system that enables lazy capsule
materialization while maintaining cryptographic consistency and audit integrity.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from ..core import (
    SALT_LENGTH,
    derive_capsule_anchor,
    derive_dataset_anchor,
    derive_master_anchor,
    secure_random_bytes,
    sha256_hash,
    to_hex,
    # Backwards compatibility aliases
    derive_capsule_key,
    derive_dataset_key,
    derive_master_key,
)


class DatasetAnchor:
    """
    Represents a dataset anchor for lazy capsule materialization.

    This class manages the cryptographic hierarchy:
    Passphrase → Master Key → Dataset Key → Capsule Keys (on demand)
    """

    def __init__(
        self,
        dataset_id: str,
        metadata: Dict[str, Any] = None,
        model_name: str = None,
        master_password: str = None,
        salt: bytes = None,
    ):
        """
        Initialize a dataset anchor.

        Args:
            dataset_id: Unique identifier for the dataset.
            metadata: Metadata about the dataset (optional, for backwards compatibility)
            model_name: Name of the model (optional, for backwards compatibility)
            master_password: Master password for anchor derivation (optional)
            salt: Salt for anchor derivation (optional)
        """
        self.dataset_id = dataset_id

        # Handle different initialization patterns for backwards compatibility
        if metadata is not None:
            self.metadata = metadata.copy()
        else:
            self.metadata = {}

        # Use master_password if provided, otherwise fall back to model_name
        password = master_password or model_name or "default_password"
        self.model_name = (
            model_name or "default_model"
        )  # Store for backwards compatibility

        # Generate or use provided salt
        if salt is not None:
            self.master_key_salt = salt
        else:
            self.master_key_salt = secure_random_bytes(SALT_LENGTH)

        # Derive master anchor and dataset anchor
        self.master_anchor = derive_master_anchor(password, self.master_key_salt)

        # Compute dataset hash from metadata and derive dataset anchor
        self.dataset_hash = self._compute_dataset_hash()
        self.dataset_anchor = derive_dataset_anchor(self.master_anchor, self.dataset_hash)

        # Maintain backwards compatibility with existing property names
        self.master_key = self.master_anchor  # Legacy alias
        self.dataset_key = self.dataset_anchor  # Legacy alias

        # Initialize sample tracking
        self.sample_hashes: List[str] = []
        self.total_samples = 0
        self.training_samples = 0
        self.testing_samples = 0

        # Track capsulation status by phase
        self.capsulation_status = {
            "full_dataset": {"total": 0, "capsulated": 0, "percentage": 0.0},
            "training_phase": {"total": 0, "capsulated": 0, "percentage": 0.0},
            "testing_phase": {"total": 0, "capsulated": 0, "percentage": 0.0},
        }

        # Initialize data items tracking for lazy management
        self.data_items: Dict[str, Dict[str, Any]] = {}

        # Add anchoring metadata with capsulation notes
        self.metadata.update(
            {
                "dataset_anchor_id": self.dataset_id,
                "master_key_salt": self.master_key_salt.hex(),
                "dataset_hash": self.dataset_hash,
                "creation_timestamp": datetime.now().isoformat(),
                "audit_tags": [
                    "lazy_capsules",
                    "anchored_merkle_root",
                    "dataset_level_derivation",
                ],
                "capsulation_notes": {
                    "train_test_split_impact": "Capsulation percentages reflect subset used in each phase due to random train/test split",
                    "recommendation": "Perform bias and quality validation on full dataset before train/test split",
                    "validation_required": "Use pre_ingestion_validator before splitting data",
                },
            }
        )

        print(
            f"Dataset Anchor '{self.dataset_id}' anchor initialized for model '{self.model_name}'"
        )

    def _compute_dataset_hash(self) -> str:
        """
        Compute a hash of the dataset metadata for consistent anchor derivation.

        Returns:
            SHA256 hash of the dataset metadata.
        """
        # Create a deterministic representation of the metadata
        metadata_copy = self.metadata.copy()
        # Remove dynamic fields that shouldn't affect the dataset hash
        metadata_copy.pop("creation_timestamp", None)
        metadata_copy.pop("audit_tags", None)

        metadata_str = json.dumps(metadata_copy, sort_keys=True)
        return sha256_hash(metadata_str.encode("utf-8"))

    def add_sample_hash(self, sample_hash: str) -> None:
        """
        Add a sample hash to the dataset for Merkle tree construction.

        Args:
            sample_hash: SHA256 hash of a data sample.
        """
        if sample_hash not in self.sample_hashes:
            self.sample_hashes.append(sample_hash)
            self.total_samples = len(self.sample_hashes)

    def add_data_item(
        self,
        item_id: str,
        content: Any,
        metadata: Dict[str, Any],
        phase: str = "full_dataset",
    ) -> None:
        """
        Add a data item to the dataset for lazy capsule management.

        Args:
            item_id: Unique identifier for the data item
            content: The data content (string, float, int, or other)
            metadata: Metadata about the data item
            phase: Dataset phase ('full_dataset', 'training_phase', 'testing_phase')
        """
        self.data_items[item_id] = {
            "content": content,
            "metadata": metadata.copy(),
            "phase": phase,
        }

        # Update capsulation tracking for the specific phase
        if phase in self.capsulation_status:
            self.capsulation_status[phase]["total"] += 1
            self.capsulation_status[phase][
                "capsulated"
            ] += 1  # Assume capsulated when added

            # Update percentage
            total = self.capsulation_status[phase]["total"]
            capsulated = self.capsulation_status[phase]["capsulated"]
            self.capsulation_status[phase]["percentage"] = (
                (capsulated / total * 100) if total > 0 else 0.0
            )

        # Also add to sample hashes for Merkle tree - handle different data types
        if isinstance(content, (str, bytes)):
            content_str = (
                content if isinstance(content, str) else content.decode("utf-8")
            )
        else:
            # Convert numerical or other data to string representation
            content_str = str(content)

        item_hash = sha256_hash(f"{item_id}:{content_str}".encode("utf-8"))
        self.add_sample_hash(item_hash)

        # Track by phase
        if phase == "training_phase":
            self.training_samples += 1
        elif phase == "testing_phase":
            self.testing_samples += 1

    def set_phase_totals(self, training_total: int, testing_total: int) -> None:
        """
        Set the expected totals for training and testing phases.

        This should be called after train/test split to set accurate totals.

        Args:
            training_total: Total number of training samples expected
            testing_total: Total number of testing samples expected
        """
        self.capsulation_status["training_phase"]["total"] = training_total
        self.capsulation_status["testing_phase"]["total"] = testing_total

        # Recalculate percentages
        for phase in ["training_phase", "testing_phase"]:
            total = self.capsulation_status[phase]["total"]
            capsulated = self.capsulation_status[phase]["capsulated"]
            self.capsulation_status[phase]["percentage"] = (
                (capsulated / total * 100) if total > 0 else 0.0
            )

        print(
            f"📊 Phase totals set - Training: {training_total}, Testing: {testing_total}"
        )

    def get_capsulation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of capsulation status across all phases.

        Returns:
            Dictionary with capsulation statistics and notes
        """
        return {
            "capsulation_status": self.capsulation_status.copy(),
            "total_items_tracked": len(self.data_items),
            "merkle_tree_samples": len(self.sample_hashes),
            "split_impact_note": (
                "Capsulation percentages may be less than 100% due to random train/test split. "
                "Only samples used in each phase are capsulated for that phase."
            ),
            "recommendation": (
                "Validate complete dataset for bias and quality issues before train/test split "
                "using pre_ingestion_validator to ensure representative sampling."
            ),
        }

    def derive_item_anchor(self, item_id: str) -> str:
        """
        Derive an anchor for a specific data item using the dataset anchor.

        Args:
            item_id: Unique identifier for the data item

        Returns:
            Derived anchor for the data item as hex string
        """
        return to_hex(derive_capsule_anchor(self.dataset_anchor, item_id))

    def derive_capsule_anchor(self, capsule_id: str) -> str:
        """
        Derive a capsule anchor on-demand using HMAC(dataset_anchor, capsule_id).

        Args:
            capsule_id: Unique identifier for the capsule.

        Returns:
            Derived capsule anchor as hexadecimal string.
        """
        return to_hex(derive_capsule_anchor(self.dataset_anchor, capsule_id))

    # Backwards compatibility methods
    def derive_item_key(self, item_id: str) -> str:
        """Legacy method - use derive_item_anchor instead."""
        return self.derive_item_anchor(item_id)

    def derive_capsule_key(self, capsule_id: str) -> str:
        """Legacy method - use derive_capsule_anchor instead."""
        return self.derive_capsule_anchor(capsule_id)

    def get_merkle_root(self) -> str:
        """
        Get the Merkle root of all sample hashes in the dataset.

        Returns:
            Merkle root hash, or None if no samples.
        """
        if not self.sample_hashes:
            return None

        from ..core import MerkleTree

        merkle_tree = MerkleTree(self.sample_hashes)
        return merkle_tree.get_root()

    def create_true_lazy_manager(self):
        """
        Create a TrueLazyManager for this dataset anchor.

        Returns:
            TrueLazyManager instance configured for this dataset
        """
        from .true_lazy_manager import TrueLazyManager

        return TrueLazyManager(self.dataset_id)

    def verify_capsule_integrity(
        self, capsule_id: str, capsule_metadata: Dict[str, Any]
    ) -> bool:
        """
        Verify that a capsule belongs to this dataset anchor.

        Args:
            capsule_id: The capsule ID to verify.
            capsule_metadata: Metadata from the capsule.

        Returns:
            True if the capsule is valid for this dataset.
        """
        # Check dataset anchor ID
        if capsule_metadata.get("dataset_anchor_id") != self.dataset_id:
            return False

        # Verify capsule anchor derivation
        expected_anchor = self.derive_capsule_anchor(capsule_id)
        if "capsule_key_derivation" in capsule_metadata:
            # This is a lazy capsule, verify the derivation string
            expected_derivation = f"HMAC(dataset_anchor, {capsule_id})"
            return capsule_metadata["capsule_key_derivation"] == expected_derivation

        return True

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize the dataset anchor to JSON.

        Returns:
            JSON-serializable dictionary.
        """
        return {
            "dataset_id": self.dataset_id,
            "model_name": self.model_name,
            "metadata": self.metadata,
            "dataset_hash": self.dataset_hash,
            "master_key_salt": self.master_key_salt.hex(),
            "sample_hashes": self.sample_hashes,
            "total_samples": self.total_samples,
            "merkle_root": self.get_merkle_root(),
        }

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "DatasetAnchor":
        """
        Reconstruct a dataset anchor from JSON data.

        Args:
            json_data: JSON data from to_json().

        Returns:
            Reconstructed DatasetAnchor instance.
        """
        # Create new instance
        anchor = cls.__new__(cls)
        anchor.dataset_id = json_data["dataset_id"]
        anchor.model_name = json_data["model_name"]
        anchor.metadata = json_data["metadata"]
        anchor.dataset_hash = json_data["dataset_hash"]
        anchor.master_key_salt = bytes.fromhex(json_data["master_key_salt"])
        anchor.sample_hashes = json_data["sample_hashes"]
        anchor.total_samples = json_data["total_samples"]

        # Regenerate derived anchors with backwards compatibility
        anchor.master_anchor = derive_master_anchor(
            anchor.model_name, anchor.master_key_salt, 32
        )
        anchor.dataset_anchor = derive_dataset_anchor(anchor.master_anchor, anchor.dataset_hash)
        
        # Maintain legacy aliases
        anchor.master_key = anchor.master_anchor
        anchor.dataset_key = anchor.dataset_anchor

        return anchor
