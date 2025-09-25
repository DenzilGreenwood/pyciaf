"""
Training snapshots for verifiable model training records.

This module contains the TrainingSnapshot class which creates tamper-evident
records of what data was used to train a specific model version.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from datetime import datetime
from typing import Any, Dict

from ..core import MerkleTree, sha256_hash


class ModelAggregationAnchor:
    """
    Represents a Model Aggregation Anchor (MAA) for authenticating training data.
    Used to generate and verify signatures for data used in model training.
    """

    def __init__(self, key_id: str, secret_material: str):
        """
        Initializes a ModelAggregationAnchor.

        Args:
            key_id: Unique identifier for this MAA.
            secret_material: Secret string used for anchor derivation.
        """
        import hashlib

        from ..core import SALT_LENGTH, derive_master_anchor, secure_random_bytes

        self.key_id = key_id
        self.salt = secure_random_bytes(SALT_LENGTH)
        
        # Use proper anchor derivation instead of legacy derive_key
        self.derived_anchor = derive_master_anchor(secret_material, self.salt)
        
        print(f"MAA '{self.key_id}' initialized with anchor-based cryptography.")

    def generate_data_signature(self, data_hash: str) -> str:
        """
        Generates a signature for a given data hash using anchor-based cryptography.

        Args:
            data_hash: SHA256 hash of the data to sign.

        Returns:
            Hexadecimal signature string.
        """
        import hashlib

        h = hashlib.sha256()
        h.update(self.derived_anchor)
        h.update(data_hash.encode("utf-8"))
        return h.hexdigest()

    def verify_data_signature(self, data_hash: str, signature: str) -> bool:
        """
        Verifies a signature against a data hash.

        Args:
            data_hash: SHA256 hash of the data.
            signature: Signature to verify.

        Returns:
            True if the signature is valid, False otherwise.
        """
        expected_signature = self.generate_data_signature(data_hash)
        return expected_signature == signature


class TrainingSnapshot:
    """
    Represents a snapshot of model training, including parameters and data provenance.
    Creates a tamper-evident record of what data was used to train a specific model version.
    """

    def __init__(
        self,
        model_version: str,
        training_parameters: dict,
        provenance_capsule_hashes: list[str],
    ):
        """
        Initializes a TrainingSnapshot.

        Args:
            model_version: Version identifier for the trained model.
            training_parameters: Dictionary of training hyperparameters.
            provenance_capsule_hashes: List of hash proofs from provenance capsules.
        """
        self.model_version = model_version
        self.training_parameters = training_parameters
        self.provenance_capsule_hashes = sorted(list(set(provenance_capsule_hashes)))
        self.timestamp = datetime.now().isoformat()
        self.merkle_tree = MerkleTree(self.provenance_capsule_hashes)
        self.merkle_root_hash = self.merkle_tree.get_root()

        # Initialize metadata dictionary for lazy capsule support
        self.metadata = {
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "merkle_root_hash": self.merkle_root_hash,
            "training_parameters": self.training_parameters,
        }

        self.snapshot_id = sha256_hash(
            f"{self.model_version}-{self.merkle_root_hash}-{self.timestamp}-{json.dumps(self.training_parameters, sort_keys=True)}".encode(
                "utf-8"
            )
        )
        print(
            f"Training Snapshot '{self.snapshot_id}' created for model '{self.model_version}'."
        )

    def to_json(self) -> dict:
        """
        Serializes the TrainingSnapshot to a JSON-compatible dictionary.

        Returns:
            Dictionary representation of the snapshot.
        """
        return {
            "snapshot_id": self.snapshot_id,
            "model_version": self.model_version,
            "training_parameters": self.training_parameters,
            "provenance_capsule_hashes": self.provenance_capsule_hashes,
            "timestamp": self.timestamp,
            "merkle_root_hash": self.merkle_root_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, json_data: dict):
        """
        Reconstructs a TrainingSnapshot from JSON data.

        Args:
            json_data: Dictionary representation from to_json().

        Returns:
            Reconstructed TrainingSnapshot instance.
        """
        snapshot = cls.__new__(cls)
        snapshot.snapshot_id = json_data["snapshot_id"]
        snapshot.model_version = json_data["model_version"]
        snapshot.training_parameters = json_data["training_parameters"]
        snapshot.provenance_capsule_hashes = json_data["provenance_capsule_hashes"]
        snapshot.timestamp = json_data["timestamp"]
        snapshot.merkle_root_hash = json_data["merkle_root_hash"]
        snapshot.metadata = json_data.get("metadata", {})
        snapshot.merkle_tree = MerkleTree(snapshot.provenance_capsule_hashes)
        return snapshot

    def verify_provenance(self, provenance_capsule_hash: str) -> bool:
        """
        Verifies that a specific provenance capsule hash was used in training.

        Args:
            provenance_capsule_hash: Hash proof from a provenance capsule.

        Returns:
            True if the hash was used in training, False otherwise.
        """
        if provenance_capsule_hash not in self.provenance_capsule_hashes:
            return False
        proof = self.merkle_tree.get_proof(provenance_capsule_hash)
        return MerkleTree.verify_proof(
            provenance_capsule_hash, self.merkle_root_hash, proof
        )
