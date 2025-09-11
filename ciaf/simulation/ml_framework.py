"""
ML Framework simulator for testing CIAF integration.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

from ..provenance import ModelAggregationAnchor, ProvenanceCapsule, TrainingSnapshot
from .mock_llm import MockLLM
import hashlib
from ..core import derive_model_anchor, to_hex


class MLFrameworkSimulator:
    """
    Simulates interactions with an ML framework (e.g., PyTorch, TensorFlow).
    Provides conceptual integration points for CIAF components.
    """

    def __init__(self, model_name: str = "MockSimulator"):
        """
        Initialize the ML Framework Simulator.

        Args:
            model_name: Name of the simulated model.
        """
        self.model_name = model_name
        self.llm = MockLLM()

    def prepare_data(self, raw_data_list: list) -> list[ProvenanceCapsule]:
        """
        Simulates preparing data by creating Provenance Capsules.

        Args:
            raw_data_list: List of dictionaries, each representing a raw data item.

        Returns:
            List of ProvenanceCapsule objects.
        """
        provenance_capsules = []
        for i, data_item in enumerate(raw_data_list):
            data_id = data_item.get("id", f"data_{i}")
            data_secret = f"secret_for_{data_id}"
            
            capsule = ProvenanceCapsule(
                original_data=data_item["content"],
                metadata=data_item["metadata"],
                data_secret=data_secret,
            )
            provenance_capsules.append(capsule)
            print(f"  Created Provenance Capsule for data ID: {data_id}")
        return provenance_capsules

    def train_model(
        self,
        training_data_capsules: list[ProvenanceCapsule],
        maa: ModelAggregationAnchor,
        training_params: dict,
        model_version: str,
    ) -> TrainingSnapshot:
        """
        Simulates model training, including MAA validation and snapshot generation.

        Args:
            training_data_capsules: List of ProvenanceCapsule objects.
            maa: The ModelAggregationAnchor to use for data authorization.
            training_params: Dictionary of training hyperparameters.
            model_version: Version identifier for the trained model.

        Returns:
            TrainingSnapshot representing the training session.
        """
        print(f"🔄 Starting training simulation for model '{self.model_name}' v{model_version}")
        print(f"   Training on {len(training_data_capsules)} data capsules")
        
        # Extract provenance hashes for snapshot
        provenance_hashes = []
        for capsule in training_data_capsules:
            hash_proof = capsule.hash_proof
            provenance_hashes.append(hash_proof)
            print(f"   ✓ Validated capsule with hash: {hash_proof[:16]}...")

        # Create training snapshot
        snapshot = TrainingSnapshot(
            model_version=model_version,
            training_parameters=training_params,
            provenance_capsule_hashes=provenance_hashes,
        )

        print(f"✅ Training simulation completed. Snapshot ID: {snapshot.snapshot_id[:16]}...")
        return snapshot

    def hash_model_parameters(self, parameters: dict) -> str:
        """
        Hash model parameters for immutable model fingerprinting.

        Args:
            parameters: Dictionary of model parameters.
        Returns:
            SHA256 hash of sorted parameters as hex string.
        """
        param_str = str(sorted(parameters.items()))
        return hashlib.sha256(param_str.encode("utf-8")).hexdigest()

    def create_model_anchor(self, master_anchor: bytes, parameters: dict) -> str:
        """
        Create a model anchor using hashed parameters.

        Args:
            master_anchor: Master anchor bytes.
            parameters: Dictionary of model parameters.
        Returns:
            Model anchor as hex string.
        """
        model_hash = self.hash_model_parameters(parameters)
        anchor_bytes = derive_model_anchor(master_anchor, model_hash)
        return to_hex(anchor_bytes)

    def get_model_info(self, parameters: dict = None, master_anchor: bytes = None) -> dict:
        """
        Get information about the simulated model, including anchor and parameter hash if provided.

        Args:
            parameters: Optional model parameters for hashing.
            master_anchor: Optional master anchor for anchor derivation.
        Returns:
            Dictionary containing model information.
        """
        info = {
            "model_name": self.model_name,
            "framework": "CIAF Simulator",
            "type": "Mock ML Framework",
            "supports_provenance": True,
            "supports_maa": True,
        }
        if parameters:
            info["model_param_hash"] = self.hash_model_parameters(parameters)
        if parameters and master_anchor:
            info["model_anchor"] = self.create_model_anchor(master_anchor, parameters)
        return info
