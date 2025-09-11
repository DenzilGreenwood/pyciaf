"""
Lazy capsule management system.

This module orchestrates lazy capsule materialization across multiple datasets
and provides high-level interfaces for dataset management and on-demand capsule materialization.

UPDATED: Now supports true lazy behavior that achieves patent-claimed performance improvements.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

from datetime import datetime
from typing import Any, Dict

from .dataset_anchor import DatasetAnchor
from .true_lazy_manager import TrueLazyManager


class LazyProvenanceManager:
    """
    Manager for lazy capsule materialization across multiple datasets.

    This class orchestrates the lazy capsule system and provides high-level
    interfaces for dataset management and on-demand capsule materialization.

    UPDATED: Now supports both legacy and true lazy behavior for backwards compatibility.
    """

    def __init__(self, use_true_lazy: bool = True):
        """
        Initialize the lazy provenance manager.

        Args:
            use_true_lazy: If True, use TrueLazyManager for patent-level performance.
                          If False, use legacy behavior for backwards compatibility.
        """
        self.dataset_anchors: Dict[str, DatasetAnchor] = {}
        self.lazy_capsule_registry: Dict[str, Dict[str, Any]] = {}
        self.true_lazy_managers: Dict[str, TrueLazyManager] = {}
        self.use_true_lazy = use_true_lazy

        print(f"🚀 LazyProvenanceManager initialized (true_lazy={use_true_lazy})")

    def create_dataset_anchor(
        self, dataset_id: str, model_name: str, dataset_metadata: Dict[str, Any]
    ) -> DatasetAnchor:
        """
        Create a new dataset anchor.

        Args:
            dataset_id: Unique identifier for the dataset.
            model_name: Name of the model.
            dataset_metadata: Metadata about the dataset.

        Returns:
            Created DatasetAnchor instance.
        """
        anchor = DatasetAnchor(
            dataset_id=dataset_id, 
            metadata=dataset_metadata, 
            model_name=model_name
        )
        self.dataset_anchors[dataset_id] = anchor

        # Create true lazy manager if enabled
        if self.use_true_lazy:
            self.true_lazy_managers[dataset_id] = anchor.create_true_lazy_manager()

        return anchor

    def register_lazy_capsule(
        self,
        dataset_id: str,
        capsule_id: str,
        sample_data: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Register a lazy capsule without full materialization.

        Args:
            dataset_id: ID of the dataset anchor.
            capsule_id: Unique identifier for the capsule.
            sample_data: The raw sample data.
            metadata: Additional metadata for the sample.

        Returns:
            Lazy capsule metadata.
        """
        if dataset_id not in self.dataset_anchors:
            raise ValueError(f"Dataset anchor '{dataset_id}' not found")

        anchor = self.dataset_anchors[dataset_id]

        if self.use_true_lazy and dataset_id in self.true_lazy_managers:
            # Use true lazy implementation
            true_lazy_manager = self.true_lazy_managers[dataset_id]
            data_secret = anchor.derive_capsule_key(capsule_id)

            lazy_ref = true_lazy_manager.create_lazy_reference(
                item_id=capsule_id,
                original_data=sample_data,
                metadata=metadata,
                data_secret=data_secret,
            )

            return lazy_ref.get_lightweight_info()
        else:
            # Use legacy implementation
            lazy_metadata = self._create_lazy_capsule_metadata(
                anchor, capsule_id, sample_data, metadata
            )

            # Register in the lazy capsule registry
            full_capsule_id = f"{dataset_id}:{capsule_id}"
            self.lazy_capsule_registry[full_capsule_id] = {
                "lazy_metadata": lazy_metadata,
                "sample_data": sample_data,
                "materialized_capsule": None,
            }

            return lazy_metadata

    def _create_lazy_capsule_metadata(
        self,
        anchor: DatasetAnchor,
        capsule_id: str,
        sample_data: Any,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create metadata for a lazy capsule without materializing the full capsule.

        Args:
            anchor: The dataset anchor.
            capsule_id: Unique identifier for the capsule.
            sample_data: The raw sample data (string, float, int, or other).
            metadata: Additional metadata for the sample.

        Returns:
            Capsule metadata with lazy materialization support.
        """
        from ..core import sha256_hash

        # Compute sample hash - handle different data types
        if isinstance(sample_data, (str, bytes)):
            sample_data_str = (
                sample_data
                if isinstance(sample_data, str)
                else sample_data.decode("utf-8")
            )
        else:
            # Convert numerical or other data to string representation
            sample_data_str = str(sample_data)

        sample_hash = sha256_hash(sample_data_str.encode("utf-8"))
        anchor.add_sample_hash(sample_hash)

        # Create lazy capsule metadata (without actually deriving the expensive key)
        lazy_metadata = {
            "capsule_id": capsule_id,
            "dataset_anchor_id": anchor.dataset_id,
            "sample_hash": sample_hash,
            "capsule_key_derivation": f"HMAC(dataset_key, {capsule_id})",
            "materialized": False,
            "creation_timestamp": datetime.now().isoformat(),
            "metadata": metadata,
        }

        return lazy_metadata

    def materialize_capsule(self, dataset_id: str, capsule_id: str):
        """
        Materialize a specific capsule on-demand.

        Args:
            dataset_id: ID of the dataset anchor.
            capsule_id: Unique identifier for the capsule.

        Returns:
            Materialized ProvenanceCapsule.

        Raises:
            ValueError: If capsule is not found or dataset anchor doesn't exist.
        """
        if dataset_id not in self.dataset_anchors:
            raise ValueError(f"Dataset anchor '{dataset_id}' not found")

        if self.use_true_lazy and dataset_id in self.true_lazy_managers:
            # Use true lazy implementation
            true_lazy_manager = self.true_lazy_managers[dataset_id]
            return true_lazy_manager.materialize_capsule(capsule_id)
        else:
            # Use legacy implementation
            full_capsule_id = f"{dataset_id}:{capsule_id}"

            if full_capsule_id not in self.lazy_capsule_registry:
                raise ValueError(f"Lazy capsule '{full_capsule_id}' not found")

            registry_entry = self.lazy_capsule_registry[full_capsule_id]

            # Check if already materialized
            if registry_entry["materialized_capsule"] is not None:
                return registry_entry["materialized_capsule"]

            # Materialize the capsule
            anchor = self.dataset_anchors[dataset_id]
            sample_data = registry_entry["sample_data"]
            metadata = registry_entry["lazy_metadata"]["metadata"]

            capsule = self._materialize_provenance_capsule(
                anchor, capsule_id, sample_data, metadata
            )

            # Cache the materialized capsule
            registry_entry["materialized_capsule"] = capsule
            registry_entry["lazy_metadata"]["materialized"] = True

            return capsule

    def _materialize_provenance_capsule(
        self,
        anchor: DatasetAnchor,
        capsule_id: str,
        sample_data: str,
        metadata: Dict[str, Any],
    ):
        """
        Materialize a full provenance capsule on-demand.

        Args:
            anchor: The dataset anchor.
            capsule_id: Unique identifier for the capsule.
            sample_data: The raw sample data.
            metadata: Additional metadata for the sample.

        Returns:
            Fully materialized ProvenanceCapsule.
        """
        from ..provenance import ProvenanceCapsule

        # Derive the capsule key
        capsule_key = anchor.derive_capsule_key(capsule_id)

        # Create enhanced metadata for the capsule
        enhanced_metadata = metadata.copy()
        enhanced_metadata.update(
            {
                "capsule_id": capsule_id,
                "dataset_anchor_id": anchor.dataset_id,
                "audit_reference": f"provenance_{anchor.model_name}_{capsule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            }
        )

        # Create the provenance capsule using the derived key as the data secret
        capsule = ProvenanceCapsule(sample_data, enhanced_metadata, capsule_key)

        print(f"Materialized capsule '{capsule_id}' for dataset '{anchor.dataset_id}'")
        return capsule

    def get_dataset_summary(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get a summary of a dataset's lazy capsule status.

        Args:
            dataset_id: ID of the dataset anchor.

        Returns:
            Summary dictionary with statistics.
        """
        if dataset_id not in self.dataset_anchors:
            raise ValueError(f"Dataset anchor '{dataset_id}' not found")

        anchor = self.dataset_anchors[dataset_id]

        if self.use_true_lazy and dataset_id in self.true_lazy_managers:
            # Use true lazy implementation
            true_lazy_manager = self.true_lazy_managers[dataset_id]
            manager_summary = true_lazy_manager.get_summary()

            return {
                "dataset_id": dataset_id,
                "model_name": anchor.model_name,
                "total_samples": anchor.total_samples,
                "lazy_implementation": "TrueLazyManager",
                "performance_improvement": manager_summary["performance"][
                    "estimated_improvement"
                ],
                "memory_efficiency": manager_summary["efficiency"],
                "merkle_root": anchor.get_merkle_root(),
                "dataset_hash": anchor.dataset_hash,
            }
        else:
            # Use legacy implementation
            lazy_count = 0
            materialized_count = 0

            for full_id, entry in self.lazy_capsule_registry.items():
                if full_id.startswith(f"{dataset_id}:"):
                    if entry["materialized_capsule"] is not None:
                        materialized_count += 1
                    else:
                        lazy_count += 1

            return {
                "dataset_id": dataset_id,
                "model_name": anchor.model_name,
                "total_samples": anchor.total_samples,
                "lazy_capsules": lazy_count,
                "materialized_capsules": materialized_count,
                "lazy_implementation": "Legacy",
                "merkle_root": anchor.get_merkle_root(),
                "dataset_hash": anchor.dataset_hash,
            }

    def audit_capsule_provenance(
        self, dataset_id: str, capsule_id: str
    ) -> Dict[str, Any]:
        """
        Perform a complete audit of a capsule's provenance.

        Args:
            dataset_id: ID of the dataset anchor.
            capsule_id: Unique identifier for the capsule.

        Returns:
            Audit results with verification status.
        """
        try:
            if self.use_true_lazy and dataset_id in self.true_lazy_managers:
                # Use true lazy implementation
                true_lazy_manager = self.true_lazy_managers[dataset_id]
                return true_lazy_manager.audit_capsule_provenance(capsule_id)
            else:
                # Use legacy implementation
                # Materialize the capsule for audit
                capsule = self.materialize_capsule(dataset_id, capsule_id)

                # Verify capsule integrity
                integrity_valid = capsule.verify_hash_proof()

                # Verify dataset anchor consistency
                anchor = self.dataset_anchors[dataset_id]
                anchor_valid = anchor.verify_capsule_integrity(
                    capsule_id, capsule.metadata
                )

                # Generate audit metadata
                audit_metadata = {
                    "capsule_id": capsule_id,
                    "dataset_anchor_id": dataset_id,
                    "audit_timestamp": datetime.now().isoformat(),
                    "integrity_verified": integrity_valid,
                    "anchor_verified": anchor_valid,
                    "audit_passed": integrity_valid and anchor_valid,
                    "capsule_hash": capsule.hash_proof,
                    "audit_reference": capsule.metadata.get(
                        "audit_reference",
                        f"audit_{capsule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    ),
                }

                return audit_metadata

        except Exception as e:
            return {
                "capsule_id": capsule_id,
                "dataset_anchor_id": dataset_id,
                "audit_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "audit_passed": False,
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics across all datasets.

        Returns:
            Comprehensive performance statistics
        """
        if not self.use_true_lazy:
            return {
                "lazy_implementation": "Legacy",
                "performance_note": "True lazy performance requires use_true_lazy=True",
            }

        stats = {
            "lazy_implementation": "TrueLazyManager",
            "total_datasets": len(self.dataset_anchors),
            "datasets": {},
        }

        total_improvement = 0.0
        dataset_count = 0

        for dataset_id, true_lazy_manager in self.true_lazy_managers.items():
            dataset_stats = true_lazy_manager.get_performance_stats()
            stats["datasets"][dataset_id] = dataset_stats

            if dataset_stats["performance_improvement_estimate"] > 1.0:
                total_improvement += dataset_stats["performance_improvement_estimate"]
                dataset_count += 1

        if dataset_count > 0:
            stats["average_performance_improvement"] = total_improvement / dataset_count
        else:
            stats["average_performance_improvement"] = 1.0

        return stats
