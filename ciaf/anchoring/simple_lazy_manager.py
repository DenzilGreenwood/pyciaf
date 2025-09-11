"""
Simple lazy manager for individual dataset anchors.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

from typing import Any, Dict

from ..provenance import ProvenanceCapsule
from .dataset_anchor import DatasetAnchor


class LazyManager:
    """
    Simple lazy manager for a single dataset anchor.
    Provides lazy capsule creation and materialization tracking.
    """

    def __init__(self, anchor: DatasetAnchor):
        """
        Initialize the lazy manager with a dataset anchor.

        Args:
            anchor: The DatasetAnchor to manage
        """
        self.anchor = anchor
        self.materialized_capsules: Dict[str, ProvenanceCapsule] = {}

    def create_lazy_capsule(
        self, item_id: str, original_data: Any, metadata: Dict[str, Any]
    ) -> ProvenanceCapsule:
        """
        Create a lazy capsule for a data item.

        Args:
            item_id: Unique identifier for the data item
            original_data: The original data content
            metadata: Metadata about the data item

        Returns:
            ProvenanceCapsule instance
        """
        # Create the capsule using dataset-level anchor derivation
        if item_id not in self.anchor.data_items:
            self.anchor.add_data_item(item_id, original_data, metadata)

        # Create the capsule with lazy anchor derivation
        data_secret = self.anchor.derive_item_key(item_id)

        capsule = ProvenanceCapsule(
            original_data=original_data, metadata=metadata, data_secret=data_secret
        )

        # Track materialized capsule
        self.materialized_capsules[item_id] = capsule

        return capsule

    def get_capsule(self, item_id: str) -> ProvenanceCapsule:
        """
        Get a materialized capsule by item ID.

        Args:
            item_id: ID of the data item

        Returns:
            ProvenanceCapsule if materialized, None otherwise
        """
        return self.materialized_capsules.get(item_id)

    def materialize_all(self) -> Dict[str, ProvenanceCapsule]:
        """
        Materialize all capsules for the dataset.

        Returns:
            Dictionary of all materialized capsules
        """
        for item_id in self.anchor.data_items:
            if item_id not in self.materialized_capsules:
                item_data = self.anchor.data_items[item_id]
                self.create_lazy_capsule(
                    item_id=item_id,
                    original_data=item_data["content"],
                    metadata=item_data["metadata"],
                )

        return self.materialized_capsules.copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the lazy manager.

        Returns:
            Dictionary with statistics
        """
        total_items = len(self.anchor.data_items)
        materialized_count = len(self.materialized_capsules)

        return {
            "total_items": total_items,
            "materialized_count": materialized_count,
            "materialization_rate": (
                materialized_count / total_items if total_items > 0 else 0
            ),
            "dataset_id": self.anchor.dataset_id,
        }
