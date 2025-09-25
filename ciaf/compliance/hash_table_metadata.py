"""
Persistent hash table metadata storage for compliance verification.

This module provides persistent storage of Merkle tree hash tables and verification
caches, enabling efficient compliance audits without rebuilding cryptographic structures.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.crypto import sha256_hash


class HashTableMetadata:
    """
    Persistent storage for hash table metadata used in compliance verification.

    This class stores:
    - Merkle tree structure and proofs
    - Verification cache results
    - Capsule integrity metadata
    - Audit trail information
    """

    def __init__(self, dataset_id: str, storage_path: str = None):
        """
        Initialize hash table metadata storage.

        Args:
            dataset_id: Unique identifier for the dataset
            storage_path: Path to store metadata files (default: ./compliance_metadata/)
        """
        self.dataset_id = dataset_id
        self.storage_path = Path(storage_path or "./compliance_metadata")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Metadata structure
        self.metadata = {
            "dataset_id": dataset_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "version": "1.0",
            "merkle_tree_data": {
                "leaves": [],
                "merkle_root": None,
                "tree_hash": None,  # Hash of the tree structure for integrity
            },
            "proof_cache": {},  # leaf_hash -> proof
            "verification_cache": {},  # (leaf_hash, root_hash) -> bool
            "capsule_metadata": {},  # item_id -> capsule metadata
            "audit_trail": [],
            "statistics": {
                "total_capsules": 0,
                "total_proofs_cached": 0,
                "total_verifications_cached": 0,
                "cache_hit_rate": 0.0,
            },
        }

        # Load existing metadata if available
        self._load_metadata()

    def _get_metadata_file_path(self) -> Path:
        """Get the file path for metadata storage."""
        return self.storage_path / f"{self.dataset_id}_hash_metadata.json"

    def _load_metadata(self) -> bool:
        """Load metadata from file if it exists."""
        file_path = self._get_metadata_file_path()

        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    loaded_metadata = json.load(f)

                # Verify dataset ID matches
                if loaded_metadata.get("dataset_id") == self.dataset_id:
                    self.metadata = loaded_metadata
                    print(
                        f"Loaded hash table metadata for dataset '{self.dataset_id}'"
                    )
                    return True
                else:
                    print(f"Dataset ID mismatch in metadata file: {file_path}")
            except Exception as e:
                print(f"Error loading metadata from {file_path}: {e}")

        return False

    def save_metadata(self) -> bool:
        """Save metadata to file."""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            file_path = self._get_metadata_file_path()

            with open(file_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

            print(f"Saved hash table metadata to: {file_path}")
            return True

        except Exception as e:
            print(f"Error saving metadata: {e}")
            return False

    def store_merkle_tree_data(self, leaves: List[str], merkle_root: str) -> None:
        """
        Store Merkle tree structure data.

        Args:
            leaves: List of leaf hashes
            merkle_root: Root hash of the Merkle tree
        """
        # Create a hash of the tree structure for integrity verification
        tree_content = json.dumps(
            {
                "leaves": sorted(leaves),  # Sort for consistent hashing
                "root": merkle_root,
            },
            sort_keys=True,
        )
        tree_hash = sha256_hash(tree_content.encode("utf-8"))

        self.metadata["merkle_tree_data"] = {
            "leaves": leaves,
            "merkle_root": merkle_root,
            "tree_hash": tree_hash,
            "stored_at": datetime.now().isoformat(),
        }

        # Add audit trail entry
        self._add_audit_entry(
            "merkle_tree_stored",
            {
                "leaf_count": len(leaves),
                "merkle_root": merkle_root,
                "tree_hash": tree_hash,
            },
        )

        print(
            f"Stored Merkle tree data: {len(leaves)} leaves, root: {merkle_root[:16]}..."
        )

    def store_proof_cache(self, proof_cache: Dict[str, List[Tuple[str, str]]]) -> None:
        """
        Store Merkle proof cache.

        Args:
            proof_cache: Dictionary mapping leaf hashes to their proofs
        """
        # Convert tuples to lists for JSON serialization
        serializable_cache = {}
        for leaf_hash, proof in proof_cache.items():
            serializable_cache[leaf_hash] = [
                [sibling, direction] for sibling, direction in proof
            ]

        self.metadata["proof_cache"] = serializable_cache
        self.metadata["statistics"]["total_proofs_cached"] = len(proof_cache)

        # Add audit trail entry
        self._add_audit_entry("proof_cache_stored", {"cache_size": len(proof_cache)})

        print(f"Stored {len(proof_cache)} Merkle proofs in cache")

    def store_verification_cache(
        self, verification_cache: Dict[Tuple[str, str], bool]
    ) -> None:
        """
        Store verification results cache.

        Args:
            verification_cache: Dictionary mapping (leaf_hash, root_hash) tuples to verification results
        """
        # Convert tuple keys to strings for JSON serialization
        serializable_cache = {}
        for (leaf_hash, root_hash), verified in verification_cache.items():
            key = f"{leaf_hash}:{root_hash}"
            serializable_cache[key] = verified

        self.metadata["verification_cache"] = serializable_cache
        self.metadata["statistics"]["total_verifications_cached"] = len(
            verification_cache
        )

        # Add audit trail entry
        self._add_audit_entry(
            "verification_cache_stored", {"cache_size": len(verification_cache)}
        )

        print(f"Stored {len(verification_cache)} verification results in cache")

    def store_capsule_metadata(
        self, item_id: str, capsule_metadata: Dict[str, Any]
    ) -> None:
        """
        Store metadata for a specific capsule.

        Args:
            item_id: Capsule identifier
            capsule_metadata: Metadata dictionary for the capsule
        """
        # Create a safe copy with compliance-relevant information
        compliance_metadata = {
            "item_id": item_id,
            "data_fingerprint": capsule_metadata.get("data_fingerprint"),
            "creation_timestamp": capsule_metadata.get("creation_timestamp"),
            "dataset_anchor_id": capsule_metadata.get("dataset_anchor_id"),
            "stored_at": datetime.now().isoformat(),
            "verification_status": "stored",
        }

        self.metadata["capsule_metadata"][item_id] = compliance_metadata
        self.metadata["statistics"]["total_capsules"] = len(
            self.metadata["capsule_metadata"]
        )

        # Add audit trail entry
        self._add_audit_entry(
            "capsule_metadata_stored",
            {
                "item_id": item_id,
                "data_fingerprint": compliance_metadata.get("data_fingerprint"),
            },
        )

    def get_merkle_tree_data(self) -> Optional[Dict[str, Any]]:
        """Get stored Merkle tree data."""
        return self.metadata.get("merkle_tree_data")

    def get_proof_cache(self) -> Dict[str, List[Tuple[str, str]]]:
        """Get stored proof cache, converting back to tuple format."""
        cached_proofs = self.metadata.get("proof_cache", {})

        # Convert back to tuple format
        proof_cache = {}
        for leaf_hash, proof_list in cached_proofs.items():
            proof_cache[leaf_hash] = [
                (sibling, direction) for sibling, direction in proof_list
            ]

        return proof_cache

    def get_verification_cache(self) -> Dict[Tuple[str, str], bool]:
        """Get stored verification cache, converting back to tuple keys."""
        cached_verifications = self.metadata.get("verification_cache", {})

        # Convert back to tuple keys
        verification_cache = {}
        for key, verified in cached_verifications.items():
            if ":" in key:
                leaf_hash, root_hash = key.split(":", 1)
                verification_cache[(leaf_hash, root_hash)] = verified

        return verification_cache

    def get_capsule_metadata(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific capsule."""
        return self.metadata["capsule_metadata"].get(item_id)

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of stored metadata.

        Returns:
            Dictionary with verification results
        """
        verification_result = {
            "verified": True,
            "issues": [],
            "statistics": self.metadata["statistics"].copy(),
        }

        # Verify Merkle tree integrity
        merkle_data = self.metadata.get("merkle_tree_data")
        if merkle_data:
            # Recalculate tree hash
            tree_content = json.dumps(
                {
                    "leaves": sorted(merkle_data["leaves"]),
                    "root": merkle_data["merkle_root"],
                },
                sort_keys=True,
            )
            calculated_hash = sha256_hash(tree_content.encode("utf-8"))

            if calculated_hash != merkle_data.get("tree_hash"):
                verification_result["verified"] = False
                verification_result["issues"].append(
                    "Merkle tree integrity hash mismatch"
                )

        # Verify capsule count consistency
        stored_capsule_count = len(self.metadata["capsule_metadata"])
        reported_count = self.metadata["statistics"]["total_capsules"]

        if stored_capsule_count != reported_count:
            verification_result["verified"] = False
            verification_result["issues"].append(
                f"Capsule count mismatch: stored={stored_capsule_count}, reported={reported_count}"
            )

        return verification_result

    def compliance_report(self) -> Dict[str, Any]:
        """
        Generate a compliance report based on stored metadata.

        Returns:
            Comprehensive compliance report
        """
        integrity_check = self.verify_integrity()

        report = {
            "dataset_id": self.dataset_id,
            "report_generated_at": datetime.now().isoformat(),
            "metadata_file": str(self._get_metadata_file_path()),
            "integrity_verified": integrity_check["verified"],
            "integrity_issues": integrity_check["issues"],
            "statistics": {
                "total_capsules": self.metadata["statistics"]["total_capsules"],
                "total_proofs_cached": self.metadata["statistics"][
                    "total_proofs_cached"
                ],
                "total_verifications_cached": self.metadata["statistics"][
                    "total_verifications_cached"
                ],
                "audit_trail_entries": len(self.metadata["audit_trail"]),
            },
            "merkle_tree_summary": None,
            "audit_trail_summary": [],
        }

        # Add Merkle tree summary
        merkle_data = self.metadata.get("merkle_tree_data")
        if merkle_data:
            report["merkle_tree_summary"] = {
                "leaf_count": len(merkle_data["leaves"]),
                "merkle_root": merkle_data["merkle_root"],
                "tree_hash": merkle_data["tree_hash"],
                "stored_at": merkle_data.get("stored_at"),
            }

        # Add recent audit trail entries (last 10)
        recent_entries = self.metadata["audit_trail"][-10:]
        report["audit_trail_summary"] = recent_entries

        return report

    def _add_audit_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Add an entry to the audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
        }

        self.metadata["audit_trail"].append(entry)

        # Keep only the last 1000 entries to prevent file bloat
        if len(self.metadata["audit_trail"]) > 1000:
            self.metadata["audit_trail"] = self.metadata["audit_trail"][-1000:]

    def export_compliance_package(self, export_path: str = None) -> str:
        """
        Export a complete compliance package for external audit.

        Args:
            export_path: Path for the export package

        Returns:
            Path to the exported compliance package
        """
        if export_path is None:
            export_path = (
                self.storage_path
                / f"{self.dataset_id}_compliance_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        # Create comprehensive compliance package
        package = {
            "package_info": {
                "dataset_id": self.dataset_id,
                "exported_at": datetime.now().isoformat(),
                "exporter": "CIAF Hash Table Metadata System",
                "version": self.metadata["version"],
            },
            "compliance_report": self.compliance_report(),
            "full_metadata": self.metadata,
            "verification_instructions": {
                "merkle_tree_verification": "Verify tree_hash matches recalculated hash of sorted leaves and root",
                "proof_verification": "Use cached proofs to verify capsule integrity",
                "audit_trail": "Review audit trail for complete operation history",
            },
        }

        try:
            with open(export_path, "w") as f:
                json.dump(package, f, indent=2)

            print(f"Exported compliance package to: {export_path}")
            return str(export_path)

        except Exception as e:
            print(f"Error exporting compliance package: {e}")
            raise
