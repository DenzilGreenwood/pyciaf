"""
Provenance Capsule implementation for the Cognitive Insight Audit Framework.

This module contains the ProvenanceCapsule class which provides verifiable data lineage
without exposing raw sensitive data. HIPAA compliant through data minimization and
consent management.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import os
from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime

from cryptography.exceptions import InvalidTag


from ..core import (
    SALT_LENGTH,
    decrypt_aes_gcm,
    encrypt_aes_gcm,
    secure_random_bytes,
    sha256_hash,
)


class ProvenanceCapsule:
    """
    Represents a Provenance Capsule for a piece of training data.
    Ensures verifiable data lineage without exposing raw sensitive data.

    HIPAA Compliance: Directly supports data minimization and patient consent requirements
    by encapsulating metadata without revealing raw PHI.
    """

    def __init__(self, original_data, metadata: dict, data_secret: str):
        """
        Initializes a ProvenanceCapsule.

        Args:
            original_data: The raw, sensitive data (e.g., patient record, financial transaction, numerical values).
            metadata: Dictionary containing verifiable metadata (e.g., source, consent_status, timestamp).
            data_secret: A secret string unique to this piece of data, used for anchor derivation.
        """
        self.original_data = original_data  # Store original data for testing

        # Handle different data types - convert to string representation first
        if isinstance(original_data, (str, bytes)):
            data_str = (
                original_data
                if isinstance(original_data, str)
                else original_data.decode("utf-8")
            )
        else:
            # Convert numerical or other data to string representation
            data_str = str(original_data)

        self.original_data_bytes = data_str.encode("utf-8")
        self.metadata = metadata
        self.data_secret_bytes = data_secret.encode("utf-8")
        self.salt = secure_random_bytes(SALT_LENGTH)
        self.derived_key = derive_key(self.salt, self.data_secret_bytes, 32)
        self.encrypted_data, self.nonce, self.tag = encrypt_aes_gcm(
            self.derived_key, self.original_data_bytes
        )
        self.hash_proof = sha256_hash(self.original_data_bytes)
        self.metadata["hash_proof"] = self.hash_proof
        self.metadata["creation_timestamp"] = datetime.now().isoformat()

    def to_json(self) -> dict:
        """
        Serializes the ProvenanceCapsule to a JSON-compatible dictionary.

        Returns:
            Dictionary representation of the capsule.
        """
        return {
            "metadata": self.metadata,
            "encrypted_data": urlsafe_b64encode(self.encrypted_data).decode("utf-8"),
            "nonce": urlsafe_b64encode(self.nonce).decode("utf-8"),
            "tag": urlsafe_b64encode(self.tag).decode("utf-8"),
            "salt": urlsafe_b64encode(self.salt).decode("utf-8"),
        }

    @classmethod
    def from_json(cls, json_data: dict, data_secret: str):
        """
        Reconstructs a ProvenanceCapsule from JSON data.

        Args:
            json_data: Dictionary representation from to_json().
            data_secret: The secret used to create the original capsule.

        Returns:
            Reconstructed ProvenanceCapsule instance.
        """
        capsule = cls.__new__(cls)
        capsule.metadata = json_data["metadata"]
        capsule.encrypted_data = urlsafe_b64decode(json_data["encrypted_data"])
        capsule.nonce = urlsafe_b64decode(json_data["nonce"])
        capsule.tag = urlsafe_b64decode(json_data["tag"])
        capsule.salt = urlsafe_b64decode(json_data["salt"])
        capsule.data_secret_bytes = data_secret.encode("utf-8")
        capsule.derived_key = derive_key(capsule.salt, capsule.data_secret_bytes, 32)
        capsule.hash_proof = capsule.metadata["hash_proof"]
        return capsule

    def decrypt_data(self) -> str:
        """
        Decrypts and returns the original data.

        Returns:
            The decrypted original data as a string.

        Raises:
            InvalidTag: If the data has been tampered with.
        """
        return decrypt_aes_gcm(
            self.derived_key, self.encrypted_data, self.nonce, self.tag
        ).decode("utf-8")

    def verify_hash_proof(self) -> bool:
        """
        Verifies the integrity of the capsule by checking the hash proof.

        Returns:
            True if the hash proof is valid, False otherwise.
        """
        try:
            decrypted_data = self.decrypt_data()
            return sha256_hash(decrypted_data.encode("utf-8")) == self.hash_proof
        except InvalidTag as e:
            print(f"Error during hash proof verification: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during hash proof verification: {e}")
            return False
