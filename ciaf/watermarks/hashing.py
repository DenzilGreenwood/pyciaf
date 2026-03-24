"""
CIAF Watermarking - Hashing Utilities

Implements multiple hashing strategies for forensic artifact matching:
1. Exact hashing - SHA-256 for cryptographic proof
2. Normalized hashing - Resilient to formatting changes
3. Perceptual hashing - Resilient to content modifications
4. SimHash - Semantic similarity for text

Created: 2026-03-24
Author: Denzil James Greenwood
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional, List
import base64


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of bytes (exact matching)."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Compute SHA-256 hash of text (exact matching)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text_for_forensics(text: str) -> str:
    """
    Normalize text for format-resilient matching.

    Removes variations in:
    - Whitespace (spaces, tabs, newlines)
    - Casing (lowercase)
    - Leading/trailing whitespace
    - Multiple spaces to single space

    This allows matching even if someone reformats the text.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalized_text_hash(text: str) -> str:
    """
    Compute normalized hash of text.

    Resilient to:
    - Whitespace changes
    - Case changes
    - Minor formatting variations

    NOT resilient to:
    - Content changes
    - Rewording
    - Paraphrasing
    """
    normalized = normalize_text_for_forensics(text)
    return sha256_text(normalized)


def strip_common_watermarks(text: str) -> str:
    """
    Strip common watermark patterns from text.

    Removes:
    - Footer watermarks
    - Header watermarks
    - Inline provenance tags

    This helps detect watermark removal by comparing
    suspect text against pre-watermark hash.
    """
    # Remove footer-style watermarks (---\nAI Provenance...)
    text = re.sub(r'\n+---+\n+AI Provenance.*$', '', text, flags=re.DOTALL | re.MULTILINE)

    # Remove header-style watermarks
    text = re.sub(r'^AI Provenance.*\n+---+\n+', '', text, flags=re.MULTILINE)

    # Remove inline tags like [AI Generated: ...]
    text = re.sub(r'\[AI Generated:.*?\]', '', text)

    # Remove verification URLs
    text = re.sub(r'Verify:\s*https?://[^\s]+', '', text)

    return text.strip()


def text_with_watermark_stripped_hash(text: str) -> str:
    """
    Hash text after stripping watermarks.

    Use case: Detect if suspect text matches original content
    even if watermark was removed.
    """
    stripped = strip_common_watermarks(text)
    normalized = normalize_text_for_forensics(stripped)
    return sha256_text(normalized)


# SimHash implementation for text similarity
class SimHash:
    """
    SimHash implementation for near-duplicate text detection.

    SimHash produces a fixed-size fingerprint where similar documents
    have similar fingerprints (low Hamming distance).

    Use for detecting:
    - Minor rewording
    - Small additions/deletions
    - Paraphrasing with similar content
    """

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization by word boundaries."""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    @staticmethod
    def _hash_token(token: str) -> int:
        """Hash a single token to 64-bit integer."""
        h = hashlib.md5(token.encode('utf-8')).digest()
        return int.from_bytes(h[:8], byteorder='big')

    @classmethod
    def compute(cls, text: str, hashbits: int = 64) -> str:
        """
        Compute SimHash fingerprint.

        Args:
            text: Input text
            hashbits: Size of hash (default 64 bits)

        Returns:
            Hex string representation of SimHash
        """
        tokens = cls._tokenize(text)

        if not tokens:
            return '0' * (hashbits // 4)

        # Initialize vector
        v = [0] * hashbits

        # Process each token
        for token in tokens:
            h = cls._hash_token(token)

            # Add/subtract from vector based on bit values
            for i in range(hashbits):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1

        # Generate fingerprint
        fingerprint = 0
        for i in range(hashbits):
            if v[i] > 0:
                fingerprint |= (1 << i)

        # Convert to hex string
        hex_len = hashbits // 4
        return format(fingerprint, f'0{hex_len}x')

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """
        Compute Hamming distance between two SimHashes.

        Lower distance = more similar content
        Typical thresholds:
        - 0-3: Near duplicates
        - 4-10: Similar content
        - 11-20: Somewhat related
        - >20: Different content
        """
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        xor = int1 ^ int2
        return bin(xor).count('1')


def simhash_text(text: str) -> str:
    """
    Compute SimHash fingerprint for text.

    Use for similarity detection across minor modifications.
    """
    return SimHash.compute(text)


def simhash_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between SimHashes.

    Returns:
        Number of differing bits (0 = identical, 64 = opposite)
    """
    return SimHash.hamming_distance(hash1, hash2)


# Perceptual hashing placeholder
# For production, integrate:
# - imagehash library for images (pHash, dHash, wHash)
# - chromaprint for audio
# - video fingerprinting libraries

def perceptual_hash_placeholder(data: bytes, algorithm: str = "phash") -> str:
    """
    Placeholder for perceptual hashing.

    For production:
    - Images: Use imagehash library (pHash, dHash, aHash, wHash)
    - Audio: Use chromaprint/AcoustID
    - Video: Use video fingerprinting (e.g., VQMT)

    Args:
        data: Binary data (image, audio, video)
        algorithm: Algorithm name

    Returns:
        Hex string of perceptual hash
    """
    # Placeholder: just return regular hash
    # TODO: Integrate proper perceptual hashing libraries
    return sha256_bytes(data)[:16]  # Truncated hash as placeholder


# MinHash implementation for document similarity
class MinHash:
    """
    MinHash for Jaccard similarity estimation.

    Good for:
    - Large documents
    - Fast similarity estimation
    - Detecting copied/plagiarized content

    Not needed for typical watermark use cases, but included for completeness.
    """

    @staticmethod
    def compute(text: str, num_perm: int = 128) -> List[int]:
        """
        Compute MinHash signatures.

        Args:
            text: Input text
            num_perm: Number of permutations (higher = more accurate)

        Returns:
            List of minimum hash values
        """
        tokens = set(re.findall(r'\w+', text.lower()))

        if not tokens:
            return [0] * num_perm

        # Simplified MinHash using Python's hash function
        # For production, use `datasketch` library
        signature = []
        for i in range(num_perm):
            min_hash = min(hash(token + str(i)) & 0xFFFFFFFF for token in tokens)
            signature.append(min_hash)

        return signature

    @staticmethod
    def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
        """
        Estimate Jaccard similarity from MinHash signatures.

        Returns:
            Similarity score (0.0 = different, 1.0 = identical)
        """
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have same length")

        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


def minhash_text(text: str, num_perm: int = 128) -> str:
    """
    Compute MinHash signature for text.

    Returns base64-encoded signature.
    """
    signature = MinHash.compute(text, num_perm)
    # Encode as bytes then base64
    sig_bytes = b''.join(sig.to_bytes(4, 'big') for sig in signature)
    return base64.b64encode(sig_bytes).decode('ascii')


def minhash_similarity(hash1: str, hash2: str) -> float:
    """
    Compute Jaccard similarity from MinHash strings.

    Returns:
        Similarity score (0.0-1.0)
    """
    # Decode base64
    sig1_bytes = base64.b64decode(hash1)
    sig2_bytes = base64.b64decode(hash2)

    # Convert back to list of ints
    sig1 = [int.from_bytes(sig1_bytes[i:i+4], 'big') for i in range(0, len(sig1_bytes), 4)]
    sig2 = [int.from_bytes(sig2_bytes[i:i+4], 'big') for i in range(0, len(sig2_bytes), 4)]

    return MinHash.jaccard_similarity(sig1, sig2)


__all__ = [
    # Exact hashing
    "sha256_bytes",
    "sha256_text",

    # Normalized hashing
    "normalize_text_for_forensics",
    "normalized_text_hash",
    "strip_common_watermarks",
    "text_with_watermark_stripped_hash",

    # Similarity hashing
    "SimHash",
    "simhash_text",
    "simhash_distance",

    # MinHash
    "MinHash",
    "minhash_text",
    "minhash_similarity",

    # Perceptual (placeholder)
    "perceptual_hash_placeholder",
]
