"""
Merkle tree implementation for tamper-evident data integrity verification.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

from .crypto import sha256_hash

class MerkleTree:
    """Deterministic Merkle tree with left/right proofs and caches."""

    def __init__(self, leaves: list[str]):
        if not leaves:
            raise ValueError("Merkle tree must have at least one leaf.")
        self.leaves = leaves
        self.tree = self._build_tree(leaves)
        self.root = self.tree[-1][0] if self.tree else None

        self._proof_cache: dict[str, list[tuple[str, str]]] = {}
        self._verification_cache: dict[tuple[str, str], bool] = {}

        self._proof_cache_hits = 0
        self._proof_cache_misses = 0
        self._verification_cache_hits = 0
        self._verification_cache_misses = 0

    def _hash_pair(self, h1: str, h2: str) -> str:
        return sha256_hash(bytes.fromhex(h1) + bytes.fromhex(h2))

    def _build_tree(self, leaves: list[str]) -> list[list[str]]:
        tree = [leaves]
        level = leaves
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level), 2):
                a = level[i]
                b = level[i + 1] if i + 1 < len(level) else a
                nxt.append(self._hash_pair(a, b))
            tree.append(nxt)
            level = nxt
        return tree

    def get_root(self) -> str:
        return self.root

    def get_proof(self, leaf_hash: str) -> list[tuple[str, str]]:
        if leaf_hash in self._proof_cache:
            self._proof_cache_hits += 1
            return self._proof_cache[leaf_hash]
        self._proof_cache_misses += 1

        try:
            idx = self.leaves.index(leaf_hash)
        except ValueError:
            self._proof_cache[leaf_hash] = []
            return []

        if len(self.leaves) == 1:
            self._proof_cache[leaf_hash] = []
            return []

        proof: list[tuple[str, str]] = []
        current_index = idx

        for level in self.tree[:-1]:
            is_right = current_index % 2 != 0
            sib_idx = current_index - 1 if is_right else current_index + 1
            sibling_hash = level[sib_idx] if sib_idx < len(level) else level[current_index]
            pos = "left" if is_right else "right"
            proof.append((sibling_hash, pos))
            current_index //= 2

        self._proof_cache[leaf_hash] = proof
        return proof

    @staticmethod
    def verify_proof(leaf_hash: str, root_hash: str, proof: list[tuple[str, str]]) -> bool:
        cur = leaf_hash
        if not proof and cur == root_hash:
            return True
        for sib, pos in proof:
            if pos == "left":
                cur = sha256_hash(bytes.fromhex(sib) + bytes.fromhex(cur))
            else:
                cur = sha256_hash(bytes.fromhex(cur) + bytes.fromhex(sib))
        return cur == root_hash

    def verify_proof_cached(self, leaf_hash: str, root_hash: str | None = None) -> bool:
        root = root_hash or self.root
        key = (leaf_hash, root)
        if key in self._verification_cache:
            self._verification_cache_hits += 1
            return self._verification_cache[key]
        self._verification_cache_misses += 1
        proof = self.get_proof(leaf_hash)
        res = self.verify_proof(leaf_hash, root, proof)
        self._verification_cache[key] = res
        return res

    def clear_cache(self) -> None:
        self._proof_cache.clear()
        self._verification_cache.clear()
        self._proof_cache_hits = 0
        self._proof_cache_misses = 0
        self._verification_cache_hits = 0
        self._verification_cache_misses = 0

    def get_cache_stats(self) -> dict[str, int]:
        return {
            "proof_cache_size": len(self._proof_cache),
            "proof_cache_hits": self._proof_cache_hits,
            "proof_cache_misses": self._proof_cache_misses,
            "verification_cache_size": len(self._verification_cache),
            "verification_cache_hits": self._verification_cache_hits,
            "verification_cache_misses": self._verification_cache_misses,
            "total_leaves": len(self.leaves),
        }
