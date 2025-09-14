#!/usr/bin/env python3
"""
CIAF Receipt Verifier

A standalone tool for verifying CIAF receipts and audit trails.
This verifier can independently validate:
- Dataset split leaves → Merkle root
- Model parameter/architecture fingerprints
- Audit chain linkage (hash-chaining event IDs)

This tool demonstrates that CIAF produces verifiable artifacts that can be
validated independently of the main framework.

Usage:
    python tools/verify_receipt.py <receipt_file.json>
    python tools/verify_receipt.py --verify-merkle <data_file.json>
    python tools/verify_receipt.py --verify-audit-chain <audit_file.json>

Created: 2025-09-12
Author: Denzil James Greenwood
"""

import json
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional


class CIAFVerifier:
    """Independent CIAF receipt and audit trail verifier."""
    
    @staticmethod
    def sha256_hash(data: str) -> str:
        """Calculate SHA256 hash of string data."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    @staticmethod
    def verify_merkle_root(leaves: List[str], expected_root: str) -> bool:
        """
        Verify Merkle root calculation from leaves.
        
        Args:
            leaves: List of leaf hashes
            expected_root: Expected Merkle root hash
            
        Returns:
            True if calculated root matches expected root
        """
        if not leaves:
            return False
        
        if len(leaves) == 1:
            return leaves[0] == expected_root
        
        # Build Merkle tree bottom-up
        current_level = leaves[:]
        
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                
                # Hash concatenation of left and right
                combined = CIAFVerifier.sha256_hash(left + right)
                next_level.append(combined)
            
            current_level = next_level
        
        calculated_root = current_level[0]
        return calculated_root == expected_root
    
    @staticmethod
    def verify_parameter_fingerprint(parameters: Dict[str, Any], expected_fingerprint: str) -> bool:
        """
        Verify model parameter fingerprint.
        
        Args:
            parameters: Model parameters dictionary
            expected_fingerprint: Expected parameter fingerprint
            
        Returns:
            True if calculated fingerprint matches expected
        """
        # Sort parameters for deterministic hashing
        sorted_params = json.dumps(parameters, sort_keys=True, separators=(',', ':'))
        calculated_fingerprint = CIAFVerifier.sha256_hash(sorted_params)
        return calculated_fingerprint == expected_fingerprint
    
    @staticmethod
    def verify_audit_chain(audit_records: List[Dict[str, Any]]) -> bool:
        """
        Verify audit chain integrity.
        
        Args:
            audit_records: List of audit records with hash chaining
            
        Returns:
            True if audit chain is valid
        """
        if not audit_records:
            return False
        
        if len(audit_records) == 1:
            # Single record - verify its hash
            record = audit_records[0]
            return CIAFVerifier._verify_record_hash(record)
        
        # Verify chain linkage
        for i in range(1, len(audit_records)):
            current = audit_records[i]
            previous = audit_records[i - 1]
            
            # Verify current record's previous_hash matches previous record's hash
            if current.get('previous_hash') != previous.get('hash'):
                print(f"❌ Chain break at record {i}: previous_hash mismatch")
                return False
            
            # Verify current record's hash
            if not CIAFVerifier._verify_record_hash(current):
                print(f"❌ Invalid hash at record {i}")
                return False
        
        return True
    
    @staticmethod
    def _verify_record_hash(record: Dict[str, Any]) -> bool:
        """Verify individual audit record hash."""
        expected_hash = record.get('hash')
        if not expected_hash:
            return False
        
        # Reconstruct record data for hashing
        event_id = record.get('event_id', '')
        event_type = record.get('event_type', '')
        timestamp = record.get('timestamp', '')
        previous_hash = record.get('previous_hash', '0' * 64)
        
        data = f"{event_id}|{event_type}|{timestamp}|{previous_hash}"
        calculated_hash = CIAFVerifier.sha256_hash(data)
        
        return calculated_hash == expected_hash
    
    @staticmethod
    def verify_receipt(receipt_data: Dict[str, Any]) -> bool:
        """
        Verify complete CIAF receipt.
        
        Args:
            receipt_data: Complete receipt data
            
        Returns:
            True if receipt is valid
        """
        print("🔍 Verifying CIAF Receipt...")
        print("=" * 40)
        
        valid = True
        
        # Verify dataset Merkle root
        if 'dataset' in receipt_data:
            dataset_info = receipt_data['dataset']
            leaves = dataset_info.get('leaves', [])
            expected_root = dataset_info.get('merkle_root')
            
            if leaves and expected_root:
                merkle_valid = CIAFVerifier.verify_merkle_root(leaves, expected_root)
                print(f"📊 Dataset Merkle root: {'✅ Valid' if merkle_valid else '❌ Invalid'}")
                valid = valid and merkle_valid
            else:
                print("⚠️  Dataset Merkle data not found")
        
        # Verify model fingerprints
        if 'model' in receipt_data:
            model_info = receipt_data['model']
            
            # Parameters fingerprint
            params = model_info.get('parameters', {})
            param_fingerprint = model_info.get('parameter_fingerprint')
            if params and param_fingerprint:
                param_valid = CIAFVerifier.verify_parameter_fingerprint(params, param_fingerprint)
                print(f"🤖 Model parameters: {'✅ Valid' if param_valid else '❌ Invalid'}")
                valid = valid and param_valid
            
            # Architecture fingerprint
            arch = model_info.get('architecture', {})
            arch_fingerprint = model_info.get('architecture_fingerprint')
            if arch and arch_fingerprint:
                arch_valid = CIAFVerifier.verify_parameter_fingerprint(arch, arch_fingerprint)
                print(f"🏗️  Model architecture: {'✅ Valid' if arch_valid else '❌ Invalid'}")
                valid = valid and arch_valid
        
        # Verify audit chain
        if 'audit_chain' in receipt_data:
            audit_records = receipt_data['audit_chain']
            if audit_records:
                audit_valid = CIAFVerifier.verify_audit_chain(audit_records)
                print(f"📋 Audit chain: {'✅ Valid' if audit_valid else '❌ Invalid'}")
                valid = valid and audit_valid
            else:
                print("⚠️  Audit chain is empty")
        
        print("=" * 40)
        print(f"🎯 Overall Receipt: {'✅ VALID' if valid else '❌ INVALID'}")
        
        return valid


def create_sample_receipt() -> Dict[str, Any]:
    """Create a sample receipt for testing."""
    return {
        "receipt_id": "sample_001",
        "timestamp": "2025-09-12T10:00:00Z",
        "dataset": {
            "dataset_id": "demo_dataset",
            "leaves": [
                "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",  # hash of "hello"
                "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",  # hash of "foo"
            ],
            "merkle_root": "c6c6ba3b82ba9c63b3f01e4bfeef8a2e2a5b7a9f3f3f2c6c6b2a82ba9c63b3f0"
        },
        "model": {
            "model_name": "demo_model",
            "parameters": {"epochs": 3, "lr": 0.01},
            "parameter_fingerprint": "calculated_param_hash",
            "architecture": {"type": "logreg"},
            "architecture_fingerprint": "calculated_arch_hash"
        },
        "audit_chain": [
            {
                "event_id": "train_001",
                "event_type": "training_started",
                "timestamp": "2025-09-12T10:00:00Z",
                "previous_hash": "0" * 64,
                "hash": "calculated_hash_1"
            },
            {
                "event_id": "train_002",
                "event_type": "training_completed",
                "timestamp": "2025-09-12T10:30:00Z",
                "previous_hash": "calculated_hash_1",
                "hash": "calculated_hash_2"
            }
        ]
    }


def main():
    """Main verifier function."""
    parser = argparse.ArgumentParser(description="CIAF Receipt Verifier")
    parser.add_argument("receipt_file", nargs="?", help="Receipt JSON file to verify")
    parser.add_argument("--create-sample", action="store_true", help="Create sample receipt")
    parser.add_argument("--verify-merkle", help="Verify Merkle root from data file")
    parser.add_argument("--verify-audit-chain", help="Verify audit chain from file")
    
    args = parser.parse_args()
    
    if args.create_sample:
        sample = create_sample_receipt()
        print("📝 Sample receipt created:")
        print(json.dumps(sample, indent=2))
        return
    
    if args.verify_merkle:
        try:
            with open(args.verify_merkle, 'r') as f:
                data = json.load(f)
            
            leaves = data.get('leaves', [])
            expected_root = data.get('expected_root')
            
            result = CIAFVerifier.verify_merkle_root(leaves, expected_root)
            print(f"🌳 Merkle verification: {'✅ Valid' if result else '❌ Invalid'}")
            
        except Exception as e:
            print(f"❌ Error verifying Merkle root: {e}")
        return
    
    if args.verify_audit_chain:
        try:
            with open(args.verify_audit_chain, 'r') as f:
                data = json.load(f)
            
            audit_records = data.get('audit_records', [])
            result = CIAFVerifier.verify_audit_chain(audit_records)
            print(f"📋 Audit chain verification: {'✅ Valid' if result else '❌ Invalid'}")
            
        except Exception as e:
            print(f"❌ Error verifying audit chain: {e}")
        return
    
    if args.receipt_file:
        try:
            with open(args.receipt_file, 'r') as f:
                receipt_data = json.load(f)
            
            result = CIAFVerifier.verify_receipt(receipt_data)
            sys.exit(0 if result else 1)
            
        except FileNotFoundError:
            print(f"❌ File not found: {args.receipt_file}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in file: {args.receipt_file}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error verifying receipt: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()