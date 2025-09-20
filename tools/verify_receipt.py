#!/usr/bin/env python3
"""
CIAF Receipt Verifier

A standalone tool for verifying CIAF receipts and audit trails.
This verifier can independently validate:
- Dataset split leaves -> Merkle root
- Model parameter/architecture fingerprints
- Audit connections linkage (hash-connecting event IDs)

This tool demonstrates that CIAF produces verifiable artifacts that can be
validated independently of the main framework.

Usage:
    python tools/verify_receipt.py <receipt_file.json>
    python tools/verify_receipt.py --verify-merkle <data_file.json>
    python tools/verify_receipt.py --verify-audit-connections <audit_file.json>

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
    def _calculate_merkle_root_from_leaves(leaves: List[str]) -> str:
        """
        Calculate Merkle root from leaves (internal helper).
        
        Args:
            leaves: List of leaf hashes
            
        Returns:
            Calculated Merkle root hash
        """
        if not leaves:
            return ""
        
        if len(leaves) == 1:
            return leaves[0]
        
        # Build Merkle tree bottom-up
        current_level = leaves[:]
        
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs of nodes
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                
                # Combine and hash
                combined = CIAFVerifier.sha256_hash(left + right)
                next_level.append(combined)
            
            current_level = next_level
        
        return current_level[0]
    
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
    def verify_audit_connections(audit_records: List[Dict[str, Any]]) -> bool:
        """
        Verify audit connections integrity.

        Args:
            audit_records: List of audit records with hash connecting

        Returns:
            True if audit connections is valid
        """
        if not audit_records:
            return False
        
        if len(audit_records) == 1:
            # Single record - verify its hash
            record = audit_records[0]
            return CIAFVerifier._verify_record_hash(record)
        
        # Verify connections linkage
        for i in range(1, len(audit_records)):
            current = audit_records[i]
            previous = audit_records[i - 1]
            
            # Verify current record's previous_hash matches previous record's hash
            if current.get('previous_hash') != previous.get('hash'):
                print(f"[FAIL] Connections break at record {i}: previous_hash mismatch")
                return False
            
            # Verify current record's hash
            if not CIAFVerifier._verify_record_hash(current):
                print(f"[FAIL] Invalid hash at record {i}")
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
        print("Verifying CIAF Receipt...")
        print("=" * 40)
        
        valid = True
        
        # Verify dataset Merkle root
        if 'dataset' in receipt_data:
            dataset_info = receipt_data['dataset']
            leaves = dataset_info.get('leaves', [])
            expected_root = dataset_info.get('merkle_root')
            
            if leaves and expected_root:
                calculated_root = CIAFVerifier._calculate_merkle_root_from_leaves(leaves)
                merkle_valid = calculated_root == expected_root
                
                print(f"Dataset Merkle root: {'Valid' if merkle_valid else 'Invalid'}")
                print(f"   [DATASET] Dataset ID: {dataset_info.get('dataset_id', 'N/A')}")
                print(f"   [LEAF] Leaf count: {len(leaves)}")
                print(f"   [EXPECTED] Expected root: {expected_root}")
                print(f"   [CALCULATED] Calculated root: {calculated_root}")
                if not merkle_valid:
                    print(f"   [FAIL] Merkle root mismatch!")
                valid = valid and merkle_valid
            else:
                print("[WARNING] Dataset Merkle data not found")
        
        # Verify model fingerprints
        if 'model' in receipt_data:
            model_info = receipt_data['model']
            
            # Parameters fingerprint
            params = model_info.get('parameters', {})
            param_fingerprint = model_info.get('parameter_fingerprint')
            if params and param_fingerprint:
                sorted_params = json.dumps(params, sort_keys=True, separators=(',', ':'))
                calculated_param_fingerprint = CIAFVerifier.sha256_hash(sorted_params)
                param_valid = calculated_param_fingerprint == param_fingerprint
                
                print(f"[MODEL] Model parameters: {'[SUCCESS] Valid' if param_valid else '[FAIL] Invalid'}")
                print(f"   [MODEL] Model name: {model_info.get('model_name', 'N/A')}")
                print(f"   [PARAMS] Parameters: {params}")
                print(f"   [EXPECTED] Expected fingerprint: {param_fingerprint}")
                print(f"   [CALCULATED] Calculated fingerprint: {calculated_param_fingerprint}")
                if not param_valid:
                    print(f"   [FAIL] Parameter fingerprint mismatch!")
                valid = valid and param_valid
            
            # Architecture fingerprint
            arch = model_info.get('architecture', {})
            arch_fingerprint = model_info.get('architecture_fingerprint')
            if arch and arch_fingerprint:
                sorted_arch = json.dumps(arch, sort_keys=True, separators=(',', ':'))
                calculated_arch_fingerprint = CIAFVerifier.sha256_hash(sorted_arch)
                arch_valid = calculated_arch_fingerprint == arch_fingerprint
                
                print(f"[ARCHITECTURE] Model architecture: {'[SUCCESS] Valid' if arch_valid else '[FAIL] Invalid'}")
                print(f"   [ARCH] Architecture: {arch}")
                print(f"   [EXPECTED] Expected fingerprint: {arch_fingerprint}")
                print(f"   [CALCULATED] Calculated fingerprint: {calculated_arch_fingerprint}")
                if not arch_valid:
                    print(f"   [FAIL] Architecture fingerprint mismatch!")
                valid = valid and arch_valid
        
        # Verify audit connections
        if 'audit_connections' in receipt_data:
            audit_records = receipt_data['audit_connections']
            if audit_records:
                audit_valid = True
                print(f"[AUDIT] Audit connections: {'[SUCCESS] Valid' if CIAFVerifier.verify_audit_connections(audit_records) else '[FAIL] Invalid'}")
                print(f"   [COUNT] Event count: {len(audit_records)}")
                
                # Show details for each audit event
                for i, record in enumerate(audit_records):
                    event_id = record.get('event_id', 'N/A')
                    event_type = record.get('event_type', 'N/A')
                    timestamp = record.get('timestamp', 'N/A')
                    expected_hash = record.get('hash', 'N/A')
                    previous_hash = record.get('previous_hash', '0' * 64)
                    
                    # Calculate expected hash
                    data = f"{event_id}|{event_type}|{timestamp}|{previous_hash}"
                    calculated_hash = CIAFVerifier.sha256_hash(data)
                    record_valid = calculated_hash == expected_hash
                    
                    print(f"   [EVENT] Event {i+1}: {event_type} ({'[SUCCESS]' if record_valid else '[FAIL]'})")
                    print(f"     [ID] Event ID: {event_id}")
                    print(f"     [TIME] Timestamp: {timestamp}")
                    print(f"     [LINK] Previous hash: {previous_hash}")
                    print(f"     [EXPECTED] Expected hash: {expected_hash}")
                    print(f"     [CALCULATED] Calculated hash: {calculated_hash}")
                    
                    if not record_valid:
                        print(f"     [FAIL] Hash verification failed!")
                        audit_valid = False
                
                valid = valid and audit_valid
            else:
                print("[WARNING] Audit connections is empty")
        
        print("=" * 40)
        print(f"[OVERALL] Overall Receipt: {'[SUCCESS] VALID' if valid else '[FAIL] INVALID'}")
        
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
        "audit_connections": [
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
    parser.add_argument("--verify-audit-connections", help="Verify audit connections from file")
    
    args = parser.parse_args()
    
    if args.create_sample:
        sample = create_sample_receipt()
        print("[SAMPLE] Sample receipt created:")
        print(json.dumps(sample, indent=2))
        return
    
    if args.verify_merkle:
        try:
            with open(args.verify_merkle, 'r') as f:
                data = json.load(f)
            
            leaves = data.get('leaves', [])
            expected_root = data.get('expected_root')
            
            result = CIAFVerifier.verify_merkle_root(leaves, expected_root)
            print(f"[TREE] Merkle verification: {'[SUCCESS] Valid' if result else '[FAIL] Invalid'}")
            
        except Exception as e:
            print(f"[FAIL] Error verifying Merkle root: {e}")
        return
    
    if args.verify_audit_connections:
        try:
            with open(args.verify_audit_connections, 'r') as f:
                data = json.load(f)
            
            audit_records = data.get('audit_records', [])
            result = CIAFVerifier.verify_audit_connections(audit_records)
            print(f"[AUDIT] Audit connections verification: {'[SUCCESS] Valid' if result else '[FAIL] Invalid'}")
            
        except Exception as e:
            print(f"[FAIL] Error verifying audit connections: {e}")
        return
    
    if args.receipt_file:
        try:
            with open(args.receipt_file, 'r') as f:
                receipt_data = json.load(f)
            
            result = CIAFVerifier.verify_receipt(receipt_data)
            sys.exit(0 if result else 1)
            
        except FileNotFoundError:
            print(f"[FAIL] File not found: {args.receipt_file}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"[FAIL] Invalid JSON in file: {args.receipt_file}")
            sys.exit(1)
        except Exception as e:
            print(f"[FAIL] Error verifying receipt: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()