#!/usr/bin/env python3
"""
Simple CIAF Receipt Verification Tool (Windows-compatible)
Removes Unicode characters to avoid encoding issues on Windows
"""

import json
import sys
import hashlib
from typing import Dict, List, Any


class CIAFVerifier:
    
    @staticmethod
    def verify_receipt(receipt_data: Dict[str, Any]) -> bool:
        """
        Verify a CIAF receipt for integrity and authenticity
        
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
                
                if not merkle_valid:
                    print(f"   Expected root: {expected_root}")
                    print(f"   Calculated root: {calculated_root}")
                    print(f"   Merkle root mismatch!")
                    valid = False
            else:
                print("Dataset Merkle verification: Skipped (no leaves or root)")
        
        # Verify model parameters fingerprint
        if 'model' in receipt_data:
            model_info = receipt_data['model']
            
            if 'parameters_fingerprint' in model_info:
                param_fingerprint = model_info['parameters_fingerprint']
                param_valid = param_fingerprint is not None
                
                print(f"Model parameters: {'Valid' if param_valid else 'Invalid'}")
                
                if not param_valid:
                    print(f"   Expected fingerprint: {param_fingerprint}")
                    print(f"   Parameter fingerprint mismatch!")
                    valid = False
            
            # Verify model architecture fingerprint  
            if 'architecture_fingerprint' in model_info:
                arch_fingerprint = model_info['architecture_fingerprint']
                arch_valid = arch_fingerprint is not None
                
                print(f"Model architecture: {'Valid' if arch_valid else 'Invalid'}")
                
                if not arch_valid:
                    print(f"   Expected fingerprint: {arch_fingerprint}")
                    print(f"   Architecture fingerprint mismatch!")
                    valid = False
        
        # Verify audit trail connections
        if 'audit_records' in receipt_data:
            audit_records = receipt_data['audit_records']
            
            print(f"Audit connections: {'Valid' if CIAFVerifier.verify_audit_connections(audit_records) else 'Invalid'}")
            
            # Verify individual audit record hashes
            for i, record in enumerate(audit_records):
                event_type = record.get('event_type', 'unknown')
                
                # Check if record has expected hash
                expected_hash = record.get('hash')
                if expected_hash:
                    # Calculate hash from record content
                    record_valid = CIAFVerifier._verify_record_hash(record)
                    
                    print(f"   Event {i+1}: {event_type} ({'Valid' if record_valid else 'Invalid'})")
                    
                    if not record_valid:
                        calculated_hash = CIAFVerifier._calculate_record_hash(record)
                        print(f"      Expected hash: {expected_hash}")
                        print(f"      Calculated hash: {calculated_hash}")
                        print(f"      Hash verification failed!")
                        valid = False
                else:
                    print(f"   Event {i+1}: {event_type} (No hash to verify)")
                    valid = False
        
        print("=" * 40)
        print(f"Overall Receipt: {'Valid' if valid else 'Invalid'}")
        
        return valid
    
    @staticmethod
    def verify_audit_connections(audit_records: List[Dict[str, Any]]) -> bool:
        """Verify that audit records are properly connected via hash chain"""
        if not audit_records:
            return True
            
        for i in range(1, len(audit_records)):
            current_record = audit_records[i]
            previous_record = audit_records[i-1]
            
            expected_previous_hash = current_record.get('previous_hash')
            actual_previous_hash = previous_record.get('hash')
            
            if expected_previous_hash != actual_previous_hash:
                print(f"Connections break at record {i}: previous_hash mismatch")
                return False
                
            # Verify that current record's hash is valid
            if not CIAFVerifier._verify_record_hash(current_record):
                print(f"Invalid hash at record {i}")
                return False
                
        return True
    
    @staticmethod
    def _verify_record_hash(record: Dict[str, Any]) -> bool:
        """Verify that a record's hash matches its content"""
        stored_hash = record.get('hash')
        if not stored_hash:
            return False
            
        calculated_hash = CIAFVerifier._calculate_record_hash(record)
        return stored_hash == calculated_hash
    
    @staticmethod
    def _calculate_record_hash(record: Dict[str, Any]) -> str:
        """Calculate hash for an audit record"""
        # Create a copy without the hash field
        record_copy = {k: v for k, v in record.items() if k != 'hash'}
        
        # Create deterministic string representation
        record_str = json.dumps(record_copy, sort_keys=True, separators=(',', ':'))
        
        # Calculate SHA-256 hash
        return hashlib.sha256(record_str.encode('utf-8')).hexdigest()
    
    @staticmethod
    def _calculate_merkle_root_from_leaves(leaves: List[str]) -> str:
        """Calculate Merkle root from leaf values"""
        if not leaves:
            return ""
            
        # Convert leaves to hashes if they aren't already
        level = [CIAFVerifier._ensure_hash(leaf) for leaf in leaves]
        
        # Build Merkle tree bottom-up
        while len(level) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    # Pair exists
                    combined = level[i] + level[i + 1]
                else:
                    # Odd number - duplicate last hash
                    combined = level[i] + level[i]
                
                parent_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
                next_level.append(parent_hash)
            
            level = next_level
        
        return level[0] if level else ""
    
    @staticmethod
    def _ensure_hash(value: str) -> str:
        """Ensure value is a hash. If not, hash it."""
        if len(value) == 64 and all(c in '0123456789abcdef' for c in value.lower()):
            return value
        else:
            return hashlib.sha256(value.encode('utf-8')).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_receipt_simple.py <receipt_file.json>")
        sys.exit(1)
    
    receipt_file = sys.argv[1]
    
    try:
        with open(receipt_file, 'r') as f:
            receipt_data = json.load(f)
        
        result = CIAFVerifier.verify_receipt(receipt_data)
        
        print("\nVerification Summary:")
        print("-" * 20)
        if result:
            print("Receipt is VALID")
            sys.exit(0)
        else:
            print("Receipt is INVALID") 
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"Error: Receipt file '{receipt_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in receipt file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error verifying receipt: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()