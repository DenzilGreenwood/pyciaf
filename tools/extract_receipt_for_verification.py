#!/usr/bin/env python3
"""
Extract and Convert CIAF Receipts for Verification

This script extracts receipts from our deferred LCM system and converts them
to the format expected by the verify_receipt.py tool.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path

def sha256_hash(data: str) -> str:
    """Calculate SHA256 hash of string data."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def calculate_merkle_root(leaves):
    """Calculate Merkle root from leaves."""
    if not leaves:
        return None
    
    if len(leaves) == 1:
        return leaves[0]
    
    current_level = leaves[:]
    
    while len(current_level) > 1:
        next_level = []
        
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else left
            combined = sha256_hash(left + right)
            next_level.append(combined)
        
        current_level = next_level
    
    return current_level[0]

def extract_ciaf_receipt_from_deferred_batch(batch_file: str):
    """Extract and convert a deferred LCM batch to verifiable receipt format."""
    
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
    
    receipts = batch_data.get('receipts', [])
    if not receipts:
        print("No receipts found in batch")
        return None
    
    # Take the first receipt as example
    receipt = receipts[0]
    
    # Create sample dataset leaves (in real scenario, these would come from training)
    sample_leaves = [
        sha256_hash("training_sample_1"),
        sha256_hash("training_sample_2"), 
        sha256_hash("training_sample_3"),
        sha256_hash("training_sample_4")
    ]
    
    merkle_root = calculate_merkle_root(sample_leaves)
    
    # Create model parameters (these would come from actual model)
    model_params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
    model_arch = {
        "architecture_type": "ensemble",
        "base_estimator": "decision_tree",
        "voting": "soft"
    }
    
    param_fingerprint = sha256_hash(json.dumps(model_params, sort_keys=True, separators=(',', ':')))
    arch_fingerprint = sha256_hash(json.dumps(model_arch, sort_keys=True, separators=(',', ':')))
    
    # Create audit chain from the receipt
    audit_events = []
    
    # First event - training started
    event1 = {
        "event_id": "training_start",
        "event_type": "training_started", 
        "timestamp": "2025-09-19T10:00:00Z",
        "previous_hash": "0" * 64
    }
    event1_data = f"{event1['event_id']}|{event1['event_type']}|{event1['timestamp']}|{event1['previous_hash']}"
    event1["hash"] = sha256_hash(event1_data)
    audit_events.append(event1)
    
    # Second event - inference
    event2 = {
        "event_id": receipt.get('request_id', 'inference_001'),
        "event_type": "inference_executed",
        "timestamp": receipt.get('timestamp', '2025-09-19T10:49:32Z'),
        "previous_hash": event1["hash"]
    }
    event2_data = f"{event2['event_id']}|{event2['event_type']}|{event2['timestamp']}|{event2['previous_hash']}"
    event2["hash"] = sha256_hash(event2_data)
    audit_events.append(event2)
    
    # Create the complete receipt
    ciaf_receipt = {
        "receipt_id": receipt.get('receipt_id', 'unknown'),
        "timestamp": receipt.get('timestamp', datetime.now().isoformat()),
        "dataset": {
            "dataset_id": "deferred_lcm_demo_dataset",
            "leaves": sample_leaves,
            "merkle_root": merkle_root
        },
        "model": {
            "model_name": "Enhanced_CIAF_Demo_Model",
            "parameters": model_params,
            "parameter_fingerprint": param_fingerprint,
            "architecture": model_arch,
            "architecture_fingerprint": arch_fingerprint
        },
        "audit_connections": audit_events,
        "deferred_lcm_metadata": {
            "priority": receipt.get('priority', 'normal'),
            "materialization_timestamp": receipt.get('materialization_timestamp'),
            "deferred": receipt.get('metadata', {}).get('deferred', False)
        }
    }
    
    return ciaf_receipt

def main():
    """Main function to extract and save a verifiable receipt."""
    
    batch_file = "../deferred_lcm_storage/audit_trails/audit_batch_20250919_104932_7805759d.json"
    
    if not Path(batch_file).exists():
        print(f"[ERROR] Batch file not found: {batch_file}")
        return
        
    print("[INFO] Extracting CIAF receipt from deferred LCM batch...")
    
    receipt = extract_ciaf_receipt_from_deferred_batch(batch_file)
    
    if receipt:
        output_file = "extracted_ciaf_receipt_for_verification.json"
        
        with open(output_file, 'w') as f:
            json.dump(receipt, f, indent=2)
            
        print(f"[SUCCESS] Receipt extracted and saved to: {output_file}")
        print(f"[INFO] Receipt ID: {receipt['receipt_id']}")
        print(f"[INFO] Timestamp: {receipt['timestamp']}")
        print(f"[INFO] Deferred LCM: {receipt['deferred_lcm_metadata']['deferred']}")
        
        return output_file
    else:
        print("[ERROR] Failed to extract receipt")
        return None

if __name__ == "__main__":
    main()