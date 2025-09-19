#!/usr/bin/env python3
"""
Deferred LCM Receipt Verification Demo

This script demonstrates the complete workflow of:
1. Extracting receipts from deferred LCM system
2. Converting to verifiable format
3. Running independent verification

Shows how the CIAF framework maintains audit integrity even
with deferred lifecycle management.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and return the result."""
    print(f"\n[INFO] {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {e}")
        print(f"Stderr: {e.stderr}")
        return False, e.stderr

def main():
    """Main demonstration workflow."""
    
    print("CIAF Deferred LCM Receipt Verification Demo")
    print("=" * 60)
    print(f"Demo started at: {datetime.now().isoformat()}")
    
    # Step 1: Check if we have deferred LCM audit trails
    audit_dir = Path("../deferred_lcm_storage/audit_trails")
    if not audit_dir.exists():
        print("\n[ERROR] No deferred LCM audit trails found.")
        print("[INFO] Please run the deferred LCM benchmark first:")
        print("   python deferred_lcm_benchmark.py")
        return
        
    audit_files = list(audit_dir.glob("*.json"))
    if not audit_files:
        print(f"\n[ERROR] No audit files found in {audit_dir}")
        return
        
    print(f"\n[SUCCESS] Found {len(audit_files)} audit trail files")
    latest_file = max(audit_files, key=lambda x: x.stat().st_mtime)
    print(f"[INFO] Using latest file: {latest_file.name}")
    
    # Step 2: Extract and convert receipt
    success, output = run_command(
        ["python", "extract_receipt_for_verification.py"],
        "Extracting CIAF receipt from deferred LCM batch"
    )
    
    if not success:
        print("[ERROR] Failed to extract receipt")
        return
        
    # Step 3: Verify the receipt
    success, output = run_command(
        ["python", "verify_receipt_simple.py", "extracted_ciaf_receipt_for_verification.json"],
        "Verifying extracted receipt with independent verifier"
    )
    
    # Step 4: Show receipt details
    if Path("extracted_ciaf_receipt_for_verification.json").exists():
        print("\nReceipt Details:")
        print("-" * 40)
        
        with open("extracted_ciaf_receipt_for_verification.json", 'r') as f:
            receipt = json.load(f)
            
        print(f"Receipt ID: {receipt['receipt_id']}")
        print(f"Timestamp: {receipt['timestamp']}")
        print(f"Dataset ID: {receipt['dataset']['dataset_id']}")
        print(f"Model Name: {receipt['model']['model_name']}")
        print(f"Audit Events: {len(receipt['audit_connections'])}")
        
        # Show deferred LCM specific metadata
        deferred_meta = receipt.get('deferred_lcm_metadata', {})
        print(f"Deferred Mode: {deferred_meta.get('deferred', 'Unknown')}")
        print(f"Priority: {deferred_meta.get('priority', 'Unknown')}")
        
        if deferred_meta.get('materialization_timestamp'):
            print(f"Materialized: {deferred_meta['materialization_timestamp']}")
            
    # Step 5: Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if success:
        print("[SUCCESS] Receipt verification: PASSED")
        print("[INFO] Audit integrity: MAINTAINED")
        print("[INFO] Deferred LCM compatibility: CONFIRMED")
        print("\nThis demonstrates that:")
        print("   - Deferred LCM receipts maintain full audit integrity")
        print("   - Independent verification tools work seamlessly") 
        print("   - CIAF compliance is preserved across all processing modes")
    else:
        print("[ERROR] Receipt verification: FAILED")
        print("[WARNING] Audit integrity: COMPROMISED")
        
    print(f"\nDemo completed at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()