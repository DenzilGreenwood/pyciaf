"""
Deferred LCM Processing System
=============================

This module implements deferred Lazy Capsule Materialization for dramatically
improved inference performance while maintaining full audit trail capabilities.
"""

import asyncio
import json
import time
import hashlib
import threading
import uuid
from queue import Queue, Empty
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
import os
from pathlib import Path

@dataclass
class LightweightReceipt:
    """Minimal receipt stored during fast inference"""
    receipt_id: str
    timestamp: str
    model_ref: str
    model_version: str
    request_id: str
    input_hash: str
    output_hash: str
    input_commitment: str
    output_commitment: str
    raw_input: Optional[str] = None  # Can be encrypted for security
    raw_output: Optional[str] = None  # Can be encrypted for security
    priority: str = "normal"
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'LightweightReceipt':
        """Create from dictionary"""
        return cls(**data)

class ReceiptQueue:
    """Persistent queue for lightweight receipts"""
    
    def __init__(self, storage_dir: str = "deferred_lcm_queue"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.memory_queue = Queue()
        self.queue_file = self.storage_dir / "receipt_queue.jsonl"
        self._load_persisted_receipts()
        
    def _load_persisted_receipts(self):
        """Load any persisted receipts on startup"""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'r') as f:
                    for line in f:
                        receipt_data = json.loads(line.strip())
                        receipt = LightweightReceipt.from_dict(receipt_data)
                        self.memory_queue.put(receipt)
                # Clear the file after loading
                self.queue_file.unlink()
            except Exception as e:
                print(f"⚠️ Error loading persisted receipts: {e}")

    def put(self, receipt: LightweightReceipt):
        """Add receipt to queue"""
        self.memory_queue.put(receipt)
        
    def get(self, timeout: Optional[float] = None) -> Optional[LightweightReceipt]:
        """Get receipt from queue"""
        try:
            return self.memory_queue.get(timeout=timeout)
        except Empty:
            return None
            
    def get_batch(self, max_size: int, timeout: float = 0.1) -> List[LightweightReceipt]:
        """Get a batch of receipts"""
        batch = []
        for _ in range(max_size):
            receipt = self.get(timeout=timeout)
            if receipt is None:
                break
            batch.append(receipt)
        return batch
        
    def size(self) -> int:
        """Get queue size"""
        return self.memory_queue.qsize()
        
    def persist_queue(self):
        """Persist current queue to disk"""
        try:
            receipts_to_persist = []
            # Drain the queue temporarily
            while not self.memory_queue.empty():
                try:
                    receipt = self.memory_queue.get_nowait()
                    receipts_to_persist.append(receipt)
                except Empty:
                    break
                    
            # Write to file
            if receipts_to_persist:
                with open(self.queue_file, 'w') as f:
                    for receipt in receipts_to_persist:
                        f.write(json.dumps(receipt.to_dict()) + '\n')
                        
                # Put them back in queue
                for receipt in receipts_to_persist:
                    self.memory_queue.put(receipt)
                    
            print(f"📀 Persisted {len(receipts_to_persist)} receipts to disk")
        except Exception as e:
            print(f"⚠️ Error persisting receipts: {e}")

class DeferredLCMProcessor:
    """Background processor for converting lightweight receipts to full LCM"""
    
    def __init__(self, 
                 batch_size: int = 50,
                 processing_interval: float = 2.0,
                 storage_dir: str = "deferred_lcm_storage",
                 max_queue_size: int = 10000):
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.max_queue_size = max_queue_size
        
        # Core components
        self.receipt_queue = ReceiptQueue()
        self.processing_thread = None
        self.running = False
        self.stats = {
            'total_processed': 0,
            'total_batches': 0,
            'average_batch_time': 0.0,
            'queue_overflows': 0
        }
        
        # Audit trail storage
        self.audit_storage = self.storage_dir / "audit_trails"
        self.audit_storage.mkdir(exist_ok=True)
        
    def start_background_processing(self):
        """Start background LCM processing thread"""
        if self.running:
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
        print("🚀 Deferred LCM processor started")
        
    def stop_background_processing(self):
        """Stop background processing gracefully"""
        if not self.running:
            return
            
        print("🛑 Stopping deferred LCM processor...")
        self.running = False
        
        # Persist any remaining receipts
        self.receipt_queue.persist_queue()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=10)
            
        print("✅ Deferred LCM processor stopped")
        
    def add_receipt(self, receipt: LightweightReceipt) -> bool:
        """Add receipt to processing queue (fast operation)"""
        if self.receipt_queue.size() >= self.max_queue_size:
            self.stats['queue_overflows'] += 1
            print(f"⚠️ Queue overflow! Size: {self.receipt_queue.size()}")
            return False
            
        self.receipt_queue.put(receipt)
        return True
        
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        stats = self.stats.copy()
        stats['queue_size'] = self.receipt_queue.size()
        stats['is_running'] = self.running
        return stats
        
    def _process_loop(self):
        """Main background processing loop"""
        print("🔄 Background LCM processing loop started")
        
        while self.running:
            try:
                # Get batch of receipts
                batch = self.receipt_queue.get_batch(self.batch_size, timeout=1.0)
                
                if batch:
                    self._process_batch(batch)
                    
                time.sleep(self.processing_interval)
                
            except Exception as e:
                print(f"❌ Error in processing loop: {e}")
                time.sleep(5.0)  # Wait before retrying
                
        print("🔄 Background LCM processing loop stopped")
        
    def _process_batch(self, receipts: List[LightweightReceipt]):
        """Process a batch of receipts into full LCM"""
        start_time = time.time()
        
        print(f"🔄 Processing batch of {len(receipts)} receipts...")
        
        full_receipts = []
        for receipt in receipts:
            try:
                full_receipt = self._materialize_full_receipt(receipt)
                full_receipts.append(full_receipt)
            except Exception as e:
                print(f"❌ Error materializing receipt {receipt.receipt_id}: {e}")
                
        # Store batch in audit trail
        if full_receipts:
            self._store_audit_batch(full_receipts)
            
        # Update statistics
        batch_time = time.time() - start_time
        self.stats['total_processed'] += len(full_receipts)
        self.stats['total_batches'] += 1
        self.stats['average_batch_time'] = (
            (self.stats['average_batch_time'] * (self.stats['total_batches'] - 1) + batch_time) / 
            self.stats['total_batches']
        )
        
        print(f"✅ Batch processing complete: {len(full_receipts)} receipts in {batch_time:.3f}s")
        
    def _materialize_full_receipt(self, light_receipt: LightweightReceipt) -> Dict:
        """Convert lightweight receipt to full LCM receipt"""
        # This simulates the heavy cryptographic work that was previously
        # done during inference - now done in background
        
        # Generate additional cryptographic digests
        receipt_content = f"{light_receipt.receipt_id}{light_receipt.timestamp}{light_receipt.input_hash}{light_receipt.output_hash}"
        receipt_digest = hashlib.sha256(receipt_content.encode()).hexdigest()
        
        connections_content = f"{light_receipt.model_ref}{receipt_digest}"
        connections_digest = hashlib.sha256(connections_content.encode()).hexdigest()
        
        # Create full receipt structure
        full_receipt = {
            "receipt_id": light_receipt.receipt_id,
            "model_anchor_ref": light_receipt.model_ref,
            "deployment_anchor_ref": "Production_Model",
            "request_id": light_receipt.request_id,
            "timestamp": light_receipt.timestamp,
            "receipt_digest": receipt_digest,
            "connections_digest": connections_digest,
            "anchor_id": f"r_{receipt_digest[:8]}...",
            "input_commitment": {
                "commitment_type": "CommitmentType.SALTED",
                "commitment_value": light_receipt.input_commitment
            },
            "output_commitment": {
                "commitment_type": "CommitmentType.SALTED",
                "commitment_value": light_receipt.output_commitment
            },
            "metadata": light_receipt.metadata or {},
            "priority": light_receipt.priority,
            "materialization_timestamp": datetime.now().isoformat(),
            "model_version": light_receipt.model_version
        }
        
        return full_receipt
        
    def _store_audit_batch(self, receipts: List[Dict]):
        """Store a batch of full receipts in audit trail"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = self.audit_storage / f"audit_batch_{timestamp}_{uuid.uuid4().hex[:8]}.json"
        
        audit_data = {
            "batch_timestamp": datetime.now().isoformat(),
            "batch_size": len(receipts),
            "receipts": receipts
        }
        
        try:
            with open(batch_file, 'w') as f:
                json.dump(audit_data, f, indent=2)
            print(f"📁 Stored audit batch: {batch_file.name}")
        except Exception as e:
            print(f"❌ Error storing audit batch: {e}")

class ReceiptHasher:
    """Utility class for creating cryptographic hashes and commitments"""
    
    @staticmethod
    def hash_data(data: Any) -> str:
        """Create SHA256 hash of data"""
        data_str = str(data) if not isinstance(data, str) else data
        return hashlib.sha256(data_str.encode()).hexdigest()
        
    @staticmethod
    def create_commitment(data: Any, salt: Optional[str] = None) -> str:
        """Create salted commitment for data"""
        if salt is None:
            salt = uuid.uuid4().hex
        commitment_data = f"{data}{salt}"
        return hashlib.sha256(commitment_data.encode()).hexdigest()
        
    @staticmethod
    def generate_receipt_id() -> str:
        """Generate unique receipt ID"""
        return hashlib.sha256(f"{time.time()}{uuid.uuid4()}".encode()).hexdigest()

if __name__ == "__main__":
    # Simple test of the deferred LCM system
    processor = DeferredLCMProcessor(batch_size=5, processing_interval=1.0)
    processor.start_background_processing()
    
    # Add some test receipts
    for i in range(10):
        receipt = LightweightReceipt(
            receipt_id=ReceiptHasher.generate_receipt_id(),
            timestamp=datetime.now().isoformat(),
            model_ref="test_model_hash",
            model_version="1.0.0-test",
            request_id=f"req_{i}",
            input_hash=ReceiptHasher.hash_data(f"input_{i}"),
            output_hash=ReceiptHasher.hash_data(f"output_{i}"),
            input_commitment=ReceiptHasher.create_commitment(f"input_{i}"),
            output_commitment=ReceiptHasher.create_commitment(f"output_{i}"),
            metadata={"test": True}
        )
        processor.add_receipt(receipt)
        
    # Let it process
    time.sleep(5)
    
    # Show stats
    stats = processor.get_stats()
    print(f"📊 Processing stats: {stats}")
    
    processor.stop_background_processing()