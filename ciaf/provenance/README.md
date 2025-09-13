# CIAF Provenance System

The provenance package provides comprehensive data lineage tracking and model training verification through cryptographic capsules and verifiable snapshots.

## Overview

The provenance system ensures complete auditability of AI model training while protecting sensitive data:

- **Provenance Capsules** — Cryptographic containers for sensitive training data
- **Training Snapshots** — Tamper-evident records of model training sessions
- **Model Aggregation Anchors (MAA)** — Authentication system for training data
- **Data Lineage** — Complete traceability from raw data to trained models
- **Privacy Protection** — HIPAA-compliant data minimization and encryption
- **Integrity Verification** — Merkle tree-based proof systems

## Components

### ProvenanceCapsule (`capsules.py`)

Cryptographic containers that encapsulate sensitive training data while maintaining verifiable lineage.

**Key Features:**
- **Encryption Protection** — AES-256-GCM encryption for sensitive data
- **Hash Proofs** — SHA-256 integrity verification without data exposure
- **Metadata Preservation** — Non-sensitive lineage information
- **HIPAA Compliance** — Data minimization and consent management
- **Tamper Detection** — Cryptographic verification of data integrity

**Usage Example:**
```python
from ciaf.provenance import ProvenanceCapsule
from datetime import datetime

# Create provenance capsule for sensitive medical data
patient_data = "Patient ID: 12345, Age: 45, Symptoms: chest pain, dyspnea"
metadata = {
    "source": "hospital_system_alpha",
    "consent_status": "explicit_consent_obtained",
    "data_classification": "PHI",
    "collection_date": "2025-09-01",
    "anonymization_level": "pseudonymized"
}
data_secret = "patient_12345_unique_secret"

# Encapsulate data
capsule = ProvenanceCapsule(
    original_data=patient_data,
    metadata=metadata,
    data_secret=data_secret
)

# Access verifiable metadata without exposing raw data
print(f"Hash proof: {capsule.hash_proof}")
print(f"Source: {capsule.metadata['source']}")
print(f"Consent status: {capsule.metadata['consent_status']}")

# Verify integrity
is_valid = capsule.verify_hash_proof()
print(f"Capsule integrity: {is_valid}")

# Serialize for storage
capsule_json = capsule.to_json()

# Later reconstruction
reconstructed_capsule = ProvenanceCapsule.from_json(
    capsule_json, 
    data_secret
)

# Decrypt only when authorized
if authorized_for_access:
    original_data = reconstructed_capsule.decrypt_data()
```

**Capsule Structure:**
```json
{
  "metadata": {
    "source": "hospital_system_alpha",
    "consent_status": "explicit_consent_obtained",
    "data_classification": "PHI",
    "collection_date": "2025-09-01",
    "hash_proof": "sha256:abc123...",
    "creation_timestamp": "2025-09-12T14:30:00.000Z"
  },
  "encrypted_data": "base64_encrypted_content",
  "nonce": "base64_nonce",
  "tag": "base64_authentication_tag",
  "salt": "base64_salt"
}
```

### TrainingSnapshot (`snapshots.py`)

Tamper-evident records of model training sessions with complete data provenance.

**Key Features:**
- **Training Verification** — Cryptographic proof of training data usage
- **Merkle Tree Integrity** — Efficient verification of large datasets
- **Parameter Recording** — Complete hyperparameter and configuration tracking
- **Temporal Anchoring** — Timestamp-based training session identification
- **Provenance Linking** — Direct connection to provenance capsules

**Usage Example:**
```python
from ciaf.provenance import TrainingSnapshot, ProvenanceCapsule

# Create multiple provenance capsules for training data
training_capsules = []
capsule_hashes = []

for i, patient_record in enumerate(training_dataset):
    capsule = ProvenanceCapsule(
        original_data=patient_record,
        metadata={
            "source": f"hospital_system_{i}",
            "consent_status": "explicit_consent_obtained",
            "data_classification": "PHI"
        },
        data_secret=f"patient_{i}_secret"
    )
    training_capsules.append(capsule)
    capsule_hashes.append(capsule.hash_proof)

# Define training parameters
training_params = {
    "algorithm": "deep_neural_network",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam",
    "loss_function": "cross_entropy",
    "regularization": "l2",
    "dropout_rate": 0.2,
    "validation_split": 0.2
}

# Create training snapshot
snapshot = TrainingSnapshot(
    model_version="diagnostic_model_v1.0",
    training_parameters=training_params,
    provenance_capsule_hashes=capsule_hashes
)

print(f"Training snapshot ID: {snapshot.snapshot_id}")
print(f"Merkle root: {snapshot.merkle_root_hash}")
print(f"Training data count: {len(snapshot.provenance_capsule_hashes)}")

# Verify specific data was used in training
data_hash = training_capsules[0].hash_proof
was_used = snapshot.verify_provenance(data_hash)
print(f"Data hash {data_hash[:16]}... was used in training: {was_used}")

# Serialize snapshot
snapshot_json = snapshot.to_json()

# Later reconstruction
restored_snapshot = TrainingSnapshot.from_json(snapshot_json)
```

**Snapshot Structure:**
```json
{
  "snapshot_id": "sha256:def456...",
  "model_version": "diagnostic_model_v1.0",
  "training_parameters": {
    "algorithm": "deep_neural_network",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
  },
  "provenance_capsule_hashes": [
    "sha256:abc123...",
    "sha256:def456...",
    "sha256:ghi789..."
  ],
  "timestamp": "2025-09-12T14:30:00.000Z",
  "merkle_root_hash": "merkle_root_sha256",
  "metadata": {
    "model_version": "diagnostic_model_v1.0",
    "timestamp": "2025-09-12T14:30:00.000Z",
    "merkle_root_hash": "merkle_root_sha256",
    "training_parameters": {...}
  }
}
```

### ModelAggregationAnchor (MAA) (`snapshots.py`)

Authentication system for verifying training data signatures.

**Key Features:**
- **Data Authentication** — Cryptographic signatures for data verification
- **Key Management** — Secure key derivation and storage
- **Signature Verification** — Validate data authenticity
- **Tamper Detection** — Detect unauthorized data modifications

**Usage Example:**
```python
from ciaf.provenance import ModelAggregationAnchor

# Create Model Aggregation Anchor
maa = ModelAggregationAnchor(
    key_id="medical_model_maa_v1",
    secret_material="secure_secret_for_medical_data"
)

# Generate signatures for training data
training_data_hashes = [
    "sha256:patient_1_hash",
    "sha256:patient_2_hash", 
    "sha256:patient_3_hash"
]

signatures = {}
for data_hash in training_data_hashes:
    signature = maa.generate_data_signature(data_hash)
    signatures[data_hash] = signature
    print(f"Data {data_hash[:16]}... signed: {signature[:16]}...")

# Later verification
for data_hash, signature in signatures.items():
    is_valid = maa.verify_data_signature(data_hash, signature)
    print(f"Signature for {data_hash[:16]}... is valid: {is_valid}")
```

## Advanced Features

### HIPAA-Compliant Data Handling

For healthcare applications requiring strict privacy:

```python
# Create HIPAA-compliant provenance capsules
def create_hipaa_compliant_capsule(patient_data, patient_id, consent_record):
    # Metadata without PHI
    hipaa_metadata = {
        "source": "authorized_healthcare_provider",
        "consent_status": "explicit_written_consent",
        "consent_date": consent_record["date"],
        "consent_scope": consent_record["scope"],
        "data_classification": "PHI",
        "retention_period": "7_years_post_treatment",
        "access_restrictions": "authorized_personnel_only",
        "audit_required": True,
        "anonymization_method": "k_anonymity_5"
    }
    
    # Use patient-specific secret
    patient_secret = f"patient_{patient_id}_training_secret_{consent_record['date']}"
    
    capsule = ProvenanceCapsule(
        original_data=patient_data,
        metadata=hipaa_metadata,
        data_secret=patient_secret
    )
    
    # Log access for HIPAA audit trail
    log_phi_access(
        patient_id=patient_id,
        access_type="provenance_capsule_creation",
        user_id="ai_training_system",
        timestamp=datetime.now(),
        purpose="ml_model_training"
    )
    
    return capsule

# Example usage
patient_record = {
    "demographics": "45-year-old male",
    "symptoms": "chest pain, shortness of breath",
    "vital_signs": "BP: 140/90, HR: 85, O2Sat: 98%",
    "lab_results": "Troponin: elevated, CRP: normal"
}

consent_info = {
    "date": "2025-09-01",
    "scope": "ai_model_training_research",
    "duration": "indefinite_with_opt_out"
}

hipaa_capsule = create_hipaa_compliant_capsule(
    patient_data=str(patient_record),
    patient_id="patient_12345",
    consent_record=consent_info
)
```

### Multi-Source Data Integration

For training with data from multiple sources:

```python
# Integrate data from multiple healthcare systems
def create_multi_source_training_snapshot(data_sources):
    all_capsules = []
    all_hashes = []
    
    # Process each data source
    for source_name, source_data in data_sources.items():
        source_capsules = []
        
        for record in source_data["records"]:
            # Create source-specific metadata
            source_metadata = {
                "source_system": source_name,
                "source_version": source_data["system_version"],
                "data_quality_score": source_data["quality_metrics"]["score"],
                "consent_framework": source_data["consent_framework"],
                "regulatory_compliance": source_data["compliance_frameworks"]
            }
            source_metadata.update(record["metadata"])
            
            # Create capsule
            capsule = ProvenanceCapsule(
                original_data=record["data"],
                metadata=source_metadata,
                data_secret=f"{source_name}_{record['id']}_secret"
            )
            
            source_capsules.append(capsule)
            all_capsules.append(capsule)
            all_hashes.append(capsule.hash_proof)
    
    # Create comprehensive training snapshot
    training_params = {
        "multi_source_training": True,
        "source_count": len(data_sources),
        "total_records": len(all_capsules),
        "data_fusion_method": "federated_learning",
        "privacy_preservation": "differential_privacy"
    }
    
    snapshot = TrainingSnapshot(
        model_version="multi_source_model_v1.0",
        training_parameters=training_params,
        provenance_capsule_hashes=all_hashes
    )
    
    return snapshot, all_capsules

# Example usage
data_sources = {
    "hospital_alpha": {
        "system_version": "Epic_2023.1",
        "quality_metrics": {"score": 0.95},
        "consent_framework": "explicit_opt_in",
        "compliance_frameworks": ["HIPAA", "FDA_21CFR11"],
        "records": [
            {"id": "pt_001", "data": "patient_data_1", "metadata": {}},
            {"id": "pt_002", "data": "patient_data_2", "metadata": {}}
        ]
    },
    "clinic_beta": {
        "system_version": "Cerner_2023.2",
        "quality_metrics": {"score": 0.92},
        "consent_framework": "broad_consent",
        "compliance_frameworks": ["HIPAA", "GDPR"],
        "records": [
            {"id": "pt_003", "data": "patient_data_3", "metadata": {}},
            {"id": "pt_004", "data": "patient_data_4", "metadata": {}}
        ]
    }
}

multi_source_snapshot, all_capsules = create_multi_source_training_snapshot(data_sources)
```

### Audit Trail Integration

Complete integration with audit systems:

```python
from ciaf.compliance import AuditTrailGenerator
from ciaf.provenance import ProvenanceCapsule, TrainingSnapshot

class ProvenanceAuditSystem:
    def __init__(self, system_id):
        self.audit_generator = AuditTrailGenerator(system_id)
        self.capsule_registry = {}
        self.snapshot_registry = {}
    
    def create_audited_capsule(self, data, metadata, secret, user_id):
        # Create capsule
        capsule = ProvenanceCapsule(data, metadata, secret)
        
        # Register in audit system
        self.audit_generator.log_data_event(
            event_type="provenance_capsule_created",
            data_hash=capsule.hash_proof,
            user_id=user_id,
            metadata=metadata,
            compliance_frameworks=["HIPAA", "EU_AI_ACT"]
        )
        
        # Store in registry
        self.capsule_registry[capsule.hash_proof] = {
            "capsule": capsule,
            "created_by": user_id,
            "created_at": datetime.now(),
            "access_log": []
        }
        
        return capsule
    
    def create_audited_snapshot(self, model_version, params, capsule_hashes, user_id):
        # Verify all capsules exist
        for hash_proof in capsule_hashes:
            if hash_proof not in self.capsule_registry:
                raise ValueError(f"Capsule {hash_proof} not found in registry")
        
        # Create snapshot
        snapshot = TrainingSnapshot(model_version, params, capsule_hashes)
        
        # Log training event
        self.audit_generator.log_training_event(
            event_type="training_snapshot_created",
            snapshot_id=snapshot.snapshot_id,
            model_version=model_version,
            data_count=len(capsule_hashes),
            user_id=user_id,
            training_parameters=params
        )
        
        # Store in registry
        self.snapshot_registry[snapshot.snapshot_id] = {
            "snapshot": snapshot,
            "created_by": user_id,
            "created_at": datetime.now(),
            "verification_log": []
        }
        
        return snapshot
    
    def verify_training_lineage(self, snapshot_id, data_hash, user_id):
        # Log verification attempt
        self.audit_generator.log_verification_event(
            event_type="provenance_verification",
            snapshot_id=snapshot_id,
            data_hash=data_hash,
            user_id=user_id
        )
        
        snapshot = self.snapshot_registry[snapshot_id]["snapshot"]
        result = snapshot.verify_provenance(data_hash)
        
        # Log verification result
        self.audit_generator.log_verification_result(
            snapshot_id=snapshot_id,
            data_hash=data_hash,
            verification_result=result,
            user_id=user_id
        )
        
        return result

# Example usage
audit_system = ProvenanceAuditSystem("medical_ai_system")

# Create audited provenance capsule
capsule = audit_system.create_audited_capsule(
    data="sensitive_patient_data",
    metadata={"source": "hospital_alpha", "consent": "obtained"},
    secret="patient_secret",
    user_id="data_scientist_alice"
)

# Create audited training snapshot
snapshot = audit_system.create_audited_snapshot(
    model_version="diagnostic_v1",
    params={"lr": 0.001, "epochs": 100},
    capsule_hashes=[capsule.hash_proof],
    user_id="ml_engineer_bob"
)

# Verify lineage with audit trail
verified = audit_system.verify_training_lineage(
    snapshot_id=snapshot.snapshot_id,
    data_hash=capsule.hash_proof,
    user_id="auditor_charlie"
)
```

## Integration Patterns

### With CIAF Framework

```python
from ciaf.api import CIAFFramework
from ciaf.provenance import ProvenanceCapsule, TrainingSnapshot

# Framework automatically manages provenance
framework = CIAFFramework("MedicalAI")

# Add training data with automatic provenance
def add_training_data_with_provenance(framework, patient_records):
    capsules = []
    
    for record in patient_records:
        # Framework creates provenance capsule
        capsule = framework.create_provenance_capsule(
            data=record["data"],
            metadata=record["metadata"],
            data_secret=record["patient_secret"]
        )
        capsules.append(capsule)
    
    return capsules

# Train with provenance tracking
training_capsules = add_training_data_with_provenance(framework, patient_data)
training_snapshot = framework.train_model_with_provenance(
    model_name="diagnostic_model",
    training_parameters=training_params,
    provenance_capsules=training_capsules
)

# Verify model lineage
for capsule in training_capsules:
    lineage_verified = framework.verify_model_lineage(
        model_name="diagnostic_model",
        data_hash=capsule.hash_proof
    )
    print(f"Lineage verified: {lineage_verified}")
```

### With LCM System

```python
from ciaf.lcm import LCMDatasetManager, LCMTrainingManager
from ciaf.provenance import ProvenanceCapsule

# LCM provides enhanced provenance management
lcm_dataset = LCMDatasetManager("medical_training_data")
lcm_training = LCMTrainingManager("diagnostic_model")

# Create LCM-managed provenance capsules
for record in training_data:
    # Create standard provenance capsule
    capsule = ProvenanceCapsule(
        original_data=record["data"],
        metadata=record["metadata"],
        data_secret=record["secret"]
    )
    
    # Register with LCM
    lcm_dataset.register_provenance_capsule(
        capsule=capsule,
        dataset_anchor_ref="medical_data_anchor_v1"
    )

# Create LCM training snapshot
lcm_snapshot = lcm_training.create_training_snapshot(
    model_version="diagnostic_v1",
    training_parameters=training_params,
    dataset_manager=lcm_dataset
)

# LCM provides enhanced verification
verification_result = lcm_training.verify_training_provenance(
    snapshot=lcm_snapshot,
    data_hash=capsule.hash_proof,
    include_lcm_chain_verification=True
)
```

### With Compliance Engine

```python
from ciaf.compliance import ComplianceValidator, BiasValidator
from ciaf.provenance import TrainingSnapshot

# Compliance validation for training snapshots
compliance_validator = ComplianceValidator("training_system")
bias_validator = BiasValidator("fairness_system")

# Validate training snapshot compliance
def validate_training_compliance(snapshot, frameworks):
    # Basic compliance check
    compliance_report = compliance_validator.validate_training_snapshot(
        snapshot=snapshot,
        frameworks=frameworks
    )
    
    # Bias assessment
    bias_report = bias_validator.assess_training_data_bias(
        provenance_hashes=snapshot.provenance_capsule_hashes,
        protected_attributes=["age", "gender", "ethnicity"]
    )
    
    # Combined report
    combined_report = {
        "snapshot_id": snapshot.snapshot_id,
        "compliance_status": compliance_report["status"],
        "bias_assessment": bias_report,
        "recommendations": []
    }
    
    # Add recommendations based on findings
    if bias_report["bias_detected"]:
        combined_report["recommendations"].append(
            "Consider bias mitigation techniques in training"
        )
    
    if not compliance_report["gdpr_compliant"]:
        combined_report["recommendations"].append(
            "Ensure GDPR compliance for EU patient data"
        )
    
    return combined_report

# Example usage
compliance_result = validate_training_compliance(
    snapshot=training_snapshot,
    frameworks=["HIPAA", "GDPR", "EU_AI_ACT"]
)
```

## Performance Considerations

### Large Dataset Optimization

```python
# Efficient handling of large training datasets
class OptimizedProvenanceSystem:
    def __init__(self, batch_size=1000, compression=True):
        self.batch_size = batch_size
        self.compression = compression
        self.capsule_cache = {}
    
    def create_batch_capsules(self, data_batch):
        capsules = []
        hashes = []
        
        # Process in smaller chunks for memory efficiency
        for chunk in self.chunk_data(data_batch, self.batch_size):
            chunk_capsules = []
            
            for record in chunk:
                capsule = ProvenanceCapsule(
                    original_data=record["data"],
                    metadata=record["metadata"],
                    data_secret=record["secret"]
                )
                chunk_capsules.append(capsule)
                hashes.append(capsule.hash_proof)
            
            # Cache management
            if self.compression:
                self._compress_and_store_capsules(chunk_capsules)
            else:
                capsules.extend(chunk_capsules)
        
        return capsules, hashes
    
    def create_efficient_snapshot(self, model_version, params, capsule_hashes):
        # Use optimized Merkle tree for large datasets
        snapshot = TrainingSnapshot(
            model_version=model_version,
            training_parameters=params,
            provenance_capsule_hashes=capsule_hashes
        )
        
        # Store snapshot with compression
        if self.compression:
            self._compress_snapshot(snapshot)
        
        return snapshot
    
    def chunk_data(self, data, chunk_size):
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
```

### Memory Management

```python
# Memory-efficient provenance handling
class MemoryEfficientProvenance:
    def __init__(self, max_memory_capsules=1000):
        self.max_memory_capsules = max_memory_capsules
        self.memory_capsules = {}
        self.disk_storage = {}
    
    def add_capsule(self, capsule):
        # Add to memory cache
        self.memory_capsules[capsule.hash_proof] = capsule
        
        # Check memory limits
        if len(self.memory_capsules) > self.max_memory_capsules:
            self._archive_old_capsules()
    
    def get_capsule(self, hash_proof):
        # Check memory first
        if hash_proof in self.memory_capsules:
            return self.memory_capsules[hash_proof]
        
        # Load from disk if needed
        if hash_proof in self.disk_storage:
            return self._load_capsule_from_disk(hash_proof)
        
        return None
    
    def _archive_old_capsules(self):
        # Move oldest capsules to disk
        oldest_capsules = sorted(
            self.memory_capsules.items(),
            key=lambda x: x[1].metadata["creation_timestamp"]
        )
        
        # Move half to disk
        archive_count = len(oldest_capsules) // 2
        for hash_proof, capsule in oldest_capsules[:archive_count]:
            self._store_capsule_to_disk(hash_proof, capsule)
            del self.memory_capsules[hash_proof]
```

## Security Features

### Access Control

```python
# Role-based access control for provenance
class SecureProvenanceManager:
    def __init__(self):
        self.access_policies = {}
        self.user_roles = {}
    
    def set_capsule_access_policy(self, capsule_hash, policy):
        self.access_policies[capsule_hash] = policy
    
    def can_access_capsule(self, user_id, capsule_hash, operation):
        user_role = self.user_roles.get(user_id, "guest")
        policy = self.access_policies.get(capsule_hash, {})
        
        allowed_roles = policy.get(operation, [])
        return user_role in allowed_roles
    
    def decrypt_capsule_with_authorization(self, user_id, capsule, secret):
        if not self.can_access_capsule(user_id, capsule.hash_proof, "decrypt"):
            raise PermissionError(f"User {user_id} not authorized to decrypt capsule")
        
        # Log access
        self._log_capsule_access(user_id, capsule.hash_proof, "decrypt")
        
        return capsule.decrypt_data()

# Example usage
secure_manager = SecureProvenanceManager()

# Set access policies
secure_manager.access_policies["patient_data_hash"] = {
    "view_metadata": ["researcher", "admin", "auditor"],
    "decrypt": ["admin", "authorized_researcher"],
    "modify": ["admin"]
}

secure_manager.user_roles["alice"] = "authorized_researcher"
secure_manager.user_roles["bob"] = "researcher"

# Controlled access
try:
    data = secure_manager.decrypt_capsule_with_authorization(
        user_id="alice",
        capsule=patient_capsule,
        secret="patient_secret"
    )
except PermissionError as e:
    print(f"Access denied: {e}")
```

### Integrity Monitoring

```python
# Continuous integrity monitoring
class ProvenanceIntegrityMonitor:
    def __init__(self):
        self.integrity_checks = {}
        self.alert_thresholds = {
            "failed_verifications": 5,
            "suspicious_access_patterns": 10
        }
    
    def monitor_capsule_integrity(self, capsule):
        # Verify hash proof
        integrity_valid = capsule.verify_hash_proof()
        
        # Log result
        self.integrity_checks[capsule.hash_proof] = {
            "last_check": datetime.now(),
            "integrity_valid": integrity_valid,
            "check_count": self.integrity_checks.get(
                capsule.hash_proof, {}
            ).get("check_count", 0) + 1
        }
        
        # Alert on failures
        if not integrity_valid:
            self._trigger_integrity_alert(capsule.hash_proof)
        
        return integrity_valid
    
    def monitor_snapshot_integrity(self, snapshot):
        # Verify Merkle tree
        merkle_valid = True
        for hash_proof in snapshot.provenance_capsule_hashes:
            if not snapshot.verify_provenance(hash_proof):
                merkle_valid = False
                break
        
        # Log snapshot check
        self.integrity_checks[snapshot.snapshot_id] = {
            "last_check": datetime.now(),
            "merkle_valid": merkle_valid,
            "data_count": len(snapshot.provenance_capsule_hashes)
        }
        
        return merkle_valid
    
    def _trigger_integrity_alert(self, item_id):
        # Security incident response
        alert = {
            "alert_type": "integrity_violation",
            "item_id": item_id,
            "timestamp": datetime.now(),
            "severity": "high",
            "action_required": "immediate_investigation"
        }
        
        # Log security event
        self._log_security_event(alert)
```

## Best Practices

### 1. Data Classification

```python
# Classify data sensitivity levels
def classify_training_data(data_record):
    classification = "public"  # Default
    
    # Check for sensitive patterns
    if contains_phi(data_record):
        classification = "phi"
    elif contains_pii(data_record):
        classification = "pii"
    elif contains_financial_info(data_record):
        classification = "financial"
    
    return classification

def create_classified_capsule(data, classification):
    metadata = {
        "data_classification": classification,
        "encryption_required": classification in ["phi", "pii", "financial"],
        "retention_policy": get_retention_policy(classification),
        "access_restrictions": get_access_restrictions(classification)
    }
    
    # Use classification-specific secret generation
    secret = generate_classified_secret(data, classification)
    
    return ProvenanceCapsule(data, metadata, secret)
```

### 2. Consent Management

```python
# Manage patient consent for training
class ConsentManager:
    def __init__(self):
        self.consent_records = {}
    
    def record_consent(self, patient_id, consent_details):
        self.consent_records[patient_id] = {
            "consent_date": datetime.now(),
            "consent_scope": consent_details["scope"],
            "consent_duration": consent_details["duration"],
            "withdrawal_allowed": consent_details.get("withdrawal_allowed", True),
            "purpose": consent_details["purpose"]
        }
    
    def verify_consent_for_training(self, patient_id, training_purpose):
        consent = self.consent_records.get(patient_id)
        if not consent:
            return False
        
        # Check scope and purpose
        if training_purpose not in consent["consent_scope"]:
            return False
        
        # Check expiration
        if self._is_consent_expired(consent):
            return False
        
        return True
    
    def create_consent_compliant_capsule(self, patient_id, data, purpose):
        if not self.verify_consent_for_training(patient_id, purpose):
            raise ValueError(f"No valid consent for patient {patient_id}")
        
        consent = self.consent_records[patient_id]
        metadata = {
            "patient_id": patient_id,
            "consent_date": consent["consent_date"].isoformat(),
            "consent_scope": consent["consent_scope"],
            "training_purpose": purpose,
            "data_classification": "phi"
        }
        
        return ProvenanceCapsule(data, metadata, f"patient_{patient_id}_secret")
```

### 3. Audit Trail Integration

```python
# Complete audit integration
def create_fully_audited_training_session():
    audit_logger = AuditTrailGenerator("medical_ai")
    
    # Log training session start
    session_id = audit_logger.start_training_session(
        model_name="diagnostic_model",
        user_id="data_scientist",
        purpose="pneumonia_detection"
    )
    
    # Create provenance capsules with audit
    capsules = []
    for patient_data in training_dataset:
        capsule = ProvenanceCapsule(
            original_data=patient_data["data"],
            metadata={
                "session_id": session_id,
                "patient_consent_verified": True,
                "data_source": patient_data["source"]
            },
            data_secret=patient_data["secret"]
        )
        
        # Log data inclusion
        audit_logger.log_data_inclusion(
            session_id=session_id,
            data_hash=capsule.hash_proof,
            consent_status="verified"
        )
        
        capsules.append(capsule)
    
    # Create training snapshot
    snapshot = TrainingSnapshot(
        model_version="diagnostic_v1",
        training_parameters=training_params,
        provenance_capsule_hashes=[c.hash_proof for c in capsules]
    )
    
    # Log training completion
    audit_logger.complete_training_session(
        session_id=session_id,
        snapshot_id=snapshot.snapshot_id,
        data_count=len(capsules)
    )
    
    return snapshot, capsules, session_id
```

## Contributing

When extending the provenance package:

1. **Maintain Cryptographic Integrity** — All data must be verifiable
2. **Privacy by Design** — Default to maximum privacy protection
3. **Regulatory Compliance** — Support HIPAA, GDPR, and emerging frameworks
4. **Performance Optimization** — Consider large-scale training scenarios
5. **Comprehensive Auditing** — Log all provenance operations

## Dependencies

The provenance package depends on:
- `ciaf.core` — Cryptographic utilities for encryption and hashing
- `cryptography` — AES-GCM encryption and cryptographic operations
- `datetime` — Timestamp generation for lineage tracking
- `json` — Serialization for capsule and snapshot data
- `typing` — Type hints for better code clarity

---

*For integration examples and advanced patterns, see the [examples folder](../examples/) and [API documentation](../api/).*