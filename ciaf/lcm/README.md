# CIAF Lifecycle Management (LCM) System

The LCM (Lazy Capsule Materialization) system provides comprehensive end-to-end lifecycle management for AI models with proper anchoring, dataset families with splits, deployment stages, and audit trails.

## Overview

The LCM system implements a complete AI model lifecycle with cryptographic provenance tracking from data ingestion through inference. It supports:

- **Dataset Family Management** — Proper representation of datasets with train/validation/test splits
- **Model Anchoring** — Immutable model metadata with parameter and architecture fingerprints
- **Training Sessions** — Comprehensive training tracking with checkpoints and metrics
- **Deployment Pipeline** — Pre-deployment preparation and deployment tracking
- **Inference Management** — Receipt generation and audit chaining for production inference
- **Root Computation** — Merkle root computation for training sessions, releases, and inference batches

## Architecture

```
LCM System Components
├── Policy Framework
│   ├── Cryptographic policies (hashing, Merkle trees)
│   ├── Domain types (dataset, model, training, deployment, inference)
│   └── Commitment types (salted, HMAC, plaintext)
├── Dataset Management
│   ├── Dataset Family Manager (multi-split datasets)
│   ├── Dataset Manager (individual dataset anchors)
│   └── Split assignment with reproducible digests
├── Model Management
│   ├── Model anchors with architecture/parameter fingerprints
│   ├── Training environment capture
│   └── Authorized dataset validation
├── Training Management
│   ├── Training sessions with comprehensive tracking
│   ├── Checkpoint management
│   └── Metrics digest computation
├── Deployment Management
│   ├── Pre-deployment anchors (artifacts, SBOM, approvals)
│   ├── Deployment anchors (infrastructure, configuration)
│   └── Rollout planning and tracking
├── Inference Management
│   ├── Receipt generation with privacy commitments
│   ├── Chain-of-custody tracking
│   └── Batch processing for time windows
├── Root Management
│   ├── Training session roots
│   ├── Release roots
│   └── Inference batch roots
└── Capsule Headers
    ├── Compact JSON state representation
    ├── Stage-by-stage lifecycle tracking
    └── Policy compliance verification
```

## Core Components

### 1. Policy Framework (`policy.py`)

Defines the canonical policies for hashing, domains, Merkle trees, and commitments.

```python
from ciaf.lcm import LCMPolicy, get_default_policy, DomainType, CommitmentType

# Get default policy
policy = get_default_policy()
print(policy.format_policy_line())

# Create custom policy
custom_policy = LCMPolicy(
    hash_algorithm="SHA-256",
    canonicalization="json(sorted,utf-8)",
    commitments=CommitmentType.SALTED
)
```

### 2. Dataset Family Management (`dataset_family_manager.py`)

Manages datasets with proper train/validation/test splits.

```python
from ciaf.lcm import LCMDatasetFamilyManager, DatasetFamilyMetadata, DatasetSplit

# Create dataset family manager
family_manager = LCMDatasetFamilyManager()

# Define family metadata
metadata = DatasetFamilyMetadata(
    name="medical_images",
    version="v1.0",
    owner="hospital_ai_team",
    license="custom",
    description="Chest X-ray dataset for pneumonia detection",
    contains_pii=True,
    privacy_level="restricted",
    compliance_frameworks=["HIPAA"]
)

# Create dataset family anchor
family_anchor = family_manager.create_dataset_family_anchor(
    family_id="med_img_v1",
    metadata=metadata,
    master_password="secure_family_password"
)

# Add dataset splits
splits_info = {
    DatasetSplit.TRAIN: {"sample_count": 8000, "record_ids": ["r001", "r002", ...]},
    DatasetSplit.VALIDATION: {"sample_count": 1000, "record_ids": ["r501", ...]},
    DatasetSplit.TEST: {"sample_count": 1000, "record_ids": ["r901", ...]}
}

split_anchors = family_manager.add_dataset_splits(
    family_anchor=family_anchor,
    splits_info=splits_info
)
```

### 3. Model Management (`model_manager.py`)

Creates immutable model anchors with comprehensive metadata.

```python
from ciaf.lcm import LCMModelManager, ModelArchitecture, TrainingEnvironment

# Create model manager
model_manager = LCMModelManager()

# Define model architecture
architecture = ModelArchitecture(
    type="CNN",
    layers=[
        {"type": "conv2d", "filters": 32, "kernel_size": 3},
        {"type": "maxpool", "pool_size": 2},
        {"type": "dense", "units": 128},
        {"type": "dense", "units": 1, "activation": "sigmoid"}
    ],
    input_dim=224*224*3,
    output_dim=1,
    total_params=1_250_000
)

# Define training environment
environment = TrainingEnvironment(
    framework="tensorflow",
    framework_version="2.13.0",
    cuda_version="11.8",
    dependencies={"numpy": "1.24.0", "pillow": "9.5.0"}
)

# Create model anchor
model_anchor = model_manager.create_model_anchor(
    model_name="pneumonia_detector",
    version="v1.0",
    architecture=architecture,
    hyperparameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "optimizer": "adam"
    },
    environment=environment,
    authorized_datasets=["med_img_v1"],
    trainer_commit="abc123def",
    master_password="secure_model_password"
)
```

### 4. Training Management (`training_manager.py`)

Manages comprehensive training sessions with checkpoints and metrics.

```python
from ciaf.lcm import LCMTrainingManager, TrainingMetrics

# Create training manager
training_manager = LCMTrainingManager()

# Start training session
training_session = training_manager.start_training_session(
    session_id="train_001",
    model_anchor=model_anchor,
    datasets_root_anchor="root_anchor_hash",
    training_config={
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "max_epochs": 50,
        "early_stopping": True,
        "seed": 42
    },
    data_splits={
        DatasetSplit.TRAIN: "train_anchor_id",
        DatasetSplit.VALIDATION: "val_anchor_id"
    }
)

# Add training checkpoints
training_manager.add_checkpoint(
    training_session=training_session,
    checkpoint_id="ckpt_epoch_10",
    epoch=10,
    step=2500,
    metrics={"loss": 0.45, "accuracy": 0.82, "val_loss": 0.48, "val_accuracy": 0.80},
    model_state_digest="model_state_hash",
    optimizer_state_digest="optimizer_state_hash"
)

# Complete training session
final_metrics = TrainingMetrics(
    train_metrics={"loss": [1.0, 0.8, 0.6, 0.45], "accuracy": [0.6, 0.7, 0.8, 0.82]},
    val_metrics={"loss": [1.1, 0.9, 0.65, 0.48], "accuracy": [0.58, 0.68, 0.78, 0.80]},
    epochs=[1, 10, 20, 50]
)

training_snapshot = training_manager.complete_training_session(
    training_session=training_session,
    final_model_digest="final_model_hash",
    metrics=final_metrics,
    user_id="data_scientist_alice"
)
```

### 5. Deployment Management (`deployment_manager.py`)

Manages pre-deployment preparation and deployment tracking.

```python
from ciaf.lcm import LCMDeploymentManager, BuildArtifact, SBOM

# Create deployment manager
deployment_manager = LCMDeploymentManager()

# Create build artifact
artifact = BuildArtifact(
    artifact_type="docker",
    artifact_digest="sha256:abc123...",
    build_timestamp=datetime.now().isoformat(),
    builder_info="jenkins-v2.401",
    size_bytes=1024*1024*500  # 500MB
)

# Create SBOM
sbom = SBOM(
    dependencies={
        "tensorflow": "2.13.0",
        "numpy": "1.24.0",
        "flask": "2.3.0"
    },
    security_scan_digest="security_scan_hash",
    vulnerability_count=0,
    compliance_status="passed"
)

# Create pre-deployment anchor
predeployment_anchor = deployment_manager.create_predeployment_anchor(
    predeployment_id="predeploy_001",
    build_artifact=artifact,
    sbom=sbom,
    approval_ticket_id="TICKET-12345",
    intended_env="production",
    intended_region="us-east-1"
)

# Create deployment anchor
deployment_anchor = deployment_manager.create_deployment_anchor(
    deployment_id="deploy_001",
    predeployment_anchor_ref=predeployment_anchor.predeployment_id,
    infrastructure={
        "cluster": "prod-k8s",
        "namespace": "ai-models",
        "replicas": 3,
        "cpu_limit": "2",
        "memory_limit": "4Gi"
    },
    network_config={
        "load_balancer": "internal",
        "security_groups": ["sg-123", "sg-456"]
    },
    deployed_by="devops_alice"
)
```

### 6. Inference Management (`inference_manager.py`)

Generates receipts and manages audit chaining for production inference.

```python
from ciaf.lcm import LCMInferenceManager, LCMInferenceCommitment, CommitmentType

# Create inference manager
inference_manager = LCMInferenceManager()

# Create input/output commitments for privacy
input_commitment = LCMInferenceCommitment(
    commitment_type=CommitmentType.SALTED,
    commitment_value="salted_input_hash",
    metadata={"query_length": 150, "content_type": "text"}
)

output_commitment = LCMInferenceCommitment(
    commitment_type=CommitmentType.SALTED,
    commitment_value="salted_output_hash",
    metadata={"confidence": 0.95, "prediction": "positive"}
)

# Create inference receipt
receipt = inference_manager.create_inference_receipt(
    receipt_id="inf_001",
    model_anchor_ref=model_anchor.model_id,
    deployment_anchor_ref=deployment_anchor.deployment_id,
    request_id="req_12345",
    query="Is there pneumonia in this chest X-ray?",
    ai_output="No pneumonia detected (confidence: 95%)",
    input_commitment=input_commitment,
    output_commitment=output_commitment,
    explanation_digests=["shap_explanation_hash"],
    prev_chain_digest=None  # First receipt in chain
)

# Process batch inference
window_id = "batch_2025_09_12_14"
receipts = [receipt]  # Add more receipts
batch_digest = inference_manager.process_inference_batch(window_id, receipts)
```

### 7. Root Management (`root_manager.py`)

Computes Merkle roots for training sessions, releases, and inference batches.

```python
from ciaf.lcm import LCMRootManager, TestEvaluationAnchor

# Create root manager
root_manager = LCMRootManager()

# Create test evaluation anchor
test_anchor = root_manager.create_test_evaluation_anchor(
    test_id="test_001",
    test_dataset_ref="test_dataset_anchor",
    evaluation_type="pre_deploy",
    metrics={
        "accuracy": 0.885,
        "precision": 0.892,
        "recall": 0.878,
        "f1_score": 0.885
    }
)

# Compute training session root
training_session_root = root_manager.compute_training_session_root(training_session)

# Compute release root
release_root = root_manager.compute_release_root(
    training_session_root=training_session_root,
    predeployment_anchor=predeployment_anchor,
    deployment_anchor=deployment_anchor,
    test_evaluation_anchor=test_anchor
)

# Compute inference batch root
inference_batch_root = root_manager.compute_inference_batch_root(
    window_id="batch_001",
    receipt_digests=[receipt.receipt_digest for receipt in receipts]
)
```

### 8. Capsule Headers (`capsule_headers.py`)

Creates compact JSON state representation for the complete lifecycle.

```python
from ciaf.lcm import LCMCapsuleManager, CapsuleHeader

# Create capsule manager
capsule_manager = LCMCapsuleManager()

# Create comprehensive capsule header
header = capsule_manager.create_capsule_header(
    dataset_anchor=family_anchor,
    model_anchor=model_anchor,
    training_session=training_session,
    predeployment_anchor=predeployment_anchor,
    deployment_anchor=deployment_anchor,
    test_evaluation_anchor=test_anchor,
    inference_receipt=receipt,
    training_session_root=training_session_root,
    release_root=release_root,
    inference_batch_root=inference_batch_root
)

# Export to JSON
print(header.to_json(indent=2))

# Create capsule from header
capsule = capsule_manager.create_capsule_from_header(header)
```

## Complete Example: End-to-End AI Model Lifecycle

```python
from ciaf.lcm import *

# 1. Create dataset family
family_manager = LCMDatasetFamilyManager()
metadata = DatasetFamilyMetadata(
    name="medical_dataset",
    version="v1.0", 
    owner="hospital",
    license="custom"
)
family_anchor = family_manager.create_dataset_family_anchor(
    "med_v1", metadata, "family_password"
)

# 2. Create model
model_manager = LCMModelManager()
architecture = ModelArchitecture(type="CNN", layers=[], total_params=1000000)
environment = TrainingEnvironment(framework="tensorflow", framework_version="2.13.0")
model_anchor = model_manager.create_model_anchor(
    "pneumonia_model", "v1.0", architecture, {"lr": 0.001}, 
    environment, ["med_v1"], "commit123", "model_password"
)

# 3. Train model
training_manager = LCMTrainingManager()
session = training_manager.start_training_session(
    "train_001", model_anchor, "datasets_root", 
    {"epochs": 50}, {DatasetSplit.TRAIN: "train_anchor"}
)
snapshot = training_manager.complete_training_session(
    session, "final_model_hash", 
    TrainingMetrics({}, {}, []), "user_alice"
)

# 4. Deploy model
deployment_manager = LCMDeploymentManager()
artifact = BuildArtifact("docker", "sha256:abc", datetime.now().isoformat(), "jenkins")
sbom = SBOM({"tensorflow": "2.13.0"}, "scan_hash")
predeploy = deployment_manager.create_predeployment_anchor(
    "predeploy_001", artifact, sbom, "TICKET-123", "prod", "us-east-1"
)
deploy = deployment_manager.create_deployment_anchor(
    "deploy_001", predeploy.predeployment_id, {}, {}, "devops_alice"
)

# 5. Run inference
inference_manager = LCMInferenceManager()
input_commit = LCMInferenceCommitment(CommitmentType.SALTED, "input_hash")
output_commit = LCMInferenceCommitment(CommitmentType.SALTED, "output_hash")
receipt = inference_manager.create_inference_receipt(
    "inf_001", model_anchor.model_id, deploy.deployment_id, 
    "req_001", "input", "output", input_commit, output_commit
)

# 6. Compute roots
root_manager = LCMRootManager()
test_anchor = root_manager.create_test_evaluation_anchor(
    "test_001", "test_dataset", "pre_deploy", {"accuracy": 0.95}
)
training_root = root_manager.compute_training_session_root(session)
release_root = root_manager.compute_release_root(
    training_root, predeploy, deploy, test_anchor
)
inference_root = root_manager.compute_inference_batch_root(
    "batch_001", [receipt.receipt_digest]
)

# 7. Create capsule header
capsule_manager = LCMCapsuleManager()
header = capsule_manager.create_capsule_header(
    family_anchor, model_anchor, session, predeploy, deploy,
    test_anchor, receipt, training_root, release_root, inference_root
)

print("Complete AI lifecycle captured in capsule header:")
print(header.to_json(indent=2))
```

## Key Features

### 1. **Cryptographic Provenance**
- All anchors use HMAC-SHA256 with salt for tamper evidence
- Merkle trees provide efficient proof-of-inclusion
- Chain-of-custody maintained throughout lifecycle

### 2. **Lazy Materialization**
- Capsules materialized on-demand to minimize storage
- Cryptographic verification without full data exposure
- Efficient audit trail generation

### 3. **Compliance Ready**
- Built-in support for HIPAA, GDPR, EU AI Act requirements
- Privacy commitments for sensitive data
- Comprehensive audit trail generation

### 4. **Reproducibility**
- RNG seed tracking for deterministic splits
- Environment capture for training reproducibility
- Immutable parameter and architecture fingerprints

### 5. **Production Ready**
- Deployment pipeline with SBOM and security scanning
- Infrastructure configuration tracking
- Real-time inference receipt generation

## Policy Configuration

The LCM system uses a centralized policy framework:

```python
from ciaf.lcm import LCMPolicy, CommitmentType, DomainType, MerklePolicy

# Create custom policy
policy = LCMPolicy(
    hash_algorithm="SHA-256",
    canonicalization="json(sorted,utf-8)",
    domains=[DomainType.DATASET, DomainType.MODEL, DomainType.TRAINING],
    merkle=MerklePolicy(fanout=2, padding="duplicate_last"),
    commitments=CommitmentType.HMAC_SHA256,
    anchor_schema_version="1.0"
)

# Set as global default
from ciaf.lcm.policy import set_default_policy
set_default_policy(policy)
```

## Integration with CIAF Framework

The LCM system integrates seamlessly with the main CIAF framework:

```python
from ciaf import CIAFFramework
from ciaf.lcm import LCMCapsuleManager

# Use LCM with CIAF framework
framework = CIAFFramework("MyProject")
lcm_manager = LCMCapsuleManager()

# Create LCM components and integrate with CIAF
# ... (create anchors, sessions, etc.)

# Generate CIAF-compatible audit trail
audit_trail = framework.get_complete_audit_trail_with_lcm(
    model_name="pneumonia_model",
    lcm_capsule_header=header
)
```

## Testing and Development

The LCM system includes comprehensive testing utilities:

```python
# Simulate complete lifecycle for testing
def simulate_ai_lifecycle():
    # ... (use managers to create mock lifecycle)
    return capsule_header

# Validate policy compliance
def validate_lcm_policy(header: CapsuleHeader):
    # ... (validate against policy requirements)
    return compliance_status
```

## Security Considerations

1. **Master Passwords**: Use strong, unique passwords for each anchor type
2. **Salt Management**: Salts are automatically generated and managed securely
3. **Commitment Types**: Choose appropriate commitment types based on privacy needs
4. **Policy Validation**: Ensure consistent policy usage across all components

## Performance Optimization

1. **Lazy Loading**: Components are materialized only when needed
2. **Batch Processing**: Inference receipts can be processed in batches
3. **Merkle Trees**: Efficient proof generation with O(log n) complexity
4. **Caching**: Computed roots and digests are cached for reuse

## Contributing

When extending the LCM system:

1. Follow the established patterns in existing managers
2. Ensure all new components support the policy framework
3. Add comprehensive tests for new functionality
4. Update this README with new features and examples

---

*For more information, see the main [CIAF documentation](../../README.md) and [compliance guides](../../docs/compliance/).*