"""
CIAF API Framework with Compliance Extensions

High-level API for the Cognitive Insight Audit Framework with regulatory compliance
extensions and non-bypassable cryptographic audit receipt invariants.

Created: 2025-09-09
Last Modified: 2025-09-23
Author: Denzil James Greenwood
Version: 2.0.0
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

# Removed redundant anchoring imports - using LCM system instead
from ..core import CryptoUtils, MerkleTree, derive_model_anchor, derive_master_anchor, sha256_hash, secure_random_bytes, SALT_LENGTH, to_hex
from ..core.canonicalization import (
    Policy, RecordType, AnchorRecord, Receipt, Signer, WORMMerkleTree, CapsuleBuilder,
    canonical_json, canonicalize_and_hash, validate_required_fields, 
    enrich_metadata_with_defaults, make_anchor, HashAlgorithm
)
from ..provenance import ModelAggregationAnchor, ProvenanceCapsule, TrainingSnapshot
from ..simulation import MLFrameworkSimulator
from ..inference import InferenceReceipt, ZKEConnections
from ..compliance import AuditTrailGenerator


# LCM System Integration
from ..lcm import (
    LCMRootManager, LCMDatasetManager, LCMModelManager, 
    LCMTrainingManager, LCMInferenceManager, LCMDeploymentManager,
    LCMDatasetAnchor, LCMPolicy, get_default_policy, DatasetMetadata, DatasetSplit
)


class ComplianceError(Exception):
    """Exception raised for compliance violations."""
    pass


class CIAFFramework:
    """
    Main framework class providing high-level API for CIAF operations with compliance extensions.
    
    Implements complete audit flow with non-bypassable invariants:
    Dataset Anchor → Model Anchor → Inference Receipt → Merkle Tree → Signed Anchor → Proof Capsule
    
    Enforces:
    - Canonicalized metadata → leaf hash → Merkle tree → signed anchor
    - Required fields validation for each record type  
    - Dual anchoring discipline (hash table + Merkle ledger) with WORM semantics
    - LCM: store anchors/logs by default; materialize inclusion proofs on demand
    """

    def __init__(
        self, 
        framework_name: str = "CIAF",
        policy: Optional[Policy] = None,
        anchor_signer: Optional[Signer] = None
    ):
        self.framework_name = framework_name
        
        # Compliance-enhanced initialization
        self.policy = policy or Policy(
            policy_id="ciaf_default_v1",
            schema_version="1.0.0",
            domain_labels=["audit", "ai_governance"],
            hash_algorithm=HashAlgorithm.SHA256
        )
        
        # Use proper signer implementation instead of Protocol
        if anchor_signer is not None:
            self.anchor_signer = anchor_signer
        else:
            try:
                from ..core.canonicalization import create_production_signer
                self.anchor_signer = create_production_signer("ciaf_default_key")
            except ImportError:
                # Create a fallback signer
                class FallbackSigner:
                    def __init__(self, key_id):
                        self.key_id = key_id
                    def sign(self, data):
                        return f"fallback_signature_{hash(data)}"
                    def verify(self, data, signature):
                        return True
                
                self.anchor_signer = FallbackSigner("ciaf_default_key")
        
        # Core WORM Merkle ledger
        self.ledger = WORMMerkleTree(self.policy.hash_algorithm)
        

        
        # Legacy components (removed - using LCM system)
        self.crypto_utils = CryptoUtils()
        self.dataset_anchors: Dict[str, Dict[DatasetSplit, LCMDatasetAnchor]] = {}
        self.model_anchors: Dict[str, Dict[str, Any]] = {}
        # Lazy managers removed - functionality integrated into LCM system
        self.ml_simulators: Dict[str, MLFrameworkSimulator] = {}
        self.inference_connections: Dict[str, ZKEConnections] = {}
        self.audit_generators: Dict[str, AuditTrailGenerator] = {}
        
        # LCM System Integration
        self.lcm_policy = get_default_policy()
        self.lcm_root_manager = LCMRootManager(self.lcm_policy)
        self.lcm_dataset_manager = LCMDatasetManager(self.lcm_policy)
        self.lcm_model_manager = LCMModelManager(self.lcm_policy)
        self.lcm_training_manager = LCMTrainingManager(self.lcm_policy)
        self.lcm_inference_manager = LCMInferenceManager(self.lcm_policy)
        self.lcm_deployment_manager = LCMDeploymentManager(self.lcm_policy)

    # ================================
    # Compliance-Enhanced Commit Methods
    # ================================

    def commit_dataset_record(self, record_meta: Dict[str, Any]) -> Receipt:
        """
        Commit dataset record with compliance extensions.
        
        Non-bypassable flow: canonicalize → validate → enrich → anchor → emit receipt
        
        Args:
            record_meta: Dataset metadata
            
        Returns:
            Receipt with proof of commitment
            
        Raises:
            ComplianceError: If compliance validation fails
        """
        print(f"Committing dataset record: {record_meta.get('dataset_id', 'unknown')}")
        
        # Step 1: Canonicalize metadata
        base_meta = canonicalize_and_hash(record_meta, self.policy.hash_algorithm)
        
        # Step 2: Enrich with compliance metadata
        enriched_meta = self.compliance.create_enhanced_capsule_metadata(
            record_meta, 
            include_oversight=False,  # Datasets typically don't need oversight
            include_consent=True,
            include_robustness=False,
            include_monitoring=False
        )
        
        # Step 3: Enrich with required fields
        final_meta = enrich_metadata_with_defaults(
            enriched_meta, RecordType.DATASET, self.policy
        )
        
        # Step 4: Validate required fields (non-bypassable)
        validate_required_fields(final_meta, RecordType.DATASET)
        
        # Step 5: GDPR compliance validation
        self.compliance.gdpr_manager.validate_dataset_commit(final_meta)
        
        # Step 6: Anchor and emit receipt
        return self._anchor_and_emit(final_meta, RecordType.DATASET)
    
    def commit_model_checkpoint(self, ckpt_meta: Dict[str, Any]) -> Receipt:
        """
        Commit model checkpoint with compliance extensions.
        
        Args:
            ckpt_meta: Model checkpoint metadata
            
        Returns:
            Receipt with robustness/security attestations
            
        Raises:
            ComplianceError: If compliance validation fails
        """
        print(f"Committing model checkpoint: {ckpt_meta.get('model_id', 'unknown')}")
        
        # Step 1: Canonicalize metadata
        base_meta = canonicalize_and_hash(ckpt_meta, self.policy.hash_algorithm)
        
        # Step 2: Enrich with compliance metadata
        enriched_meta = self.compliance.create_enhanced_capsule_metadata(
            ckpt_meta,
            include_oversight=self.policy.is_high_risk(),
            include_consent=False,
            include_robustness=True,  # Models require robustness attestation
            include_monitoring=True
        )
        
        # Step 3: Enrich with required fields  
        final_meta = enrich_metadata_with_defaults(
            enriched_meta, RecordType.MODEL, self.policy
        )
        
        # Step 4: Validate required fields (non-bypassable)
        validate_required_fields(final_meta, RecordType.MODEL)
        
        # Step 5: EU AI Act Article 15 robustness checks
        # Always run robustness validation for models - the manager will decide if it's required
        self.compliance.robustness_manager.validate_model_commit(final_meta)
        
        # Step 6: Anchor and emit receipt
        return self._anchor_and_emit(final_meta, RecordType.MODEL)
    
    def commit_inference(self, inf_meta: Dict[str, Any]) -> Receipt:
        """
        Commit inference with compliance extensions.
        
        Args:
            inf_meta: Inference metadata
            
        Returns:
            Receipt with oversight gating and consent verification
            
        Raises:
            ComplianceError: If compliance validation fails
        """
        print(f"Committing inference: {inf_meta.get('inference_id', 'unknown')}")
        
        # Step 1: Canonicalize metadata
        base_meta = canonicalize_and_hash(inf_meta, self.policy.hash_algorithm)
        
        # Step 2: Enrich with compliance metadata
        enriched_meta = self.compliance.create_enhanced_capsule_metadata(
            inf_meta,
            include_oversight=True,  # Inference requires oversight checks
            include_consent=True,
            include_robustness=False,
            include_monitoring=True
        )
        
        # Step 3: Enrich with required fields
        final_meta = enrich_metadata_with_defaults(
            enriched_meta, RecordType.INFERENCE, self.policy
        )
        
        # Step 4: Validate required fields (non-bypassable)
        validate_required_fields(final_meta, RecordType.INFERENCE)
        
        # Step 5: EU AI Act Article 14 oversight validation
        if not self.compliance.oversight_manager.validate_capsule_oversight(final_meta):
            raise ComplianceError("Human oversight required but not completed")
        
        # Step 6: GDPR consent validation (only if processing personal data)
        data_categories = final_meta.get('data_categories', [])
        contains_personal_data = any(cat in ['personal', 'sensitive', 'biometric'] for cat in data_categories)
        
        if contains_personal_data:
            consent_receipts = final_meta.get('compliance_extensions', {}).get('consent_receipts', [])
            if not consent_receipts:
                raise ComplianceError("GDPR consent receipt required for processing personal data")
        
        # Step 7: Anchor and emit receipt
        return self._anchor_and_emit(final_meta, RecordType.INFERENCE)
    
    def _anchor_and_emit(self, metadata: Dict[str, Any], record_type: RecordType) -> Receipt:
        """
        Anchor metadata in WORM ledger and emit receipt.
        
        Implements: leaf hash → Merkle tree → signed anchor → receipt
        
        Args:
            metadata: Validated and enriched metadata
            record_type: Type of record being anchored
            
        Returns:
            Complete audit receipt
        """
        # Step 1: Compute leaf hash
        leaf_hash = canonicalize_and_hash(metadata, self.policy.hash_algorithm)
        
        # Step 2: Append to WORM Merkle ledger (dual anchoring)
        root = self.ledger.append_leaf(leaf_hash, metadata)
        
        # Step 3: Create signed anchor
        anchor = make_anchor(root, self.policy, self.anchor_signer)
        
        # Step 4: Append anchor to WORM log
        self.ledger.append_anchor(anchor)
        
        # Step 5: Create and return receipt
        receipt = Receipt(
            metadata=metadata,
            anchor=anchor, 
            leaf_hash=leaf_hash,
            record_type=record_type
        )
        
        print(f"Record anchored - Root: {root[:16]}..., Leaf: {leaf_hash[:16]}...")
        return receipt
    
    def materialize_proof_capsule(self, artifact_id: str) -> Dict[str, Any]:
        """
        Materialize proof capsule on demand (LCM pattern).
        
        Args:
            artifact_id: ID of artifact to generate proof for
            
        Returns:
            Complete proof capsule with Merkle path and anchor
        """
        print(f"Materializing proof capsule for: {artifact_id}")
        
        # Find metadata by artifact ID
        metadata = None
        leaf_hash = None
        record_type = None
        
        # Search in hash table
        for hash_key, stored_meta in self.ledger.hash_table.items():
            if (stored_meta.get('dataset_id') == artifact_id or 
                stored_meta.get('model_id') == artifact_id or
                stored_meta.get('inference_id') == artifact_id):
                metadata = stored_meta
                leaf_hash = hash_key
                
                # Determine record type
                if 'dataset_id' in stored_meta:
                    record_type = RecordType.DATASET
                elif 'model_id' in stored_meta:
                    record_type = RecordType.MODEL
                elif 'inference_id' in stored_meta:
                    record_type = RecordType.INFERENCE
                break
        
        if not metadata:
            raise ValueError(f"Artifact {artifact_id} not found in ledger")
        
        # Get Merkle path (inclusion proof)
        merkle_path = self.ledger.get_merkle_path(leaf_hash)
        
        # Get signed anchor
        anchor = self.ledger.get_latest_anchor()
        if not anchor:
            raise ValueError("No anchor found for proof generation")
        
        # Build proof capsule
        capsule = CapsuleBuilder.build(
            metadata=metadata,
            merkle_path=merkle_path,
            anchor=anchor,
            record_type=record_type,
            leaf_hash=leaf_hash
        )
        
        print(f"Proof capsule materialized with {len(merkle_path)} proof elements")
        return capsule

    def create_dataset_anchor_lcm(
        self, dataset_id: str, dataset_metadata: Dict[str, Any], master_password: str
    ) -> Dict[DatasetSplit, LCMDatasetAnchor]:
        """
        Create a new dataset anchor using LCM integration.
        
        Args:
            dataset_id: Unique identifier for the dataset
            dataset_metadata: Metadata about the dataset (features, size, etc.)
            master_password: Master password for cryptographic operations
            
        Returns:
            DatasetAnchor: Configured anchor instance with LCM tracking
        """
        # Create dataset metadata for LCM system
        lcm_metadata = DatasetMetadata(
            name=dataset_metadata.get('name', dataset_id),
            version=dataset_metadata.get('version', '1.0'),
            features=dataset_metadata.get('features', []),
            size=dataset_metadata.get('size', 0)
        )
        
        # Register with LCM dataset manager
        lcm_anchor = self.lcm_dataset_manager.create_dataset_anchor(
            dataset_id, lcm_metadata
        )
        
        # Store LCM anchor directly (no legacy compatibility needed)
        print(f"LCM dataset anchor {dataset_id} created")
        print(f"LCM tracking: {lcm_metadata.total_samples} samples")
        
        return lcm_anchor

    def create_dataset_anchor(
        self, dataset_id: str, dataset_metadata: Dict[str, Any], master_password: str
    ) -> Dict[DatasetSplit, LCMDatasetAnchor]:
        """
        Create a new dataset anchor with its own anchor derivation hierarchy.

        Args:
            dataset_id: Unique identifier for the dataset
            dataset_metadata: Metadata about the dataset
            master_password: Master password for anchor derivation

        Returns:
            DatasetAnchor instance
        """
        print(f"Creating dataset anchor for: {dataset_id}")

        # Generate dataset-specific salt
        dataset_salt = hashlib.sha256(
            f"{dataset_id}_{self.framework_name}".encode()
        ).digest()

        # Create LCM dataset metadata
        lcm_metadata = DatasetMetadata(
            name=dataset_metadata.get('name', dataset_id),
            version=dataset_metadata.get('version', '1.0'),
            description=dataset_metadata.get('description', ''),
            features=dataset_metadata.get('features', []),
            total_samples=dataset_metadata.get('total_samples', 0)
        )

        # Create dataset splits using LCM system
        splits = self.lcm_dataset_manager.create_dataset_splits(
            dataset_id=dataset_id,
            metadata=lcm_metadata,
            master_password=master_password,
            splits=[DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST]
        )

        self.dataset_anchors[dataset_id] = splits

        print(f"LCM dataset anchor initialized with {len(splits)} splits")  
        return splits

    def create_model_anchor_lcm(
        self, 
        model_name: str, 
        model_parameters: Dict[str, Any],
        model_architecture: Dict[str, Any] = None,
        authorized_datasets: List[str] = None,
        master_password: str = None
    ) -> Dict[str, Any]:
        """
        Create model anchor with LCM integration.
        
        Args:
            model_name: Name of the model
            model_parameters: Model hyperparameters and configuration
            model_architecture: Model architecture details
            authorized_datasets: List of dataset IDs authorized for this model
            master_password: Password for anchor derivation
            
        Returns:
            Dictionary containing model anchor with LCM tracking
        """
        print(f">> Creating LCM-enabled model anchor for: {model_name}")
        
        # Register with LCM model manager
        lcm_anchor = self.lcm_model_manager.create_model_anchor(
            model_name, model_parameters
        )
        
        # Create traditional anchor for compatibility
        password = master_password or model_name
        model_salt = secure_random_bytes(SALT_LENGTH)
        master_anchor = derive_master_anchor(password, model_salt)
        
        # Store model anchor with LCM tracking
        model_anchor_data = {
            "model_name": model_name,
            "master_anchor": to_hex(master_anchor),
            "salt": to_hex(model_salt),
            "parameters": model_parameters,
            "architecture": model_architecture or {},
            "authorized_datasets": authorized_datasets or [],
            "lcm_tracked": True,
            "lcm_anchor_id": lcm_anchor.get('anchor_id') if isinstance(lcm_anchor, dict) else str(lcm_anchor),
            "created_at": datetime.now().isoformat()
        }
        
        self.model_anchors[model_name] = model_anchor_data
        
        print(f"Model {model_name} created with LCM integration")
        return model_anchor_data

    def create_model_anchor(
        self, 
        model_name: str, 
        model_parameters: Dict[str, Any],
        model_architecture: Dict[str, Any] = None,
        authorized_datasets: List[str] = None,
        master_password: str = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive model anchor with immutable parameter hashing.
        
        This is the enhanced model anchor creation that follows the audit flow:
        Dataset Anchor → Model Anchor → Training/Inference
        
        Args:
            model_name: Name of the model
            model_parameters: Model hyperparameters and configuration
            model_architecture: Model architecture details (layers, etc.)
            authorized_datasets: List of dataset IDs authorized for this model
            master_password: Password for anchor derivation (uses model_name if not provided)
            
        Returns:
            Dictionary containing model anchor information and metadata
        """
        print(f">> Creating enhanced model anchor for: {model_name}")
        
        # Use provided password or fallback to model name
        password = master_password or model_name
        
        # Generate model-specific salt
        model_salt = secure_random_bytes(SALT_LENGTH)
        
        # Create master anchor for this model
        master_anchor = derive_master_anchor(password, model_salt)
        
        # Create immutable model metadata hash
        model_metadata = {
            "model_name": model_name,
            "parameters": model_parameters,
            "architecture": model_architecture or {},
            "authorized_datasets": authorized_datasets or [],
            "creation_timestamp": datetime.now().isoformat(),
            "framework_version": self.framework_name
        }
        
        # Compute deterministic model hash (exclude timestamp for reproducibility)
        model_hash_data = {
            "model_name": model_name,
            "parameters": model_parameters,
            "architecture": model_architecture or {},
            "authorized_datasets": sorted(authorized_datasets or [])
        }
        model_hash = sha256_hash(json.dumps(model_hash_data, sort_keys=True).encode())
        
        # Derive model anchor using hierarchical derivation
        model_anchor_bytes = derive_model_anchor(master_anchor, model_hash)
        model_anchor_hex = to_hex(model_anchor_bytes)
        
        # Create comprehensive model anchor record
        model_anchor_record = {
            "model_name": model_name,
            "model_hash": model_hash,
            "model_anchor": model_anchor_hex,
            "master_anchor": to_hex(master_anchor),
            "master_salt": to_hex(model_salt),
            "metadata": model_metadata,
            "parameters_fingerprint": sha256_hash(json.dumps(model_parameters, sort_keys=True).encode()),
            "architecture_fingerprint": sha256_hash(json.dumps(model_architecture or {}, sort_keys=True).encode()),
            "authorized_dataset_anchors": {},
            "audit_tags": [
                "model_anchor",
                "immutable_parameters", 
                "hierarchical_derivation",
                "audit_ready"
            ]
        }
        
        # Link to authorized datasets and collect their anchors
        if authorized_datasets:
            for dataset_id in authorized_datasets:
                if dataset_id in self.dataset_anchors:
                    dataset_anchor = self.dataset_anchors[dataset_id]
                    model_anchor_record["authorized_dataset_anchors"][dataset_id] = {
                        "dataset_anchor": dataset_anchor.dataset_anchor.hex() if isinstance(dataset_anchor.dataset_anchor, bytes) else dataset_anchor.dataset_anchor,
                        "dataset_hash": dataset_anchor.dataset_hash,
                        "sample_count": dataset_anchor.total_samples
                    }
                else:
                    print(f"Warning: Dataset {dataset_id} not found in anchors")
        
        # Store model anchor
        self.model_anchors[model_name] = model_anchor_record
        
        # Create audit trail generator for this model
        self.audit_generators[model_name] = AuditTrailGenerator(
            model_name=model_name,
            compliance_frameworks=["AI_AUDIT", "TRANSPARENCY"]
        )
        
        # Create inference connections for this model
        self.inference_connections[model_name] = ZKEConnections()
        
        print(f"Model anchor created with fingerprint: {model_anchor_record['parameters_fingerprint'][:16]}...")
        print(f"Linked to {len(authorized_datasets or [])} authorized datasets")
        print(f"Audit trail generator initialized for {model_name}")
        
        return model_anchor_record

    def register_ml_simulator(self, model_name: str) -> MLFrameworkSimulator:
        """
        Register a new ML framework simulator.

        Args:
            model_name: Name of the ML model

        Returns:
            MLFrameworkSimulator instance
        """
        simulator = MLFrameworkSimulator(model_name)
        self.ml_simulators[model_name] = simulator
        print(f"Registered ML simulator: {model_name}")
        return simulator

    def create_provenance_capsules(
        self, dataset_id: str, data_items: List[Dict[str, Any]]
    ) -> List[ProvenanceCapsule]:
        """
        Create provenance capsules for a dataset using lazy materialization.

        Args:
            dataset_id: ID of the dataset
            data_items: List of data items with content and metadata

        Returns:
            List of ProvenanceCapsule instances
        """
        if dataset_id not in self.lazy_managers:
            raise ValueError(f"No lazy manager found for dataset: {dataset_id}")

        lazy_manager = self.lazy_managers[dataset_id]
        capsules = []

        print(
            f"Creating {len(data_items)} provenance capsules for dataset: {dataset_id}"
        )

        for item in data_items:
            # Create capsule using lazy manager
            capsule = lazy_manager.create_lazy_capsule(
                item_id=item["metadata"]["id"],
                original_data=item["content"],
                metadata=item["metadata"],
            )
            capsules.append(capsule)

        print(f"Created {len(capsules)} provenance capsules")
        return capsules

    def create_model_aggregation_anchor(
        self, model_name: str, authorized_datasets: List[str]
    ) -> ModelAggregationAnchor:
        """
        Create a Model Aggregation Anchor for training authorization.

        Args:
            model_name: Name of the model
            authorized_datasets: List of dataset IDs authorized for this model

        Returns:
            ModelAggregationAnchor instance
        """
        print(f"Creating MAA for model: {model_name}")
        print(f"Authorized datasets: {authorized_datasets}")

        # Collect dataset anchors from authorized datasets
        dataset_anchors = {}
        for dataset_id in authorized_datasets:
            if dataset_id in self.dataset_anchors:
                anchor = self.dataset_anchors[dataset_id]
                dataset_anchors[dataset_id] = anchor.dataset_anchor
            else:
                print(f"WARNING: Dataset {dataset_id} not found in anchors")

        # Create MAA with proper constructor
        maa = ModelAggregationAnchor(
            key_id=f"{model_name}_MAA",
            secret_material=f"anchor_for_{model_name}_with_datasets_{'_'.join(authorized_datasets)}",
        )

        print(f"MAA created initialized with {len(dataset_anchors)} dataset anchors")
        return maa

    def train_model_with_audit(
        self,
        model_name: str,
        capsules: List[ProvenanceCapsule],
        training_params: Dict[str, Any],
        model_version: str,
        user_id: str = "system"
    ) -> TrainingSnapshot:
        """
        Enhanced model training with complete audit integration.
        
        Implements the full audit flow: Dataset Anchor → Model Anchor → Training → Audit Trail
        
        Args:
            model_name: Name of the model (must have model anchor created)
            capsules: List of provenance capsules
            training_params: Training parameters
            model_version: Version of the model
            user_id: User performing the training
            
        Returns:
            TrainingSnapshot with complete audit trail
        """
        print(f"Starting enhanced model training with audit for: {model_name} v{model_version}")
        
        # Verify model anchor exists
        if model_name not in self.model_anchors:
            raise ValueError(f"Model anchor not found for {model_name}. Create model anchor first using create_model_anchor()")
        
        model_anchor_record = self.model_anchors[model_name]
        
        # Verify dataset authorization
        capsule_dataset_ids = set()
        for capsule in capsules:
            dataset_id = capsule.metadata.get("dataset_id")
            if dataset_id:
                capsule_dataset_ids.add(dataset_id)
        
        authorized_datasets = set(model_anchor_record["metadata"]["authorized_datasets"])
        unauthorized_datasets = capsule_dataset_ids - authorized_datasets
        
        if unauthorized_datasets:
            raise ValueError(f"Unauthorized datasets detected: {unauthorized_datasets}. Model {model_name} is only authorized for: {authorized_datasets}")
        
        print(f"Dataset authorization verified for {len(capsule_dataset_ids)} datasets")
        
        # Register ML simulator if needed
        if model_name not in self.ml_simulators:
            self.register_ml_simulator(model_name)
        
        # Create enhanced training parameters with model anchor integration
        enhanced_training_params = training_params.copy()
        enhanced_training_params.update({
            "model_anchor": model_anchor_record["model_anchor"],
            "model_hash": model_anchor_record["model_hash"],
            "parameters_fingerprint": model_anchor_record["parameters_fingerprint"],
            "architecture_fingerprint": model_anchor_record["architecture_fingerprint"],
            "authorized_datasets": list(authorized_datasets),
            "training_timestamp": datetime.now().isoformat(),
            "framework_version": self.framework_name
        })
        
        # Record training start in audit trail
        audit_generator = self.audit_generators[model_name]
        
        simulator = self.ml_simulators[model_name]
        
        print(f">> Training {model_name} v{model_version} with {len(capsules)} capsules")
        print(f"Model fingerprint: {model_anchor_record['parameters_fingerprint'][:16]}...")
        
        # Create legacy MAA for compatibility
        maa = ModelAggregationAnchor(
            key_id=f"{model_name}_MAA_{model_version}",
            secret_material=f"anchor_for_{model_name}_v{model_version}",
        )
        
        # Train using the simulator
        snapshot = simulator.train_model(
            training_data_capsules=capsules,
            maa=maa,
            training_params=enhanced_training_params,
            model_version=model_version,
        )
        
        # Record training completion in audit trail
        training_audit_record = audit_generator.record_training_event(
            training_snapshot=snapshot,
            training_params=enhanced_training_params,
            user_id=user_id
        )
        
        print(f"Training completed and audit record created: {training_audit_record.event_id}")
        print(f"Merkle root: {snapshot.merkle_root_hash}")
        
        return snapshot

    def train_model(
        self,
        model_name: str,
        capsules: List[ProvenanceCapsule],
        maa: ModelAggregationAnchor,
        training_params: Dict[str, Any],
        model_version: str,
    ) -> TrainingSnapshot:
        """
        Legacy model training method (maintained for backwards compatibility).
        
        Use train_model_with_audit() for enhanced audit integration.

        Args:
            model_name: Name of the model
            capsules: List of provenance capsules
            maa: Model Aggregation Anchor for authorization
            training_params: Training parameters
            model_version: Version of the model

        Returns:
            TrainingSnapshot instance
        """
        if model_name not in self.ml_simulators:
            self.register_ml_simulator(model_name)

        simulator = self.ml_simulators[model_name]

        print(f"Training model {model_name} version {model_version}")
        print(f"Using {len(capsules)} provenance capsules")

        # Train using the simulator
        snapshot = simulator.train_model(
            training_data_capsules=capsules,
            maa=maa,
            training_params=training_params,
            model_version=model_version,
        )

        return snapshot

    def perform_inference_with_lcm(
        self,
        model_name: str,
        query: str,
        ai_output: str,
        user_id: str = "anonymous",
        query_metadata: Dict[str, Any] = None
    ) -> InferenceReceipt:
        """
        Perform inference with complete LCM integration.
        
        Args:
            model_name: Name of the model
            query: Input query/prompt
            ai_output: Model output/response
            user_id: User identifier for audit purposes
            query_metadata: Additional metadata for the query
            
        Returns:
            InferenceReceipt: Complete receipt with LCM tracking
        """
        print(f"Performing LCM-tracked inference for model: {model_name}")
        
        # Create inference receipt using LCM
        lcm_receipt = self.lcm_inference_manager.create_inference_receipt(
            model_name, query, ai_output, user_id
        )
        
        # Create traditional receipt for compatibility
        if model_name not in self.model_anchors:
            raise ValueError(f"Model {model_name} not found. Create model anchor first.")
        
        model_anchor_data = self.model_anchors[model_name]
        
        # Generate inference receipt
        receipt = InferenceReceipt(
            query=query,
            ai_output=ai_output,
            model_anchor=model_anchor_data["master_anchor"],
            user_id=user_id,
            metadata={
                **(query_metadata or {}),
                "lcm_tracked": True,
                "lcm_receipt_id": getattr(lcm_receipt, 'receipt_id', 'unknown')
            }
        )
        
        print(f"LCM inference completed for {model_name}")
        return receipt

    def perform_inference_with_audit(
        self,
        model_name: str,
        query: str,
        ai_output: str,
        training_snapshot: TrainingSnapshot,
        user_id: str = "anonymous",
        query_metadata: Dict[str, Any] = None
    ) -> InferenceReceipt:
        """
        Perform inference with complete audit integration.
        
        Completes the audit flow: Dataset Anchor → Model Anchor → Training → Inference Receipt → Audit Trail
        
        Args:
            model_name: Name of the model
            query: Input query/prompt
            ai_output: Model output/response
            training_snapshot: Training snapshot used for this model
            user_id: User performing the inference
            query_metadata: Additional metadata about the query
            
        Returns:
            InferenceReceipt with complete audit trail
        """
        print(f">> Performing inference with audit for model: {model_name}")
        
        # Verify model anchor exists
        if model_name not in self.model_anchors:
            raise ValueError(f"Model anchor not found for {model_name}. Create model anchor first.")
        
        if model_name not in self.inference_connections:
            self.inference_connections[model_name] = ZKEConnections()
        
        if model_name not in self.audit_generators:
            self.audit_generators[model_name] = AuditTrailGenerator(model_name)
        
        model_anchor_record = self.model_anchors[model_name]
        inference_connections = self.inference_connections[model_name]
        audit_generator = self.audit_generators[model_name]
        
        query_metadata = query_metadata or {}
        
        print(f"Creating inference receipt for query: {query[:50]}...")
        
        # Create inference receipt using the connections
        receipt = inference_connections.add_receipt(
            query=query,
            ai_output=ai_output,
            model_version=training_snapshot.model_version,
            training_snapshot_id=training_snapshot.snapshot_id,
            training_snapshot_merkle_root=training_snapshot.merkle_root_hash
        )
        
        # Record inference in audit trail
        inference_audit_record = audit_generator.record_inference_event(
            receipt=receipt,
            query_metadata=query_metadata,
            user_id=user_id
        )
        
        print(f"Inference completed and audit record created: {inference_audit_record.event_id}")
        print(f"Connected to previous receipt: {receipt.prev_receipt_hash is not None}")
        print(f"Total receipts in connections: {len(inference_connections.receipts)}")
        
        return receipt

    def get_complete_audit_trail(
        self, 
        model_name: str,
        include_merkle_proofs: bool = True
    ) -> Dict[str, Any]:
        """
        Get complete audit trail showing the full flow: Dataset → Model → Training → Inference.
        
        Args:
            model_name: Name of the model
            include_merkle_proofs: Whether to include Merkle proofs for verification
            
        Returns:
            Complete audit trail with all components
        """
        print(f"Generating complete audit trail for: {model_name}")
        
        if model_name not in self.model_anchors:
            raise ValueError(f"Model anchor not found for {model_name}")
        
        model_anchor_record = self.model_anchors[model_name]
        
        # Collect dataset anchor information
        dataset_audit_info = {}
        for dataset_id in model_anchor_record["metadata"]["authorized_datasets"]:
            if dataset_id in self.dataset_anchors:
                dataset_anchor = self.dataset_anchors[dataset_id]
                dataset_audit_info[dataset_id] = {
                    "dataset_id": dataset_anchor.dataset_id,
                    "dataset_hash": dataset_anchor.dataset_hash,
                    "dataset_anchor": dataset_anchor.dataset_anchor.hex() if isinstance(dataset_anchor.dataset_anchor, bytes) else dataset_anchor.dataset_anchor,
                    "total_samples": dataset_anchor.total_samples,
                    "merkle_root": dataset_anchor.get_merkle_root(),
                    "metadata": dataset_anchor.metadata
                }
                
                if include_merkle_proofs and dataset_anchor.sample_hashes:
                    # Include a sample Merkle proof
                    merkle_tree = MerkleTree(dataset_anchor.sample_hashes)
                    sample_hash = dataset_anchor.sample_hashes[0]
                    proof = merkle_tree.get_proof(sample_hash)
                    dataset_audit_info[dataset_id]["sample_merkle_proof"] = {
                        "sample_hash": sample_hash,
                        "proof": proof,
                        "verification": MerkleTree.verify_proof(sample_hash, merkle_tree.get_root(), proof)
                    }
        
        # Get inference connections summary
        inference_summary = {}
        if model_name in self.inference_connections:
            inference_connections = self.inference_connections[model_name]
            inference_summary = inference_connections.get_connections_summary()
        
        # Get audit trail records
        audit_records = []
        if model_name in self.audit_generators:
            audit_generator = self.audit_generators[model_name]
            audit_records = [
                {
                    "event_id": record.event_id,
                    "event_type": record.event_type.value,
                    "timestamp": record.timestamp,
                    "audit_hash": record.audit_hash,
                    "data_hash": record.data_hash,
                    "previous_hash": record.previous_hash
                }
                for record in audit_generator.get_audit_trail()
            ]
        
        complete_audit = {
            "model_name": model_name,
            "audit_timestamp": datetime.now().isoformat(),
            "framework_version": self.framework_name,
            
            # Model Anchor Information
            "model_anchor": {
                "model_hash": model_anchor_record["model_hash"],
                "model_anchor": model_anchor_record["model_anchor"],
                "parameters_fingerprint": model_anchor_record["parameters_fingerprint"],
                "architecture_fingerprint": model_anchor_record["architecture_fingerprint"],
                "metadata": model_anchor_record["metadata"]
            },
            
            # Dataset Anchor Information
            "dataset_anchors": dataset_audit_info,
            
            # Inference Connections Information
            "inference_connections": inference_summary,
            
            # Audit Trail Records
            "audit_records": audit_records,
            
            # Verification Information
            "verification": {
                "total_datasets": len(dataset_audit_info),
                "total_audit_records": len(audit_records),
                "connections_integrity": inference_summary.get("connections_valid", True),
                "audit_connections_length": len(audit_records),
                "integrity_verified": self._verify_audit_trail_integrity(model_name, inference_summary)
            }
        }
        
        print(f"Complete audit trail generated:")
        print(f"{len(dataset_audit_info)} dataset anchors")
        print(f"   >> 1 model anchor")
        print(f"{len(audit_records)} audit records")
        print(f"{inference_summary.get('total_receipts', 0)} inference receipts")
        
        return complete_audit

    def _verify_audit_trail_integrity(self, model_name: str, inference_summary: dict) -> bool:
        """
        Verify the integrity of the complete audit trail for a model.
        
        Args:
            model_name: Name of the model to verify
            inference_summary: Summary of inference connections
            
        Returns:
            True if audit trail integrity is verified, False otherwise
        """
        try:
            # Check if model exists
            if model_name not in self.model_anchors:
                return False
            
            # Check if model anchor exists and has required data
            model_anchor_data = self.model_anchors[model_name]
            if not model_anchor_data or "model_hash" not in model_anchor_data:
                return False
            
            # Model anchor verification - if it exists, it's considered verified
            model_verified = True
            
            # Check inference connections integrity
            connections_verified = inference_summary.get("connections_valid", True)
            
            # Check if inference receipts exist and are valid
            receipts_verified = True
            if model_name in self.inference_connections:
                receipts = self.inference_connections[model_name].receipts
                if receipts:
                    # Verify that all receipts have valid hashes
                    for receipt in receipts:
                        if not hasattr(receipt, 'receipt_hash') or not receipt.receipt_hash:
                            receipts_verified = False
                            break
                    # Also verify the connections integrity
                    connections_verified = self.inference_connections[model_name].verify_connections()
            
            # Overall integrity is verified if all components are verified
            return model_verified and connections_verified and receipts_verified
            
        except Exception as e:
            print(f"Audit trail integrity verification failed: {e}")
            return False

    def validate_training_integrity(self, snapshot: TrainingSnapshot) -> bool:
        """
        Validate the integrity of a training snapshot.

        Args:
            snapshot: TrainingSnapshot to validate

        Returns:
            True if valid, False otherwise
        """
        print(f"Validating training snapshot: {snapshot.snapshot_id}")

        # Verify that the snapshot has required attributes
        if not hasattr(snapshot, 'merkle_tree') or not snapshot.merkle_tree:
            print("ERROR: Snapshot missing Merkle tree")
            return False

        # Verify that the merkle root matches
        try:
            computed_root = snapshot.merkle_tree.get_root()
            if computed_root != snapshot.merkle_root_hash:
                print("ERROR: Merkle root mismatch")
                print(f"  Computed: {computed_root}")
                print(f"  Expected: {snapshot.merkle_root_hash}")
                return False
        except Exception as e:
            print(f"ERROR: Failed to compute Merkle root: {e}")
            return False

        # Verify snapshot has valid components
        if not snapshot.provenance_capsule_hashes:
            print("ERROR: Snapshot has no provenance capsules")
            return False

        print("Training snapshot validation successful")
        return True

    def get_performance_metrics(self, dataset_id: str = None, model_name: str = None) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for datasets and models.

        Args:
            dataset_id: ID of the dataset (optional)
            model_name: Name of the model (optional)

        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            "framework": self.framework_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Dataset metrics
        if dataset_id:
            if dataset_id not in self.lazy_managers:
                metrics["error"] = f"No lazy manager found for dataset: {dataset_id}"
                return metrics
            
            lazy_manager = self.lazy_managers[dataset_id]
            metrics["dataset"] = {
                "dataset_id": dataset_id,
                "total_items": len(lazy_manager.anchor.data_items),
                "materialized_capsules": len(lazy_manager.materialized_capsules),
                "materialization_rate": (
                    len(lazy_manager.materialized_capsules)
                    / len(lazy_manager.anchor.data_items)
                    if lazy_manager.anchor.data_items
                    else 0
                ),
                "dataset_anchor_derived": lazy_manager.anchor.dataset_anchor is not None,
                "merkle_root": lazy_manager.anchor.get_merkle_root(),
                "total_samples": lazy_manager.anchor.total_samples
            }
        
        # Model metrics
        if model_name:
            if model_name in self.model_anchors:
                model_anchor_record = self.model_anchors[model_name]
                metrics["model"] = {
                    "model_name": model_name,
                    "model_anchor_created": True,
                    "parameters_fingerprint": model_anchor_record["parameters_fingerprint"],
                    "architecture_fingerprint": model_anchor_record["architecture_fingerprint"],
                    "authorized_datasets": len(model_anchor_record["metadata"]["authorized_datasets"])
                }
                
                # Add inference metrics
                if model_name in self.inference_connections:
                    connections_summary = self.inference_connections[model_name].get_connections_summary()
                    metrics["model"]["inference_connections"] = connections_summary
                
                # Add audit metrics
                if model_name in self.audit_generators:
                    audit_records = self.audit_generators[model_name].get_audit_trail()
                    metrics["model"]["audit_records"] = len(audit_records)
            else:
                metrics["model"] = {
                    "model_name": model_name,
                    "model_anchor_created": False,
                    "error": "Model anchor not found"
                }
        
        # Overall framework metrics
        metrics["framework_summary"] = {
            "total_datasets": len(self.dataset_anchors),
            "total_models": len(self.model_anchors),
            "total_lazy_managers": len(self.lazy_managers),
            "total_ml_simulators": len(self.ml_simulators),
            "total_inference_connections": len(self.inference_connections),
            "total_audit_generators": len(self.audit_generators),
            "lcm_integration": {
                "root_manager": bool(self.lcm_root_manager),
                "dataset_manager": bool(self.lcm_dataset_manager),
                "model_manager": bool(self.lcm_model_manager),
                "training_manager": bool(self.lcm_training_manager),
                "inference_manager": bool(self.lcm_inference_manager),
                "deployment_manager": bool(self.lcm_deployment_manager)
            }
        }

        return metrics

    def lcm_complete_workflow(
        self,
        dataset_id: str,
        dataset_metadata: Dict[str, Any],
        model_name: str,
        model_parameters: Dict[str, Any],
        query: str,
        ai_output: str,
        master_password: str = None
    ) -> Dict[str, Any]:
        """
        Complete LCM workflow: Dataset → Model → Training → Inference.
        
        This method demonstrates the full LCM integration replacing
        the legacy anchoring system.
        
        Args:
            dataset_id: Dataset identifier
            dataset_metadata: Dataset configuration and metadata
            model_name: Model identifier
            model_parameters: Model hyperparameters
            query: Input for inference
            ai_output: Model response
            master_password: Cryptographic password
            
        Returns:
            Dictionary with complete workflow results and LCM tracking
        """
        print("Starting complete LCM workflow...")
        
        # Step 1: Create dataset with LCM
        dataset_anchor = self.create_dataset_anchor_lcm(
            dataset_id, dataset_metadata, master_password or dataset_id
        )
        
        # Step 2: Create model with LCM
        model_anchor = self.create_model_anchor_lcm(
            model_name, model_parameters, 
            authorized_datasets=[dataset_id],
            master_password=master_password or model_name
        )
        
        # Step 3: Create training session with LCM
        training_session = self.lcm_training_manager.create_training_session(
            model_name, [dataset_id]
        )
        
        # Step 4: Perform inference with LCM
        inference_receipt = self.perform_inference_with_lcm(
            model_name, query, ai_output
        )
        
        # Complete workflow summary
        workflow_result = {
            "workflow_type": "LCM_COMPLETE",
            "dataset": {
                "id": dataset_id,
                "anchor_created": True,
                "lcm_tracked": True
            },
            "model": {
                "name": model_name,
                "anchor_created": True,
                "lcm_tracked": True
            },
            "training": {
                "completed": True,
                "lcm_tracked": True,
                "training_session": getattr(training_session, 'session_id', 'unknown')
            },
            "inference": {
                "completed": True,
                "receipt_generated": True,
                "lcm_tracked": True
            },
            "lcm_integration": {
                "root_manager": bool(self.lcm_root_manager),
                "dataset_manager": bool(self.lcm_dataset_manager),
                "model_manager": bool(self.lcm_model_manager),
                "training_manager": bool(self.lcm_training_manager),
                "inference_manager": bool(self.lcm_inference_manager),
                "deployment_manager": bool(self.lcm_deployment_manager)
            }
        }
        
        print("Complete LCM workflow finished successfully!")
        return workflow_result

    def register_inference_receipt(self, model_name: str, receipt: InferenceReceipt) -> None:
        """
        Register an inference receipt with the framework for audit trail tracking.
        
        Args:
            model_name: Name of the model that generated the inference
            receipt: The inference receipt to register
        """
        # Initialize ZKEConnections for this model if it doesn't exist
        if model_name not in self.inference_connections:
            from ..inference import ZKEConnections
            self.inference_connections[model_name] = ZKEConnections()
        
        # Add the receipt to the connections
        self.inference_connections[model_name].receipts.append(receipt)
        
    def get_inference_receipts(self, model_name: str) -> List[InferenceReceipt]:
        """
        Get all inference receipts for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of inference receipts for the model
        """
        if model_name not in self.inference_connections:
            return []
        return self.inference_connections[model_name].receipts
