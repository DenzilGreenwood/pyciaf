"""
CIAF LCM Dataset Manager

Enhanced dataset management supporting train/validation/test splits with proper anchoring.

Created: 2025-09-09
Last Modified: 2026-03-30
Author: Denzil James Greenwood
Version: 2.0.0 - Converted to Pydantic models
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from ..core import sha256_hash, SALT_LENGTH, to_hex
from .policy import (
    LCMPolicy,
    get_default_policy,
    CommitmentType,
    create_commitment,
    canonical_hash,
)

if TYPE_CHECKING:
    pass


def compute_split_assignment_digest(
    record_ids: List[str], salt: Optional[bytes] = None
) -> str:
    """
    Compute split assignment digest for audit reproducibility.

    Args:
        record_ids: List of stable record IDs or pre-hashed row digests
        salt: Optional salt for privacy-preserving commitment

    Returns:
        SHA-256 digest of the sorted record assignment
    """
    # Sort record IDs for deterministic output
    items = sorted(record_ids)

    if salt:
        # Privacy-preserving variant: hash each record ID with salt
        items = [sha256_hash(salt + rid.encode("utf-8")) for rid in items]

    # Option 1: Flat digest (simple approach)
    concat = "".join(items).encode("utf-8")
    return sha256_hash(concat)

    # Option 2: Merkle root (stronger proof, uncomment if preferred)
    # return MerkleTree(items).get_root()


class DatasetSplit(Enum):
    """Dataset split types."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
    FULL = "full"


class DatasetMetadata(BaseModel):
    """Enhanced dataset metadata for LCM with comprehensive dataset information."""

    name: str = Field(..., min_length=1, description="Dataset name")
    owner: str = Field("unknown", description="Dataset owner")
    license: str = Field("unknown", description="Dataset license")
    schema_digest: str = Field("", description="Schema digest")
    sampling_rules: Dict[str, Any] = Field(
        default_factory=dict, description="Sampling rules"
    )
    version: str = Field("1.0.0", description="Dataset version")
    content_root: str = Field("", description="Merkle root or rolling hash")

    # Privacy and compliance
    contains_pii: bool = Field(False, description="Contains PII")
    privacy_level: str = Field("public", description="Privacy level")
    compliance_frameworks: List[str] = Field(
        default_factory=list, description="Compliance frameworks"
    )

    # Additional metadata
    creation_date: Optional[str] = Field(None, description="Creation date")
    description: str = Field("", description="Dataset description")
    tags: List[str] = Field(default_factory=list, description="Dataset tags")

    # RNG reproducibility
    rng_seed: Optional[int] = Field(None, description="RNG seed")
    rng_source: Optional[str] = Field(None, description="RNG source")

    # Stratification
    stratify_by: Optional[List[str]] = Field(None, description="Stratification columns")

    # Split assignment commitment
    split_assignment_digest: Optional[str] = Field(
        None, description="Split assignment digest"
    )

    # Dataset structure and features
    num_samples: Optional[int] = Field(None, ge=0, description="Number of samples")
    num_features: Optional[int] = Field(None, ge=0, description="Number of features")
    feature_names: Optional[List[str]] = Field(None, description="Feature names")
    features: Optional[List[str]] = Field(None, description="Feature names (legacy)")
    total_samples: Optional[int] = Field(
        None, ge=0, description="Total samples (legacy)"
    )
    feature_types: Optional[Dict[str, str]] = Field(None, description="Feature types")
    target_column: Optional[str] = Field(None, description="Target column")
    target_type: Optional[str] = Field(None, description="Target type")

    # Feature statistics
    feature_statistics: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, description="Feature statistics"
    )
    missing_values: Optional[Dict[str, int]] = Field(
        None, description="Missing values count"
    )
    categorical_mappings: Optional[Dict[str, Dict[str, int]]] = Field(
        None, description="Categorical mappings"
    )

    # Data quality metrics
    duplicate_rows: Optional[int] = Field(
        None, ge=0, description="Duplicate rows count"
    )
    data_quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Quality score"
    )
    outlier_count: Optional[Dict[str, int]] = Field(
        None, description="Outlier count per feature"
    )

    # Dataset shape and dimensions
    data_shape: Optional[tuple] = Field(None, description="Data shape (rows, columns)")
    file_format: Optional[str] = Field(None, description="File format")
    encoding: Optional[str] = Field(None, description="File encoding")

    # Domain-specific metadata
    domain: Optional[str] = Field(
        None, description="Domain (healthcare, finance, etc.)"
    )
    task_type: Optional[str] = Field(None, description="Task type")
    benchmark_dataset: Optional[bool] = Field(False, description="Is benchmark dataset")

    # Data lineage and provenance
    source_datasets: Optional[List[str]] = Field(None, description="Source datasets")
    preprocessing_steps: Optional[List[str]] = Field(
        None, description="Preprocessing steps"
    )
    data_collection_method: Optional[str] = Field(
        None, description="Data collection method"
    )

    # Temporal information
    temporal_coverage: Optional[Dict[str, str]] = Field(
        None, description="Temporal coverage"
    )
    update_frequency: Optional[str] = Field(None, description="Update frequency")
    last_updated: Optional[str] = Field(None, description="Last updated timestamp")

    # Geographical information
    geographical_coverage: Optional[str] = Field(
        None, description="Geographical coverage"
    )

    # Bias and fairness considerations
    known_biases: Optional[List[str]] = Field(None, description="Known biases")
    protected_attributes: Optional[List[str]] = Field(
        None, description="Protected attributes"
    )
    fairness_constraints: Optional[Dict[str, Any]] = Field(
        None, description="Fairness constraints"
    )

    @model_validator(mode="after")
    def initialize_defaults_and_legacy_compatibility(self) -> "DatasetMetadata":
        """Initialize default values and handle legacy compatibility."""
        if self.creation_date is None:
            self.creation_date = datetime.now().isoformat()
        if self.last_updated is None:
            self.last_updated = self.creation_date
        if self.source_datasets is None:
            self.source_datasets = []
        if self.preprocessing_steps is None:
            self.preprocessing_steps = []
        if self.known_biases is None:
            self.known_biases = []
        if self.protected_attributes is None:
            self.protected_attributes = []

        # Legacy compatibility mappings
        if self.features is not None and self.feature_names is None:
            self.feature_names = self.features
        elif self.feature_names is not None and self.features is None:
            self.features = self.feature_names

        if self.total_samples is not None and self.num_samples is None:
            self.num_samples = self.total_samples
        elif self.num_samples is not None and self.total_samples is None:
            self.total_samples = self.num_samples

        return self

    @property
    def sample_count(self) -> Optional[int]:
        """Return total_samples as sample_count (backward compatibility)."""
        return self.total_samples

    def add_feature_statistics(self, feature_name: str, stats: Dict[str, Any]) -> None:
        """Add statistics for a specific feature."""
        if self.feature_statistics is None:
            self.feature_statistics = {}
        self.feature_statistics[feature_name] = stats

    def set_feature_type(self, feature_name: str, feature_type: str) -> None:
        """Set the type for a specific feature."""
        if self.feature_types is None:
            self.feature_types = {}
        self.feature_types[feature_name] = feature_type

    def add_categorical_mapping(
        self, feature_name: str, mapping: Dict[str, int]
    ) -> None:
        """Add categorical value mapping for a feature."""
        if self.categorical_mappings is None:
            self.categorical_mappings = {}
        self.categorical_mappings[feature_name] = mapping

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get a summary of dataset features."""
        return {
            "num_features": self.num_features,
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "target_column": self.target_column,
            "target_type": self.target_type,
            "has_missing_values": bool(
                self.missing_values
                and any(count > 0 for count in self.missing_values.values())
            ),
            "categorical_features": (
                list(self.categorical_mappings.keys())
                if self.categorical_mappings
                else []
            ),
        }

    def validate_metadata(self) -> List[str]:
        """Validate metadata consistency and return any issues found."""
        issues = []

        # Check feature consistency
        if self.feature_names and self.num_features:
            if len(self.feature_names) != self.num_features:
                issues.append(
                    f"Feature count mismatch: {len(self.feature_names)} names vs {self.num_features} declared"
                )

        # Check data shape consistency
        if self.data_shape and self.num_samples:
            if self.data_shape[0] != self.num_samples:
                issues.append(
                    f"Sample count mismatch: {self.data_shape[0]} in shape vs {self.num_samples} declared"
                )

        if self.data_shape and self.num_features:
            if len(self.data_shape) > 1 and self.data_shape[1] != self.num_features:
                issues.append(
                    f"Feature count mismatch: {self.data_shape[1]} in shape vs {self.num_features} declared"
                )

        # Check target column exists in features
        if self.target_column and self.feature_names:
            if self.target_column not in self.feature_names:
                issues.append(
                    f"Target column '{self.target_column}' not found in feature names"
                )

        # Check feature statistics consistency
        if self.feature_statistics and self.feature_names:
            stats_features = set(self.feature_statistics.keys())
            declared_features = set(self.feature_names)
            if not stats_features.issubset(declared_features):
                extra_stats = stats_features - declared_features
                issues.append(
                    f"Feature statistics for undeclared features: {list(extra_stats)}"
                )

        return issues


def create_dataset_metadata_from_dataframe(
    df,
    name: str,
    owner: str,
    license: str = "unknown",
    target_column: str = None,
    domain: str = None,
    description: str = "",
    **kwargs,
) -> DatasetMetadata:
    """
    Create DatasetMetadata from a pandas DataFrame or similar data structure.

    Args:
        df: DataFrame-like object with data
        name: Dataset name
        owner: Dataset owner
        license: License information
        target_column: Name of the target/label column
        domain: Domain of the dataset
        description: Dataset description
        **kwargs: Additional metadata fields

    Returns:
        DatasetMetadata instance with auto-populated fields
    """
    import hashlib

    # Basic information
    num_samples, num_features = (
        df.shape
        if hasattr(df, "shape")
        else (len(df), len(df.columns) if hasattr(df, "columns") else 0)
    )
    feature_names = list(df.columns) if hasattr(df, "columns") else None

    # Generate schema digest
    schema_info = {
        "columns": feature_names,
        "dtypes": (
            {col: str(df[col].dtype) for col in feature_names} if feature_names else {}
        ),
        "shape": (num_samples, num_features),
    }
    schema_str = str(sorted(schema_info.items()))
    schema_digest = hashlib.sha256(schema_str.encode()).hexdigest()

    # Auto-detect feature types
    feature_types = {}
    missing_values = {}
    feature_statistics = {}
    categorical_mappings = {}

    if feature_names and hasattr(df, "dtypes"):
        for col in feature_names:
            dtype = str(df[col].dtype)
            series = df[col]

            # Determine feature type
            if "int" in dtype or "float" in dtype:
                feature_types[col] = "numerical"
                # Basic statistics for numerical features
                if hasattr(series, "describe"):
                    stats = series.describe()
                    feature_statistics[col] = {
                        "mean": float(stats["mean"]) if "mean" in stats else None,
                        "std": float(stats["std"]) if "std" in stats else None,
                        "min": float(stats["min"]) if "min" in stats else None,
                        "max": float(stats["max"]) if "max" in stats else None,
                        "median": (
                            float(series.median())
                            if hasattr(series, "median")
                            else None
                        ),
                    }
            elif "object" in dtype or "category" in dtype or "string" in dtype:
                feature_types[col] = "categorical"
                # Categorical statistics
                if hasattr(series, "value_counts"):
                    value_counts = series.value_counts()
                    categorical_mappings[col] = dict(
                        value_counts.head(20)
                    )  # Top 20 categories
                    feature_statistics[col] = {
                        "unique_values": len(value_counts),
                        "most_common": (
                            value_counts.index[0] if len(value_counts) > 0 else None
                        ),
                        "most_common_count": (
                            value_counts.iloc[0] if len(value_counts) > 0 else None
                        ),
                    }
            elif "datetime" in dtype:
                feature_types[col] = "datetime"
                if hasattr(series, "min") and hasattr(series, "max"):
                    feature_statistics[col] = {
                        "min_date": str(series.min()),
                        "max_date": str(series.max()),
                        "date_range_days": (
                            (series.max() - series.min()).days
                            if hasattr(series.max() - series.min(), "days")
                            else None
                        ),
                    }
            else:
                feature_types[col] = "other"

            # Count missing values
            if hasattr(series, "isnull"):
                missing_values[col] = int(series.isnull().sum())

    # Detect target type if target column specified
    target_type = None
    if target_column and target_column in feature_types:
        if feature_types[target_column] == "numerical":
            # Check if it looks like classification (few unique values) or regression
            if hasattr(df[target_column], "nunique"):
                unique_count = df[target_column].nunique()
                if unique_count <= 20:  # Heuristic for classification
                    target_type = "classification"
                else:
                    target_type = "regression"
        elif feature_types[target_column] == "categorical":
            target_type = "classification"

    # Calculate data quality score
    total_cells = num_samples * num_features if num_features > 0 else 0
    total_missing = sum(missing_values.values()) if missing_values else 0
    data_quality_score = 1.0 - (total_missing / total_cells) if total_cells > 0 else 1.0

    # Count duplicate rows
    duplicate_rows = None
    if hasattr(df, "duplicated"):
        duplicate_rows = int(df.duplicated().sum())

    return DatasetMetadata(
        name=name,
        owner=owner,
        license=license,
        schema_digest=schema_digest,
        sampling_rules=kwargs.get("sampling_rules", {}),
        version=kwargs.get("version", "1.0.0"),
        content_root=kwargs.get("content_root", ""),
        description=description,
        domain=domain,
        # Auto-detected fields
        num_samples=num_samples,
        num_features=num_features,
        feature_names=feature_names,
        feature_types=feature_types,
        target_column=target_column,
        target_type=target_type,
        feature_statistics=feature_statistics,
        missing_values=missing_values,
        categorical_mappings=categorical_mappings,
        data_quality_score=data_quality_score,
        duplicate_rows=duplicate_rows,
        data_shape=(num_samples, num_features),
        # Pass through any additional kwargs
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["sampling_rules", "version", "content_root"]
        },
    )


class LCMDatasetAnchor:
    """Enhanced dataset anchor for LCM with split support using protocol interfaces."""

    def __init__(
        self,
        dataset_id: str,
        split: DatasetSplit,
        metadata: DatasetMetadata,
        master_password: str,
        policy: LCMPolicy = None,
        salt: bytes = None,
    ):
        """
        Initialize LCM dataset anchor using protocol interfaces.

        Args:
            dataset_id: Unique identifier for the dataset
            split: Dataset split type (train/val/test)
            metadata: Dataset metadata
            master_password: Master password for anchor derivation
            policy: LCM policy (uses default if None)
            salt: Salt for anchor derivation (generates if None)
        """
        self.dataset_id = dataset_id
        self.split = split
        self.metadata = metadata
        self.policy = policy or get_default_policy()

        # Get protocol implementations from policy
        self.rng = self.policy.rng
        self.anchor_deriver = self.policy.anchor_deriver
        self.merkle_factory = self.policy.merkle_factory

        # Generate or use provided salt
        if salt is not None:
            self.master_salt = salt
        else:
            self.master_salt = self.rng.random_bytes(SALT_LENGTH)

        # Derive anchors using protocol
        self.master_anchor = self.anchor_deriver.derive_master_anchor(
            master_password, self.master_salt
        )

        # Compute dataset hash including split information
        self.dataset_hash = self._compute_dataset_hash()
        self.dataset_anchor = self.anchor_deriver.derive_dataset_anchor(
            self.master_anchor, self.dataset_hash
        )

        # Sample tracking with Merkle tree
        self._merkle_tree = self.merkle_factory([])
        self.sample_hashes: List[str] = []
        self.total_samples = 0

        # Generate anchor ID with split prefix
        split_prefix = {
            DatasetSplit.TRAIN: "t_",
            DatasetSplit.VALIDATION: "v_",
            DatasetSplit.TEST: "x_",
            DatasetSplit.FULL: "f_",
        }.get(
            split, "u_"
        )  # Use 'u_' for unknown splits

        self.anchor_id = f"{split_prefix}{to_hex(self.dataset_anchor)[:8]}..."

        print(
            f"LCM Dataset Anchor '{self.dataset_id}' ({self.split.value}) initialized with anchor: {self.anchor_id}"
        )

    @property
    def split_type(self) -> str:
        """Return split type as string (backward compatibility property)."""
        return self.split.value

    def _compute_dataset_hash(self) -> str:
        """Compute canonical hash of dataset metadata including split and RNG info."""
        # Create canonical representation including RNG reproducibility fields
        hash_data = {
            "dataset_id": self.dataset_id,
            "split": self.split.value,
            "name": self.metadata.name,
            "owner": self.metadata.owner,
            "license": self.metadata.license,
            "schema_digest": self.metadata.schema_digest,
            "sampling_rules": self.metadata.sampling_rules,
            "version": self.metadata.version,
            "content_root": self.metadata.content_root,
            "contains_pii": self.metadata.contains_pii,
            "privacy_level": self.metadata.privacy_level,
            "compliance_frameworks": sorted(self.metadata.compliance_frameworks),
            "tags": sorted(self.metadata.tags),
            "rng_seed": self.metadata.rng_seed,
            "rng_source": self.metadata.rng_source,
            "stratify_by": self.metadata.stratify_by,
            "split_assignment_digest": self.metadata.split_assignment_digest,
        }

        return canonical_hash(hash_data)

    def add_sample_hash(self, sample_hash: str) -> None:
        """Add a sample hash to the dataset."""
        if sample_hash not in self.sample_hashes:
            self.sample_hashes.append(sample_hash)
            self.total_samples = len(self.sample_hashes)
            # Rebuild Merkle tree with new samples
            self._merkle_tree = self.merkle_factory(self.sample_hashes)

    def get_merkle_root(self) -> Optional[str]:
        """Get Merkle root of all sample hashes."""
        if not self.sample_hashes:
            return None
        return self._merkle_tree.get_root()

    def create_commitment(
        self, data: Any, commitment_type: CommitmentType = None
    ) -> str:
        """Create commitment for data according to policy using protocol RNG."""
        commitment_type = commitment_type or self.policy.commitments
        return create_commitment(data, commitment_type, self.dataset_anchor, self.rng)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_id": self.dataset_id,
            "name": f"{self.metadata.name}_{self.split.value}",
            "anchor": self.anchor_id,
            "commitment": self.policy.commitments.value,
            "split": self.split.value,
            "dataset_hash": self.dataset_hash,
            "total_samples": self.total_samples,
            "merkle_root": self.get_merkle_root(),
            "metadata": {
                "owner": self.metadata.owner,
                "version": self.metadata.version,
                "contains_pii": self.metadata.contains_pii,
                "privacy_level": self.metadata.privacy_level,
                "compliance_frameworks": self.metadata.compliance_frameworks,
            },
        }

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-compatible dictionary."""
        return self.to_dict()


class LCMDatasetManager:
    """
    Enhanced dataset manager supporting train/validation/test splits.
    """

    def __init__(self, policy: LCMPolicy = None):
        """Initialize LCM dataset manager."""
        self.policy = policy or get_default_policy()
        self.dataset_anchors: Dict[str, Dict[DatasetSplit, LCMDatasetAnchor]] = {}

    def create_dataset_splits(
        self,
        dataset_id: str,
        metadata: DatasetMetadata,
        master_password: str,
        splits: List[DatasetSplit] = None,
    ) -> Dict[DatasetSplit, LCMDatasetAnchor]:
        """
        Create dataset anchors for train/validation/test splits.

        Args:
            dataset_id: Unique identifier for the dataset
            metadata: Dataset metadata
            master_password: Master password for anchor derivation
            splits: List of splits to create (defaults to train/val/test)

        Returns:
            Dictionary mapping splits to their anchors
        """
        if splits is None:
            splits = [DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST]

        print(
            f"🗃️ Creating dataset anchors for {dataset_id} with splits: {[s.value for s in splits]}"
        )

        dataset_splits = {}

        base_seed = 42  # Default seed for reproducible splits

        for split in splits:
            # Generate deterministic seed for this split
            split_seed = base_seed + hash(split.value) % 10000

            # Generate mock record IDs for this split (simulate actual data assignment)
            num_samples = {"train": 1000, "val": 200, "test": 200}.get(split.value, 100)
            record_ids = [
                f"{dataset_id}_{split.value}_record_{i}" for i in range(num_samples)
            ]

            # Compute split assignment digest
            assignment_digest = compute_split_assignment_digest(record_ids)

            # Create split-specific metadata with RNG info
            split_metadata = DatasetMetadata(
                name=f"{metadata.name}_{split.value}",
                owner=metadata.owner,
                license=metadata.license,
                schema_digest=metadata.schema_digest,
                sampling_rules=metadata.sampling_rules,
                version=metadata.version,
                content_root=f"{metadata.content_root}_{split.value}",
                contains_pii=metadata.contains_pii,
                privacy_level=metadata.privacy_level,
                compliance_frameworks=metadata.compliance_frameworks,
                description=f"{metadata.description} ({split.value} split)",
                tags=metadata.tags + [f"split:{split.value}"],
                rng_seed=split_seed,
                rng_source="random",  # Using Python's random module
                stratify_by=metadata.stratify_by,
                split_assignment_digest=assignment_digest,
            )

            # Create anchor for this split
            anchor = LCMDatasetAnchor(
                dataset_id=f"{dataset_id}_{split.value}",
                split=split,
                metadata=split_metadata,
                master_password=master_password,
                policy=self.policy,
            )

            dataset_splits[split] = anchor

        self.dataset_anchors[dataset_id] = dataset_splits

        print(f"✅ Created {len(splits)} dataset anchors for {dataset_id}")
        return dataset_splits

    def create_dataset_anchor(
        self,
        dataset_id: str,
        dataset_path: str,
        split_type: str,
        master_password: str = "test_password",
    ) -> LCMDatasetAnchor:
        """
        Create a single dataset anchor (backward compatibility helper).

        Args:
            dataset_id: Dataset identifier
            dataset_path: Path to dataset
            split_type: Split type (train, val, test, full)
            master_password: Master password for anchor derivation

        Returns:
            LCMDatasetAnchor for the specified split
        """
        # Convert string to DatasetSplit enum
        split_map = {
            "train": DatasetSplit.TRAIN,
            "val": DatasetSplit.VALIDATION,
            "validation": DatasetSplit.VALIDATION,
            "test": DatasetSplit.TEST,
            "full": DatasetSplit.FULL,
        }
        split = split_map.get(split_type.lower(), DatasetSplit.TRAIN)

        # Create metadata  
        metadata = DatasetMetadata(
            name=dataset_id,
            source_path=dataset_path,
            row_count=1000,
            column_count=10,
            dataset_hash=f"hash_{dataset_id}_{split_type}",
        )

        # Create anchor directly (don't modify dataset_id)
        anchor = LCMDatasetAnchor(
            dataset_id=dataset_id,  # Keep original ID
            split=split,
            metadata=metadata,
            master_password=master_password,
            policy=self.policy,
        )

        return anchor

    def get_datasets_root_anchor(self, dataset_id: str) -> str:
        """
        Compute datasets root anchor from all splits.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Merkle root of all split anchors
        """
        if dataset_id not in self.dataset_anchors:
            raise ValueError(f"Dataset {dataset_id} not found")

        splits = self.dataset_anchors[dataset_id]

        # Collect all split anchor hashes
        split_hashes = []
        for split in [DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST]:
            if split in splits:
                split_hashes.append(splits[split].dataset_hash)

        if not split_hashes:
            raise ValueError(f"No splits found for dataset {dataset_id}")

        # Compute Merkle root using protocol
        policy = get_default_policy()
        merkle_tree = policy.merkle_factory(split_hashes)
        return merkle_tree.get_root()

    def compute_enhanced_split_map_digest(self, dataset_id: str) -> str:
        """
        Compute enhanced split map digest including assignment digests.

        This provides stronger audit guarantees by including actual membership
        information, not just that splits existed.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Enhanced split map digest
        """
        if dataset_id not in self.dataset_anchors:
            raise ValueError(f"Dataset {dataset_id} not found")

        splits = self.dataset_anchors[dataset_id]

        split_map = {}
        for split, anchor in splits.items():
            split_map[split.value] = {
                "anchor": anchor.dataset_hash,  # existing: proves split exists
                "assignment": anchor.metadata.split_assignment_digest,  # new: proves membership
            }

        # Compute digest: H(canon(split_map))
        return canonical_hash(split_map)

    def get_dataset_anchor(
        self, dataset_id: str, split: DatasetSplit
    ) -> Optional[LCMDatasetAnchor]:
        """Get dataset anchor for specific split."""
        if (
            dataset_id in self.dataset_anchors
            and split in self.dataset_anchors[dataset_id]
        ):
            return self.dataset_anchors[dataset_id][split]
        return None

    def get_all_splits(self, dataset_id: str) -> Dict[DatasetSplit, LCMDatasetAnchor]:
        """Get all splits for a dataset."""
        return self.dataset_anchors.get(dataset_id, {})

    def add_samples_to_split(
        self, dataset_id: str, split: DatasetSplit, samples: List[Dict[str, Any]]
    ) -> None:
        """Add samples to a specific dataset split."""
        anchor = self.get_dataset_anchor(dataset_id, split)
        if not anchor:
            raise ValueError(f"Dataset anchor not found: {dataset_id}_{split.value}")

        for sample in samples:
            # Create sample hash
            sample_data = json.dumps(sample, sort_keys=True)
            sample_hash = sha256_hash(sample_data.encode("utf-8"))
            anchor.add_sample_hash(sample_hash)

        print(f"✅ Added {len(samples)} samples to {dataset_id}_{split.value}")

    def format_datasets_summary(self, dataset_id: str) -> str:
        """Format dataset splits summary for pretty printing."""
        if dataset_id not in self.dataset_anchors:
            return f"Dataset {dataset_id} not found"

        splits = self.dataset_anchors[dataset_id]
        lines = []

        for split in [DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST]:
            if split in splits:
                anchor = splits[split]
                commitment = anchor.create_commitment(anchor.metadata.name)
                lines.append(
                    f"  ✅ {split.value:<5}: {anchor.metadata.name:<20} (anchor= {anchor.anchor_id}, commitment={commitment})"
                )

        datasets_root = self.get_datasets_root_anchor(dataset_id)
        lines.append(
            f"  🌳 datasets_root_anchor: {datasets_root[:4]}...{datasets_root[-4:]}"
        )

        return "\n".join(lines)

    def simulate_dataset_anchor(
        self, dataset_id: str, dataset_path: str, split_type: str = "train"
    ) -> LCMDatasetAnchor:
        """
        Simulate dataset anchor creation for demonstration.

        Args:
            dataset_id: Dataset identifier
            dataset_path: Path to dataset file
            split_type: Type of split (train, validation, test)

        Returns:
            LCMDatasetAnchor instance
        """
        print(f"📊 Creating dataset anchor: {dataset_id} ({split_type})")

        # Convert string to enum
        if split_type == "train":
            split = DatasetSplit.TRAIN
        elif split_type == "validation":
            split = DatasetSplit.VALIDATION
        elif split_type == "test":
            split = DatasetSplit.TEST
        else:
            split = DatasetSplit.TRAIN

        # Create mock dataset metadata
        metadata = DatasetMetadata(
            name=dataset_id,
            owner="ciaf_user",
            license="Example License",
            schema_digest="",  # Will be computed
            sampling_rules={},
            version="1.0.0",
            content_root="",  # Will be computed
            description=f"Mock dataset for {split_type}",
        )

        # Create dataset splits if not exists
        if dataset_id not in self.dataset_anchors:
            self.create_dataset_splits(
                dataset_id=dataset_id,
                metadata=metadata,
                master_password="demo_password",
            )

        # Return the requested split
        anchor = self.get_dataset_anchor(dataset_id, split)
        print(f"✅ Dataset anchor created: {anchor.anchor_id}")
        return anchor
