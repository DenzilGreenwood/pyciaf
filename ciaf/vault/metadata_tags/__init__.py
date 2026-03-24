"""
CIAF Metadata Tags Module

This module provides metadata tagging capabilities for AI outputs,
similar to EXIF data for images. These tags enable deepfake detection,
misinformation defense, and regulatory compliance tracking.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class CIAFTagVersion(Enum):
    """Versions of CIAF tag format."""

    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


class ContentType(Enum):
    """Types of AI-generated content."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    DATA = "data"
    MULTIMODAL = "multimodal"


class AIModelType(Enum):
    """Types of AI models."""

    LLM = "large_language_model"
    DIFFUSION = "diffusion_model"
    GAN = "generative_adversarial_network"
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    ENSEMBLE = "ensemble"
    TRANSFORMER = "transformer"
    CNN = "convolutional_neural_network"
    RNN = "recurrent_neural_network"


@dataclass
class CIAFMetadataTag:
    """CIAF metadata tag structure."""

    # Core identification
    ciaf_version: str
    tag_id: str
    timestamp: str
    content_type: ContentType

    # Model information
    model_name: str
    model_version: str
    model_type: AIModelType
    model_hash: str

    # Provenance information
    training_snapshot_id: str
    dataset_anchor_id: str
    inference_receipt_hash: str

    # Compliance and governance
    compliance_level: str
    regulatory_frameworks: List[str]
    intended_use: str
    restrictions: List[str]

    # Technical metadata
    confidence_score: float
    uncertainty_estimate: Dict[str, float]
    explainability_available: bool

    # Ownership and licensing
    creator: str
    organization: str
    license: str
    contact_info: str

    # Optional extended metadata
    custom_fields: Optional[Dict[str, Any]] = None
    parent_content_hash: Optional[str] = None
    generation_params: Optional[Dict[str, Any]] = None
    watermark_data: Optional[str] = None


class CIAFTagGenerator:
    """Generator for CIAF metadata tags."""

    def __init__(
        self,
        default_creator: str = "CIAF Framework",
        default_organization: str = "Unknown",
        default_license: str = "Proprietary",
    ):
        """
        Initialize tag generator.

        Args:
            default_creator: Default creator name
            default_organization: Default organization name
            default_license: Default license type
        """
        self.default_creator = default_creator
        self.default_organization = default_organization
        self.default_license = default_license

    def create_tag(
        self,
        content: Any,
        content_type: ContentType,
        model_name: str,
        model_version: str,
        model_type: AIModelType,
        training_snapshot_id: str,
        dataset_anchor_id: str,
        inference_receipt_hash: str,
        confidence_score: float = 0.0,
        uncertainty_estimate: Optional[Dict[str, float]] = None,
        explainability_available: bool = False,
        intended_use: str = "General AI application",
        restrictions: Optional[List[str]] = None,
        regulatory_frameworks: Optional[List[str]] = None,
        creator: Optional[str] = None,
        organization: Optional[str] = None,
        license: Optional[str] = None,
        contact_info: str = "contact@ciaf.org",  # Default contact info need to review this and update
        custom_fields: Optional[Dict[str, Any]] = None,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> CIAFMetadataTag:
        """Create a CIAF metadata tag for AI-generated content."""

        # Generate unique tag ID
        tag_id = self._generate_tag_id(content, model_name, training_snapshot_id)

        # Generate model hash for integrity
        model_hash = self._generate_model_hash(
            model_name, model_version, training_snapshot_id
        )

        # Set defaults
        uncertainty_estimate = uncertainty_estimate or {
            "total": 0.5,
            "aleatoric": 0.3,
            "epistemic": 0.2,
        }
        restrictions = restrictions or ["Commercial use requires licensing"]
        regulatory_frameworks = regulatory_frameworks or ["GDPR", "EU AI Act"]
        creator = creator or self.default_creator
        organization = organization or self.default_organization
        license = license or self.default_license

        # Determine compliance level
        compliance_level = self._determine_compliance_level(
            confidence_score, uncertainty_estimate, explainability_available
        )

        tag = CIAFMetadataTag(
            ciaf_version=CIAFTagVersion.V2_0.value,
            tag_id=tag_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_type=content_type,
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            model_hash=model_hash,
            training_snapshot_id=training_snapshot_id,
            dataset_anchor_id=dataset_anchor_id,
            inference_receipt_hash=inference_receipt_hash,
            compliance_level=compliance_level,
            regulatory_frameworks=regulatory_frameworks,
            intended_use=intended_use,
            restrictions=restrictions,
            confidence_score=confidence_score,
            uncertainty_estimate=uncertainty_estimate,
            explainability_available=explainability_available,
            creator=creator,
            organization=organization,
            license=license,
            contact_info=contact_info,
            custom_fields=custom_fields,
            generation_params=generation_params,
        )

        return tag

    def _generate_tag_id(
        self, content: Any, model_name: str, training_snapshot_id: str
    ) -> str:
        """Generate unique tag ID."""
        content_str = str(content)[:100]  # Limit content for hash
        timestamp = datetime.now(timezone.utc).isoformat()
        combined = f"{content_str}_{model_name}_{training_snapshot_id}_{timestamp}"
        return "CIAF_" + hashlib.sha256(combined.encode()).hexdigest()[:16].upper()

    def _generate_model_hash(
        self, model_name: str, model_version: str, training_snapshot_id: str
    ) -> str:
        """Generate model integrity hash."""
        combined = f"{model_name}_{model_version}_{training_snapshot_id}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def _determine_compliance_level(
        self, confidence: float, uncertainty: Dict[str, float], explainable: bool
    ) -> str:
        """Determine compliance level based on model capabilities."""
        if confidence >= 0.9 and uncertainty.get("total", 1.0) <= 0.1 and explainable:
            return "HIGH_ASSURANCE"
        elif confidence >= 0.7 and uncertainty.get("total", 1.0) <= 0.3:
            return "MEDIUM_ASSURANCE"
        elif confidence >= 0.5:
            return "BASIC_ASSURANCE"
        else:
            return "LOW_ASSURANCE"


class CIAFTagEncoder:
    """Encoder for embedding CIAF tags in content."""

    @staticmethod
    def encode_tag(tag: CIAFMetadataTag) -> str:
        """Encode CIAF tag to base64 string."""
        tag_dict = asdict(tag)
        tag_json = json.dumps(tag_dict, separators=(",", ":"))
        tag_bytes = tag_json.encode("utf-8")
        return base64.b64encode(tag_bytes).decode("ascii")

    @staticmethod
    def decode_tag(encoded_tag: str) -> CIAFMetadataTag:
        """Decode CIAF tag from base64 string."""
        try:
            tag_bytes = base64.b64decode(encoded_tag.encode("ascii"))
            tag_json = tag_bytes.decode("utf-8")
            tag_dict = json.loads(tag_json)

            # Convert string enums back to enum objects
            tag_dict["content_type"] = ContentType(tag_dict["content_type"])
            tag_dict["model_type"] = AIModelType(tag_dict["model_type"])

            return CIAFMetadataTag(**tag_dict)
        except Exception as e:
            raise ValueError(f"Invalid CIAF tag format: {e}")

    @staticmethod
    def embed_in_text(text: str, tag: CIAFMetadataTag) -> str:
        """Embed CIAF tag in text content."""
        encoded_tag = CIAFTagEncoder.encode_tag(tag)
        # Use zero-width characters for invisible embedding
        tag_marker = "​"  # Zero-width space
        embedded_tag = f"{tag_marker}CIAF:{encoded_tag}{tag_marker}"
        return text + embedded_tag

    @staticmethod
    def extract_from_text(text: str) -> Optional[CIAFMetadataTag]:
        """Extract CIAF tag from text content."""
        try:
            tag_marker = "​"  # Zero-width space
            start_marker = f"{tag_marker}CIAF:"
            end_marker = tag_marker

            start_idx = text.find(start_marker)
            if start_idx == -1:
                return None

            start_idx += len(start_marker)
            end_idx = text.find(end_marker, start_idx)
            if end_idx == -1:
                return None

            encoded_tag = text[start_idx:end_idx]
            return CIAFTagEncoder.decode_tag(encoded_tag)
        except Exception:
            return None

    @staticmethod
    def create_json_metadata(tag: CIAFMetadataTag) -> str:
        """Create JSON metadata for external files."""
        tag_dict = asdict(tag)
        return json.dumps(tag_dict, indent=2)


class CIAFTagValidator:
    """Validator for CIAF metadata tags."""

    @staticmethod
    def validate_tag(tag: CIAFMetadataTag) -> Dict[str, Any]:
        """Validate CIAF metadata tag."""
        issues = []
        warnings = []

        # Check required fields
        if not tag.tag_id:
            issues.append("Missing tag ID")
        if not tag.model_name:
            issues.append("Missing model name")
        if not tag.training_snapshot_id:
            issues.append("Missing training snapshot ID")
        if not tag.inference_receipt_hash:
            issues.append("Missing inference receipt hash")

        # Check data quality
        if tag.confidence_score < 0 or tag.confidence_score > 1:
            warnings.append("Confidence score should be between 0 and 1")

        if tag.uncertainty_estimate:
            total_uncertainty = tag.uncertainty_estimate.get("total", 0)
            if total_uncertainty < 0 or total_uncertainty > 1:
                warnings.append("Total uncertainty should be between 0 and 1")

        # Check compliance level consistency
        if tag.compliance_level == "HIGH_ASSURANCE":
            if tag.confidence_score < 0.9:
                warnings.append("High assurance level inconsistent with low confidence")
            if not tag.explainability_available:
                warnings.append("High assurance level should have explainability")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "compliance_level": tag.compliance_level,
            "regulatory_ready": len(issues) == 0 and len(warnings) <= 1,
        }

    @staticmethod
    def verify_integrity(tag: CIAFMetadataTag, expected_model_hash: str) -> bool:
        """Verify tag integrity against expected model hash."""
        return tag.model_hash == expected_model_hash


class CIAFWatermarkGenerator:
    """Generator for CIAF watermarks."""

    @staticmethod
    def generate_text_watermark(tag: CIAFMetadataTag) -> str:
        """Generate text watermark from CIAF tag."""
        watermark_data = {
            "model": tag.model_name,
            "version": tag.model_version,
            "timestamp": tag.timestamp[:19],  # Remove timezone for brevity
            "compliance": tag.compliance_level,
            "tag_id": tag.tag_id[:8],  # Short form
        }

        watermark = f"[Generated by {watermark_data['model']} v{watermark_data['version']} on {watermark_data['timestamp']} | CIAF ID: {watermark_data['tag_id']} | Compliance: {watermark_data['compliance']}]"
        return watermark

    @staticmethod
    def generate_qr_metadata(tag: CIAFMetadataTag) -> Dict[str, str]:
        """Generate QR code metadata for CIAF tag."""
        return {
            "url": f"https://ciaf.verify/{tag.tag_id}",
            "data": CIAFTagEncoder.encode_tag(tag),
            "format": "CIAF_QR_V2",
        }


# Global tag generator instance
tag_generator = CIAFTagGenerator()


def create_text_tag(
    text: str,
    model_name: str,
    model_version: str,
    training_snapshot_id: str,
    dataset_anchor_id: str,
    inference_receipt_hash: str,
    **kwargs,
) -> CIAFMetadataTag:
    """Create CIAF tag for text content."""
    return tag_generator.create_tag(
        content=text,
        content_type=ContentType.TEXT,
        model_name=model_name,
        model_version=model_version,
        model_type=AIModelType.LLM,
        training_snapshot_id=training_snapshot_id,
        dataset_anchor_id=dataset_anchor_id,
        inference_receipt_hash=inference_receipt_hash,
        **kwargs,
    )


def create_classification_tag(
    prediction: Any,
    model_name: str,
    model_version: str,
    training_snapshot_id: str,
    dataset_anchor_id: str,
    inference_receipt_hash: str,
    **kwargs,
) -> CIAFMetadataTag:
    """Create CIAF tag for classification output."""
    return tag_generator.create_tag(
        content=prediction,
        content_type=ContentType.DATA,
        model_name=model_name,
        model_version=model_version,
        model_type=AIModelType.CLASSIFIER,
        training_snapshot_id=training_snapshot_id,
        dataset_anchor_id=dataset_anchor_id,
        inference_receipt_hash=inference_receipt_hash,
        **kwargs,
    )
