# CIAF Metadata Tags Module

The `metadata_tags` module provides comprehensive metadata tagging capabilities for AI-generated content, similar to EXIF data for images. These tags enable deepfake detection, misinformation defense, regulatory compliance tracking, and complete AI output provenance within the Cognitive Insight Audit Framework (CIAF).

## 📋 Overview

This module creates standardized metadata tags that embed crucial information about AI-generated content directly into the output, enabling:
- **Content Authentication**: Verify AI-generated content authenticity
- **Regulatory Compliance**: Meet EU AI Act, GDPR, and NIST AI RMF requirements
- **Provenance Tracking**: Complete audit trail from training to inference
- **Misinformation Defense**: Combat deepfakes and manipulated content
- **License Management**: Track usage rights and restrictions

## 🏗️ Architecture

```
metadata_tags/
├── __init__.py              # Core metadata tagging classes and utilities
└── __pycache__/            # Compiled Python files
```

## 🔧 Core Components

### CIAFMetadataTag (Data Structure)
Comprehensive metadata structure containing all AI output information:

```python
from ciaf.metadata_tags import CIAFMetadataTag, ContentType, AIModelType

# Create metadata tag structure
tag = CIAFMetadataTag(
    ciaf_version="2.0",
    tag_id="CIAF_A1B2C3D4E5F6G7H8",
    timestamp="2025-01-15T10:30:00+00:00",
    content_type=ContentType.TEXT,
    model_name="GPT-4-CIAF",
    model_version="1.2.3",
    model_type=AIModelType.LLM,
    model_hash="abc123def456...",
    training_snapshot_id="snapshot_789",
    dataset_anchor_id="anchor_456",
    inference_receipt_hash="receipt_123",
    compliance_level="HIGH_ASSURANCE",
    regulatory_frameworks=["EU AI Act", "GDPR", "NIST AI RMF"],
    intended_use="Content generation",
    restrictions=["Commercial use requires licensing"],
    confidence_score=0.95,
    uncertainty_estimate={"total": 0.05, "aleatoric": 0.02, "epistemic": 0.03},
    explainability_available=True,
    creator="CIAF Framework",
    organization="AI Research Lab",
    license="MIT",
    contact_info="contact@ciaf.org"
)
```

### CIAFTagGenerator
Intelligent tag generation with automatic compliance assessment:

```python
from ciaf.metadata_tags import CIAFTagGenerator, ContentType, AIModelType

# Initialize tag generator
generator = CIAFTagGenerator(
    default_creator="AI Research Team",
    default_organization="Tech Company",
    default_license="Proprietary"
)

# Generate tag for AI text
tag = generator.create_tag(
    content="This is AI-generated text content",
    content_type=ContentType.TEXT,
    model_name="Custom-LLM",
    model_version="2.1.0",
    model_type=AIModelType.LLM,
    training_snapshot_id="training_abc123",
    dataset_anchor_id="dataset_def456",
    inference_receipt_hash="inference_ghi789",
    confidence_score=0.92,
    uncertainty_estimate={"total": 0.08, "aleatoric": 0.03, "epistemic": 0.05},
    explainability_available=True,
    intended_use="Educational content generation",
    restrictions=["Non-commercial use only"],
    regulatory_frameworks=["EU AI Act", "GDPR"],
    custom_fields={"prompt_type": "educational", "domain": "science"}
)

print(f"Generated tag ID: {tag.tag_id}")
print(f"Compliance level: {tag.compliance_level}")
```

### CIAFTagEncoder
Embedding and extraction of metadata tags in content:

```python
from ciaf.metadata_tags import CIAFTagEncoder, create_text_tag

# Create tag for text content
tag = create_text_tag(
    text="AI-generated article content",
    model_name="NewsBot-Pro",
    model_version="1.5.2",
    training_snapshot_id="news_training_001",
    dataset_anchor_id="news_data_anchor",
    inference_receipt_hash="receipt_news_123",
    confidence_score=0.87,
    intended_use="News article generation"
)

# Embed tag invisibly in text
original_text = "This is an AI-generated news article about technology trends."
tagged_text = CIAFTagEncoder.embed_in_text(original_text, tag)

print(f"Original: {len(original_text)} chars")
print(f"Tagged: {len(tagged_text)} chars")
# Text appears identical but contains embedded metadata

# Extract tag from content
extracted_tag = CIAFTagEncoder.extract_from_text(tagged_text)
if extracted_tag:
    print(f"Model: {extracted_tag.model_name}")
    print(f"Confidence: {extracted_tag.confidence_score}")
    print(f"Compliance: {extracted_tag.compliance_level}")

# Encode tag as base64 for external storage
encoded = CIAFTagEncoder.encode_tag(tag)
print(f"Base64 tag: {encoded[:50]}...")

# Decode tag from base64
decoded_tag = CIAFTagEncoder.decode_tag(encoded)
```

### CIAFTagValidator
Comprehensive tag validation and integrity checking:

```python
from ciaf.metadata_tags import CIAFTagValidator

# Validate tag completeness and consistency
validation_result = CIAFTagValidator.validate_tag(tag)

print(f"Valid: {validation_result['valid']}")
print(f"Issues: {validation_result['issues']}")
print(f"Warnings: {validation_result['warnings']}")
print(f"Regulatory ready: {validation_result['regulatory_ready']}")

# Verify tag integrity
expected_hash = "expected_model_hash_123"
is_intact = CIAFTagValidator.verify_integrity(tag, expected_hash)
print(f"Tag integrity verified: {is_intact}")
```

### CIAFWatermarkGenerator
Generate visible watermarks and QR codes for metadata:

```python
from ciaf.metadata_tags import CIAFWatermarkGenerator

# Generate text watermark
watermark = CIAFWatermarkGenerator.generate_text_watermark(tag)
print(f"Watermark: {watermark}")
# Output: [Generated by Custom-LLM v2.1.0 on 2025-01-15T10:30:00 | CIAF ID: A1B2C3D4 | Compliance: HIGH_ASSURANCE]

# Generate QR metadata for verification
qr_data = CIAFWatermarkGenerator.generate_qr_metadata(tag)
print(f"QR URL: {qr_data['url']}")
print(f"QR format: {qr_data['format']}")
```

## 🚀 Quick Start Examples

### Text Content Tagging
```python
from ciaf.metadata_tags import create_text_tag, CIAFTagEncoder

# Generate AI text with metadata
ai_text = "Climate change poses significant challenges to global agriculture..."

# Create comprehensive tag
tag = create_text_tag(
    text=ai_text,
    model_name="ClimateGPT",
    model_version="3.1.0",
    training_snapshot_id="climate_model_v3",
    dataset_anchor_id="climate_dataset_2024",
    inference_receipt_hash="climate_inference_456",
    confidence_score=0.91,
    uncertainty_estimate={"total": 0.09, "aleatoric": 0.04, "epistemic": 0.05},
    explainability_available=True,
    intended_use="Educational climate information",
    restrictions=["Attribution required"],
    regulatory_frameworks=["EU AI Act", "GDPR"],
    creator="Climate AI Research Lab",
    organization="Environmental Tech Institute",
    custom_fields={
        "domain": "climate_science",
        "fact_checked": True,
        "sources": ["IPCC", "NOAA", "NASA"]
    }
)

# Embed metadata invisibly
tagged_content = CIAFTagEncoder.embed_in_text(ai_text, tag)

# Save as JSON metadata file
json_metadata = CIAFTagEncoder.create_json_metadata(tag)
with open("climate_article_metadata.json", "w") as f:
    f.write(json_metadata)
```

### Classification Output Tagging
```python
from ciaf.metadata_tags import create_classification_tag

# Tag classification prediction
prediction_result = {
    "class": "positive_sentiment",
    "probability": 0.94,
    "confidence_interval": [0.89, 0.97]
}

tag = create_classification_tag(
    prediction=prediction_result,
    model_name="SentimentAnalyzer-Pro",
    model_version="2.3.1",
    training_snapshot_id="sentiment_training_2024",
    dataset_anchor_id="sentiment_data_v2",
    inference_receipt_hash="sentiment_inference_789",
    confidence_score=0.94,
    uncertainty_estimate={"total": 0.06, "aleatoric": 0.03, "epistemic": 0.03},
    explainability_available=True,
    intended_use="Customer feedback analysis",
    regulatory_frameworks=["GDPR", "CCPA"]
)
```

### Multi-Modal Content Tagging
```python
from ciaf.metadata_tags import CIAFTagGenerator, ContentType, AIModelType

generator = CIAFTagGenerator()

# Tag for image generation
image_tag = generator.create_tag(
    content="Generated landscape image",
    content_type=ContentType.IMAGE,
    model_name="DiffusionArt-Pro",
    model_version="4.2.0",
    model_type=AIModelType.DIFFUSION,
    training_snapshot_id="art_training_v4",
    dataset_anchor_id="art_dataset_curated",
    inference_receipt_hash="image_inference_abc",
    confidence_score=0.88,
    intended_use="Artistic content creation",
    generation_params={
        "prompt": "serene mountain landscape at sunset",
        "style": "photorealistic",
        "resolution": "1024x768",
        "guidance_scale": 7.5
    }
)

# Tag for code generation
code_tag = generator.create_tag(
    content="def fibonacci(n): ...",
    content_type=ContentType.CODE,
    model_name="CodeGPT-Advanced",
    model_version="1.8.0",
    model_type=AIModelType.LLM,
    training_snapshot_id="code_training_2024",
    dataset_anchor_id="code_repository_anchor",
    inference_receipt_hash="code_inference_def",
    confidence_score=0.96,
    intended_use="Educational programming assistance",
    custom_fields={
        "language": "python",
        "complexity": "beginner",
        "tested": True
    }
)
```

## 🔗 Integration with CIAF Ecosystem

### LCM Integration
```python
from ciaf.lcm import InferenceManager
from ciaf.metadata_tags import CIAFTagGenerator

# Integrate tagging with inference management
inference_manager = InferenceManager()
tag_generator = CIAFTagGenerator()

# Automatic tag generation during inference
def generate_with_tags(model_id, input_data):
    # Get inference receipt
    receipt = inference_manager.create_inference_receipt(model_id, input_data)
    
    # Generate output
    output = inference_manager.predict(model_id, input_data)
    
    # Create metadata tag
    tag = tag_generator.create_tag(
        content=output,
        content_type=ContentType.TEXT,
        model_name=model_id,
        training_snapshot_id=receipt.training_snapshot_id,
        dataset_anchor_id=receipt.dataset_anchor_id,
        inference_receipt_hash=receipt.receipt_hash,
        # ... other parameters
    )
    
    return output, tag
```

### Compliance Integration
```python
from ciaf.compliance import AuditTrailGenerator
from ciaf.metadata_tags import CIAFTagValidator

# Audit tag compliance
audit_trail = AuditTrailGenerator()
validator = CIAFTagValidator()

def audit_tagged_content(tagged_content):
    # Extract tag
    tag = CIAFTagEncoder.extract_from_text(tagged_content)
    
    if tag:
        # Validate tag
        validation = validator.validate_tag(tag)
        
        # Log audit event
        audit_trail.log_compliance_check(
            content_id=tag.tag_id,
            compliance_level=tag.compliance_level,
            validation_result=validation,
            regulatory_frameworks=tag.regulatory_frameworks
        )
        
        return validation
    return None
```

### Provenance Integration
```python
from ciaf.provenance import ProvenanceCapsule
from ciaf.metadata_tags import create_text_tag

# Link tags with provenance system
def create_provenance_tagged_content(model, input_data, training_capsule):
    # Generate content
    output = model.generate(input_data)
    
    # Create tag with provenance information
    tag = create_text_tag(
        text=output,
        model_name=model.name,
        model_version=model.version,
        training_snapshot_id=training_capsule.snapshot_id,
        dataset_anchor_id=training_capsule.dataset_anchor_id,
        inference_receipt_hash=training_capsule.receipt_hash,
        # Provenance metadata
        custom_fields={
            "provenance_capsule_id": training_capsule.capsule_id,
            "lineage_verified": True,
            "chain_of_custody": training_capsule.custody_chain
        }
    )
    
    return CIAFTagEncoder.embed_in_text(output, tag)
```

## 📊 Content Types and Model Support

### Supported Content Types
```python
from ciaf.metadata_tags import ContentType, AIModelType

# Text content
ContentType.TEXT        # Articles, stories, code, etc.
ContentType.CODE        # Generated source code
ContentType.DATA        # Predictions, classifications

# Media content  
ContentType.IMAGE       # Generated images
ContentType.AUDIO       # Generated audio/speech
ContentType.VIDEO       # Generated video content
ContentType.MULTIMODAL  # Combined content types

# Supported AI model types
AIModelType.LLM                           # Large Language Models
AIModelType.DIFFUSION                     # Image generation models
AIModelType.GAN                          # Generative Adversarial Networks
AIModelType.CLASSIFIER                   # Classification models
AIModelType.REGRESSOR                    # Regression models
AIModelType.TRANSFORMER                  # Transformer architectures
AIModelType.CNN                          # Convolutional Neural Networks
AIModelType.RNN                          # Recurrent Neural Networks
AIModelType.ENSEMBLE                     # Ensemble methods
```

## 🔒 Security and Privacy Features

### Tamper Detection
```python
from ciaf.core import CryptoManager
from ciaf.metadata_tags import CIAFTagValidator

# Cryptographic integrity verification
crypto_manager = CryptoManager()

def verify_tag_integrity(tag, content):
    # Generate content hash
    content_hash = crypto_manager.hash_data(content.encode())
    
    # Verify tag hasn't been tampered with
    verification_data = {
        "tag_id": tag.tag_id,
        "model_hash": tag.model_hash,
        "content_hash": content_hash,
        "timestamp": tag.timestamp
    }
    
    # Check cryptographic signature
    is_valid = crypto_manager.verify_signature(
        verification_data, 
        tag.custom_fields.get("signature", "")
    )
    
    return is_valid
```

### Privacy-Preserving Tags
```python
# Create tags with minimal personal information
privacy_safe_tag = generator.create_tag(
    content=output,
    content_type=ContentType.TEXT,
    model_name="PrivacyLLM",
    # Anonymized identifiers
    creator="Anonymous",
    organization="Redacted",
    contact_info="privacy@ciaf.org",
    # No sensitive custom fields
    custom_fields={
        "privacy_level": "high",
        "pii_removed": True,
        "anonymized": True
    },
    restrictions=[
        "No personal data collection",
        "GDPR Article 25 compliant",
        "Privacy by design"
    ]
)
```

## 🧪 Compliance Validation

### Regulatory Framework Checking
```python
def validate_regulatory_compliance(tag):
    """Validate tag against regulatory requirements."""
    compliance_results = {}
    
    # EU AI Act compliance
    if "EU AI Act" in tag.regulatory_frameworks:
        compliance_results["eu_ai_act"] = {
            "transparency_required": tag.explainability_available,
            "risk_level": tag.compliance_level,
            "documentation_complete": bool(tag.intended_use and tag.restrictions),
            "conformity_assessment": tag.confidence_score >= 0.8
        }
    
    # GDPR compliance  
    if "GDPR" in tag.regulatory_frameworks:
        compliance_results["gdpr"] = {
            "lawful_basis_documented": "consent" in tag.intended_use.lower(),
            "data_minimization": len(tag.custom_fields or {}) <= 10,
            "purpose_limitation": bool(tag.intended_use),
            "transparency": tag.explainability_available
        }
    
    # NIST AI RMF compliance
    if "NIST AI RMF" in tag.regulatory_frameworks:
        compliance_results["nist_ai_rmf"] = {
            "govern_function": bool(tag.restrictions),
            "map_function": bool(tag.model_type and tag.intended_use),
            "measure_function": bool(tag.confidence_score and tag.uncertainty_estimate),
            "manage_function": tag.compliance_level in ["HIGH_ASSURANCE", "MEDIUM_ASSURANCE"]
        }
    
    return compliance_results
```

## 🔮 Advanced Usage

### Custom Tag Extensions
```python
from dataclasses import dataclass
from ciaf.metadata_tags import CIAFMetadataTag

@dataclass
class ExtendedCIAFTag(CIAFMetadataTag):
    """Extended CIAF tag with domain-specific fields."""
    
    # Medical AI extensions
    medical_classification: Optional[str] = None
    hipaa_compliance: bool = False
    clinical_validation: Optional[str] = None
    
    # Financial AI extensions  
    risk_category: Optional[str] = None
    sox_compliance: bool = False
    financial_disclaimer: Optional[str] = None
    
    # Legal AI extensions
    jurisdiction: Optional[str] = None
    legal_precedent: Optional[List[str]] = None
    attorney_review: bool = False

# Usage
extended_tag = ExtendedCIAFTag(
    # Standard CIAF fields
    ciaf_version="2.0",
    tag_id="EXTENDED_123",
    # ... other standard fields ...
    
    # Domain-specific extensions
    medical_classification="diagnostic_assistance",
    hipaa_compliance=True,
    clinical_validation="FDA_cleared"
)
```

### Batch Tag Processing
```python
def process_batch_content(content_list, model_info):
    """Process multiple AI outputs with consistent tagging."""
    tags = []
    
    for i, content in enumerate(content_list):
        tag = generator.create_tag(
            content=content,
            content_type=ContentType.TEXT,
            model_name=model_info["name"],
            model_version=model_info["version"],
            model_type=model_info["type"],
            training_snapshot_id=f"{model_info['snapshot']}_{i}",
            dataset_anchor_id=model_info["dataset_anchor"],
            inference_receipt_hash=f"batch_inference_{i}",
            # Batch-specific metadata
            custom_fields={
                "batch_id": model_info["batch_id"],
                "batch_index": i,
                "batch_size": len(content_list),
                "batch_timestamp": datetime.now().isoformat()
            }
        )
        tags.append(tag)
    
    return tags
```

## 🤝 Contributing

When contributing to the metadata_tags module:

1. **Maintain Schema Compatibility**: Ensure tag structure remains backward compatible
2. **Add Comprehensive Validation**: Include validation for new fields and formats
3. **Consider Privacy**: Implement privacy-preserving features for sensitive domains
4. **Support New Content Types**: Extend ContentType and AIModelType enums appropriately
5. **Document Regulatory Impact**: Update compliance validation for new requirements

## 📚 Related Documentation

- [CIAF Core Framework](../api/README.md) - Main framework integration
- [Compliance Engine](../compliance/README.md) - Regulatory compliance features
- [Inference System](../inference/README.md) - Integration with inference receipts
- [Provenance System](../provenance/README.md) - Provenance tracking integration
- [LCM System](../lcm/README.md) - Lifecycle management integration

---

The metadata_tags module provides the foundation for trustworthy AI output identification, enabling complete content provenance, regulatory compliance, and defense against AI-generated misinformation while maintaining seamless integration with existing AI workflows.