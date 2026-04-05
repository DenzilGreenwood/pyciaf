# Unified Watermarking Interface Design

**Purpose**: Enable watermarking of ANY AI output at the point of inference with a single API call.

**Status**: 🚧 DESIGN DOCUMENT - Not yet implemented

---

## Problem Statement

**Current State**: Type-specific functions require knowing artifact type in advance
```python
# Must call different functions
build_text_artifact_evidence(raw_text=...)
build_image_artifact_evidence(image_bytes=...)
build_pdf_artifact_evidence(pdf_bytes=...)
```

**Desired State**: Single unified interface at inference point
```python
# One function handles all types
evidence, watermarked = watermark_ai_output(
    artifact=ai_model_output,  # Any type
    model_id="gpt-4",
    ...
)
```

---

## Proposed Solution

### Architecture: Three-Layer Design

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Inference Point Integration              │
│  └─ watermark_ai_output(artifact, ...)             │
│     - Auto-detects artifact type                    │
│     - Dispatches to appropriate watermarker         │
│     - Returns (evidence, watermarked)               │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 2: Type Detection & Dispatch                │
│  └─ ArtifactTypeDetector                           │
│     - detect_artifact_type(artifact)                │
│     - Returns: ArtifactType enum                    │
│  └─ WatermarkDispatcher                            │
│     - dispatch(artifact, type, config)              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 3: Type-Specific Watermarkers               │
│  └─ TextWatermarker                                │
│  └─ ImageWatermarker                               │
│  └─ PDFWatermarker                                 │
│  └─ JSONWatermarker (future)                       │
│  └─ BinaryWatermarker (future)                     │
└─────────────────────────────────────────────────────┘
```

---

## Implementation Design

### 1. Unified Entry Point Function

```python
from typing import Union, Tuple, Optional
from ciaf.watermarks.models import ArtifactEvidence, ArtifactType

def watermark_ai_output(
    artifact: Union[str, bytes],
    model_id: str,
    model_version: str,
    actor_id: str,
    prompt: str,
    verification_base_url: str,
    artifact_type: Optional[ArtifactType] = None,  # Auto-detect if None
    watermark_config: Optional[dict] = None,
    enable_forensic_fragments: bool = True,
    store_in_vault: bool = False,
) -> Tuple[ArtifactEvidence, Union[str, bytes]]:
    """
    Watermark any AI-generated artifact at the point of inference.

    This is the UNIFIED INTERFACE for watermarking all artifact types.

    Args:
        artifact: The raw AI output (text string or bytes)
        model_id: Model identifier
        model_version: Model version
        actor_id: User/system that requested generation
        prompt: Input prompt (or hash of prompt)
        verification_base_url: Base URL for verification
        artifact_type: Explicit type (or None to auto-detect)
        watermark_config: Type-specific watermark configuration
        enable_forensic_fragments: Enable DNA sampling (default True)
        store_in_vault: Auto-store in vault (default False)

    Returns:
        (evidence, watermarked_artifact)
        - evidence: Complete ArtifactEvidence record
        - watermarked_artifact: Artifact with watermark applied

    Examples:
        # Text
        evidence, watermarked = watermark_ai_output(
            artifact="AI generated text...",
            model_id="gpt-4",
            model_version="2026-03",
            actor_id="user:analyst-17",
            prompt="Summarize the report",
            verification_base_url="https://vault.example.com"
        )

        # Image
        evidence, watermarked = watermark_ai_output(
            artifact=image_bytes,
            model_id="stable-diffusion-v3",
            model_version="2026-03",
            actor_id="user:artist-42",
            prompt="Generate landscape",
            verification_base_url="https://vault.example.com",
            watermark_config={
                "mode": "visual",
                "opacity": 0.4,
                "position": "bottom_right",
                "include_qr": True,
            }
        )

        # Auto-detects type and applies appropriate watermarking
    """
    # Step 1: Detect artifact type
    if artifact_type is None:
        artifact_type = detect_artifact_type(artifact)

    # Step 2: Dispatch to type-specific watermarker
    dispatcher = WatermarkDispatcher()
    evidence, watermarked = dispatcher.dispatch(
        artifact=artifact,
        artifact_type=artifact_type,
        model_id=model_id,
        model_version=model_version,
        actor_id=actor_id,
        prompt=prompt,
        verification_base_url=verification_base_url,
        watermark_config=watermark_config,
        enable_forensic_fragments=enable_forensic_fragments,
    )

    # Step 3: Optional vault storage
    if store_in_vault:
        from ciaf.watermarks import create_watermark_vault
        vault = create_watermark_vault()
        vault.store_evidence(evidence)

    return evidence, watermarked
```

---

### 2. Type Detection

```python
from ciaf.watermarks.models import ArtifactType

def detect_artifact_type(artifact: Union[str, bytes]) -> ArtifactType:
    """
    Auto-detect the type of artifact.

    Detection Logic:
    1. If string → TEXT
    2. If bytes → Inspect magic bytes:
       - PNG: b'\x89PNG'
       - JPEG: b'\xff\xd8\xff'
       - PDF: b'%PDF'
       - MP4: b'ftyp' (offset 4)
       - Fallback: BINARY

    Args:
        artifact: Raw artifact (string or bytes)

    Returns:
        ArtifactType enum

    Examples:
        >>> detect_artifact_type("Hello world")
        ArtifactType.TEXT

        >>> detect_artifact_type(b'\x89PNG\r\n\x1a\n...')
        ArtifactType.IMAGE

        >>> detect_artifact_type(b'%PDF-1.4...')
        ArtifactType.PDF
    """
    # Text detection
    if isinstance(artifact, str):
        return ArtifactType.TEXT

    # Bytes detection - check magic bytes
    if not isinstance(artifact, bytes):
        raise TypeError(f"Artifact must be str or bytes, got {type(artifact)}")

    # PNG
    if artifact.startswith(b'\x89PNG\r\n\x1a\n'):
        return ArtifactType.IMAGE

    # JPEG
    if artifact.startswith(b'\xff\xd8\xff'):
        return ArtifactType.IMAGE

    # WebP
    if artifact.startswith(b'RIFF') and b'WEBP' in artifact[:16]:
        return ArtifactType.IMAGE

    # PDF
    if artifact.startswith(b'%PDF'):
        return ArtifactType.PDF

    # MP4/MOV
    if len(artifact) > 8 and artifact[4:8] == b'ftyp':
        return ArtifactType.VIDEO

    # WAV
    if artifact.startswith(b'RIFF') and b'WAVE' in artifact[:16]:
        return ArtifactType.AUDIO

    # MP3
    if artifact.startswith(b'ID3') or artifact.startswith(b'\xff\xfb'):
        return ArtifactType.AUDIO

    # JSON (heuristic: starts with { or [)
    try:
        text = artifact.decode('utf-8')
        text_stripped = text.strip()
        if text_stripped.startswith(('{', '[')):
            import json
            json.loads(text)  # Validate JSON
            return ArtifactType.JSON
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass

    # Fallback
    return ArtifactType.BINARY
```

---

### 3. Dispatcher

```python
class WatermarkDispatcher:
    """
    Dispatches watermarking requests to type-specific handlers.
    """

    def dispatch(
        self,
        artifact: Union[str, bytes],
        artifact_type: ArtifactType,
        model_id: str,
        model_version: str,
        actor_id: str,
        prompt: str,
        verification_base_url: str,
        watermark_config: Optional[dict] = None,
        enable_forensic_fragments: bool = True,
    ) -> Tuple[ArtifactEvidence, Union[str, bytes]]:
        """
        Dispatch to appropriate watermarker based on artifact type.
        """
        # Get watermarker for artifact type
        watermarker = self._get_watermarker(artifact_type)

        # Common parameters
        common_params = {
            "model_id": model_id,
            "model_version": model_version,
            "actor_id": actor_id,
            "prompt": prompt,
            "verification_base_url": verification_base_url,
        }

        # Type-specific watermarking
        if artifact_type == ArtifactType.TEXT:
            from ciaf.watermarks import build_text_artifact_evidence
            return build_text_artifact_evidence(
                raw_text=artifact,
                **common_params,
                enable_forensic_fragments=enable_forensic_fragments,
            )

        elif artifact_type == ArtifactType.IMAGE:
            from ciaf.watermarks import build_image_artifact_evidence, ImageWatermarkSpec

            # Parse watermark config
            spec = ImageWatermarkSpec(**(watermark_config or {}))

            return build_image_artifact_evidence(
                image_bytes=artifact,
                **common_params,
                watermark_spec=spec,
                include_perceptual_hashes=True,
            )

        elif artifact_type == ArtifactType.PDF:
            from ciaf.watermarks import build_pdf_artifact_evidence
            return build_pdf_artifact_evidence(
                pdf_bytes=artifact,
                **common_params,
                additional_metadata=watermark_config or {},
            )

        elif artifact_type == ArtifactType.JSON:
            # TODO: Implement JSON watermarking
            raise NotImplementedError("JSON watermarking not yet implemented")

        elif artifact_type == ArtifactType.AUDIO:
            # Audio not implemented
            raise NotImplementedError("Audio watermarking not yet implemented")

        elif artifact_type == ArtifactType.VIDEO:
            # Video not implemented
            raise NotImplementedError("Video watermarking not yet implemented")

        elif artifact_type == ArtifactType.BINARY:
            # Binary not implemented
            raise NotImplementedError("Binary watermarking not yet implemented")

        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")

    def _get_watermarker(self, artifact_type: ArtifactType):
        """Get watermarker instance for artifact type."""
        # Future: Return watermarker instances if we create classes
        return None
```

---

## Integration with Model Wrappers

### At Inference Point

```python
from ciaf.wrappers import CIAFModelWrapper
from ciaf.watermarks import watermark_ai_output

# Wrap your model
wrapped_model = CIAFModelWrapper(
    model=your_model,
    model_id="gpt-4",
    model_version="2026-03"
)

# Generate AI output
response = wrapped_model.generate(
    prompt="Summarize the quarterly report",
    user_id="analyst-17"
)

# Watermark automatically at inference point
evidence, watermarked_output = watermark_ai_output(
    artifact=response.content,      # Auto-detects type
    model_id=wrapped_model.model_id,
    model_version=wrapped_model.model_version,
    actor_id=f"user:{response.user_id}",
    prompt=response.prompt,
    verification_base_url="https://vault.example.com",
    store_in_vault=True,            # Auto-store evidence
)

# Return watermarked output to user
return watermarked_output
```

### Automatic Watermarking Hook

```python
class CIAFModelWrapper:
    """Enhanced model wrapper with automatic watermarking."""

    def __init__(
        self,
        model,
        model_id: str,
        model_version: str,
        auto_watermark: bool = True,
        verification_base_url: str = "https://vault.example.com"
    ):
        self.model = model
        self.model_id = model_id
        self.model_version = model_version
        self.auto_watermark = auto_watermark
        self.verification_base_url = verification_base_url

    def generate(self, prompt: str, user_id: str, **kwargs):
        """Generate AI output and automatically watermark."""

        # Step 1: Generate AI output (original)
        raw_output = self.model.generate(prompt, **kwargs)

        # Step 2: Automatically watermark if enabled
        if self.auto_watermark:
            evidence, watermarked_output = watermark_ai_output(
                artifact=raw_output,
                model_id=self.model_id,
                model_version=self.model_version,
                actor_id=f"user:{user_id}",
                prompt=prompt,
                verification_base_url=self.verification_base_url,
                store_in_vault=True,  # Auto-store
            )

            return ModelResponse(
                content=watermarked_output,
                evidence=evidence,
                watermarked=True,
            )
        else:
            return ModelResponse(
                content=raw_output,
                evidence=None,
                watermarked=False,
            )
```

---

## Benefits of Unified Interface

### 1. Simplicity
```python
# OLD: Must know type and call specific function
if isinstance(output, str):
    evidence, watermarked = build_text_artifact_evidence(...)
elif magic_bytes == PNG:
    evidence, watermarked = build_image_artifact_evidence(...)
# etc...

# NEW: One function handles all
evidence, watermarked = watermark_ai_output(output, ...)
```

### 2. Type Safety
```python
# Auto-detection prevents errors
evidence, watermarked = watermark_ai_output(
    artifact=unknown_ai_output,  # Don't need to know type
    model_id="model",
    ...
)
```

### 3. Inference Point Integration
```python
# Seamless integration with model wrappers
class MyAIModel(CIAFModelWrapper):
    def generate(self, prompt):
        output = super().generate(prompt)
        # Watermarking happens automatically
        return output
```

### 4. Consistent API
```python
# Same parameters for all types
common_config = {
    "model_id": "gpt-4",
    "model_version": "2026-03",
    "actor_id": "user:123",
    "verification_base_url": "https://vault.example.com"
}

# Text
evidence, text = watermark_ai_output(text_output, **common_config)

# Images
evidence, image = watermark_ai_output(image_output, **common_config)

# PDFs
evidence, pdf = watermark_ai_output(pdf_output, **common_config)
```

---

## Configuration Flexibility

### Global Defaults
```python
# Set default watermark configuration
from ciaf.watermarks import set_default_watermark_config

set_default_watermark_config({
    "verification_base_url": "https://vault.mycompany.com",
    "store_in_vault": True,
    "enable_forensic_fragments": True,
    "text": {
        "style": "footer",
        "include_simhash": True,
    },
    "image": {
        "mode": "visual",
        "opacity": 0.4,
        "position": "bottom_right",
        "include_qr": True,
    },
    "pdf": {
        "mode": "metadata",
        "add_visual_qr": True,
    }
})

# Then use simplified calls
evidence, watermarked = watermark_ai_output(
    artifact=ai_output,
    model_id="gpt-4",
    model_version="2026-03",
    actor_id="user:123",
    prompt="Generate content",
    # Uses global defaults
)
```

### Per-Call Overrides
```python
# Override defaults for specific call
evidence, watermarked = watermark_ai_output(
    artifact=ai_output,
    model_id="gpt-4",
    model_version="2026-03",
    actor_id="user:123",
    prompt="Generate content",
    verification_base_url="https://vault.mycompany.com",
    watermark_config={
        "mode": "hybrid",  # Override default
        "opacity": 0.6,    # Override default
    }
)
```

---

## Implementation Checklist

To build this unified interface:

- [ ] **Step 1**: Create `detect_artifact_type()` function
  - [ ] Implement magic byte detection
  - [ ] Add JSON heuristics
  - [ ] Add comprehensive tests

- [ ] **Step 2**: Create `WatermarkDispatcher` class
  - [ ] Implement dispatch logic
  - [ ] Map ArtifactType to watermarker functions
  - [ ] Handle watermark_config parsing

- [ ] **Step 3**: Create `watermark_ai_output()` entry point
  - [ ] Integrate detection + dispatch
  - [ ] Add vault auto-storage option
  - [ ] Add configuration defaults system

- [ ] **Step 4**: Integrate with `CIAFModelWrapper`
  - [ ] Add auto_watermark parameter
  - [ ] Hook into generate() method
  - [ ] Return enhanced ModelResponse

- [ ] **Step 5**: Add configuration system
  - [ ] Global defaults
  - [ ] Per-call overrides
  - [ ] Type-specific configs

- [ ] **Step 6**: Tests & Documentation
  - [ ] Unit tests for each component
  - [ ] Integration tests for complete workflow
  - [ ] Update README with unified interface examples

---

## Estimated Implementation Time

- **Type detection**: 2-3 hours
- **Dispatcher**: 3-4 hours
- **Unified entry point**: 2-3 hours
- **Model wrapper integration**: 3-4 hours
- **Configuration system**: 2-3 hours
- **Tests**: 4-5 hours
- **Documentation**: 2-3 hours

**Total**: ~20-25 hours (2-3 days)

---

## Backward Compatibility

The unified interface is **additive** - existing code continues to work:

```python
# OLD API (still works)
from ciaf.watermarks import build_text_artifact_evidence
evidence, watermarked = build_text_artifact_evidence(...)

# NEW API (recommended)
from ciaf.watermarks import watermark_ai_output
evidence, watermarked = watermark_ai_output(...)
```

---

## Status

**Current**: 🚧 DESIGN DOCUMENT
**Next Step**: Review and approval
**Implementation**: Not started

**Questions for Review**:
1. Should we auto-detect type or require explicit artifact_type parameter?
2. Should vault storage be automatic at inference point?
3. What should the default watermark configurations be?
4. Should this be a separate module or integrated into main watermarks package?

---

**Document Version**: 1.0
**Created**: 2026-04-04
**Author**: Unified Interface Design
**Status**: Proposal for Implementation
