# CIAF Watermarking System - Complete Documentation

**Version**: 1.7.0  
**Last Updated**: April 4, 2026  
**Author**: Denzil James Greenwood  
**Status**: Production-ready for text, images, PDF; Beta for video/audio

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Features](#core-features)
4. [Artifact Types](#artifact-types)
5. [Hashing Strategies](#hashing-strategies)
6. [Verification & Forensics](#verification--forensics)
7. [Advanced Features](#advanced-features)
8. [API Reference](#api-reference)
9. [Installation](#installation)
10. [Usage Examples](#usage-examples)
11. [Testing](#testing)
12. [Performance](#performance)
13. [Security & Limitations](#security--limitations)
14. [Roadmap](#roadmap)

---

## Overview

The CIAF Watermarking module implements a **forensic provenance layer** for AI-generated artifacts. Unlike traditional watermarking, CIAF makes **watermark removal attempts forensically detectable** through dual-state cryptographic evidence.

### Key Differentiator

CIAF doesn't just mark AI content — it creates **tamper-evident audit trails** that prove:
- ✅ This is the exact distributed copy (watermarked version)
- ⚠️ This is the original with watermark removed (forensic evidence of removal)
- ❌ This content has been modified (tampering detection)
- ❓ This is not from our system (no match)

### Core Innovation: Dual-State Artifact Integrity

CIAF stores **two cryptographic hashes** for each artifact:
1. **Before watermark**: Hash of original AI output (pre-distribution)
2. **After watermark**: Hash of distributed version with provenance tag

This dual-state model enables **forensic detection even when watermarks are removed**.

---

## Architecture

### Module Structure

```
ciaf/watermarks/
├── __init__.py                      # Main exports & version
├── models.py                        # Data models (ArtifactEvidence, etc.)
├── hashing.py                       # Hashing strategies (exact, normalized, SimHash)
├── text.py                          # Text watermarking implementation
├── verify.py                        # Verification and forensic matching
├── vault_adapter.py                 # Storage integration with CIAF vault
├── signature_envelope.py            # Cryptographic signing
├── fragment_selection.py            # Forensic fragment sampling
├── fragment_verification.py         # Fragment-based verification
├── hierarchical_verification.py     # Multi-layer verification strategy
├── advanced_features.py             # Advanced audio/video features
├── images/                          # Image watermarking package
│   ├── __init__.py
│   ├── visual.py                    # Visual overlays (text + QR)
│   ├── fingerprints.py              # Perceptual hashing
│   ├── qr.py                        # QR code generation
│   └── steganography.py             # LSB steganographic embedding
└── pdf/                             # PDF watermarking package
    ├── __init__.py
    ├── metadata.py                  # PDF metadata watermarking
    └── visual.py                    # PDF visible stamps
```

### Data Flow

```
AI Output → Hash (before) → Add Watermark → Hash (after) → Store Evidence
                                    ↓
                              Distribute Content
                                    ↓
                          Suspect Content Received
                                    ↓
                            Forensic Verification
                                    ↓
            Match: Watermarked | Stripped | Modified | Unknown
```

---

## Core Features

### Production-Ready Features ✅

1. **Text Watermarking**
   - Dual-state hashing (pre/post watermark)
   - Multiple verification strategies (exact, normalized, SimHash)
   - Watermark styles: Footer, Header, Inline
   - Watermark removal detection
   - Vault integration

2. **Image Watermarking**
   - Visual text overlays (9 positions, opacity control)
   - QR codeoverlays (verification URLs)
   - LSB steganographic embedding (invisible watermarks)
   - Perceptual hashing (pHash, aHash, dHash, wHash)
   - Fragment-based verification (64x64 patches)
   - Dual-state hashing

3. **PDF Watermarking**
   - Metadata watermarking (invisible)
   - Visual stamps (headers/footers)
   - QR code placement
   - Dual-state hashing

4. **Forensic Fragment Verification**
   - Entropy-based patch selection
   - Multi-point sampling (configurable patches)
   - Spatial search with adaptive stride
   - Confidence scoring
   - Tampering detection

5. **Hierarchical Verification**
   - Multi-layer verification strategy
   - Content integrity validation
   - Watermark presence/removal detection
   - Fragment-based tampering detection
   - Overall confidence assessment

### Beta Features ⚠️

1. **Video Watermarking**
   - Keyframe extraction (temporal sampling)
   - Patch-based hashing (high-entropy regions)
   - Motion signature computation
   - Scene change detection
   - Optical flow analysis

2. **Audio Watermarking**
   - Spectral segment extraction
   - Frequency feature computation (centroid, flatness)
   - MFCCs (Mel-frequency cepstral coefficients)
   - Chroma features (pitch classes)
   - Beat tracking and tempo detection

### Advanced Features (Optional Dependencies)

1. **Advanced Spectral Analysis** (librosa)
   - 13 MFCC coefficients
   - 12-bin chroma profiles
   - Spectral centroid, bandwidth, rolloff
   - Tempo detection (BPM)
   - Zero-crossing rate

2. **Multi-Format Support** (ffmpeg)
   - Audio: MP3, AAC, FLAC, OGG → WAV
   - Video: AVI, MKV, MOV → MP4
   - Codec support: H.264, H.265, VP9

3. **Perceptual Hashing**
   - Video pHash (imagehash)
   - Audio chromaprint (pyacoustid)
   - Image similarity detection

4. **Cloud Storage** (AWS S3 / Azure Blob)
   - Fragment storage in cloud
   - Metadata preservation
   - Retrieval by artifact ID

---

## Artifact Types

### 1. Text Artifacts

**Watermark Style Options**:
```python
# Footer (default)
"""
Original text content...

---
AI Provenance Tag: wmk-a1b2c3d4-e5f6-7890
Verify: https://vault.example.com/verify/abc123
Generated with CIAF
"""

# Header
"""
---
AI Generated Content | Verify: https://vault.example.com/verify/abc123
---

Original text content...
"""

# Inline (after first paragraph)
"""
First paragraph text...
[AI Provenance: wmk-a1b2c3d4 | Verify]

Remaining content...
"""
```

**Hashing Strategies**:
- SHA-256 (exact matching)
- Normalized hash (format-resilient)
- SimHash (similarity matching)

**Verification Capabilities**:
- Exact match detection
- Watermark removal detection
- Content modification detection
- Similarity scoring

### 2. Image Artifacts

**Visual Watermarking**:
```python
# Text overlay
- 9 positions (corners, edges, center)
- Opacity control (0.0-1.0)
- Font, size, color customization

# QR code
- 9 positions
- Configurable size
- Verification URL embedded
```

**LSB Steganography**:
```python
# Invisible watermark
- Embedded in RGB/RGBA channels
- JSON payload with metadata
- SHA-256 checksum
- Lossless PNG output
- Automatic mode conversion
```

**Perceptual Hashing**:
```python
# Four algorithms
- pHash: DCT-based (general forensics)
- aHash: Average hash (fast screening)
- dHash: Gradient-based (edit detection)
- wHash: Wavelet-based (robust to modifications)
```

**Fragment Verification**:
```python
# Patch-based sampling
- Entropy-driven selection
- 64x64 pixel patches
- Spatial coordinates tracking
- SHA-256 hash per patch
```

### 3. PDF Artifacts

**Metadata Watermarking**:
```python
# Invisible embedded metadata
- Custom properties
- Watermark ID
- Verification URL
- Timestamp
- Actor/model information
```

**Visual Stamps**:
```python
# Visible header/footer
- Text stamps
- QR codes
- Page-level overlays
```

### 4. Video Artifacts (Beta)

**Forensic Keyframes**:
```python
# Temporal sampling
- Keyframes at 25%, 50%, 75%
- High-entropy region selection
- 4-patch extraction per frame
- Motion signature between frames
```

**Advanced Analysis**:
```python
# Motion & scene analysis
- Dense optical flow
- Scene change detection
- Transition classification
- Object detection (basic)
```

### 5. Audio Artifacts (Beta)

**Spectral Segments**:
```python
# Temporal sampling
- Segments at even intervals
- Frequency feature extraction
- Spectral centroid/flatness
- SHA-256 hashing
```

**Advanced Analysis**:
```python
# Librosa features
- MFCCs (13 coefficients)
- Chroma features (12 bins)
- Beat tracking
- Tempo detection (BPM)
```

---

## Hashing Strategies

### 1. Exact Hashing (SHA-256)

**Purpose**: Cryptographic proof of identity

```python
from ciaf.watermarks import sha256_text

hash1 = sha256_text("Exact content")
hash2 = sha256_text("Exact content")
hash3 = sha256_text("Different content")

assert hash1 == hash2  # Perfect match
assert hash1 != hash3  # Different
```

**Characteristics**:
- Deterministic
- Collision-resistant
- Fast (~1M hashes/sec)
- Byte-level precision

**Use Cases**:
- Exact duplicate detection
- Integrity verification
- Legal proof of authenticity

### 2. Normalized Hashing

**Purpose**: Resilient to formatting changes

```python
from ciaf.watermarks import normalized_text_hash

text1 = "This is some text."
text2 = "  THIS   IS    SOME   TEXT.  "

hash1 = normalized_text_hash(text1)
hash2 = normalized_text_hash(text2)

assert hash1 == hash2  # Same despite formatting
```

**Normalization Steps**:
1. Lowercase conversion
2. Whitespace collapse
3. Punctuation removal (optional)
4. Strip leading/trailing space

**Use Cases**:
- Copy-paste detection
- Format-independent matching
- Cross-platform comparison

### 3. SimHash (Similarity)

**Purpose**: Detect minor rewording or edits

```python
from ciaf.watermarks import simhash_text, simhash_distance

text1 = "The quick brown fox jumps over the lazy dog."
text2 = "The fast brown fox leaps over the sleepy dog."

hash1 = simhash_text(text1)
hash2 = simhash_text(text2)
distance = simhash_distance(hash1, hash2)

print(f"Distance: {distance}/64")  # Low = similar
```

**Distance Thresholds**:
- **0-3**: Near duplicates (>95% similar)
- **4-10**: Similar content (80-95%)
- **11-20**: Somewhat related (50-80%)
- **>20**: Different content (<50%)

**Use Cases**:
- Paraphrase detection
- Content similarity
- Derivative work identification

### 4. Perceptual Hashing (Images)

**Purpose**: Image similarity despite modifications

```python
from ciaf.watermarks.images import compute_all_hashes, hamming_distance

# Compute all four algorithms
phash, ahash, dhash, whash = compute_all_hashes(image_bytes)

# Compare with modified image
phash2, ahash2, dhash2, whash2 = compute_all_hashes(modified_bytes)

# Check similarity
for orig, mod in [(phash, phash2), (ahash, ahash2), ...]:
    dist = hamming_distance(orig, mod)
    if dist <= 10:
        print("MATCH: Same source image")
```

**Algorithm Comparison**:

| Algorithm | Speed | Robustness | Best For |
|-----------|-------|------------|----------|
| **pHash** | Medium | Very Good | General forensics (RECOMMENDED) |
| **aHash** | Very Fast | Moderate | Quick duplicate detection |
| **dHash** | Fast | Good | Edit/modification detection |
| **wHash** | Slower | Excellent | Heavy modifications |

**Hamming Distance Thresholds**:
- **0-5**: Near identical (99.9%+ similar)
- **6-10**: Forensic match likely (same source)
- **11-15**: Similar content or derivative
- **16-20**: Somewhat similar
- **>20**: Different images

---

## Verification & Forensics

### Verification Workflow

```python
from ciaf.watermarks import (
    build_text_artifact_evidence,
    verify_text_artifact,
    create_watermark_vault
)

# Step 1: Generate and store watermarked content
evidence, watermarked = build_text_artifact_evidence(
    raw_text="AI output",
    model_id="production-model",
    model_version="1.0",
    actor_id="system",
    prompt="Generate content",
    verification_base_url="https://vault.example.com"
)

vault = create_watermark_vault()
vault.store_evidence(evidence)

# Step 2: Verify suspect content
suspect_text = "..."  # Received from external source

result = verify_text_artifact(suspect_text, evidence)

print(f"Authentic: {result.is_authentic()}")
print(f"Confidence: {result.confidence:.1%}")

if result.likely_tag_removed:
    print("⚠ WARNING: Watermark was removed!")
if result.content_modified:
    print("⚠ WARNING: Content was modified!")
```

### Forensic Scenarios

#### Scenario 1: Exact Match (Watermarked)

```python
# Someone shares the exact watermarked version
result = verify_text_artifact(watermarked_text, evidence)

assert result.exact_match_after_watermark == True
assert result.watermark_present == True
assert result.confidence == 1.0
# ✓ Perfect match - this is the distributed version
```

#### Scenario 2: Watermark Removed

```python
# Someone removed the watermark but kept the content
from ciaf.watermarks import remove_watermark

stripped_text = remove_watermark(watermarked_text)
result = verify_text_artifact(stripped_text, evidence)

assert result.exact_match_before_watermark == True
assert result.watermark_present == False
assert result.likely_tag_removed == True
# ⚠ Watermark removed - forensic evidence of tampering!
```

#### Scenario 3: Content Modified

```python
# Someone changed the content
modified_text = watermarked_text.replace("word", "different")
result = verify_text_artifact(modified_text, evidence)

# May detect via SimHash similarity
if result.perceptual_similarity_score > 0.8:
    print("⚠ Content modified but likely derived from original")
```

#### Scenario 4: Completely Unrelated

```python
# Random text not from our system
fake_text = "This is unrelated content."
result = verify_text_artifact(fake_text, evidence)

assert result.exact_match_before_watermark == False
assert result.exact_match_after_watermark == False
assert result.confidence < 0.5
# ✗ Not from our system
```

### Hierarchical Verification Strategy

```python
from ciaf.watermarks import hierarchical_verify_artifact

result = hierarchical_verify_artifact(
    suspect_text=suspect_text,
    evidence=evidence,
    check_fragments=True,  # Enable fragment verification
    fragment_threshold=0.8  # 80% fragments must match
)

# Multi-layer verification
print(f"Content integrity: {result.content_integrity_valid}")
print(f"Watermark status: {result.watermark_status}")
print(f"Fragment matches: {result.fragment_verification.matched}/{result.fragment_verification.total}")
print(f"Overall confidence: {result.overall_confidence:.1%}")
print(f"Legally defensible: {result.legally_defensible}")
```

### Fragment-Based Verification

```python
from ciaf.watermarks import (
    select_forensic_fragments_text,
    verify_forensic_fragments
)

# Create fragments from original
fragments = select_forensic_fragments_text(
    raw_text=original_text,
    num_fragments=10,
    fragment_length=100
)

# Verify suspect text
match_result = verify_forensic_fragments(
    suspect_text=suspect_text,
    fragments=fragments
)

print(f"Matched: {match_result.fragments_matched}/{match_result.total_fragments}")
print(f"Confidence: {match_result.overall_confidence:.1%}")

for match in match_result.matches:
    print(f"  Fragment {match.fragment_index}: {match.confidence:.1%} at position {match.position}")
```

---

## Advanced Features

### 1. Advanced Spectral Analysis (librosa)

**Installation**:
```bash
pip install librosa soundfile
```

**Usage**:
```python
from ciaf.watermarks.advanced_features import extract_advanced_audio_features

features = extract_advanced_audio_features(
    audio_path="audio.wav",
    include_mfcc=True,
    include_chroma=True,
    include_tempo=True
)

print(f"MFCCs: {features['mfccs'].shape}")  # (13, frames)
print(f"Chroma: {features['chroma'].shape}")  # (12, frames)
print(f"Tempo: {features['tempo']} BPM")
print(f"Spectral centroid: {features['spectral_centroid']}")
```

**Features**:
- 13 MFCC coefficients per frame
- 12-bin chroma profiles
- Spectral centroid, bandwidth, rolloff
- Tempo detection (BPM)
- Zero-crossing rate
- Onset strength

### 2. Multi-Format Support (ffmpeg)

**Installation**:
```bash
pip install ffmpeg-python
# Also install ffmpeg binary from https://ffmpeg.org/
```

**Usage**:
```python
from ciaf.watermarks.advanced_features import convert_audio_format, convert_video_format

# Audio conversion
wav_path = convert_audio_format(
    input_path="audio.mp3",
    output_format="wav",
    sample_rate=44100
)

# Video conversion
mp4_path = convert_video_format(
    input_path="video.avi",
    output_format="mp4",
    codec="h264"
)
```

**Supported Formats**:
- **Audio**: MP3, AAC, FLAC, OGG, WAV
- **Video**: MP4, AVI, MKV, MOV
- **Codecs**: H.264, H.265, VP9

### 3. Perceptual Hashing

**Video pHash**:
```python
from ciaf.watermarks.advanced_features import compute_video_phash, compare_video_phashes

hash1 = compute_video_phash("video1.mp4")
hash2 = compute_video_phash("video2.mp4")

similarity = compare_video_phashes(hash1, hash2)
print(f"Similarity: {similarity:.1%}")
```

**Audio Chromaprint**:
```python
from ciaf.watermarks.advanced_features import compute_audio_fingerprint, compare_audio_fingerprints

fp1 = compute_audio_fingerprint("audio1.wav")
fp2 = compute_audio_fingerprint("audio2.wav")

similarity = compare_audio_fingerprints(fp1, fp2)
print(f"Similarity: {similarity:.1%}")
```

### 4. Cloud Storage Integration

**AWS S3**:
```python
from ciaf.watermarks.advanced_features import upload_to_s3, download_from_s3

# Upload fragments
s3_key = upload_to_s3(
    file_path="fragments.json",
    bucket_name="ciaf-watermarks",
    s3_key="artifacts/abc123/fragments.json",
    metadata={"artifact_id": "abc123"}
)

# Download
local_path = download_from_s3(
    bucket_name="ciaf-watermarks",
    s3_key=s3_key,
    local_path="downloaded_fragments.json"
)
```

**Azure Blob Storage**:
```python
from ciaf.watermarks.advanced_features import upload_to_azure, download_from_azure

# Upload
blob_name = upload_to_azure(
    file_path="fragments.json",
    container_name="ciaf-watermarks",
    blob_name="artifacts/abc123/fragments.json",
    account_name="myaccount",
    account_key="key..."
)

# Download
local_path = download_from_azure(
    container_name="ciaf-watermarks",
    blob_name=blob_name,
    local_path="downloaded.json",
    account_name="myaccount",
    account_key="key..."
)
```

### 5. Video Motion Analysis (v1.7.0)

**Optical Flow**:
```python
from ciaf.watermarks.advanced_features import compute_dense_optical_flow

flow = compute_dense_optical_flow(
    video_path="video.mp4",
    frame_indices=[10, 11]  # Consecutive frames
)

magnitude, angle = flow['magnitude'], flow['angle']
print(f"Average motion: {flow['average_magnitude']}")
```

**Scene Change Detection**:
```python
from ciaf.watermarks.advanced_features import detect_scene_changes

changes = detect_scene_changes(
    video_path="video.mp4",
    threshold=30.0  # Histogram difference threshold
)

for change in changes:
    print(f"Scene change at frame {change['frame_number']} (time: {change['timestamp']}s)")
```

### 6. Audio Beat Tracking (v1.7.0)

```python
from ciaf.watermarks.advanced_features import track_beats

beat_info = track_beats("audio.wav")

print(f"Tempo: {beat_info['tempo']} BPM")
print(f"Beat frames: {beat_info['beat_frames']}")
print(f"Beat times: {beat_info['beat_times']} seconds")
```

### 7. A/V Synchronization (v1.7.0)

```python
from ciaf.watermarks.advanced_features import verify_av_sync

sync_result = verify_av_sync(
    video_path="video.mp4",
    audio_path="audio.wav",
    tolerance_ms=100  # 100ms tolerance
)

print(f"Synchronized: {sync_result['synchronized']}")
print(f"Offset: {sync_result['offset_ms']}ms")
```

---

## API Reference

### Core Functions

#### Text Watermarking

```python
# High-level
build_text_artifact_evidence(raw_text, model_id, model_version, actor_id, prompt, verification_base_url, ...) -> (ArtifactEvidence, str)
quick_watermark_text(text, model_id, ...) -> (str, str)
apply_text_watermark(raw_text, watermark_id, verification_url, style="footer") -> str
remove_watermark(watermarked_text) -> str

# Verification
verify_text_artifact(suspect_text, evidence, ...) -> VerificationResult
analyze_suspect_text(text) -> dict
hierarchical_verify_artifact(suspect_text, evidence, ...) -> HierarchicalVerificationResult

# Watermark detection
has_watermark(text) -> bool
extract_watermark_id(text) -> Optional[str]
extract_verification_url(text) -> Optional[str]
```

#### Image Watermarking

```python
# Visual watermarks
build_image_artifact_evidence(image_bytes, model_id, ...) -> (ArtifactEvidence, bytes)
apply_visual_watermark(image_bytes, text, ...) -> bytes
add_qr_watermark(image_bytes, url, ...) -> bytes

# LSB Steganography
embed_watermark_lsb(image_bytes, watermark_data, ...) -> bytes
extract_watermark_lsb(image_bytes) -> Optional[dict]
verify_lsb_watermark(image_bytes, expected_id) -> bool
has_lsb_watermark(image_bytes) -> bool

# Perceptual hashing
compute_all_hashes(image_bytes) -> (str, str, str, str)  # pHash, aHash, dHash, wHash
hamming_distance(hash1, hash2) -> int
similarity_score(hash1, hash2) -> float
```

#### PDF Watermarking

```python
build_pdf_artifact_evidence(pdf_bytes, model_id, ...) -> (ArtifactEvidence, bytes)
apply_pdf_metadata_watermark(pdf_bytes, metadata) -> bytes
extract_pdf_metadata(pdf_bytes) -> dict
```

#### Fragment Verification

```python
# Text fragments
select_forensic_fragments_text(raw_text, num_fragments, fragment_length) -> List[TextForensicFragment]
verify_forensic_fragments(suspect_text, fragments) -> ForensicVerificationSummary

# Image fragments
select_forensic_fragments_image(image_bytes, num_patches, patch_size, entropy_threshold) -> List[ImageForensicFragment]
verify_image_fragments(suspect_bytes, fragments) -> FragmentMatchResult
```

#### Hashing

```python
# Text hashing
sha256_text(text) -> str
normalized_text_hash(text) -> str
simhash_text(text) -> int
simhash_distance(hash1, hash2) -> int

# Image hashing
perceptual_hash_image(image_bytes, algorithm="phash") -> str  # phash, ahash, dhash, whash

# Video/Audio hashing
hash_video_keyframe(frame_data) -> str
hash_audio_segment(audio_data) -> str
```

### Data Models

```python
@dataclass
class ArtifactEvidence:
    artifact_id: str
    artifact_type: str  # text, image, pdf, video, audio
    model_id: str
    model_version: str
    actor_id: str
    prompt_hash: str
    output_hash_raw: str
    output_hash_distributed: str
    watermark: WatermarkDescriptor
    hashes: ArtifactHashes
    fingerprints: List[ForensicFingerprint]
    fragments: Optional[List[ForensicFragment]]
    timestamp: datetime

@dataclass
class WatermarkDescriptor:
    watermark_id: str
    watermark_type: str  # visible, invisible, metadata
    verification_url: str
    removal_resistance: str  # low, medium, high
    created_at: datetime

@dataclass
class VerificationResult:
    artifact_id: str
    exact_match_after_watermark: bool
    exact_match_before_watermark: bool
    likely_tag_removed: bool
    watermark_present: bool
    watermark_intact: bool
    confidence: float  # 0.0-1.0
    perceptual_similarity_score: Optional[float]
    notes: List[str]
```

---

## Installation

### Core Dependencies (Required)

```bash
pip install Pillow qrcode[pil] pypdf
```

### Optional Advanced Features

```bash
# Advanced audio analysis
pip install librosa soundfile

# Multi-format support
pip install ffmpeg-python

# Audio fingerprinting
pip install pyacoustid

# Cloud storage
pip install boto3 azure-storage-blob

# Image perceptual hashing
pip install imagehash

# Video processing
pip install opencv-python
```

### Complete Installation

```bash
pip install Pillow qrcode[pil] pypdf imagehash opencv-python librosa soundfile ffmpeg-python pyacoustid boto3 azure-storage-blob
```

### External Dependencies

- **FFmpeg**: https://ffmpeg.org/download.html
- **Chromaprint**: https://acoustid.org/chromaprint

---

## Usage Examples

### Text Watermarking

```python
from ciaf.watermarks import build_text_artifact_evidence, verify_text_artifact

# Generate watermarked text
evidence, watermarked = build_text_artifact_evidence(
    raw_text="AI generated summary of quarterly results.",
    model_id="gpt-4",
    model_version="2026.03",
    actor_id="user:analyst-42",
    prompt="Summarize Q4 results",
    verification_base_url="https://vault.example.com"
)

print(f"Artifact ID: {evidence.artifact_id}")
print(f"Watermarked:\n{watermarked}")

# Verify later
result = verify_text_artifact(watermarked, evidence)
print(f"Authentic: {result.is_authentic()}")
print(f"Confidence: {result.confidence:.1%}")
```

### Image Watermarking

```python
from ciaf.watermarks import build_image_artifact_evidence, ImageWatermarkSpec

# Read image
with open("ai_generated.png", "rb") as f:
    image_bytes = f.read()

# Configure watermark
spec = ImageWatermarkSpec(
    mode="visual",
    text="AI Generated",
    opacity=0.4,
    position="bottom_right",
    include_qr=True,
    qr_position="top_right"
)

# Create watermarked image
evidence, watermarked = build_image_artifact_evidence(
    image_bytes=image_bytes,
    model_id="stable-diffusion-xl",
    model_version="1.0",
    actor_id="user:artist-7",
    prompt="A sunset over mountains",
    verification_base_url="https://vault.example.com",
    watermark_spec=spec,
    include_perceptual_hashes=True
)

# Save
with open("watermarked.png", "wb") as f:
    f.write(watermarked)
```

### LSB Steganography

```python
from ciaf.watermarks.images import embed_watermark_lsb, extract_watermark_lsb

# Embed invisible watermark
watermarked = embed_watermark_lsb(
    image_bytes=image_bytes,
    watermark_data={
        "watermark_id": "wmk-invisible-123",
        "artifact_id": "art-456",
        "verification_url": "https://vault.example.com/verify/art-456"
    }
)

# Extract later
extracted = extract_watermark_lsb(watermarked)
if extracted:
    print(f"Watermark ID: {extracted['watermark_id']}")
    print(f"Verification: {extracted['verification_url']}")
```

### Fragment Verification

```python
from ciaf.watermarks import select_forensic_fragments_text, verify_forensic_fragments

# Create fragments
fragments = select_forensic_fragments_text(
    raw_text=original_text,
    num_fragments=10,
    fragment_length=100
)

# Store fragments with evidence
evidence.fragments = fragments

# Verify suspect text
result = verify_forensic_fragments(
    suspect_text=suspect_text,
    fragments=fragments
)

print(f"Matched: {result.fragments_matched}/{result.total_fragments}")
print(f"Confidence: {result.overall_confidence:.1%}")
```

---

## Testing

### Test Suite

```bash
# Core watermarking
python tests/test_watermarks.py

# Phase 1 (images + PDF)
python tests/test_watermarks_phase1.py

# Comprehensive tests
python tests/test_watermarks_comprehensive.py

# Advanced features
python tests/test_advanced_features.py

# Steganography
python tests/test_steganography.py

# Fragment verification
python tests/test_image_fragment_verification.py

# Video/audio
python tests/test_video_audio_fragments.py
```

### Test Coverage

**Core Watermarking** (7/7 passing):
- Text watermarking and verification
- Watermark removal detection
- Similarity matching (SimHash)
- Vault storage integration

**Phase 1** (5/5 passing):
- Image visual watermarking
- QR code overlays
- PDF metadata watermarking
- Perceptual hashing

**Advanced Features** (30/30 passing):
- LSB steganography
- Fragment verification
- Hierarchical verification
- Spectral analysis
- Multi-format support
- Cloud storage
- Motion analysis
- Scene detection

---

## Performance

### Hashing Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| SHA-256 text | ~1M/sec | Cryptographic |
| Normalized hash | ~500K/sec | With processing |
| SimHash | ~100K/sec | 64-bit fingerprint |
| pHash (image) | ~40/sec | DCT-based |
| All 4 perceptual | ~10/sec | pHash+aHash+dHash+wHash |

### Watermarking Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Text watermarking | ~10K/sec | Simple append |
| Text verification | ~5K/sec | Multiple strategies |
| Image text overlay | ~20/sec | PIL operations |
| Image QR code | ~15/sec | Generation + overlay |
| LSB embedding | ~10/sec | Pixel manipulation |
| PDF metadata | ~50/sec | Metadata injection |

### Fragment Verification Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| Fragment selection | ~1K/sec | Entropy calculation |
| Fragment verification | ~100/sec | Sliding window search |
| Image patch extraction | ~50/sec | Entropy-based |

---

## Security & Limitations

### Known Limitations

1. **Visual Watermarks are Easy to Remove**
   - Removal resistance: LOW
   - Mitigation: Dual-state hashing detects removal
   - Future: Steganographic watermarking

2. **Text Paraphrasing Defeats Exact Hashing**
   - Mitigation: SimHash provides similarity detection
   - Limitation: SimHash can be fooled by extensive rewrites
   - Future: Embedding-based similarity (neural networks)

3. **Image Cropping Can Remove Watermarks**
   - Perceptual hashing survives many modifications
   - Fragment verification helps detect partial matches
   - Dual-state hashing detects removal

4. **PDF Metadata Can Be Stripped**
   - Removal resistance: MEDIUM (requires tools)
   - Mitigation: Dual-state hashing detects stripping
   - Future: Combined metadata + visible stamps

5. **LSB Steganography Vulnerable to Compression**
   - JPEG compression destroys LSB watermarks
   - Only works with lossless formats (PNG)
   - Not suitable for social media distribution

### Security Best Practices

1. **Store Evidence Securely**
   - Use encrypted vault (PostgreSQL with TDE)
   - Implement access controls
   - Regular backups

2. **Use HTTPS for Verification URLs**
   - Prevent man-in-the-middle attacks
   - TLS 1.3 recommended

3. **Log All Verification Attempts**
   - Audit trail for forensic analysis
   - Rate limiting to prevent abuse

4. **Combine Multiple Strategies**
   - Visual + steganographic
   - Exact + normalized + SimHash
   - Fragment-based verification

5. **Sign Evidence Records**
   - Cryptographic integrity of stored evidence
   - Use SignatureEnvelope pattern

---

## Roadmap

### Version 1.8.0 (Q2 2026)
- [ ] Neural embedding-based similarity
- [ ] Batch watermarking API
- [ ] REST API for verification
- [ ] Improved fragment verification (Bug #161 fix)
- [ ] Enhanced PDF visual stamps

### Version 1.9.0 (Q3 2026)
- [ ] Advanced steganographic techniques
- [ ] Video frame-level watermarking
- [ ] Audio spectral watermarking
- [ ] Blockchain anchoring
- [ ] Zero-knowledge proofs

### Version 2.0.0 (Q4 2026)
- [ ] Federated verification (cross-organization)
- [ ] Machine learning-based tampering detection
- [ ] Real-time verification streaming
- [ ] Mobile SDK (iOS/Android)
- [ ] Browser extension

---

## License

See main CIAF LICENSE file.

---

**Version**: 1.7.0  
**Last Updated**: April 4, 2026  
**Author**: Denzil James Greenwood  
**Contact**: founder@cognitiveinsight.ai
