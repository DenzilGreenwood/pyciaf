"""
CIAF Watermarking - Advanced Features Examples

Demonstrates the advanced watermarking capabilities:
1. Advanced spectral analysis with librosa (MFCCs, chroma)
2. Multi-format support via ffmpeg (MP3, AAC, H.265)
3. Perceptual hashing (video pHash, audio chromaprint)
4. Cloud storage integration (AWS S3, Azure Blob)
5. Optical flow & motion analysis
6. Scene change detection
7. Keyframe transition analysis
8. Audio beat tracking
9. A/V synchronization analysis

Created: 2026-04-04
"""

import os
import sys
from pathlib import Path

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ciaf.watermarks.advanced_features import (
    get_available_features,
    print_feature_status,
    LIBROSA_AVAILABLE,
    FFMPEG_AVAILABLE,
    IMAGEHASH_AVAILABLE,
    CHROMAPRINT_AVAILABLE,
    BOTO3_AVAILABLE,
    AZURE_AVAILABLE,
)

# Conditional imports
if LIBROSA_AVAILABLE:
    from ciaf.watermarks.advanced_features import (
        extract_advanced_audio_features,
        compare_spectral_features,
    )

if FFMPEG_AVAILABLE:
    from ciaf.watermarks.advanced_features import (
        convert_audio_to_wav,
        convert_video_to_mp4,
    )

if IMAGEHASH_AVAILABLE:
    from ciaf.watermarks.advanced_features import (
        compute_video_phash,
        compare_perceptual_hashes,
    )

if CHROMAPRINT_AVAILABLE:
    from ciaf.watermarks.advanced_features import (
        compute_audio_chromaprint,
    )

from ciaf.watermarks.advanced_features import (
    CloudStorageConfig,
    CloudFragmentStorage,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_test_audio_wav(duration_seconds=2.0):
    """Create a test WAV audio file."""
    import wave
    import struct
    import math
    import tempfile
    
    sample_rate = 44100
    num_samples = int(duration_seconds * sample_rate)
    frequency = 440.0  # A4 note
    
    # Generate sine wave
    samples = []
    for i in range(num_samples):
        sample = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        samples.append(struct.pack('<hh', sample, sample))  # Stereo
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    # Write WAV (ensures file handle is closed)
    with wave.open(tmp_path, 'wb') as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b''.join(samples))
    
    # Read back as bytes
    with open(tmp_path, 'rb') as f:
        wav_bytes = f.read()
    
    # Delete temp file (with Windows error handling)
    try:
        os.unlink(tmp_path)
    except (PermissionError, OSError):
        # Windows file locking - ignore cleanup error
        pass
    
    return wav_bytes


def create_test_video(duration_seconds=2.0, width=320, height=240, fps=30):
    """Create a test video file."""
    try:
        import cv2
        import numpy as np
        import tempfile
    except ImportError:
        print("⚠️  OpenCV not available - skipping video creation")
        return None
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        output_path = tmp.name
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = int(duration_seconds * fps)
    
    for i in range(num_frames):
        # Gradient frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color_value = int((i / num_frames) * 255)
        frame[:, :] = (color_value, 255 - color_value, 128)
        out.write(frame)
    
    out.release()
    
    with open(output_path, 'rb') as f:
        video_bytes = f.read()
    
    # Delete temp file (with Windows error handling)
    try:
        os.unlink(output_path)
    except (PermissionError, OSError):
        # Windows file locking - ignore cleanup error
        pass
    
    return video_bytes


# ============================================================================
# EXAMPLE 1: ADVANCED SPECTRAL ANALYSIS
# ============================================================================

def example_1_advanced_spectral_analysis():
    """
    Example 1: Extract advanced spectral features using librosa.
    
    Demonstrates:
    - MFCC extraction (Mel-frequency cepstral coefficients)
    - Chroma features (pitch class profiles)
    - Spectral features (centroid, bandwidth, rolloff)
    - Tempo detection
    - Feature comparison
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Advanced Spectral Analysis with Librosa")
    print("=" * 70)
    
    if not LIBROSA_AVAILABLE:
        print("⚠️  Librosa not available. Install with:")
        print("   pip install librosa")
        return
    
    # Create test audio
    print("\n1. Creating test audio (2-second sine wave at 440 Hz)...")
    audio_bytes = create_test_audio_wav(duration_seconds=2.0)
    print(f"   ✓ Created audio: {len(audio_bytes):,} bytes")
    
    # Extract advanced features
    print("\n2. Extracting advanced spectral features...")
    features = extract_advanced_audio_features(audio_bytes, duration=2.0)
    
    print(f"\n   📊 MFCCs (13 coefficients):")
    print(f"      Mean: {[f'{x:.2f}' for x in features.mfcc_mean[:5]]}... (showing first 5)")
    print(f"      Std:  {[f'{x:.2f}' for x in features.mfcc_std[:5]]}... (showing first 5)")
    
    print(f"\n   🎵 Chroma Features (12 pitch classes):")
    print(f"      Mean: {[f'{x:.3f}' for x in features.chroma_mean[:6]]}... (showing first 6)")
    print(f"      Std:  {[f'{x:.3f}' for x in features.chroma_std[:6]]}... (showing first 6)")
    
    print(f"\n   📈 Spectral Features:")
    print(f"      Centroid:  {features.spectral_centroid:.1f} Hz")
    print(f"      Bandwidth: {features.spectral_bandwidth:.1f} Hz")
    print(f"      Rolloff:   {features.spectral_rolloff:.1f} Hz")
    print(f"      ZCR:       {features.zero_crossing_rate:.4f}")
    
    print(f"\n   🎼 Temporal Features:")
    print(f"      Tempo:          {features.tempo:.1f} BPM")
    print(f"      Onset Strength: {features.onset_strength:.4f}")
    
    print(f"\n   🔒 Feature Hash: {features.features_hash[:16]}...")
    
    # Compare with itself
    print("\n3. Comparing features with identical audio...")
    features2 = extract_advanced_audio_features(audio_bytes, duration=2.0)
    similarity = compare_spectral_features(features, features2)
    print(f"   ✓ Similarity: {similarity:.4f} (should be ~1.0)")
    
    print("\n✅ Example 1 complete!")


# ============================================================================
# EXAMPLE 2: MULTI-FORMAT AUDIO CONVERSION
# ============================================================================

def example_2_multi_format_audio():
    """
    Example 2: Convert audio between formats using ffmpeg.
    
    Demonstrates:
    - WAV to WAV conversion (sample rate change)
    - Format detection and conversion
    - Sample rate adjustment
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Multi-Format Audio Conversion")
    print("=" * 70)
    
    if not FFMPEG_AVAILABLE:
        print("⚠️  FFmpeg not available. Install with:")
        print("   pip install ffmpeg-python")
        print("   And install ffmpeg binary: https://ffmpeg.org/download.html")
        return
    
    # Create test audio at 44.1 kHz
    print("\n1. Creating test audio at 44,100 Hz...")
    audio_bytes = create_test_audio_wav(duration_seconds=1.0)
    print(f"   ✓ Created: {len(audio_bytes):,} bytes")
    
    # Convert to 22.05 kHz
    print("\n2. Converting to 22,050 Hz...")
    try:
        converted = convert_audio_to_wav(audio_bytes, output_sample_rate=22050)
        print(f"   ✓ Converted: {len(converted):,} bytes")
        print(f"   ℹ️  Size ratio: {len(converted)/len(audio_bytes):.2f}x")
    except Exception as e:
        print(f"   ⚠️  Conversion failed: {e}")
        print("   (This may require ffmpeg binary to be installed)")
    
    print("\n✅ Example 2 complete!")


# ============================================================================
# EXAMPLE 3: VIDEO PERCEPTUAL HASHING
# ============================================================================

def example_3_video_perceptual_hashing():
    """
    Example 3: Compute perceptual hashes for video frames.
    
    Demonstrates:
    - Frame extraction and pHash computation
    - Robustness to minor modifications
    - Similarity comparison
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Video Perceptual Hashing")
    print("=" * 70)
    
    if not IMAGEHASH_AVAILABLE:
        print("⚠️  ImageHash not available. Install with:")
        print("   pip install imagehash")
        return
    
    # Create test video
    print("\n1. Creating test video (2 seconds, 320x240)...")
    video_bytes = create_test_video(duration_seconds=2.0)
    
    if video_bytes is None:
        return
    
    print(f"   ✓ Created: {len(video_bytes):,} bytes")
    
    # Compute perceptual hashes
    print("\n2. Computing perceptual hashes for 5 keyframes...")
    phashes = compute_video_phash(video_bytes, num_frames=5)
    
    print(f"   ✓ Extracted {len(phashes)} pHashes:")
    for i, phash in enumerate(phashes):
        print(f"      Frame {i}: {phash}")
    
    # Verify consistency
    print("\n3. Verifying hash consistency...")
    phashes2 = compute_video_phash(video_bytes, num_frames=5)
    
    all_match = all(h1 == h2 for h1, h2 in zip(phashes, phashes2))
    print(f"   {'✓' if all_match else '✗'} All hashes {'match' if all_match else 'differ'}")
    
    # Compare similarity
    print("\n4. Computing pHash similarities...")
    for i, (h1, h2) in enumerate(zip(phashes[:3], phashes2[:3])):
        sim = compare_perceptual_hashes(h1, h2, hash_type="phash")
        print(f"   Frame {i}: {sim:.4f} similarity")
    
    print("\n✅ Example 3 complete!")


# ============================================================================
# EXAMPLE 4: AUDIO CHROMAPRINT FINGERPRINTING
# ============================================================================

def example_4_audio_chromaprint():
    """
    Example 4: Generate Chromaprint fingerprint for audio.
    
    Demonstrates:
    - Chromaprint extraction (AcousticID)
    - Robust audio identification
    - Fingerprint consistency
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Audio Chromaprint Fingerprinting")
    print("=" * 70)
    
    if not CHROMAPRINT_AVAILABLE:
        print("⚠️  Chromaprint not available. Install with:")
        print("   pip install pyacoustid")
        print("   And install chromaprint library: https://acoustid.org/chromaprint")
        return
    
    # Create test audio
    print("\n1. Creating test audio (3 seconds)...")
    audio_bytes = create_test_audio_wav(duration_seconds=3.0)
    print(f"   ✓ Created: {len(audio_bytes):,} bytes")
    
    # Compute chromaprint
    print("\n2. Computing Chromaprint fingerprint...")
    try:
        fingerprint = compute_audio_chromaprint(audio_bytes, duration=3)
        print(f"   ✓ Fingerprint: {fingerprint[:80]}...")
        print(f"   ℹ️  Length: {len(fingerprint)} characters")
        
        # Verify consistency
        print("\n3. Verifying fingerprint consistency...")
        fingerprint2 = compute_audio_chromaprint(audio_bytes, duration=3)
        match = (fingerprint == fingerprint2)
        print(f"   {'✓' if match else '✗'} Fingerprints {'match' if match else 'differ'}")
    except Exception as e:
        print(f"   ⚠️  Chromaprint failed: {e}")
        print("   (This requires chromaprint library to be installed)")
    
    print("\n✅ Example 4 complete!")


# ============================================================================
# EXAMPLE 5: AWS S3 CLOUD STORAGE (MOCKED)
# ============================================================================

def example_5_aws_s3_storage():
    """
    Example 5: Store forensic fragments in AWS S3.
    
    Demonstrates:
    - S3 client configuration
    - Fragment upload/download
    - Metadata storage
    - Fragment listing and deletion
    
    Note: This example uses mocked S3 for demonstration.
    In production, provide real AWS credentials.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: AWS S3 Cloud Storage (Demonstration)")
    print("=" * 70)
    
    if not BOTO3_AVAILABLE:
        print("⚠️  Boto3 not available. Install with:")
        print("   pip install boto3")
        return
    
    print("\n⚠️  Note: This is a demonstration with mock configuration.")
    print("   In production, use real AWS credentials:\n")
    
    # Configure S3
    print("1. Configuring AWS S3 connection...")
    config = CloudStorageConfig(
        provider="aws",
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",  # Mock key
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",  # Mock
        aws_region="us-east-1",
        s3_bucket="ciaf-forensic-fragments",
    )
    print(f"   ✓ Provider: {config.provider}")
    print(f"   ✓ Region: {config.aws_region}")
    print(f"   ✓ Bucket: {config.s3_bucket}")
    
    # Example code (would fail with mock credentials)
    print("\n2. Example upload code:")
    print("""
    # Initialize storage client
    storage = CloudFragmentStorage(config)
    
    # Upload fragment
    fragment_data = b"SHA256:abcd1234...fragment_bytes..."
    url = storage.upload_fragment(
        fragment_id="video_keyframe_0_250ms",
        fragment_data=fragment_data,
        metadata={
            "type": "video_keyframe",
            "format": "mp4",
            "timestamp_ms": "250",
            "entropy": "0.87",
        }
    )
    print(f"Uploaded to: {url}")
    
    # Download fragment
    retrieved = storage.download_fragment("video_keyframe_0_250ms")
    assert retrieved == fragment_data
    
    # List fragments
    fragments = storage.list_fragments(prefix="video_")
    print(f"Found {len(fragments)} video fragments")
    
    # Delete fragment
    storage.delete_fragment("video_keyframe_0_250ms")
    """)
    
    print("\n✅ Example 5 complete!")


# ============================================================================
# EXAMPLE 6: AZURE BLOB STORAGE (MOCKED)
# ============================================================================

def example_6_azure_blob_storage():
    """
    Example 6: Store forensic fragments in Azure Blob Storage.
    
    Demonstrates:
    - Azure Blob client configuration
    - Fragment upload/download
    - Container management
    
    Note: This example uses mocked Azure for demonstration.
    In production, provide real Azure connection string.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Azure Blob Storage (Demonstration)")
    print("=" * 70)
    
    if not AZURE_AVAILABLE:
        print("⚠️  Azure Storage not available. Install with:")
        print("   pip install azure-storage-blob")
        return
    
    print("\n⚠️  Note: This is a demonstration with mock configuration.")
    print("   In production, use real Azure connection string:\n")
    
    # Configure Azure Blob
    print("1. Configuring Azure Blob Storage...")
    config = CloudStorageConfig(
        provider="azure",
        azure_connection_string="DefaultEndpointsProtocol=https;AccountName=ciafforensics;AccountKey=MOCKKEY==;EndpointSuffix=core.windows.net",
        azure_container="forensic-fragments",
    )
    print(f"   ✓ Provider: {config.provider}")
    print(f"   ✓ Container: {config.azure_container}")
    
    # Example code
    print("\n2. Example upload code:")
    print("""
    # Initialize storage client  
    storage = CloudFragmentStorage(config)
    
    # Upload fragment
    fragment_data = b"SHA256:xyz789...fragment_bytes..."
    url = storage.upload_fragment(
        fragment_id="audio_segment_0_500ms",
        fragment_data=fragment_data,
        metadata={
            "type": "audio_segment",
            "format": "wav",
            "start_ms": "500",
            "duration_ms": "2000",
        }
    )
    print(f"Uploaded to: {url}")
    
    # Download fragment
    retrieved = storage.download_fragment("audio_segment_0_500ms")
    
    # List fragments
    fragments = storage.list_fragments(prefix="audio_")
    """)
    
    print("\n✅ Example 6 complete!")


# ============================================================================
# EXAMPLE 7: COMBINED WORKFLOW
# ============================================================================

def example_7_combined_workflow():
    """
    Example 7: Complete workflow using multiple advanced features.
    
    Demonstrates:
    - Video processing with pHash
    - Audio processing with librosa
    - Feature extraction and comparison
    - Simulated cloud storage
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Combined Advanced Features Workflow")
    print("=" * 70)
    
    # Check available features
    features = get_available_features()
    available_count = sum(features.values())
    total_count = len(features)
    
    print(f"\n📊 Available Features: {available_count}/{total_count}")
    for name, available in features.items():
        status = "✓" if available else "✗"
        print(f"   {status} {name}")
    
    # Video workflow (if available)
    if IMAGEHASH_AVAILABLE:
        print("\n1. Video Processing:")
        video_bytes = create_test_video(duration_seconds=1.0)
        if video_bytes:
            phashes = compute_video_phash(video_bytes, num_frames=3)
            print(f"   ✓ Extracted {len(phashes)} perceptual hashes")
            print(f"   ℹ️  First hash: {phashes[0]}")
    else:
        print("\n1. Video Processing: ⚠️  Skipped (imagehash not available)")
    
    # Audio workflow (if available)
    if LIBROSA_AVAILABLE:
        print("\n2. Audio Processing:")
        audio_bytes = create_test_audio_wav(duration_seconds=1.0)
        features = extract_advanced_audio_features(audio_bytes)
        print(f"   ✓ Extracted advanced spectral features")
        print(f"   ℹ️  Tempo: {features.tempo:.1f} BPM")
        print(f"   ℹ️  Spectral centroid: {features.spectral_centroid:.1f} Hz")
    else:
        print("\n2. Audio Processing: ⚠️  Skipped (librosa not available)")
    
    # Cloud storage workflow (if available)
    if BOTO3_AVAILABLE:
        print("\n3. Cloud Storage:")
        print("   ✓ AWS S3 available (configuration required)")
        print("   ℹ️  See Example 5 for S3 usage")
    elif AZURE_AVAILABLE:
        print("\n3. Cloud Storage:")
        print("   ✓ Azure Blob available (configuration required)")
        print("   ℹ️  See Example 6 for Azure usage")
    else:
        print("\n3. Cloud Storage: ⚠️  Not available")
    
    print("\n✅ Example 7 complete!")


# ============================================================================
# EXAMPLE 8: FEATURE STATUS REPORT
# ============================================================================

def example_8_feature_status():
    """
    Example 8: Generate comprehensive feature status report.
    
    Demonstrates:
    - Feature availability checking
    - Installation recommendations
    - System diagnostics
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Feature Status Report")
    print("=" * 70)
    
    # Print detailed status
    print_feature_status()
    
    # Get feature dictionary
    features = get_available_features()
    
    # Installation recommendations
    print("\n📦 Installation Recommendations:")
    
    recommendations = {
        'librosa_spectral_analysis': (
            "pip install librosa",
            "Advanced audio analysis with MFCCs, chroma, tempo detection"
        ),
        'ffmpeg_conversion': (
            "pip install ffmpeg-python + ffmpeg binary",
            "Support for MP3, AAC, H.265, and other formats"
        ),
        'video_perceptual_hash': (
            "pip install imagehash",
            "Video frame perceptual hashing (pHash)"
        ),
        'audio_chromaprint': (
            "pip install pyacoustid + chromaprint library",
            "Audio fingerprinting for robust identification"
        ),
        'aws_s3_storage': (
            "pip install boto3",
            "Amazon S3 cloud storage for fragments"
        ),
        'azure_blob_storage': (
            "pip install azure-storage-blob",
            "Microsoft Azure Blob storage for fragments"
        ),
        'optical_flow_analysis': (
            "pip install opencv-python",
            "Dense optical flow for motion analysis"
        ),
        'scene_change_detection': (
            "pip install opencv-python",
            "Detect scene changes and cuts in video"
        ),
        'keyframe_transitions': (
            "pip install opencv-python",
            "Analyze transitions between keyframes"
        ),
        'audio_beat_tracking': (
            "pip install librosa",
            "Track beats and rhythm in audio"
        ),
        'av_synchronization': (
            "pip install opencv-python librosa",
            "Audio-video synchronization analysis"
        ),
        'object_detection': (
            "pip install opencv-python",
            "Basic object detection in video frames"
        ),
    }
    
    for feature, available in features.items():
        if not available:
            cmd, desc = recommendations[feature]
            print(f"\n   ⚠️  {feature}:")
            print(f"      Install: {cmd}")
            print(f"      Purpose: {desc}")
    
    print("\n✅ Example 8 complete!")


# ============================================================================
# EXAMPLE 9: OPTICAL FLOW & MOTION ANALYSIS
# ============================================================================

def example_9_optical_flow():
    """Demonstrate optical flow analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Optical Flow & Motion Analysis")
    print("=" * 70)
    
    try:
        from ciaf.watermarks.advanced_features import (
            compute_optical_flow,
            CV2_AVAILABLE,
        )
        
        if not CV2_AVAILABLE:
            print("\n❌ OpenCV not available. Install with: pip install opencv-python")
            return
        
        # Create two test frames
        import numpy as np
        import cv2
        
        print("\n1️⃣  Creating test frames with motion...")
        # Frame1: White square at position (50, 50)
        frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (50, 50), (100, 100), (255, 255, 255), -1)
        
        # Frame 2: White square moved to (70, 60)
        frame2 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (70, 60), (120, 110), (255, 255, 255), -1)
        
        # Encode to bytes
        _, frame1_bytes = cv2.imencode('.png', frame1)
        _, frame2_bytes = cv2.imencode('.png', frame2)
        
        print("   ✓ Created 200x200 frames with moving square")
        
        print("\n2️⃣  Computing optical flow...")
        result = compute_optical_flow(frame1_bytes.tobytes(), frame2_bytes.tobytes())
        
        print(f"   ✓ Motion detected!")
        print(f"     - Mean magnitude: {result.magnitude_mean:.2f} pixels")
        print(f"     - Max magnitude: {result.magnitude_max:.2f} pixels")
        print(f"     - Motion score: {result.motion_score:.2f} (0-1)")
        print(f"     - Flow hash: {result.dense_flow_hash[:16]}...")
        
        print("\n✅ Example 9 complete!")
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")


# ============================================================================
# EXAMPLE 10: SCENE CHANGE DETECTION
# ============================================================================

def example_10_scene_detection():
    """Demonstrate scene change detection."""
    print("\n" + "=" * 70)
    print("EXAMPLE 10: Scene Change Detection")
    print("=" * 70)
    
    try:
        from ciaf.watermarks.advanced_features import (
            detect_scene_changes,
            CV2_AVAILABLE,
        )
        
        if not CV2_AVAILABLE:
            print("\n❌ OpenCV not available. Install with: pip install opencv-python")
            return
        
        print("\n1️⃣  Creating test video with scene changes...")
        video_bytes = create_test_video_with_cuts()
        print(f"   ✓ Created test video ({len(video_bytes):,} bytes)")
        
        print("\n2️⃣  Detecting scene changes...")
        changes = detect_scene_changes(video_bytes, threshold=0.3, check_interval=2)
        
        print(f"   ✓ Found {len(changes)} scene changes")
        
        for i, change in enumerate(changes[:5], 1):  # Show first 5
            print(f"\n   Scene Change #{i}:")
            print(f"     - Frame: {change.frame_index}")
            print(f"     - Time: {change.timestamp_ms}ms")
            print(f"     - Score: {change.change_score:.3f}")
            print(f"     - Type: {'Hard cut' if change.is_hard_cut else 'Soft transition'}")
        
        print("\n✅ Example 10 complete!")
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")


# ============================================================================
# EXAMPLE 11: KEYFRAME TRANSITION ANALYSIS
# ============================================================================

def example_11_keyframe_transitions():
    """Demonstrate keyframe transition analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 11: Keyframe Transition Analysis")
    print("=" * 70)
    
    try:
        from ciaf.watermarks.advanced_features import (
            analyze_keyframe_transitions,
            CV2_AVAILABLE,
        )
        
        if not CV2_AVAILABLE:
            print("\n❌ OpenCV not available. Install with: pip install opencv-python")
            return
        
        print("\n1️⃣  Creating test video...")
        video_bytes = create_test_video_with_cuts()
        
        print("\n2️⃣  Analyzing keyframe transitions (frames 0, 10, 20, 30)...")
        transitions = analyze_keyframe_transitions(video_bytes, [0, 10, 20, 30])
        
        print(f"   ✓ Analyzed {len(transitions)} transitions")
        
        for i, trans in enumerate(transitions, 1):
            print(f"\n   Transition #{i} (Frame {trans.from_frame_index} → {trans.to_frame_index}):")
            print(f"     - Type: {trans.transition_type}")
            print(f"     - Confidence: {trans.confidence:.2f}")
            print(f"     - Optical flow score: {trans.optical_flow_score:.2f}")
            print(f"     - Brightness change: {trans.brightness_change:.2f}")
        
        print("\n✅ Example 11 complete!")
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")


# ============================================================================
# EXAMPLE 12: AUDIO BEAT TRACKING
# ============================================================================

def example_12_beat_tracking():
    """Demonstrate audio beat tracking."""
    print("\n" + "=" * 70)
    print("EXAMPLE 12: Audio Beat Tracking")
    print("=" * 70)
    
    try:
        from ciaf.watermarks.advanced_features import (
            track_audio_beats,
            LIBROSA_AVAILABLE,
        )
        
        if not LIBROSA_AVAILABLE:
            print("\n❌ Librosa not available. Install with: pip install librosa")
            return
        
        print("\n1️⃣  Creating test audio with rhythm...")
        audio_bytes = create_test_audio_wav(duration_seconds=5.0)
        print(f"   ✓ Created 5-second audio ({len(audio_bytes):,} bytes)")
        
        print("\n2️⃣  Tracking beats...")
        result = track_audio_beats(audio_bytes)
        
        print(f"\n   ✓ Beat tracking complete!")
        print(f"     - Tempo: {result.tempo:.1f} BPM")
        print(f"     - Beats detected: {len(result.beat_times)}")
        print(f"     - Rhythm regularity: {result.rhythm_regularity:.2f} (0-1)")
        
        print(f"\n   First 10 beats:")
        for i, (time, strength) in enumerate(zip(result.beat_times[:10], result.beat_strength[:10]), 1):
            print(f"     Beat {i}: {time:.2f}s (strength: {strength:.2f})")
        
        print("\n✅ Example 12 complete!")
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")


# ============================================================================
# EXAMPLE 13: A/V SYNCHRONIZATION ANALYSIS
# ============================================================================

def example_13_av_synchronization():
    """Demonstrate audio-video synchronization analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 13: Audio-Video Synchronization")
    print("=" * 70)
    
    try:
        from ciaf.watermarks.advanced_features import (
            analyze_av_synchronization,
            CV2_AVAILABLE,
            LIBROSA_AVAILABLE,
        )
        
        if not (CV2_AVAILABLE and LIBROSA_AVAILABLE):
            print("\n❌ Requires both OpenCV and Librosa")
            print("   Install with: pip install opencv-python librosa")
            return
        
        print("\n1️⃣  Creating test media...")
        video_bytes = create_test_video_with_cuts()
        audio_bytes = create_test_audio_wav(duration_seconds=3.0)
        print("   ✓ Created test video and audio")
        
        print("\n2️⃣  Analyzing synchronization...")
        result = analyze_av_synchronization(video_bytes, audio_bytes, sync_threshold_ms=300.0)
        
        print(f"\n   ✓ Synchronization analysis complete!")
        print(f"     - Audio beats: {len(result.audio_beats)}")
        print(f"     - Video cuts: {len(result.video_cuts)}")
        print(f"     - Synchronized events: {len(result.synchronized_events)}")
        print(f"     - Sync score: {result.sync_score:.2f} (0-1)")
        print(f"     - Avg offset: {result.avg_offset_ms:.1f}ms")
        
        if result.synchronized_events:
            print(f"\n   First 5 synchronized events:")
            for i, (audio_t, video_t) in enumerate(result.synchronized_events[:5], 1):
                offset = abs(audio_t - video_t) * 1000
                print(f"     #{i}: Audio {audio_t:.2f}s ↔ Video {video_t:.2f}s (offset: {offset:.1f}ms)")
        
        print("\n✅ Example 13 complete!")
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")


# ============================================================================
# HELPER FUNCTIONS FOR NEW EXAMPLES
# ============================================================================

def create_test_video_with_cuts():
    """Create a test video with clear scene changes."""
    import cv2
    import numpy as np
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_path, fourcc, 10.0, (160, 120))
        
        # Write 40 frames with scene changes at frame 10, 20, 30
        for i in range(40):
            if i < 10:
                # Scene 1: Black
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
            elif i < 20:
                # Scene 2: White
                frame = np.full((120, 160, 3), 255, dtype=np.uint8)
            elif i < 30:
                # Scene 3: Gray
                frame = np.full((120, 160, 3), 128, dtype=np.uint8)
            else:
                # Scene 4: Dark gray
                frame = np.full((120, 160, 3), 64, dtype=np.uint8)
            
            out.write(frame)
        
        out.release()
        
        # Read back as bytes
        with open(tmp_path, 'rb') as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CIAF WATERMARKING - ADVANCED FEATURES EXAMPLES")
    print("=" * 70)
    print("\nDemonstrating advanced capabilities for forensic watermarking:")
    print("• Advanced spectral analysis (librosa)")
    print("• Multi-format support (ffmpeg)")
    print("• Perceptual hashing (pHash, chromaprint)")
    print("• Cloud storage (AWS S3, Azure Blob)")
    print("• Optical flow & motion analysis (OpenCV)")
    print("• Scene change detection (OpenCV)")
    print("• Keyframe transition analysis (OpenCV)")
    print("• Audio beat tracking (librosa)")
    print("• A/V synchronization analysis (OpenCV + librosa)")
    
    # Run examples
    examples = [
        example_1_advanced_spectral_analysis,
        example_2_multi_format_audio,
        example_3_video_perceptual_hashing,
        example_4_audio_chromaprint,
        example_5_aws_s3_storage,
        example_6_azure_blob_storage,
        example_7_combined_workflow,
        example_8_feature_status,
        example_9_optical_flow,
        example_10_scene_detection,
        example_11_keyframe_transitions,
        example_12_beat_tracking,
        example_13_av_synchronization,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 70)
    
    # Final summary
    features = get_available_features()
    available = sum(features.values())
    total = len(features)
    print(f"\n📊 Final Status: {available}/{total} advanced features available")
    print("\nFor production use:")
    print("  1. Install missing dependencies as needed")
    print("  2. Configure cloud storage credentials")
    print("  3. Test with real media files")
    print("  4. Integrate with CIAF watermarking pipeline")
    print()


if __name__ == "__main__":
    main()
