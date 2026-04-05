"""
Test suite for CIAF Watermarking Advanced Features

Tests for:
1. Advanced spectral analysis with librosa
2. Multi-format support via ffmpeg
3. Perceptual hashing (video pHash, audio chromaprint)
4. Cloud storage integration (mocked)

Created: 2026-04-04
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import hashlib

# Import advanced features module
from ciaf.watermarks.advanced_features import (
    get_available_features,
    LIBROSA_AVAILABLE,
    FFMPEG_AVAILABLE,
    IMAGEHASH_AVAILABLE,
    CHROMAPRINT_AVAILABLE,
    BOTO3_AVAILABLE,
    AZURE_AVAILABLE,
)

# Conditional imports based on availability
if LIBROSA_AVAILABLE:
    from ciaf.watermarks.advanced_features import (
        extract_advanced_audio_features,
        compare_spectral_features,
        AdvancedSpectralFeatures,
        track_audio_beats,
        BeatTrackingResults,
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

# Try to import new video analysis features
try:
    from ciaf.watermarks.advanced_features import (
        compute_optical_flow,
        detect_scene_changes,
        analyze_keyframe_transitions,
        analyze_av_synchronization,
        detect_objects_in_video,
        OpticalFlowAnalysis,
        SceneChange,
        KeyframeTransition,
        SynchronizationAnalysis,
        CV2_AVAILABLE,
    )
except ImportError:
    CV2_AVAILABLE = False

if CHROMAPRINT_AVAILABLE:
    from ciaf.watermarks.advanced_features import (
        compute_audio_chromaprint,
    )

from ciaf.watermarks.advanced_features import (
    CloudStorageConfig,
    CloudFragmentStorage,
)


# ============================================================================
# TEST HELPERS
# ============================================================================

def create_test_wav_audio(duration_seconds=1.0, sample_rate=44100):
    """Create a simple WAV audio file for testing."""
    import wave
    import struct
    import math
    
    num_samples = int(duration_seconds * sample_rate)
    frequency = 440.0  # A4 note
    
    # Generate sine wave
    samples = []
    for i in range(num_samples):
        sample = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        # Stereo: duplicate sample for both channels
        samples.append(struct.pack('<hh', sample, sample))
    
    # Write to WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    # Write WAV (ensures file handle is closed)
    with wave.open(tmp_path, 'wb') as wav_file:
        wav_file.setnchannels(2)  # Stereo
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b''.join(samples))
    
    # Read back as bytes
    with open(tmp_path, 'rb') as f:
        wav_bytes = f.read()
    
    # Delete temp file (with Windows error handling)
    try:
        os.unlink(tmp_path)
    except (PermissionError, OSError):
        pass
    
    return wav_bytes


def create_test_video(duration_seconds=2.0, width=320, height=240, fps=30):
    """Create a simple test video using OpenCV."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        pytest.skip("OpenCV not available")
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        output_path = tmp.name
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = int(duration_seconds * fps)
    
    for i in range(num_frames):
        # Create frame with changing colors
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color_value = int((i / num_frames) * 255)
        frame[:, :] = (color_value, 255 - color_value, 128)
        out.write(frame)
    
    out.release()
    
    # Read video as bytes
    with open(output_path, 'rb') as f:
        video_bytes = f.read()
    
    # Delete temp file (with Windows error handling)
    try:
        os.unlink(output_path)
    except (PermissionError, OSError):
        pass
    
    return video_bytes


# ============================================================================
# ADVANCED SPECTRAL ANALYSIS TESTS
# ============================================================================

@pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="librosa not installed")
class TestAdvancedSpectralAnalysis:
    """Test advanced spectral analysis with librosa."""
    
    def test_extract_advanced_features_basic(self):
        """Test extraction of advanced spectral features."""
        # Create test audio
        audio_bytes = create_test_wav_audio(duration_seconds=1.0)
        
        # Extract features
        features = extract_advanced_audio_features(audio_bytes, duration=1.0)
        
        # Verify feature structure
        assert isinstance(features, AdvancedSpectralFeatures)
        assert len(features.mfcc_mean) == 13  # 13 MFCC coefficients
        assert len(features.mfcc_std) == 13
        assert len(features.chroma_mean) == 12  # 12 chroma bins
        assert len(features.chroma_std) == 12
        
        # Verify spectral features are reasonable
        assert features.spectral_centroid > 0
        assert features.spectral_bandwidth > 0
        assert features.spectral_rolloff > 0
        assert 0 <= features.zero_crossing_rate <= 1
        assert features.tempo > 0
        
        # Verify hash is generated
        assert len(features.features_hash) == 64  # SHA-256 hex
    
    def test_mfcc_values_reasonable(self):
        """Test that MFCC values are in reasonable ranges."""
        audio_bytes = create_test_wav_audio(duration_seconds=0.5)
        features = extract_advanced_audio_features(audio_bytes)
        
        # MFCCs should be finite numbers
        for coeff in features.mfcc_mean:
            assert not (coeff is None or str(coeff) == 'nan' or str(coeff) == 'inf')
        
        for coeff in features.mfcc_std:
            assert coeff >= 0  # Standard deviation is non-negative
    
    def test_chroma_features_valid(self):
        """Test that chroma features are valid."""
        audio_bytes = create_test_wav_audio(duration_seconds=0.5)
        features = extract_advanced_audio_features(audio_bytes)
        
        # Chroma features should be non-negative
        for chroma_val in features.chroma_mean:
            assert chroma_val >= 0
        
        for chroma_val in features.chroma_std:
            assert chroma_val >= 0
    
    def test_tempo_detection(self):
        """Test tempo detection functionality."""
        audio_bytes = create_test_wav_audio(duration_seconds=2.0)
        features = extract_advanced_audio_features(audio_bytes)
        
        # Tempo should be in reasonable BPM range (30-300 BPM)
        assert 30 <= features.tempo <= 300
    
    def test_compare_identical_features(self):
        """Test comparison of identical spectral features."""
        audio_bytes = create_test_wav_audio(duration_seconds=1.0)
        features1 = extract_advanced_audio_features(audio_bytes)
        features2 = extract_advanced_audio_features(audio_bytes)
        
        # Should be very similar (>0.95)
        similarity = compare_spectral_features(features1, features2)
        assert similarity > 0.95
    
    def test_compare_different_features(self):
        """Test comparison of different spectral features."""
        # Create two different audio samples
        audio1 = create_test_wav_audio(duration_seconds=0.5)
        
        # Create a different audio (will be different due to randomness in timestamps)
        import time
        time.sleep(0.01)  # Small delay to ensure different random state
        audio2 = create_test_wav_audio(duration_seconds=0.5)
        
        features1 = extract_advanced_audio_features(audio1)
        features2 = extract_advanced_audio_features(audio2)
        
        # Should still be somewhat similar since both are sine waves
        similarity = compare_spectral_features(features1, features2)
        assert 0.0 <= similarity <= 1.0
    
    def test_feature_hash_consistency(self):
        """Test that feature hash is consistent for same audio."""
        audio_bytes = create_test_wav_audio(duration_seconds=0.5)
        features1 = extract_advanced_audio_features(audio_bytes)
        features2 = extract_advanced_audio_features(audio_bytes)
        
        # Hashes should be identical
        assert features1.features_hash == features2.features_hash


# ============================================================================
# MULTI-FORMAT SUPPORT TESTS
# ============================================================================

@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg-python not installed")
class TestMultiFormatSupport:
    """Test multi-format conversion with ffmpeg."""
    
    def test_wav_audio_passthrough(self):
        """Test converting WAV to WAV (should work smoothly)."""
        # Create WAV audio
        input_wav = create_test_wav_audio(duration_seconds=0.5)
        
        # Convert WAV to WAV
        output_wav = convert_audio_to_wav(input_wav, input_format="wav")
        
        # Should produce valid WAV output
        assert len(output_wav) > 0
        assert output_wav[:4] == b'RIFF'  # WAV header
    
    def test_audio_sample_rate_conversion(self):
        """Test converting audio to different sample rate."""
        input_wav = create_test_wav_audio(duration_seconds=0.5)
        
        # Convert to 22050 Hz
        output_wav = convert_audio_to_wav(input_wav, output_sample_rate=22050)
        
        # Verify output is valid WAV
        assert len(output_wav) > 0
        assert output_wav[:4] == b'RIFF'
        
        # Verify sample rate in WAV header (bytes 24-27)
        import struct
        sample_rate = struct.unpack('<I', output_wav[24:28])[0]
        assert sample_rate == 22050
    
    @pytest.mark.skip(reason="Requires actual ffmpeg binary installed")
    def test_mp3_to_wav_conversion(self):
        """Test converting MP3 to WAV (requires ffmpeg binary)."""
        # This test would require actual MP3 file
        # Skipping in CI/CD unless ffmpeg is installed
        pass
    
    def test_video_format_basic(self):
        """Test basic video format handling."""
        # Create test video
        video_bytes = create_test_video(duration_seconds=1.0)
        
        # Convert to MP4
        output_mp4 = convert_video_to_mp4(video_bytes, input_format="mp4")
        
        # Should produce valid MP4
        assert len(output_mp4) > 0
        # MP4 files typically start with ftyp box
        assert b'ftyp' in output_mp4[:100]
    
    @pytest.mark.skip(reason="Requires actual ffmpeg binary installed")
    def test_h265_encoding(self):
        """Test H.265 (HEVC) video encoding."""
        # Would require actual ffmpeg with libx265
        pass


# ============================================================================
# PERCEPTUAL HASHING TESTS
# ============================================================================

@pytest.mark.skipif(not IMAGEHASH_AVAILABLE, reason="imagehash not installed")
class TestPerceptualHashing:
    """Test perceptual hashing for videos and audio."""
    
    def test_video_phash_extraction(self):
        """Test extraction of video perceptual hashes."""
        video_bytes = create_test_video(duration_seconds=1.0, width=320, height=240)
        
        # Compute pHash for 3 frames
        phashes = compute_video_phash(video_bytes, num_frames=3)
        
        # Should get 3 hashes
        assert len(phashes) == 3
        
        # Each hash should be hex string (typically 16 characters for 8x8 pHash)
        for phash in phashes:
            assert isinstance(phash, str)
            assert len(phash) >= 8  # At least 8 hex chars
    
    def test_video_phash_consistency(self):
        """Test that pHash is consistent for same video."""
        video_bytes = create_test_video(duration_seconds=0.5)
        
        phashes1 = compute_video_phash(video_bytes, num_frames=2)
        phashes2 = compute_video_phash(video_bytes, num_frames=2)
        
        # Should get identical hashes
        assert phashes1 == phashes2
    
    def test_phash_comparison_identical(self):
        """Test pHash comparison for identical hashes."""
        hash1 = "0123456789abcdef"
        hash2 = "0123456789abcdef"
        
        similarity = compare_perceptual_hashes(hash1, hash2, hash_type="phash")
        
        # Should be identical (similarity = 1.0)
        assert similarity == 1.0
    
    def test_phash_comparison_different(self):
        """Test pHash comparison for different hashes."""
        hash1 = "0000000000000000"
        hash2 = "ffffffffffffffff"
        
        similarity = compare_perceptual_hashes(hash1, hash2, hash_type="phash")
        
        # Should be completely different (similarity = 0.0)
        assert similarity == 0.0
    
    def test_phash_comparison_similar(self):
        """Test pHash comparison for similar hashes (1 bit different)."""
        hash1 = "0123456789abcdef"
        hash2 = "0123456789abcdee"  # Last char different (f -> e)
        
        similarity = compare_perceptual_hashes(hash1, hash2, hash_type="phash")
        
        # Should be very similar but not identical
        assert 0.9 < similarity < 1.0
    
    @pytest.mark.skipif(not CHROMAPRINT_AVAILABLE, reason="chromaprint not installed")
    def test_audio_chromaprint_extraction(self):
        """Test extraction of audio chromaprint."""
        audio_bytes = create_test_wav_audio(duration_seconds=2.0)
        
        # Compute chromaprint
        fingerprint = compute_audio_chromaprint(audio_bytes, duration=2)
        
        # Should get a base64-encoded fingerprint
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0
    
    @pytest.mark.skipif(not CHROMAPRINT_AVAILABLE, reason="chromaprint not installed")
    def test_chromaprint_consistency(self):
        """Test that chromaprint is consistent for same audio."""
        audio_bytes = create_test_wav_audio(duration_seconds=1.0)
        
        fp1 = compute_audio_chromaprint(audio_bytes, duration=1)
        fp2 = compute_audio_chromaprint(audio_bytes, duration=1)
        
        # Should be identical
        assert fp1 == fp2


# ============================================================================
# CLOUD STORAGE TESTS (MOCKED)
# ============================================================================

class TestCloudStorageMocked:
    """Test cloud storage integration with mocked backends."""
    
    def test_aws_config_initialization(self):
        """Test AWS S3 configuration."""
        config = CloudStorageConfig(
            provider="aws",
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_region="us-east-1",
            s3_bucket="test-bucket",
        )
        
        assert config.provider == "aws"
        assert config.s3_bucket == "test-bucket"
    
    def test_azure_config_initialization(self):
        """Test Azure Blob configuration."""
        config = CloudStorageConfig(
            provider="azure",
            azure_connection_string="DefaultEndpointsProtocol=https;AccountName=test;...",
            azure_container="test-container",
        )
        
        assert config.provider == "azure"
        assert config.azure_container == "test-container"
    
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
    @patch('boto3.client')
    def test_aws_upload_fragment(self, mock_boto_client):
        """Test uploading fragment to AWS S3 (mocked)."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        
        config = CloudStorageConfig(
            provider="aws",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_region="us-east-1",
            s3_bucket="test-bucket",
        )
        
        storage = CloudFragmentStorage(config)
        
        # Upload fragment
        fragment_data = b"test fragment data"
        url = storage.upload_fragment(
            "fragment_001",
            fragment_data,
            metadata={"type": "video_keyframe"}
        )
        
        # Verify S3 put_object was called
        mock_s3.put_object.assert_called_once()
        assert "s3://" in url
        assert "test-bucket" in url
    
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
    @patch('boto3.client')
    def test_aws_download_fragment(self, mock_boto_client):
        """Test downloading fragment from AWS S3 (mocked)."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_response = {'Body': MagicMock()}
        mock_response['Body'].read.return_value = b"test fragment data"
        mock_s3.get_object.return_value = mock_response
        mock_boto_client.return_value = mock_s3
        
        config = CloudStorageConfig(
            provider="aws",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_region="us-east-1",
            s3_bucket="test-bucket",
        )
        
        storage = CloudFragmentStorage(config)
        
        # Download fragment
        data = storage.download_fragment("fragment_001")
        
        # Verify get_object was called
        mock_s3.get_object.assert_called_once()
        assert data == b"test fragment data"
    
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
    @patch('boto3.client')
    def test_aws_list_fragments(self, mock_boto_client):
        """Test listing fragments from AWS S3 (mocked)."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'fragments/fragment_001'},
                {'Key': 'fragments/fragment_002'},
                {'Key': 'fragments/fragment_003'},
            ]
        }
        mock_boto_client.return_value = mock_s3
        
        config = CloudStorageConfig(
            provider="aws",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_region="us-east-1",
            s3_bucket="test-bucket",
        )
        
        storage = CloudFragmentStorage(config)
        
        # List fragments
        fragments = storage.list_fragments()
        
        # Verify list_objects_v2 was called
        mock_s3.list_objects_v2.assert_called_once()
        assert len(fragments) == 3
        assert "fragment_001" in fragments
    
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
    @patch('boto3.client')
    def test_aws_delete_fragment(self, mock_boto_client):
        """Test deleting fragment from AWS S3 (mocked)."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        
        config = CloudStorageConfig(
            provider="aws",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_region="us-east-1",
            s3_bucket="test-bucket",
        )
        
        storage = CloudFragmentStorage(config)
        
        # Delete fragment
        result = storage.delete_fragment("fragment_001")
        
        # Verify delete_object was called
        mock_s3.delete_object.assert_called_once()
        assert result is True
    
    @pytest.mark.skipif(not AZURE_AVAILABLE, reason="azure-storage-blob not installed")
    @patch('ciaf.watermarks.advanced_features.BlobServiceClient')
    def test_azure_upload_fragment(self, mock_blob_service):
        """Test uploading fragment to Azure Blob (mocked)."""
        # Mock Azure Blob client
        mock_blob_client = MagicMock()
        mock_container = MagicMock()
        mock_service = MagicMock()
        mock_service.account_name = "testaccount"
        mock_service.get_blob_client.return_value = mock_blob_client
        mock_blob_service.from_connection_string.return_value = mock_service
        
        config = CloudStorageConfig(
            provider="azure",
            azure_connection_string="test_connection_string",
            azure_container="test-container",
        )
        
        storage = CloudFragmentStorage(config)
        
        # Upload fragment
        fragment_data = b"test fragment data"
        url = storage.upload_fragment(
            "fragment_001",
            fragment_data,
            metadata={"type": "video_keyframe"}
        )
        
        # Verify upload_blob was called
        mock_blob_client.upload_blob.assert_called_once()
        assert "https://" in url
        assert "testaccount" in url


# ============================================================================
# FEATURE AVAILABILITY TESTS
# ============================================================================

class TestFeatureAvailability:
    """Test feature availability detection."""
    
    def test_get_available_features(self):
        """Test getting available features dictionary."""
        features = get_available_features()
        
        # Should have all expected keys
        expected_keys = {
            'librosa_spectral_analysis',
            'ffmpeg_conversion',
            'video_perceptual_hash',
            'audio_chromaprint',
            'aws_s3_storage',
            'azure_blob_storage',
            'optical_flow_analysis',
            'scene_change_detection',
            'keyframe_transitions',
            'audio_beat_tracking',
            'av_synchronization',
            'object_detection',
        }
        
        assert set(features.keys()) == expected_keys
        
        # All values should be boolean
        for key, value in features.items():
            assert isinstance(value, bool)
    
    def test_feature_flags_consistent(self):
        """Test that feature flags are consistent."""
        features = get_available_features()
        
        # Check individual flags match module-level constants
        assert features['librosa_spectral_analysis'] == LIBROSA_AVAILABLE
        assert features['ffmpeg_conversion'] == FFMPEG_AVAILABLE
        assert features['video_perceptual_hash'] == IMAGEHASH_AVAILABLE
        assert features['audio_chromaprint'] == CHROMAPRINT_AVAILABLE
        assert features['aws_s3_storage'] == BOTO3_AVAILABLE
        assert features['azure_blob_storage'] == AZURE_AVAILABLE


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.skipif(not (LIBROSA_AVAILABLE and IMAGEHASH_AVAILABLE), 
                    reason="requires librosa and imagehash")
class TestAdvancedFeaturesIntegration:
    """Integration tests combining multiple advanced features."""
    
    def test_audio_advanced_workflow(self):
        """Test complete workflow with advanced audio features."""
        # Create test audio
        audio_bytes = create_test_wav_audio(duration_seconds=2.0)
        
        # Extract advanced features
        features = extract_advanced_audio_features(audio_bytes, duration=2.0)
        
        # Verify we got comprehensive analysis
        assert len(features.mfcc_mean) == 13
        assert len(features.chroma_mean) == 12
        assert features.tempo > 0
        assert len(features.features_hash) == 64
        
        # Extract again and compare
        features2 = extract_advanced_audio_features(audio_bytes, duration=2.0)
        similarity = compare_spectral_features(features, features2)
        
        # Should be very similar
        assert similarity > 0.95
    
    def test_video_advanced_workflow(self):
        """Test complete workflow with advanced video features."""
        # Create test video
        video_bytes = create_test_video(duration_seconds=1.0)
        
        # Compute perceptual hashes
        phashes = compute_video_phash(video_bytes, num_frames=3)
        
        # Verify we got hashes
        assert len(phashes) == 3
        
        # Compute again and verify consistency
        phashes2 = compute_video_phash(video_bytes, num_frames=3)
        assert phashes == phashes2
        
        # Compare similarity
        for h1, h2 in zip(phashes, phashes2):
            similarity = compare_perceptual_hashes(h1, h2, hash_type="phash")
            assert similarity == 1.0


# ============================================================================
# OPTICAL FLOW & MOTION ANALYSIS TESTS
# ============================================================================

@pytest.mark.skipif(not CV2_AVAILABLE, reason="requires opencv-python")
class TestOpticalFlowAnalysis:
    """Tests for optical flow analysis."""
    
    def test_optical_flow_basic(self):
        """Test basic optical flow computation."""
        # Create two similar test frames
        import numpy as np
        import cv2
        
        # Frame 1: Black with white square
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (25, 25), (50, 50), (255, 255, 255), -1)
        
        # Frame 2: Black with white square slightly shifted
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (30, 30), (55, 55), (255, 255, 255), -1)
        
        # Encode frames to bytes
        _, frame1_bytes = cv2.imencode('.png', frame1)
        _, frame2_bytes = cv2.imencode('.png', frame2)
        
        # Compute optical flow
        result = compute_optical_flow(frame1_bytes.tobytes(), frame2_bytes.tobytes())
        
        # Verify structure
        assert isinstance(result, OpticalFlowAnalysis)
        assert result.magnitude_mean >= 0
        assert result.motion_score >= 0
        assert result.motion_score <= 1.0
        assert len(result.dense_flow_hash) == 64  # SHA-256
    
    def test_optical_flow_static_frames(self):
        """Test optical flow with identical frames (no motion)."""
        import numpy as np
        import cv2
        
        # Create identical frames
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _, frame_bytes = cv2.imencode('.png', frame)
        
        result = compute_optical_flow(frame_bytes.tobytes(), frame_bytes.tobytes())
        
        # Should have minimal motion
        assert result.magnitude_mean < 1.0
        assert result.motion_score < 0.1


# ============================================================================
# SCENE CHANGE DETECTION TESTS
# ============================================================================

@pytest.mark.skipif(not CV2_AVAILABLE, reason="requires opencv-python")
class TestSceneChangeDetection:
    """Tests for scene change detection."""
    
    def test_scene_detection_basic(self):
        """Test basic scene change detection."""
        # Create minimal test video
        video_bytes = create_test_video_simple()
        
        # Detect scene changes
        changes = detect_scene_changes(video_bytes, threshold=0.2, check_interval=1)
        
        # Verify structure
        assert isinstance(changes, list)
        for change in changes:
            assert isinstance(change, SceneChange)
            assert change.frame_index >= 0
            assert 0 <= change.change_score <= 1.0
            assert isinstance(change.is_hard_cut, bool)
    
    def test_scene_detection_no_changes(self):
        """Test with uniform video (no scene changes)."""
        # Create video with static frames
        video_bytes = create_static_video()
        
        changes = detect_scene_changes(video_bytes, threshold=0.9)
        
        # Should detect few or no changes
        assert len(changes) < 3


# ============================================================================
# KEYFRAME TRANSITION ANALYSIS TESTS
# ============================================================================

@pytest.mark.skipif(not CV2_AVAILABLE, reason="requires opencv-python")
class TestKeyframeTransitionAnalysis:
    """Tests for keyframe transition analysis."""
    
    def test_transition_analysis_basic(self):
        """Test basic transition analysis."""
        video_bytes = create_test_video_simple()
        
        # Analyze transitions between frames 0, 10, 20
        transitions = analyze_keyframe_transitions(video_bytes, [0, 10, 20])
        
        # Should have 2 transitions (0->10, 10->20)
        assert isinstance(transitions, list)
        for trans in transitions:
            assert isinstance(trans, KeyframeTransition)
            assert trans.transition_type in ["cut", "fade", "dissolve", "motion"]
            assert 0 <= trans.confidence <= 1.0
    
    def test_transition_types(self):
        """Test different transition type detection."""
        video_bytes = create_test_video_simple()
        
        transitions = analyze_keyframe_transitions(video_bytes, [0, 5])
        
        # Should detect at least one transition
        if len(transitions) > 0:
            assert hasattr(transitions[0], 'transition_type')


# ============================================================================
# AUDIO BEAT TRACKING TESTS
# ============================================================================

@pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="requires librosa")
class TestAudioBeatTracking:
    """Tests for audio beat tracking."""
    
    def test_beat_tracking_basic(self):
        """Test basic beat tracking."""
        # Create test audio with clear rhythm
        audio_bytes = create_test_wav_audio(duration_seconds=3.0, frequency=440.0)
        
        result = track_audio_beats(audio_bytes)
        
        # Verify structure
        assert isinstance(result, BeatTrackingResults)
        assert result.tempo > 0
        assert isinstance(result.beat_frames, list)
        assert isinstance(result.beat_times, list)
        assert len(result.beat_times) == len(result.beat_frames)
        assert 0 <= result.rhythm_regularity <= 1.0
    
    def test_beat_tracking_tempo_range(self):
        """Test that detected tempo is in reasonable range."""
        audio_bytes = create_test_wav_audio(duration_seconds=5.0)
        
        result = track_audio_beats(audio_bytes)
        
        # Tempo should be in typical range (30-300 BPM)
        assert 30 <= result.tempo <= 300


# ============================================================================
# CROSS-MODAL SYNCHRONIZATION TESTS
# ============================================================================

@pytest.mark.skipif(not (CV2_AVAILABLE and LIBROSA_AVAILABLE), 
                    reason="requires opencv and librosa")
class TestAVSynchronization:
    """Tests for audio-video synchronization analysis."""
    
    def test_av_sync_basic(self):
        """Test basic A/V synchronization analysis."""
        video_bytes = create_test_video_simple()
        audio_bytes = create_test_wav_audio(duration_seconds=2.0)
        
        result = analyze_av_synchronization(video_bytes, audio_bytes)
        
        # Verify structure
        assert isinstance(result, SynchronizationAnalysis)
        assert isinstance(result.audio_beats, list)
        assert isinstance(result.video_cuts, list)
        assert isinstance(result.synchronized_events, list)
        assert 0 <= result.sync_score <= 1.0
        assert result.avg_offset_ms >= 0
    
    def test_av_sync_score_range(self):
        """Test that sync score is properly bounded."""
        video_bytes = create_test_video_simple()
        audio_bytes = create_test_wav_audio(duration_seconds=1.0)
        
        result = analyze_av_synchronization(video_bytes, audio_bytes, sync_threshold_ms=500.0)
        
        assert result.sync_score >= 0.0
        assert result.sync_score <= 1.0


# ============================================================================
# OBJECT DETECTION TESTS
# ============================================================================

@pytest.mark.skipif(not CV2_AVAILABLE, reason="requires opencv-python")
class TestObjectDetection:
    """Tests for object detection."""
    
    def test_object_detection_structure(self):
        """Test object detection returns proper structure."""
        video_bytes = create_test_video_simple()
        
        results = detect_objects_in_video(video_bytes, [0, 10])
        
        # Should return results even if empty (no models)
        assert isinstance(results, list)
        assert len(results) == 2  # Two frames requested
        
        for result in results:
            assert hasattr(result, 'frame_index')
            assert hasattr(result, 'objects')
            assert hasattr(result, 'num_objects')


# ============================================================================
# HELPER FUNCTIONS FOR TESTS
# ============================================================================

def create_test_video_simple():
    """Create a simple test video with OpenCV."""
    import cv2
    import numpy as np
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_path, fourcc, 10.0, (160, 120))
        
        # Write 30 frames
        for i in range(30):
            # Create frame with changing brightness
            brightness = int((i / 30) * 255)
            frame = np.full((120, 160, 3), brightness, dtype=np.uint8)
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


def create_static_video():
    """Create a video with static content (no changes)."""
    import cv2
    import numpy as np
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_path, fourcc, 10.0, (160, 120))
        
        # Write identical frames
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for i in range(20):
            out.write(frame)
        
        out.release()
        
        with open(tmp_path, 'rb') as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
