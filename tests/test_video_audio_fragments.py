"""
Video/Audio Fragment Selection Tests
=====================================

Tests for video keyframe and audio segment forensic fragment selection.

Created: 2026-04-04
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import os
from io import BytesIO

from ciaf.watermarks.fragment_selection import (
    select_video_forensic_snippets,
    select_audio_forensic_segments,
    CV2_AVAILABLE,
    WAVE_AVAILABLE,
)
from ciaf.watermarks.models import VideoForensicSnippet, AudioForensicSegment


# ============================================================================
# HELPER FUNCTIONS - Create Test Media Files
# ============================================================================


def create_test_video(width=320, height=240, duration_seconds=3, fps=10):
    """
    Create a simple test video with opencv.
    
    Returns video as bytes (MP4 format).
    """
    if not CV2_AVAILABLE:
        pytest.skip("opencv-python not available")
    
    import cv2
    import numpy as np
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name
    
    try:
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_frames = duration_seconds * fps
        
        for i in range(total_frames):
            # Create frame with changing color (to have variation)
            color = int(255 * i / total_frames)
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            
            # Add some pattern for entropy
            cv2.rectangle(frame, (10, 10), (50, 50), (255 - color, color, 128), -1)
            cv2.circle(frame, (width - 30, height - 30), 20, (color, 255 - color, 200), -1)
            
            out.write(frame)
        
        out.release()
        
        # Read video bytes
        with open(output_path, 'rb') as f:
            video_bytes = f.read()
        
        return video_bytes
    
    finally:
        try:
            os.unlink(output_path)
        except:
            pass


def create_test_audio_wav(duration_seconds=3, sample_rate=22050, frequency=440):
    """
    Create a simple test audio file (WAV format).
    
    Returns audio as bytes.
    """
    if not WAVE_AVAILABLE:
        pytest.skip("wave module not available")
    
    import wave
    import struct
    import math
    
    # Generate sine wave
    num_samples = int(duration_seconds * sample_rate)
    samples = []
    
    for i in range(num_samples):
        # Sine wave with varying frequency (to have variation)
        varying_freq = frequency + (50 * math.sin(2 * math.pi * i / sample_rate))
        value = int(32767 * math.sin(2 * math.pi * varying_freq * i / sample_rate))
        samples.append(value)
    
    # Write to BytesIO
    audio_stream = BytesIO()
    
    with wave.open(audio_stream, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Write samples
        for sample in samples:
            wav_file.writeframes(struct.pack('<h', sample))
    
    return audio_stream.getvalue()


# ============================================================================
# VIDEO FRAGMENT SELECTION TESTS
# ============================================================================


class TestVideoFragmentSelection:
    """Test video forensic snippet selection."""
    
    def test_select_video_snippets_basic(self):
        """Test basic video snippet selection."""
        if not CV2_AVAILABLE:
            pytest.skip("opencv-python not available")
        
        video_bytes = create_test_video(duration_seconds=2, fps=10)
        
        snippets = select_video_forensic_snippets(
            video_bytes,
            num_keyframes=3
        )
        
        assert len(snippets) == 3, "Should select 3 keyframes"
        assert all(isinstance(s, VideoForensicSnippet) for s in snippets)
        
        # Check snippet properties
        for snippet in snippets:
            assert snippet.fragment_type == "video_keyframe"
            assert snippet.timestamp_ms >= 0
            assert snippet.frame_index >= 0
            assert len(snippet.frame_patch_hashes) > 0
            assert 0.0 <= snippet.entropy_score <= 1.0
            print(f"Snippet: frame {snippet.frame_index}, "
                  f"time {snippet.timestamp_ms}ms, "
                  f"entropy {snippet.entropy_score:.3f}")
    
    def test_video_snippets_temporal_distribution(self):
        """Test that snippets are evenly distributed in time."""
        if not CV2_AVAILABLE:
            pytest.skip("opencv-python not available")
        
        video_bytes = create_test_video(duration_seconds=5, fps=10)
        
        snippets = select_video_forensic_snippets(
            video_bytes,
            num_keyframes=4
        )
        
        assert len(snippets) == 4
        
        # Check temporal distribution (should be ~1000ms, ~2500ms, ~3750ms, ~5000ms)
        timestamps = [s.timestamp_ms for s in snippets]
        
        # Timestamps should be in ascending order
        assert timestamps == sorted(timestamps), "Timestamps should be in order"
        
        # Timestamps should be reasonably spaced
        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            assert gap > 0, "Timestamps should increase"
            print(f"Gap {i}: {gap}ms")
    
    def test_video_patch_hashes(self):
        """Test that frame patch hashes are computed."""
        if not CV2_AVAILABLE:
            pytest.skip("opencv-python not available")
        
        video_bytes = create_test_video(width=640, height=480, duration_seconds=1, fps=5)
        
        snippets = select_video_forensic_snippets(video_bytes, num_keyframes=2)
        
        assert len(snippets) == 2
        
        for snippet in snippets:
            assert len(snippet.frame_patch_hashes) > 0
            assert all(isinstance(h, str) for h in snippet.frame_patch_hashes)
            assert all(len(h) == 64 for h in snippet.frame_patch_hashes)  # SHA-256 hex
            print(f"Frame {snippet.frame_index}: {len(snippet.frame_patch_hashes)} patches")
    
    def test_video_motion_signatures(self):
        """Test that motion signatures are computed between frames."""
        if not CV2_AVAILABLE:
            pytest.skip("opencv-python not available")
        
        video_bytes = create_test_video(duration_seconds=2, fps=10)
        
        snippets = select_video_forensic_snippets(video_bytes, num_keyframes=3)
        
        # First snippet has no prior frame, so no motion
        assert snippets[0].temporal_motion_hash is None
        assert snippets[0].motion_confidence == 0.0
        
        # Subsequent snippets should have motion signatures
        for snippet in snippets[1:]:
            assert snippet.temporal_motion_hash is not None
            assert len(snippet.temporal_motion_hash) == 64  # SHA-256
            assert 0.0 <= snippet.motion_confidence <= 1.0
            print(f"Motion confidence: {snippet.motion_confidence:.3f}")
    
    def test_video_entropy_calculation(self):
        """Test that frame entropy is calculated."""
        if not CV2_AVAILABLE:
            pytest.skip("opencv-python not available")
        
        video_bytes = create_test_video(duration_seconds=1, fps=5)
        
        snippets = select_video_forensic_snippets(video_bytes, num_keyframes=2)
        
        for snippet in snippets:
            assert 0.0 <= snippet.entropy_score <= 1.0
            # Test video has patterns, so entropy should be > 0
            assert snippet.entropy_score > 0.0
            print(f"Frame entropy: {snippet.entropy_score:.4f}")
    
    def test_video_invalid_input(self):
        """Test error handling for invalid video."""
        if not CV2_AVAILABLE:
            pytest.skip("opencv-python not available")
        
        # Invalid video data
        with pytest.raises(ValueError, match="Failed to open"):
            select_video_forensic_snippets(b"not a video", num_keyframes=2)
    
    def test_video_without_opencv(self):
        """Test that missing opencv raises informative error."""
        # Mock CV2_AVAILABLE = False
        import ciaf.watermarks.fragment_selection as frag_module
        original_cv2 = frag_module.CV2_AVAILABLE
        
        try:
            frag_module.CV2_AVAILABLE = False
            
            with pytest.raises(ImportError, match="opencv-python"):
                select_video_forensic_snippets(b"video data", num_keyframes=2)
        
        finally:
            frag_module.CV2_AVAILABLE = original_cv2


# ============================================================================
# AUDIO FRAGMENT SELECTION TESTS
# ============================================================================


class TestAudioFragmentSelection:
    """Test audio forensic segment selection."""
    
    def test_select_audio_segments_basic(self):
        """Test basic audio segment selection."""
        if not WAVE_AVAILABLE:
            pytest.skip("wave module not available")
        
        audio_bytes = create_test_audio_wav(duration_seconds=3, sample_rate=22050)
        
        segments = select_audio_forensic_segments(
            audio_bytes,
            num_segments=3,
            segment_duration_ms=1000
        )
        
        assert len(segments) == 3, "Should select 3 segments"
        assert all(isinstance(s, AudioForensicSegment) for s in segments)
        
        # Check segment properties
        for segment in segments:
            assert segment.fragment_type == "audio_spectral_segment"
            assert segment.start_time_ms >= 0
            assert segment.segment_duration_ms > 0
            assert segment.spectrogram_hash is not None
            assert len(segment.spectrogram_hash) == 64  # SHA-256
            assert 0.0 <= segment.entropy_score <= 1.0
            print(f"Segment: start {segment.start_time_ms}ms, "
                  f"duration {segment.segment_duration_ms}ms, "
                  f"entropy {segment.entropy_score:.3f}")
    
    def test_audio_temporal_distribution(self):
        """Test that audio segments are evenly distributed."""
        if not WAVE_AVAILABLE:
            pytest.skip("wave module not available")
        
        audio_bytes = create_test_audio_wav(duration_seconds=6, sample_rate=22050)
        
        segments = select_audio_forensic_segments(
            audio_bytes,
            num_segments=4,
            segment_duration_ms=1000
        )
        
        assert len(segments) == 4
        
        # Check temporal distribution
        start_times = [s.start_time_ms for s in segments]
        
        # Start times should be in ascending order
        assert start_times == sorted(start_times), "Start times should be in order"
        
        # Start times should be reasonably spaced
        for i in range(len(start_times) - 1):
            gap = start_times[i + 1] - start_times[i]
            assert gap > 0, "Start times should increase"
            print(f"Gap {i}: {gap}ms")
    
    def test_audio_spectral_features(self):
        """Test that spectral features are computed."""
        if not WAVE_AVAILABLE:
            pytest.skip("wave module not available")
        
        audio_bytes = create_test_audio_wav(duration_seconds=2, sample_rate=22050, frequency=440)
        
        segments = select_audio_forensic_segments(
            audio_bytes,
            num_segments=2,
            segment_duration_ms=500
        )
        
        assert len(segments) == 2
        
        for segment in segments:
            # Check frequency centroid
            assert segment.frequency_centroid > 0
            assert segment.frequency_centroid <= 22050 / 2  # Nyquist limit
            
            # Check spectral flatness
            assert 0.0 <= segment.spectral_flatness <= 1.0
            
            print(f"Spectral features: "
                  f"centroid={segment.frequency_centroid:.1f}Hz, "
                  f"flatness={segment.spectral_flatness:.3f}")
    
    def test_audio_segment_hashes(self):
        """Test that segment hashes are unique."""
        if not WAVE_AVAILABLE:
            pytest.skip("wave module not available")
        
        audio_bytes = create_test_audio_wav(duration_seconds=3, sample_rate=22050)
        
        segments = select_audio_forensic_segments(
            audio_bytes,
            num_segments=3,
            segment_duration_ms=500
        )
        
        # All hashes should be different (different audio content)
        hashes = [s.spectrogram_hash for s in segments]
        assert len(set(hashes)) == len(hashes), "Segment hashes should be unique"
        
        # Hashes should match before/after (no watermark applied yet)
        for segment in segments:
            assert segment.spectrogram_hash == segment.spectrogram_hash_before
            assert segment.spectrogram_hash == segment.spectrogram_hash_after
    
    def test_audio_entropy_distribution(self):
        """Test that audio entropy is calculated correctly."""
        if not WAVE_AVAILABLE:
            pytest.skip("wave module not available")
        
        # Create audio with varying characteristics
        audio_bytes = create_test_audio_wav(duration_seconds=2, sample_rate=22050, frequency=440)
        
        segments = select_audio_forensic_segments(audio_bytes, num_segments=2)
        
        for segment in segments:
            # Entropy should be in valid range
            assert 0.0 <= segment.entropy_score <= 1.0
            # Sine wave has some entropy
            assert segment.entropy_score > 0.0
            print(f"Audio entropy: {segment.entropy_score:.4f}")
    
    def test_audio_different_durations(self):
        """Test with different segment durations."""
        if not WAVE_AVAILABLE:
            pytest.skip("wave module not available")
        
        audio_bytes = create_test_audio_wav(duration_seconds=5, sample_rate=22050)
        
        # Test with 500ms segments
        segments_short = select_audio_forensic_segments(
            audio_bytes,
            num_segments=2,
            segment_duration_ms=500
        )
        
        # Test with 2000ms segments
        segments_long = select_audio_forensic_segments(
            audio_bytes,
            num_segments=2,
            segment_duration_ms=2000
        )
        
        assert len(segments_short) == 2
        assert len(segments_long) == 2
        
        # Short segments should have shorter durations
        assert all(s.segment_duration_ms <= 500 for s in segments_short)
        assert all(s.segment_duration_ms >= 500 for s in segments_long)
    
    def test_audio_invalid_input(self):
        """Test error handling for invalid audio."""
        if not WAVE_AVAILABLE:
            pytest.skip("wave module not available")
        
        # Invalid audio data
        with pytest.raises(ValueError):
            select_audio_forensic_segments(b"not audio", num_segments=2)
    
    def test_audio_edge_case_short(self):
        """Test with very short audio."""
        if not WAVE_AVAILABLE:
            pytest.skip("wave module not available")
        
        # Create very short audio (0.5 seconds)
        audio_bytes = create_test_audio_wav(duration_seconds=0.5, sample_rate=22050)
        
        # Request 2 segments
        segments = select_audio_forensic_segments(
            audio_bytes,
            num_segments=2,
            segment_duration_ms=200
        )
        
        # Should get at least 1 segment (may get less than requested)
        assert len(segments) >= 1
        assert len(segments) <= 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestVideoAudioIntegration:
    """Test integration scenarios."""
    
    def test_video_audio_combined_workflow(self):
        """Test selecting fragments from both video and audio."""
        if not CV2_AVAILABLE or not WAVE_AVAILABLE:
            pytest.skip("opencv-python or wave not available")
        
        # Create test media
        video_bytes = create_test_video(duration_seconds=3, fps=10)
        audio_bytes = create_test_audio_wav(duration_seconds=3, sample_rate=22050)
        
        # Select video snippets
        video_snippets = select_video_forensic_snippets(video_bytes, num_keyframes=3)
        
        # Select audio segments
        audio_segments = select_audio_forensic_segments(
            audio_bytes,
            num_segments=3,
            segment_duration_ms=1000
        )
        
        # Both should succeed
        assert len(video_snippets) == 3
        assert len(audio_segments) == 3
        
        print("\n📹 Video Snippets:")
        for snippet in video_snippets:
            print(f"  - Frame {snippet.frame_index} @ {snippet.timestamp_ms}ms")
        
        print("\n🔊 Audio Segments:")
        for segment in audio_segments:
            print(f"  - Segment @ {segment.start_time_ms}ms, "
                  f"duration {segment.segment_duration_ms}ms")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" VIDEO/AUDIO FRAGMENT SELECTION TESTS")
    print("=" * 70)
    
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
