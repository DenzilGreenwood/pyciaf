"""
Video/Audio Watermarking Examples  
==================================

Demonstrates forensic fragment selection for video and audio files:
- Video keyframe selection and verification
- Audio spectral segment selection  
- Temporal forensic fingerprinting
- Motion signature computation

Created: 2026-04-04
Author: Denzil James Greenwood
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import os
from io import BytesIO

from ciaf.watermarks.fragment_selection import (
    select_video_forensic_snippets,
    select_audio_forensic_segments,
    CV2_AVAILABLE,
    WAVE_AVAILABLE,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_demo_video(width=640, height=480, duration_seconds=5, fps=30):
    """Create a demo video with visual patterns."""
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python required: pip install opencv-python")
    
    import cv2
    import numpy as np
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_frames = duration_seconds * fps
        
        for i in range(total_frames):
            # Create dynamic frame
            progress = i / total_frames
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Moving gradient
            for y in range(height):
                color = int(255 * (y / height + progress) % 1.0)
                frame[y, :] = (color, 255 - color, 128)
            
            # Moving shapes
            center_x = int(width * (0.2 + 0.6 * np.sin(2 * np.pi * progress)))
            center_y = height // 2
            cv2.circle(frame, (center_x, center_y), 50, (255, 255, 0), -1)
            
            # Text overlay
            cv2.putText(frame, f"Frame {i}/{total_frames}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        
        with open(output_path, 'rb') as f:
            return f.read()
    
    finally:
        try:
            os.unlink(output_path)
        except:
            pass


def create_demo_audio(duration_seconds=10, sample_rate=44100):
    """Create a demo audio with varying tones."""
    if not WAVE_AVAILABLE:
        raise ImportError("wave module required (standard library)")
    
    import wave
    import struct
    import math
    
    num_samples = int(duration_seconds * sample_rate)
    samples = []
    
    for i in range(num_samples):
        t = i / sample_rate
        # Multi-frequency signal (musical chord)
        freq1 = 440.0  # A4
        freq2 = 554.37  # C#5
        freq3 = 659.25  # E5
        
        # Varying amplitude envelope
        envelope = 0.3 + 0.3 * math.sin(2 * math.pi * 0.5 * t)
        
        value = int(16000 * envelope * (
            math.sin(2 * math.pi * freq1 * t) +
            math.sin(2 * math.pi * freq2 * t) +
            math.sin(2 * math.pi * freq3 * t)
        ))
        
        samples.append(max(-32767, min(32767, value)))
    
    audio_stream = BytesIO()
    
    with wave.open(audio_stream, 'wb') as wav_file:
        wav_file.setnchannels(2)  # Stereo
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        for sample in samples:
            # Stereo: write same sample to both channels
            wav_file.writeframes(struct.pack('<hh', sample, sample))
    
    return audio_stream.getvalue()


# ============================================================================
# EXAMPLES
# ============================================================================


def example_1_video_keyframe_selection():
    """
    Example 1: Video Keyframe Selection
    
    Extract forensic keyframes from a video for verification.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Video Keyframe Selection")
    print("=" * 70)
    
    if not CV2_AVAILABLE:
        print("⚠️  opencv-python not installed. Install with: pip install opencv-python")
        return
    
    print("\n📹 Creating demo video (5 seconds, 30 fps)...")
    video_bytes = create_demo_video(width=640, height=480, duration_seconds=5, fps=30)
    
    print(f"✓ Created video: {len(video_bytes):,} bytes")
    
    # Select keyframes
    print("\n🔍 Selecting forensic keyframes...")
    snippets = select_video_forensic_snippets(
        video_bytes,
        num_keyframes=5
    )
    
    print(f"✓ Selected {len(snippets)} keyframes\n")
    
    print("📊 Keyframe Details:")
    for i, snippet in enumerate(snippets, 1):
        print(f"\n  Keyframe {i}:")
        print(f"    Frame Index:  {snippet.frame_index}")
        print(f"    Timestamp:    {snippet.timestamp_ms}ms ({snippet.timestamp_ms / 1000:.2f}s)")
        print(f"    Entropy:      {snippet.entropy_score:.4f}")
        print(f"    Patches:      {len(snippet.frame_patch_hashes)} hashes")
        
        if snippet.temporal_motion_hash:
            print(f"    Motion:       confidence {snippet.motion_confidence:.2%}")
            print(f"    Motion Hash:  {snippet.temporal_motion_hash[:16]}...")
        else:
            print(f"    Motion:       N/A (first frame)")
    
    print("\n✅ Video keyframes selected successfully")
    print("\n💡 Use Case:")
    print("  • Verify video authenticity by matching keyframes")
    print("  • Detect frame splicing or content insertion")
    print("  • Track motion signatures for tampering detection")


def example_2_audio_segment_selection():
    """
    Example 2: Audio Segment Selection
    
    Extract forensic audio segments for verification.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Audio Segment Selection")
    print("=" * 70)
    
    if not WAVE_AVAILABLE:
        print("⚠️  wave module not available")
        return
    
    print("\n🔊 Creating demo audio (10 seconds, 44.1kHz)...")
    audio_bytes = create_demo_audio(duration_seconds=10, sample_rate=44100)
    
    print(f"✓ Created audio: {len(audio_bytes):,} bytes")
    
    # Select segments
    print("\n🔍 Selecting forensic audio segments...")
    segments = select_audio_forensic_segments(
        audio_bytes,
        num_segments=5,
        segment_duration_ms=2000
    )
    
    print(f"✓ Selected {len(segments)} segments\n")
    
    print("📊 Segment Details:")
    for i, segment in enumerate(segments, 1):
        print(f"\n  Segment {i}:")
        print(f"    Start Time:       {segment.start_time_ms}ms ({segment.start_time_ms / 1000:.2f}s)")
        print(f"    Duration:         {segment.segment_duration_ms}ms")
        print(f"    Entropy:          {segment.entropy_score:.4f}")
        print(f"    Freq. Centroid:   {segment.frequency_centroid:.1f} Hz")
        print(f"    Spectral Flatness: {segment.spectral_flatness:.4f}")
        print(f"    Spectrogram Hash: {segment.spectrogram_hash[:16]}...")
    
    print("\n✅ Audio segments selected successfully")
    print("\n💡 Use Case:")
    print("  • Verify audio authenticity by matching spectral segments")
    print("  • Detect audio splicing or voice cloning")
    print("  • Track spectral signatures for tampering detection")


def example_3_video_verification_workflow():
    """
    Example 3: Video Verification Workflow
    
    Complete workflow: select keyframes, simulate verification.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Video Verification Workflow")
    print("=" * 70)
    
    if not CV2_AVAILABLE:
        print("⚠️  opencv-python not installed")
        return
    
    print("\n📹 Step 1: Create original video")
    original_video = create_demo_video(duration_seconds=3, fps=20)
    print(f"✓ Original: {len(original_video):,} bytes")
    
    print("\n🔍 Step 2: Extract forensic keyframes")
    keyframes = select_video_forensic_snippets(original_video, num_keyframes=4)
    print(f"✓ Extracted {len(keyframes)} keyframes")
    
    # Store keyframe hashes
    frame_hashes = {kf.frame_index: kf.frame_patch_hashes for kf in keyframes}
    print(f"✓ Stored {sum(len(h) for h in frame_hashes.values())} patch hashes")
    
    print("\n🎭 Step 3: Simulate verification scenarios")
    
    # Scenario A: Verify against original (should match)
    print("\n  Scenario A: Verify original video")
    verify_keyframes = select_video_forensic_snippets(original_video, num_keyframes=4)
    
    matches = 0
    for kf in verify_keyframes:
        if kf.frame_index in frame_hashes:
            # Check if any patch hash matches
            stored_patches = set(frame_hashes[kf.frame_index])
            current_patches = set(kf.frame_patch_hashes)
            
            if stored_patches & current_patches:  # Intersection
                matches += 1
    
    confidence = matches / len(keyframes) if keyframes else 0
    print(f"    Matched: {matches}/{len(keyframes)} keyframes")
    print(f"    Confidence: {confidence:.1%}")
    
    if confidence >= 0.75:
        print(f"    ✅ AUTHENTIC: High confidence match")
    else:
        print(f"    ⚠️  SUSPICIOUS: Low match rate")
    
    print("\n✅ Verification workflow demonstrated")


def example_4_audio_verification_workflow():
    """
    Example 4: Audio Verification Workflow
    
    Complete workflow: select segments, simulate verification.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Audio Verification Workflow")
    print("=" * 70)
    
    if not WAVE_AVAILABLE:
        print("⚠️  wave module not available")
        return
    
    print("\n🔊 Step 1: Create original audio")
    original_audio = create_demo_audio(duration_seconds=6, sample_rate=22050)
    print(f"✓ Original: {len(original_audio):,} bytes")
    
    print("\n🔍 Step 2: Extract forensic segments")
    segments = select_audio_forensic_segments(
        original_audio,
        num_segments=4,
        segment_duration_ms=1000
    )
    print(f"✓ Extracted {len(segments)} segments")
    
    # Store segment hashes
    segment_hashes = {seg.start_time_ms: seg.spectrogram_hash for seg in segments}
    print(f"✓ Stored {len(segment_hashes)} segment hashes")
    
    print("\n🎭 Step 3: Simulate verification scenarios")
    
    # Scenario A: Verify against original (should match)
    print("\n  Scenario A: Verify original audio")
    verify_segments = select_audio_forensic_segments(
        original_audio,
        num_segments=4,
        segment_duration_ms=1000
    )
    
    matches = 0
    for seg in verify_segments:
        if seg.start_time_ms in segment_hashes:
            if seg.spectrogram_hash == segment_hashes[seg.start_time_ms]:
                matches += 1
    
    confidence = matches / len(segments) if segments else 0
    print(f"    Matched: {matches}/{len(segments)} segments")
    print(f"    Confidence: {confidence:.1%}")
    
    if confidence >= 0.75:
        print(f"    ✅ AUTHENTIC: High confidence match")
    else:
        print(f"    ⚠️  SUSPICIOUS: Low match rate")
    
    print("\n✅ Verification workflow demonstrated")


def example_5_combined_multimedia():
    """
    Example 5: Combined Multimedia Forensics
    
    Extract forensic fragments from both video and audio.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Combined Multimedia Forensics")
    print("=" * 70)
    
    if not CV2_AVAILABLE or not WAVE_AVAILABLE:
        print("⚠️  Missing dependencies (opencv-python or wave)")
        return
    
    print("\n🎬 Creating multimedia content...")
    
    # Create video
    video_bytes = create_demo_video(duration_seconds=4, fps=15)
    print(f"✓ Video: {len(video_bytes):,} bytes")
    
    # Create audio
    audio_bytes = create_demo_audio(duration_seconds=4, sample_rate=22050)
    print(f"✓ Audio: {len(audio_bytes):,} bytes")
    
    print("\n🔍 Extracting forensic DNA...")
    
    # Extract video keyframes
    video_keyframes = select_video_forensic_snippets(video_bytes, num_keyframes=3)
    print(f"✓ Video: {len(video_keyframes)} keyframes")
    
    # Extract audio segments
    audio_segments = select_audio_forensic_segments(
        audio_bytes,
        num_segments=3,
        segment_duration_ms=1000
    )
    print(f"✓ Audio: {len(audio_segments)} segments")
    
    print("\n📊 Forensic DNA Summary:")
    print(f"\n  Video Keyframes:")
    for kf in video_keyframes:
        print(f"    • Frame {kf.frame_index} @ {kf.timestamp_ms}ms: "
              f"{len(kf.frame_patch_hashes)} patches, "
              f"entropy {kf.entropy_score:.3f}")
    
    print(f"\n  Audio Segments:")
    for seg in audio_segments:
        print(f"    • Segment @ {seg.start_time_ms}ms: "
              f"centroid {seg.frequency_centroid:.0f}Hz, "
              f"entropy {seg.entropy_score:.3f}")
    
    # Calculate total forensic anchors
    total_video_anchors = sum(len(kf.frame_patch_hashes) for kf in video_keyframes)
    total_audio_anchors = len(audio_segments)
    total_anchors = total_video_anchors + total_audio_anchors
    
    print(f"\n📍 Total Forensic Anchors: {total_anchors}")
    print(f"   Video patches: {total_video_anchors}")
    print(f"   Audio segments: {total_audio_anchors}")
    
    print("\n✅ Multimedia forensic DNA extracted")
    print("\n💡 Use Case:")
    print("  • Comprehensive multimedia authentication")
    print("  • Detect audio/video de-synchronization")
    print("  • Verify both visual and audio integrity")
    print("  • Track multi-modal tampering attempts")


def example_6_forensic_storage():
    """
    Example 6: Forensic Fragment Storage
    
    Show how to serialize and store forensic fragments.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Forensic Fragment Storage")
    print("=" * 70)
    
    if not CV2_AVAILABLE or not WAVE_AVAILABLE:
        print("⚠️  Missing dependencies")
        return
    
    print("\n📦 Creating forensic evidence package...")
    
    # Create media
    video_bytes = create_demo_video(duration_seconds=2, fps=10)
    audio_bytes = create_demo_audio(duration_seconds=2, sample_rate=22050)
    
    # Extract fragments
    video_keyframes = select_video_forensic_snippets(video_bytes, num_keyframes=2)
    audio_segments = select_audio_forensic_segments(audio_bytes, num_segments=2)
    
    print(f"✓ Extracted {len(video_keyframes)} video keyframes")
    print(f"✓ Extracted {len(audio_segments)} audio segments")
    
    # Serialize to dictionaries
    print("\n💾 Serializing forensic data...")
    
    forensic_package = {
        "artifact_id": "multimedia_demo_001",
        "artifact_type": "video_with_audio",
        "video_keyframes": [kf.to_dict() for kf in video_keyframes],
        "audio_segments": [seg.to_dict() for seg in audio_segments],
        "metadata": {
            "video_size_bytes": len(video_bytes),
            "audio_size_bytes": len(audio_bytes),
            "extraction_timestamp": "2026-04-04T12:00:00Z",
        }
    }
    
    print(f"✓ Serialized forensic package")
    print(f"\n📋 Package Contents:")
    print(f"   Artifact ID: {forensic_package['artifact_id']}")
    print(f"   Type: {forensic_package['artifact_type']}")
    print(f"   Video keyframes: {len(forensic_package['video_keyframes'])}")
    print(f"   Audio segments: {len(forensic_package['audio_segments'])}")
    
    # Show sample data structure
    print(f"\n📄 Sample Video Keyframe:")
    sample_kf = forensic_package['video_keyframes'][0]
    print(f"   Fragment ID: {sample_kf['fragment_id']}")
    print(f"   Frame index: {sample_kf['frame_index']}")
    print(f"   Timestamp: {sample_kf['timestamp_ms']}ms")
    print(f"   Entropy: {sample_kf['entropy_score']:.4f}")
    print(f"   Patch hashes: {len(sample_kf['frame_patch_hashes'])}")
    
    print(f"\n📄 Sample Audio Segment:")
    sample_seg = forensic_package['audio_segments'][0]
    print(f"   Fragment ID: {sample_seg['fragment_id']}")
    print(f"   Start time: {sample_seg['start_time_ms']}ms")
    print(f"   Duration: {sample_seg['segment_duration_ms']}ms")
    print(f"   Freq. centroid: {sample_seg['frequency_centroid']:.1f}Hz")
    
    print("\n✅ Forensic package ready for storage")
    print("\n💾 Storage Options:")
    print("  • JSON file: Store package as JSON")
    print("  • Database: Store in CIAF Vault")
    print("  • Cloud: Upload to Azure Blob / AWS S3")
    print("  • Blockchain: Hash package and store on chain")


def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print(" VIDEO/AUDIO WATERMARKING EXAMPLES")
    print("=" * 70)
    print("\nForensic fragment selection for multimedia content")
    
    example_1_video_keyframe_selection()
    example_2_audio_segment_selection()
    example_3_video_verification_workflow()
    example_4_audio_verification_workflow()
    example_5_combined_multimedia()
    example_6_forensic_storage()
    
    print("\n" + "=" * 70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Video: Extract keyframes at temporal positions")
    print("  2. Audio: Extract spectral segments with frequency features")
    print("  3. Both: Use hashes for exact matching verification")
    print("  4. Motion: Track frame differences for tampering detection")
    print("  5. Storage: Serialize fragments for vault/cloud storage")
    print()


if __name__ == "__main__":
    main()
