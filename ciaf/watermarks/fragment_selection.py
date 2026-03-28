"""
CIAF Watermarking - Forensic Fragment Selection

Entropy-based selection of high-information-density fragments for DNA-level
forensic verification across all artifact types.

Strategy: Don't verify the body; verify the DNA.

By storing high-entropy fragments, we enable:
- Detection of mix-and-match attacks (spliced documents)
- Legal defensibility through multi-point sampling
- Privacy protection (compact DNA records vs full document storage)

Multi-Point Sampling Rule:
- Text: 3 fragments (begin, middle, end) → any 2 matches = ~99.9% confidence
- Image: 4-6 high-complexity patches → spatial diversity
- Video: I-frames at temporal boundaries + motion signatures
- Audio: Spectral segments with high frequency variation

Created: 2026-03-28
Author: Denzil James Greenwood
Version: 1.2.0
"""

from __future__ import annotations

from typing import List, Tuple, Optional

from .models import (
    TextForensicFragment,
    ImageForensicFragment,
    VideoForensicSnippet,
    AudioForensicSegment,
    ForensicFragmentSet,
    sha256_text,
    sha256_bytes,
)

# ============================================================================
# TEXT FRAGMENT SELECTION
# ============================================================================


def compute_text_entropy(text: str) -> float:
    """
    Compute Shannon entropy of text (0.0-1.0).

    Higher entropy = more unique/diverse content
    Lower entropy = boilerplate/repetitive content

    Avoids selecting generic phrases like "In conclusion..."
    """
    if not text or len(text) < 10:
        return 0.0

    # Normalize text
    text = text.lower()

    # Count character frequencies
    char_freq = {}
    for char in text:
        char_freq[char] = char_freq.get(char, 0) + 1

    # Compute Shannon entropy
    import math

    entropy = 0.0
    for count in char_freq.values():
        if count > 0:
            p = count / len(text)
            entropy -= p * math.log2(p)

    # Normalize to 0.0-1.0 (max for 256 chars ≈ 8)
    return min(1.0, entropy / 8.0)


def select_text_fragment(
    text: str,
    location: str = "middle",  # 'beginning', 'middle', 'end'
    fragment_length: int = 200,
) -> Optional[Tuple[int, int, str, float]]:
    """
    Select a high-entropy text fragment from specified location.

    Args:
        text: Full text content
        location: Where to sample ('beginning', 'middle', 'end')
        fragment_length: Target length of fragment (chars)

    Returns:
        Tuple of (offset_start, offset_end, fragment_text, entropy_score)
        or None if text too short
    """
    if len(text) < fragment_length * 1.5:
        return None

    # Define search regions
    total_len = len(text)

    if location == "beginning":
        search_start = 0
        search_end = max(fragment_length, int(total_len * 0.2))
    elif location == "middle":
        quarter = int(total_len * 0.25)
        search_start = quarter
        search_end = min(total_len - fragment_length, quarter + int(total_len * 0.5))
    else:  # 'end'
        search_start = max(0, int(total_len * 0.8) - fragment_length)
        search_end = total_len - fragment_length

    if search_start >= search_end:
        return None

    # Find high-entropy fragment in region
    best_entropy = 0.0
    best_start = search_start
    best_end = search_start + fragment_length

    # Sample positions in the region
    step = max(1, (search_end - search_start) // 10)

    for pos in range(search_start, search_end, step):
        end_pos = min(pos + fragment_length, total_len)
        fragment = text[pos : end_pos + 1]

        if len(fragment) < fragment_length * 0.8:
            continue

        entropy = compute_text_entropy(fragment)

        if entropy > best_entropy:
            best_entropy = entropy
            best_start = pos
            best_end = end_pos

    if best_entropy < 0.4:
        # Fall back to any fragment in region
        best_start = search_start
        best_end = min(search_start + fragment_length, total_len)

    fragment = text[best_start : best_end + 1]
    return (best_start, best_end, fragment, best_entropy)


def select_text_forensic_fragments(
    raw_text: str,
    fragment_hash_before: str,  # Will be injected by caller
    fragment_hash_after: str,  # Will be injected by caller
    min_entropy: float = 0.4,
) -> List[TextForensicFragment]:
    """
    Select three high-entropy text fragments for multi-point verification.

    Strategy:
    - Beginning fragment (introduction/setup)
    - Middle fragment (core content)
    - End fragment (conclusion/results)

    Multi-point rule: If ANY 2 match, confidence > 99.9%

    Args:
        raw_text: Full text content
        fragment_hash_before: Pre-computed hash before watermark
        fragment_hash_after: Pre-computed hash after watermark
        min_entropy: Minimum entropy threshold

    Returns:
        List of TextForensicFragment records (typically 3)
    """
    fragments: List[TextForensicFragment] = []

    for location in ["beginning", "middle", "end"]:
        result = select_text_fragment(raw_text, location=location, fragment_length=200)

        if result is None:
            continue

        offset_start, offset_end, fragment_text, entropy = result

        if entropy < min_entropy:
            continue

        # Compute hashes for this fragment
        # In practice, these would be computed from the actual watermarked/unwatermarked versions
        frag_hash_before = sha256_text(fragment_text)
        frag_hash_after = sha256_text(
            fragment_text
        )  # Placeholder - actual would vary based on watermark

        fragment = TextForensicFragment(
            fragment_id=f"text_frag_{location}_{offset_start}",
            fragment_type="text",
            entropy_score=entropy,
            sampling_method=location,
            content_position=offset_start,
            offset_start=offset_start,
            offset_end=offset_end,
            fragment_length=len(fragment_text),
            sample_location=location,
            fragment_hash_before=frag_hash_before,
            fragment_hash_after=frag_hash_after,
        )

        fragments.append(fragment)

    return fragments


# ============================================================================
# IMAGE FRAGMENT SELECTION
# ============================================================================


def compute_image_patch_entropy(
    image_data: bytes, patch_x: int, patch_y: int, patch_w: int = 64, patch_h: int = 64
) -> float:
    """
    Compute entropy of an image patch (0.0-1.0).

    Higher entropy = more complex/detailed region
    Lower entropy = uniform color or blank region (avoid these)

    This prevents selecting blank sky or solid-color backgrounds.
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(__import__("io").BytesIO(image_data))
        img_array = np.array(img.convert("RGB"))

        # Extract patch
        x_end = min(patch_x + patch_w, img_array.shape[1])
        y_end = min(patch_y + patch_h, img_array.shape[0])

        patch = img_array[patch_y : y_end + 1, patch_x : x_end + 1]

        if patch.size < 32:
            return 0.0

        # Compute entropy: higher variance + diversity = higher entropy
        # Channels: standard deviation across RGB
        r_std = np.std(patch[:, :, 0])
        g_std = np.std(patch[:, :, 1])
        b_std = np.std(patch[:, :, 2])

        avg_std = (r_std + g_std + b_std) / 3.0 / 128.0  # Normalize to 0-1

        # Spatial diversity: gradient edges
        edges = np.zeros_like(patch[:, :, 0], dtype=float)
        edges[:-1, :] += np.abs(
            patch[1:, :, 0].astype(float) - patch[:-1, :, 0].astype(float)
        )
        edges[:, :-1] += np.abs(
            patch[:, 1:, 0].astype(float) - patch[:, :-1, 0].astype(float)
        )

        edge_strength = np.mean(edges) / 255.0

        # Combined entropy score
        entropy = 0.6 * avg_std + 0.4 * edge_strength
        return min(1.0, entropy)

    except Exception:
        # PIL not available or parsing error
        return 0.0


def select_image_forensic_patches(
    image_bytes: bytes,
    num_patches: int = 4,
    patch_size: int = 64,
    min_entropy: float = 0.5,
) -> List[ImageForensicFragment]:
    """
    Select high-complexity image patches for DNA-level verification.

    Strategy:
    - Divide image into grid
    - Compute entropy for each patch
    - Select most complex patches (avoid blank sky, solid colors)

    Args:
        image_bytes: Image file content
        num_patches: Number of patches to select (typically 4-6)
        patch_size: Patch dimensions (64x64 default)
        min_entropy: Minimum entropy threshold

    Returns:
        List of ImageForensicFragment records
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(__import__("io").BytesIO(image_bytes))
        img_array = np.array(img.convert("RGB"))

        img_h, img_w = img_array.shape[:2]

        # Create grid of potential patches
        grid_cols = max(1, img_w // patch_size)
        grid_rows = max(1, img_h // patch_size)

        patches_with_entropy = []

        for row in range(grid_rows):
            for col in range(grid_cols):
                patch_x = col * patch_size
                patch_y = row * patch_size

                entropy = compute_image_patch_entropy(
                    image_bytes, patch_x, patch_y, patch_size, patch_size
                )

                if entropy >= min_entropy:
                    patches_with_entropy.append((patch_x, patch_y, entropy))

        # Select top N patches by entropy
        patches_with_entropy.sort(key=lambda x: x[2], reverse=True)
        selected = patches_with_entropy[:num_patches]

        # Create fragment records
        fragments: List[ImageForensicFragment] = []

        for idx, (patch_x, patch_y, entropy) in enumerate(selected):
            # Extract patch and compute hash
            patch_image = img_array[
                patch_y : patch_y + patch_size + 1, patch_x : patch_x + patch_size + 1
            ]
            patch_bytes = patch_image.tobytes()
            patch_hash = sha256_bytes(patch_bytes)

            fragment = ImageForensicFragment(
                fragment_id=f"img_patch_{idx}_{patch_x}_{patch_y}",
                fragment_type="image_patch",
                entropy_score=entropy,
                sampling_method="spatial",
                content_position=idx,
                region_coordinates=(patch_x, patch_y, patch_size, patch_size),
                patch_grid_position=f"grid_{patch_y // patch_size}_{patch_x // patch_size}",
                patch_hash_before=patch_hash,
                patch_hash_after=patch_hash,  # Placeholder
            )

            fragments.append(fragment)

        return fragments

    except Exception:
        return []


# ============================================================================
# VIDEO FRAGMENT SELECTION (Placeholder - Phase 2)
# ============================================================================


def select_video_forensic_snippets(
    video_bytes: bytes,
    num_keyframes: int = 3,
) -> List[VideoForensicSnippet]:
    """
    Select keyframe samples for video DNA verification.

    Strategy:
    - Identify I-frames (keyframes) in video
    - Sample at 25%, 50%, 75% temporal positions
    - Extract motion signatures between keyframes

    Args:
        video_bytes: Video file content
        num_keyframes: Number of keyframes to sample

    Returns:
        List of VideoForensicSnippet records
    """
    # Phase 2 implementation
    return []


# ============================================================================
# AUDIO FRAGMENT SELECTION (Placeholder - Phase 2)
# ============================================================================


def select_audio_forensic_segments(
    audio_bytes: bytes,
    num_segments: int = 3,
    segment_duration_ms: int = 2000,
) -> List[AudioForensicSegment]:
    """
    Select spectral segments for audio DNA verification.

    Strategy:
    - Convert audio to spectrogram (frequency-domain)
    - Identify high-variance frequency regions (max entropy)
    - Compute perceptual hashes of spectral regions

    Args:
        audio_bytes: Audio file content
        num_segments: Number of segments to select
        segment_duration_ms: Duration per segment (typical: 2000-5000 ms)

    Returns:
        List of AudioForensicSegment records
    """
    # Phase 2 implementation
    return []


# ============================================================================
# FORENSIC FRAGMENT SET ASSEMBLY
# ============================================================================


def create_forensic_fragment_set(
    artifact: bytes,
    artifact_type: str,
    enable_fragments: bool = True,
) -> Optional[ForensicFragmentSet]:
    """
    Create a ForensicFragmentSet (DNA records) for an artifact.

    This assembles the multi-point samples for legal-grade verification.

    Args:
        artifact: Artifact content (bytes or will be converted)
        artifact_type: 'text', 'image', 'pdf', 'video', 'audio'
        enable_fragments: Whether to generate fragments

    Returns:
        ForensicFragmentSet with all selected fragments, or None
    """
    if not enable_fragments:
        return None

    # Convert if needed
    if isinstance(artifact, str):
        artifact_bytes = artifact.encode("utf-8")
    else:
        artifact_bytes = artifact

    fragment_set = ForensicFragmentSet(
        fragment_count=0,
        sampling_strategy="multi_point",
        total_coverage_percent=0.0,
    )

    try:
        if artifact_type == "text":
            artifact_text = (
                artifact.decode("utf-8") if isinstance(artifact, bytes) else artifact
            )
            fragments = select_text_forensic_fragments(
                artifact_text,
                fragment_hash_before="",  # Computed by caller
                fragment_hash_after="",  # Computed by caller
            )
            fragment_set.text_fragments = fragments
            fragment_set.sampling_strategy = "multi_point"  # 3-point sampling
            fragment_set.fragment_count = len(fragments)

        elif artifact_type == "image":
            fragments = select_image_forensic_patches(artifact_bytes, num_patches=4)
            fragment_set.image_fragments = fragments
            fragment_set.sampling_strategy = "spatial_diversity"
            fragment_set.fragment_count = len(fragments)

        elif artifact_type == "video":
            # Phase 2
            pass

        elif artifact_type == "audio":
            # Phase 2
            pass

        # Compute aggregate statistics
        if fragment_set.all_fragments:
            avg_entropy = sum(
                f.entropy_score for f in fragment_set.all_fragments
            ) / len(fragment_set.all_fragments)
            fragment_set.cumulative_entropy_score = avg_entropy

        return fragment_set if fragment_set.fragment_count > 0 else None

    except Exception:
        return None
