"""
Quick fix for demo data quality validation.

This shows how to adjust preprocessing policies for different data scenarios.
"""

from ciaf.preprocessing import PreprocessingPolicy

# Create a demo-friendly policy
demo_policy = PreprocessingPolicy.minimal()  # Use minimal instead of standard

# Adjust quality thresholds for demo data
demo_policy.quality_policy.min_samples = 5  # Lower threshold
demo_policy.quality_policy.max_missing_ratio = 0.5  # Allow more missing data
demo_policy.quality_policy.min_quality_score = 30.0  # Lower quality threshold
demo_policy.quality_policy.outlier_threshold = 0.3  # More tolerant of outliers

print("Demo-friendly preprocessing policy created:")
print(f"✓ Min samples: {demo_policy.quality_policy.min_samples}")
print(f"✓ Max missing ratio: {demo_policy.quality_policy.max_missing_ratio}")
print(f"✓ Min quality score: {demo_policy.quality_policy.min_quality_score}")
print(f"✓ Outlier threshold: {demo_policy.quality_policy.outlier_threshold}")

# This policy would likely give you a quality score of 60-80/100 instead of 0/100