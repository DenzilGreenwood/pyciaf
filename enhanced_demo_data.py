"""
Enhanced demo with better explainability for small datasets.

This version creates more samples and uses explainability methods
that work better with smaller datasets.
"""

def create_enhanced_demo_dataset():
    """Create a larger, more explainability-friendly dataset."""
    
    # Create 50 samples instead of 16 for better explainability
    enhanced_text_samples = []
    enhanced_numerical_samples = []
    
    # Text samples with more variety
    text_templates = [
        "The product quality is {}",
        "Customer service was {}",
        "The features are {} and useful",
        "Performance is {} than expected",
        "The interface is {} to use",
    ]
    
    quality_words = [
        "excellent", "outstanding", "superb", "amazing", "fantastic",
        "good", "decent", "acceptable", "satisfactory", "adequate",
        "poor", "terrible", "awful", "disappointing", "unsatisfactory"
    ]
    
    for i in range(50):
        template = text_templates[i % len(text_templates)]
        word = quality_words[i % len(quality_words)]
        text = template.format(word)
        target = 1 if i % 2 == 0 else 0
        
        enhanced_text_samples.append({
            "text": text,
            "target": target
        })
        
        # Corresponding numerical data
        enhanced_numerical_samples.append({
            "features": [
                float(i % 10),
                float((i * 2) % 8),
                float((i * 3) % 6),
                float((i * 4) % 9),
                float((i * 5) % 7)
            ],
            "target": float(target)
        })
    
    return enhanced_text_samples, enhanced_numerical_samples

print("Enhanced dataset creator ready!")
print("This would create 50 samples instead of 16 for better explainability.")