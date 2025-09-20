# CIAF Coding Standards

## Overview
This document outlines coding standards and best practices for the CIAF (Compliant and Interpretable AI Framework) project to ensure consistent, maintainable, and cross-platform compatible code.

## Character Encoding Standards

### ✅ DO: Use ASCII Characters Only
- Use only standard ASCII characters (0-127) in code
- Use standard punctuation and symbols: `!@#$%^&*()_+-=[]{}|;:'"<>?,./`
- Use standard letters: `a-z`, `A-Z`
- Use standard numbers: `0-9`

### ❌ DON'T: Use Unicode or Emoji Characters
- **NEVER** use emoji characters: 🔥🎯📊🧾📋🔮⚠️🚀💪🎉✅❌etc.
- **NEVER** use Unicode symbols: →•€£¥©®™
- **NEVER** use special Unicode characters: ±≤≥≠∞∂∇

### Acceptable Alternatives

Instead of Unicode/emoji characters, use clear ASCII alternatives:

| ❌ Don't Use | ✅ Use Instead |
|--------------|----------------|
| ✅ | `[SUCCESS]` or `[OK]` |
| ❌ | `[ERROR]` or `[FAIL]` |
| ⚠️ | `[WARNING]` or `[WARN]` |
| 🔍 | `[SEARCH]` or `[FIND]` |
| 📊 | `[DATA]` or `[STATS]` |
| 🚀 | `[START]` or `[LAUNCH]` |
| 🎯 | `[TARGET]` or `[GOAL]` |
| → | `->` or `-->` |
| • | `-` or `*` |

## Print Statement Guidelines

### ✅ DO: Use Clear ASCII Prefixes
```python
# Good examples
print("[SUCCESS] Model training completed")
print("[ERROR] Failed to load dataset")
print("[WARNING] Configuration file not found")
print("[INFO] Processing 1000 samples")
print("[DEBUG] Variable value: ", value)
```

### ❌ DON'T: Use Unicode Characters
```python
# Bad examples - NEVER do this
print("✅ Model training completed")
print("❌ Failed to load dataset") 
print("⚠️ Configuration file not found")
print("🔍 Processing 1000 samples")
```

## Logging Standards

### Use Standard Logging Levels
```python
import logging

logger = logging.getLogger(__name__)

# Good logging practices
logger.info("[INIT] Initializing CIAF model wrapper")
logger.warning("[CONFIG] Using default configuration")
logger.error("[VALIDATION] Input validation failed")
logger.debug("[PROCESS] Processing batch of %d items", batch_size)
```

## Documentation Standards

### Comments and Docstrings
- Use only ASCII characters in all comments
- Use clear, descriptive language
- Avoid Unicode bullet points in documentation

```python
# Good documentation
def train_model(data):
    """
    Train the CIAF model with provided data.
    
    Args:
        data: Training dataset in standard format
        
    Returns:
        Training snapshot with model state
        
    Raises:
        ValueError: If data format is invalid
    """
    pass
```

## File Naming Standards

### ✅ DO: Use ASCII File Names
- Use lowercase letters, numbers, and underscores
- Example: `model_wrapper.py`, `test_framework.py`

### ❌ DON'T: Use Special Characters
- Avoid Unicode characters in file names
- Avoid spaces (use underscores instead)

## Cross-Platform Compatibility

### Why ASCII Only?
1. **Windows Compatibility**: Windows terminals often have encoding issues with Unicode
2. **CI/CD Systems**: Many build systems expect ASCII-only output
3. **Log Processing**: Log aggregation tools work better with ASCII
4. **Terminal Support**: Not all terminals support Unicode properly
5. **Accessibility**: Screen readers work better with ASCII text

## Enforcement

### Pre-commit Checks
Consider adding a pre-commit hook to check for Unicode characters:

```bash
# Check for Unicode characters in Python files
find . -name "*.py" -exec grep -P "[^\x00-\x7F]" {} \;
```

### Code Review Guidelines
- Reviewers should check for Unicode characters
- Use find-and-replace to convert Unicode to ASCII
- Reject PRs that introduce Unicode characters

## Migration from Unicode

### If You Find Unicode Characters
1. Replace with ASCII alternatives from the table above
2. Test that functionality remains unchanged
3. Update any related documentation

### Example Migration
```python
# Before (with Unicode)
print("🚀 Starting model training...")
if success:
    print("✅ Training completed successfully!")
else:
    print("❌ Training failed!")

# After (ASCII only)
print("[START] Starting model training...")
if success:
    print("[SUCCESS] Training completed successfully!")
else:
    print("[ERROR] Training failed!")
```

## Tools and Utilities

### Unicode Detection Script
```python
import re
import os

def find_unicode_in_file(filepath):
    """Find Unicode characters in a Python file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find non-ASCII characters
    unicode_chars = re.findall(r'[^\x00-\x7F]', content)
    if unicode_chars:
        print(f"Unicode found in {filepath}: {set(unicode_chars)}")
```

## Conclusion

By following these standards, we ensure that CIAF code is:
- Cross-platform compatible
- Accessible to all developers
- Compatible with various terminal environments
- Easy to process by automated tools
- Professional and consistent

Remember: **ASCII characters only** - no exceptions!