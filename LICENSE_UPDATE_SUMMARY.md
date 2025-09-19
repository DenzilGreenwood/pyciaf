# License Update Summary

## Changes Made

### 1. Updated LICENSE File
- **File:** `LICENSE`
- **Changed from:** MIT License
- **Changed to:** Proprietary License – CognitiveInsight.AI
- **Key terms:**
  - Non-commercial research and evaluation use only
  - No redistribution or commercial use without written consent
  - Contact: legal@cognitiveinsight.ai for commercial licensing

### 2. Updated Package Configuration
- **File:** `pyproject.toml` (root)
  - Changed `license = "MIT"` to `license-files = ["LICENSE"]`
  - Removed deprecated `License :: OSI Approved :: MIT License` classifier
  - Used modern setuptools license configuration

- **File:** `ciaf/pyproject.toml` (nested)
  - Removed deprecated `License :: OSI Approved :: MIT License` classifier

### 3. Updated Documentation
- **File:** `README.md`
  - Updated license badge from MIT to Proprietary
  - Updated license section with new terms and contact information
  - Added key restrictions section

### 4. Updated Changelog
- **File:** `ciaf/CHANGELOG.md`
  - Updated license change description
  - Changed commercial contact from old email to legal@cognitiveinsight.ai

## Verification
✅ Package builds successfully without warnings
✅ LICENSE file properly included in wheel package
✅ No deprecated license configuration warnings
✅ All license references updated consistently

## Key License Terms
- **Permitted:** Non-commercial research and evaluation use only
- **Prohibited:** Commercial use, redistribution, reverse engineering
- **Contact:** legal@cognitiveinsight.ai for commercial licensing

## Files Updated
1. `LICENSE` - New proprietary license text
2. `pyproject.toml` - Modern license configuration
3. `ciaf/pyproject.toml` - Removed deprecated classifier
4. `README.md` - Updated license information and badge
5. `ciaf/CHANGELOG.md` - Updated license change documentation

All old build artifacts and egg-info directories have been cleaned and regenerated with the new license information.