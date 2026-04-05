# CIAF Installation Guide

This guide explains how to install CIAF and its dependencies based on your needs.

## Table of Contents
- [Quick Start](#quick-start)
- [Installation Options](#installation-options)
- [External Dependencies](#external-dependencies)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Installation (Core Features)

```bash
pip install -e .
```

This installs:
- Core cryptography and hashing
- Basic data validation
- Image and video watermarking
- Cloud storage support (AWS S3, Azure Blob)

### Full Installation (All Features)

```bash
pip install -e ".[all]"
```

This installs everything including ML, explainability, visualization, and advanced watermarking.

---

## Installation Options

### Core Package Only
```bash
pip install -e .
```

### With Machine Learning
```bash
pip install -e ".[ml]"
```

### With Explainability (SHAP, LIME)
```bash
pip install -e ".[explainability]"
```

### With Visualization
```bash
pip install -e ".[viz]"
```

### With Advanced Audio Analysis
```bash
pip install -e ".[audio-advanced]"
```

### With Multi-Format Support (requires ffmpeg)
```bash
pip install -e ".[multiformat]"
```

### With Web Interface (CIAF Vault)
```bash
pip install -e ".[web]"
```

### Full ML Stack
```bash
pip install -e ".[full]"
```

### All Advanced Watermarking Features
```bash
pip install -e ".[watermarking-full]"
```

### Development Environment
```bash
pip install -e ".[dev]"
```

### Combine Multiple Options
```bash
pip install -e ".[ml,viz,web,dev]"
```

---

## External Dependencies

Some features require external binaries to be installed separately:

### FFmpeg (for multi-format audio/video support)

**Windows:**
1. Download from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to PATH
4. Verify: `ffmpeg -version`

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### Chromaprint (for audio fingerprinting)

**Windows:**
1. Download from https://acoustid.org/chromaprint
2. Extract and add to PATH
3. Verify: `fpcalc -version`

**Linux:**
```bash
sudo apt install libchromaprint-dev
```

**macOS:**
```bash
brew install chromaprint
```

---

## Using requirements.txt Files

### Core Dependencies
```bash
pip install -r requirements.txt
```

### Optional Dependencies
```bash
pip install -r requirements-optional.txt
```

### Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### Install All
```bash
pip install -r requirements.txt -r requirements-optional.txt -r requirements-dev.txt
```

---

## Verification

### Verify Installation
```bash
python -c "import ciaf; print(f'CIAF version: {ciaf.__version__}')"
```

### Check Available Features
```python
from ciaf.watermarks.advanced_features import print_feature_status

print_feature_status()
```

Output:
```
CIAF Watermarking - Advanced Features Status
==================================================

📊 Spectral Analysis:
  Librosa (MFCCs/Chroma): ✓ Available

🎬 Multi-Format Support:
  FFmpeg conversion: ✓ Available

🔍 Perceptual Hashing:
  Video pHash: ✓ Available
  Audio Chromaprint: ✓ Available

☁️ Cloud Storage:
  AWS S3: ✓ Available
  Azure Blob: ✓ Available

==================================================
Overall: 6/6 features available (100%)
```

### Run Tests
```bash
# All tests
pytest

# Specific test suite
pytest tests/test_advanced_features.py -v

# With coverage
pytest --cov=ciaf tests/
```

### Run Examples
```bash
# Basic watermarking
python examples/example_video_audio_watermarking.py

# Advanced features
python examples/example_advanced_features.py
```

---

## Troubleshooting

### ImportError: No module named 'ciaf'

**Solution:** Install the package in editable mode:
```bash
pip install -e .
```

### librosa not found

**Solution:**
```bash
pip install librosa soundfile
```

### ffmpeg not found / FFmpeg conversion failed

**Cause:** ffmpeg binary not installed or not in PATH

**Solution:**
1. Install ffmpeg binary (see External Dependencies section)
2. Verify it's in PATH: `ffmpeg -version`
3. Restart terminal/IDE

### Chromaprint not found

**Cause:** chromaprint library not installed

**Solution:**
1. Install chromaprint (see External Dependencies section)
2. Install Python wrapper: `pip install pyacoustid`

### AWS S3 / Azure Blob errors

**Cause:** Missing credentials or invalid configuration

**Solution:**
```bash
# AWS
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"

# Azure
export AZURE_STORAGE_CONNECTION_STRING="your-connection-string"
```

### Windows file locking errors

**Cause:** Windows file handles not properly released

**Solution:** Already fixed in code. If you still see issues, try:
```python
import gc
gc.collect()  # Force garbage collection
```

### Pydantic validation warnings

**Status:** Fixed in v1.6.0

If you see warnings about model validators, update to latest version:
```bash
pip install -e . --upgrade
```

---

## Platform-Specific Notes

### Windows

- Use PowerShell or Command Prompt
- Some features may require Visual C++ redistributables
- File paths use backslashes: `C:\path\to\file`

### Linux

- Most external dependencies available via apt/yum
- May need build-essential for some packages
- File paths use forward slashes: `/path/to/file`

### macOS

- Use Homebrew for external dependencies
- May need Xcode Command Line Tools
- File paths use forward slashes: `/path/to/file`

---

## Environment Setup

### Virtual Environment (Recommended)

```bash
# Create venv
python -m venv venv_ciaf

# Activate
# Windows:
venv_ciaf\Scripts\activate
# Linux/macOS:
source venv_ciaf/bin/activate

# Install CIAF
pip install -e ".[all]"
```

### Conda Environment

```bash
# Create environment
conda create -n ciaf python=3.12

# Activate
conda activate ciaf

# Install CIAF
pip install -e ".[all]"
```

---

## Production Deployment

### Minimal Production Requirements
```bash
pip install -r requirements.txt
```

### With ML Features
```bash
pip install -e ".[ml,viz]"
```

### With Web Interface
```bash
pip install -e ".[web,ml,viz]"
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libchromaprint-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install CIAF
WORKDIR /app
COPY . /app
RUN pip install -e ".[all]"

CMD ["python", "your_app.py"]
```

Build and run:
```bash
docker build -t ciaf:latest .
docker run -it ciaf:latest
```

---

## GPU Support (for ML features)

### CUDA Support (NVIDIA GPUs)

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow with CUDA
pip install tensorflow[and-cuda]
```

### Apple Silicon (M1/M2)

```bash
# TensorFlow for macOS
pip install tensorflow-macos tensorflow-metal

# PyTorch already supports M1/M2
```

---

## Next Steps

After installation:

1. **Run tests:** `pytest tests/`
2. **Try examples:** `python examples/example_advanced_features.py`
3. **Read documentation:** Check `docs/` folder
4. **Configure cloud storage:** Set up AWS/Azure credentials
5. **Start building:** Import CIAF and start using it!

```python
from ciaf.watermarks import create_forensic_fragment_set
from ciaf.watermarks.advanced_features import (
    extract_advanced_audio_features,
    compute_video_phash,
    CloudFragmentStorage,
)

# Your code here...
```

---

## Support

- **Issues:** https://github.com/DenzilGreenwood/pyciaf/issues
- **Documentation:** https://github.com/DenzilGreenwood/pyciaf/blob/main/docs/
- **Examples:** `examples/` folder

---

**Last Updated:** April 4, 2026  
**Version:** 1.6.0
