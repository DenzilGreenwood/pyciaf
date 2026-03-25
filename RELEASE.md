# CIAF PyPI Publishing Guide

This document explains how to publish new versions of CIAF to PyPI using GitHub Actions with trusted publishing.

## Prerequisites

### 1. Configure PyPI Trusted Publishing

Before the first release, you need to configure trusted publishing on PyPI:

1. **Go to PyPI**: https://pypi.org/manage/account/publishing/
2. **Add a new trusted publisher** with:
   - **PyPI Project Name**: `ciaf`
   - **Owner**: `DenzilGreenwood` (your GitHub username)
   - **Repository name**: `pyciaf`
   - **Workflow name**: `publish-pypi.yml`
   - **Environment name**: `pypi`

This allows GitHub Actions to publish to PyPI without API tokens!

### 2. Create PyPI Environment in GitHub

1. Go to your repo: https://github.com/DenzilGreenwood/pyciaf/settings/environments
2. Click **New environment**
3. Name it: `pypi`
4. Add protection rules (optional):
   - Required reviewers
   - Wait timer
   - Deployment branches (only `main`)

## Publishing a New Release

### Step 1: Update Version

Update the version in these files:
- `pyproject.toml` → `version = "X.Y.Z"`
- `ciaf/__init__.py` → `Version: X.Y.Z`
- `ciaf/core/__init__.py` → `Version: X.Y.Z`
- `CHANGELOG.md` → Add release notes

### Step 2: Commit Changes

```bash
git add -A
git commit -m "chore: Bump version to X.Y.Z"
git push origin main
```

### Step 3: Create and Push Tag

```bash
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

### Step 4: Monitor GitHub Actions

1. Go to: https://github.com/DenzilGreenwood/pyciaf/actions
2. Watch the "Publish to PyPI" workflow run
3. Workflow will:
   - ✅ Run tests on Python 3.9-3.12
   - ✅ Build wheel and sdist
   - ✅ Check with twine
   - ✅ Publish to PyPI (if on main branch with tag)

### Step 5: Verify Publication

```bash
# Wait 1-2 minutes for PyPI to index
pip install --upgrade ciaf
python -c "import ciaf; print(ciaf.__version__)"
```

Check on PyPI: https://pypi.org/project/ciaf/

## Manual Publication (Fallback)

If GitHub Actions fails, you can publish manually:

```bash
# Build package
python -m build

# Check package
python -m twine check dist/*

# Upload to PyPI (requires API token)
python -m twine upload dist/*
```

## Versioning Guidelines

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (1.X.0): New features, backwards compatible
- **PATCH** (1.1.X): Bug fixes, backwards compatible

Examples:
- `1.1.0` → `1.1.1`: Bug fix
- `1.1.1` → `1.2.0`: New feature (LCM improvements)
- `1.2.0` → `2.0.0`: Breaking change (API redesign)

## Release Checklist

Before creating a release tag:

- [ ] Version updated in all files
- [ ] CHANGELOG.md updated with release notes
- [ ] All tests passing locally
- [ ] No uncommitted changes
- [ ] Changes committed to `main` branch
- [ ] CI tests passing on GitHub
- [ ] Documentation updated (if needed)

## Troubleshooting

### "Trusted publishing not configured"

- Make sure you've set up the trusted publisher on PyPI (see Prerequisites)
- Check that the workflow name and environment name match exactly

### "Tests failed"

- Run tests locally: `python tests/test_capsule_bugfix.py`
- Fix any failing tests before pushing tag

### "Package already exists"

- You can't re-upload the same version
- Bump the version and create a new tag
- Delete the old tag: `git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`

## Current Version: 1.1.0

Last published: 2026-03-25

Next planned release: 1.2.0 (Website integration examples)
