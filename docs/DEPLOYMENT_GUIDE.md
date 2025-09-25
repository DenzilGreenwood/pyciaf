# CIAF Package Deployment Guide

## Overview
This guide covers rebuilding and deploying the CIAF package after the major v1.1.0 update that replaced all mock implementations with production-ready code.

## Why Rebuild is Necessary

### 1. **Significant Functional Changes**
- ✅ Replaced all `simulate_*` methods with `create_*` methods
- ✅ Added missing cryptographic functions (`derive_model_anchor`, `derive_dataset_anchor`)
- ✅ Enhanced all LCM managers with realistic implementations
- ✅ Fixed import issues and method call problems
- ✅ Eliminated all mock/static data

### 2. **Version Update**
- **From**: 1.0.0 (demo/mock framework)
- **To**: 1.1.0 (production-ready system)

### 3. **Package Quality Improvements**
- All modules now provide realistic, functional implementations
- Proper error handling and statistical modeling
- Production-ready patterns throughout codebase

## Deployment Steps

### Step 1: Install Development Dependencies

```powershell
# Install build tools
pip install -r requirements-dev.txt

# Or install individually:
pip install build twine pytest toml
```

### Step 2: Clean Build (Recommended)

```powershell
# Using the provided build script
python build_and_deploy.py --clean-only
```

### Step 3: Run Tests

```powershell
# Test the package
python build_and_deploy.py --test-only
```

### Step 4: Build the Package

```powershell
# Build only (no deployment)
python build_and_deploy.py --build-only
```

### Step 5: Deploy to TestPyPI (Recommended First)

```powershell
# Deploy to TestPyPI for testing
python build_and_deploy.py --deploy-test
```

**Test the TestPyPI installation:**
```powershell
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ ciaf==1.1.0

# Test import
python -c "import ciaf; print(f'CIAF v{ciaf.__version__} works!')"
```

### Step 6: Deploy to Production PyPI

```powershell
# Deploy to production PyPI (after testing)
python build_and_deploy.py --deploy-prod
```

## Alternative Manual Approach

If you prefer manual control:

### 1. Clean Previous Builds
```powershell
Remove-Item -Recurse -Force build, dist, *.egg-info -ErrorAction SilentlyContinue
```

### 2. Build Package
```powershell
python -m build
```

### 3. Check Package Quality
```powershell
python -m twine check dist/*
```

### 4. Upload to TestPyPI
```powershell
python -m twine upload --repository testpypi dist/*
```

### 5. Upload to PyPI
```powershell
python -m twine upload dist/*
```

## Verification Steps

### 1. Local Verification
```powershell
# Verify version consistency
python -c "import ciaf; print(ciaf.__version__)"
python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])"
```

### 2. Installation Testing
```powershell
# Test fresh installation
pip uninstall ciaf -y
pip install ciaf==1.1.0
python -c "import ciaf; print('Installation successful')"
```

### 3. Functionality Testing
```powershell
# Test core functionality
python -c "
from ciaf import CIAFFramework
framework = CIAFFramework()
print('Framework initialization successful')
"
```

## Important Notes

### 1. **Authentication Requirements**
- **TestPyPI**: Register at https://test.pypi.org/account/register/
- **PyPI**: Register at https://pypi.org/account/register/
- **API Tokens**: Use API tokens instead of passwords (recommended)

### 2. **Version Management**
- Current version: **1.1.0**
- Each deployment requires a unique version number
- Consider patch versions (1.1.1, 1.1.2) for hotfixes
- Consider minor versions (1.2.0) for feature additions

### 3. **Package Size**
```
Expected package sizes:
- Source distribution (.tar.gz): ~200-300 KB
- Wheel distribution (.whl): ~250-350 KB
```

### 4. **Dependencies**
The package will automatically install:
- `cryptography>=3.4.8`
- `numpy>=1.21.0`
- `scikit-learn>=1.0.0`

## Troubleshooting

### Common Issues

#### 1. **Build Failures**
```
Error: Module not found
Solution: Ensure all imports are correct and circular imports resolved
```

#### 2. **Upload Failures**
```
Error: 403 Forbidden
Solution: Check API tokens and repository permissions
```

#### 3. **Version Conflicts**
```
Error: Version already exists
Solution: Bump version number in pyproject.toml and __init__.py
```

### Getting Help

1. **Build Issues**: Check `python -m build --help`
2. **Upload Issues**: Check `python -m twine upload --help`
3. **Package Issues**: Use the provided `build_and_deploy.py` script

## Success Indicators

✅ **Build Success**: Both `.tar.gz` and `.whl` files created in `dist/`
✅ **Upload Success**: Package visible on PyPI/TestPyPI
✅ **Installation Success**: `pip install ciaf==1.1.0` works
✅ **Import Success**: `import ciaf` works without errors
✅ **Functionality Success**: Core features work as expected

## Next Steps After Deployment

1. **Update Documentation**: Ensure README.md reflects new version
2. **Tag Release**: Create git tag for v1.1.0
3. **Notify Users**: Announce the significant improvements
4. **Monitor**: Watch for installation/usage issues
5. **Plan Roadmap**: Consider future enhancements

---

**Remember**: This v1.1.0 release represents a major quality improvement from demo code to production-ready implementation. Users upgrading will get significantly enhanced functionality!