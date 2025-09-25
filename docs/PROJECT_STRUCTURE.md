# CIAF Project Structure

This document describes the organized structure of the CIAF (Cognitive Insight Audit Framework) codebase after cleanup.

## Directory Structure

```
PYPI/                           # Root project directory
├── .git/                       # Git version control
├── .gitignore                  # Git ignore patterns
├── .mypy_cache/               # MyPy type checking cache
├── .pytest_cache/             # Pytest cache
├── .vscode/                   # VS Code settings
│
├── LICENSE                    # Project license
├── README.md                  # Main project README
├── MODEL_BUILDING_GUIDE.md    # Primary model building guide
├── pyproject.toml             # Python project configuration (PEP 518)
├── setup.py                   # Python setup script (legacy)
├── MANIFEST.in               # Package manifest
├── requirements-dev.txt      # Development requirements
├── build_and_deploy.py       # Build and deployment script
│
├── ciaf/                     # Main CIAF package
│   ├── __init__.py
│   ├── py.typed              # PEP 561 typing marker
│   ├── CHANGELOG.md          # Package changelog
│   ├── SECURITY.md           # Security policy
│   ├── .gitignore            # Package-specific ignores
│   │
│   ├── adaptive_lcm.py       # Adaptive LCM implementation
│   ├── deferred_lcm.py       # Deferred LCM implementation
│   ├── deferred_lcm_design.py # Deferred LCM design documentation
│   ├── cli.py                # Command-line interface
│   ├── metadata_config.py    # Metadata configuration
│   ├── metadata_integration.py # Metadata integration
│   ├── metadata_storage.py   # Metadata storage
│   ├── metadata_storage_compressed.py # Compressed storage
│   ├── metadata_storage_optimized.py  # Optimized storage
│   │
│   ├── anchoring/            # Cryptographic anchoring
│   ├── api/                  # High-level API framework
│   ├── compliance/           # Compliance and regulatory support
│   ├── core/                 # Core CIAF functionality
│   ├── examples/             # CIAF usage examples
│   ├── explainability/       # Model explainability features
│   ├── extensions/           # Framework extensions
│   ├── inference/            # Inference management
│   ├── lcm/                  # Lifecycle Management (LCM)
│   ├── metadata_tags/        # Metadata tagging system
│   ├── preprocessing/        # Data preprocessing
│   ├── provenance/           # Data and model provenance
│   ├── simulation/           # Simulation utilities
│   ├── uncertainty/          # Uncertainty quantification
│   └── wrappers/             # Model wrapper implementations
│
├── examples/                 # Usage examples and demos
│   ├── comparison_demo_model_benchmark.py # Model comparison benchmarks
│   ├── deployable_model_demo.py          # Deployable model demonstration
│   ├── enterprise_features_demo.py       # Enterprise features showcase
│   └── quickstart_deferred_lcm.py       # Quick start with deferred LCM
│
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_anchors.py       # Anchoring tests
│   ├── test_audit_chain.py   # Audit chain tests
│   ├── test_ciaf_integration.py # Integration tests
│   ├── test_demo_integration.py # Demo integration tests
│   ├── test_framework.py     # Framework tests
│   ├── test_lcm_pickle_preservation.py # LCM pickle tests
│   ├── test_merkle.py        # Merkle tree tests
│   ├── test_optimized_storage.py # Storage optimization tests
│   └── compliance/           # Compliance-specific tests
│       └── test_acceptance.py # Compliance acceptance tests
│
├── docs/                     # Documentation
│   ├── index.md             # Documentation index
│   ├── quickstart.md        # Quick start guide
│   ├── concepts.md          # Core concepts
│   ├── receipts.md          # Receipt documentation
│   ├── compliance-mapping.md # Compliance mapping
│   ├── WHITEPAPER.md        # Technical whitepaper
│   ├── CODING_STANDARDS.md  # Coding standards
│   ├── DEFERRED_LCM_README.md # Deferred LCM documentation
│   ├── DEPLOYABLE_MODEL_DEMO_GUIDE.md # Deployment guide
│   ├── DEPLOYMENT_GUIDE.md  # General deployment guide
│   ├── ENTERPRISE_FEATURES_IMPLEMENTATION_SUMMARY.md # Enterprise features
│   ├── PERFORMANCE_OPTIMIZATION_SUMMARY.md # Performance guide
│   ├── MODEL_BUILDING_GUIDE_V1_1_0.md # Version-specific guide
│   ├── compliance/          # Compliance documentation
│   ├── Documents/           # Additional documents
│   └── schemas/             # Schema definitions
│
├── tools/                    # Development and utility tools
│   ├── ciaf_demo.py         # CIAF demonstration script
│   ├── compliance_demo_standalone.py # Standalone compliance demo
│   ├── deferred_lcm_benchmark.py # Performance benchmarking
│   ├── demo_receipt_verification.py # Receipt verification demo
│   ├── extract_receipt_for_verification.py # Receipt extraction
│   ├── verify_receipt.py    # Receipt verification
│   ├── verify_receipt_simple.py # Simple verification
│   ├── verification_enhancement_summary.py # Verification enhancements
│   ├── examples/            # Tool examples
│   └── production_models/   # Production model utilities
│
├── complete_documentation/   # Complete documentation archive
├── dist/                    # Distribution files (generated)
└── ciaf.egg-info/          # Package metadata (generated)
```

## Key Organizational Principles

### 1. **Clear Separation of Concerns**
- **Source code**: `ciaf/` package
- **Examples**: `examples/` directory  
- **Tests**: `tests/` directory
- **Documentation**: `docs/` directory
- **Tools**: `tools/` directory

### 2. **Standard Python Package Structure**
- Follows PEP 518 with `pyproject.toml`
- Proper package layout with `__init__.py` files
- Type hints support with `py.typed`

### 3. **Logical Grouping**
- **Core functionality**: `ciaf/core/`, `ciaf/api/`
- **Specialized features**: `ciaf/explainability/`, `ciaf/uncertainty/`
- **Integration points**: `ciaf/wrappers/`, `ciaf/extensions/`
- **Compliance**: `ciaf/compliance/`, `tests/compliance/`

### 4. **Development Support**
- **Testing**: Comprehensive test suite in `tests/`
- **Documentation**: Organized in `docs/`
- **Tools**: Development utilities in `tools/`
- **Examples**: Working examples in `examples/`

## File Types and Locations

### Configuration Files (Root)
- `pyproject.toml` - Modern Python project configuration
- `setup.py` - Legacy setup (for compatibility)
- `MANIFEST.in` - Package manifest
- `requirements-dev.txt` - Development dependencies
- `.gitignore` - Git ignore patterns

### Source Code (`ciaf/`)
- Core implementation modules
- Modular design with clear interfaces
- Type hints throughout
- Comprehensive docstrings

### Tests (`tests/`)
- Unit tests for all modules
- Integration tests
- Compliance tests
- Performance tests

### Documentation (`docs/`)
- User guides and tutorials
- API documentation
- Compliance mapping
- Performance guides

### Examples (`examples/`)
- Working code examples
- Demo applications
- Benchmark scripts
- Quick start tutorials

### Tools (`tools/`)
- Development utilities
- Benchmarking tools
- Verification scripts
- Production utilities

## Benefits of This Structure

1. **Maintainability**: Clear organization makes code easier to maintain
2. **Discoverability**: Users can easily find what they need
3. **Standard Compliance**: Follows Python packaging standards
4. **Scalability**: Structure supports growth and additional features
5. **Professional**: Enterprise-ready organization
6. **CI/CD Friendly**: Easy integration with automated tools

## Migration Notes

The following items were moved during cleanup:
- Test files moved from root to `tests/`
- Demo files moved from root to `examples/`
- Documentation consolidated in `docs/`
- Temporary/cache directories removed
- Duplicate configuration files removed
- Added comprehensive `.gitignore`

This structure provides a solid foundation for the CIAF framework's continued development and enterprise deployment.