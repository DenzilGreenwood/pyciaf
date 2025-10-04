# CIAF Regulatory Compliance Documentation

This folder contains LaTeX documents that map the CIAF core framework to specific regulatory requirements.

## Documents

1. **`regulatory_compliance_mapping.tex`** - Comprehensive multi-page document with detailed mappings
2. **`assurance_map_one_page.tex`** - Executive summary and quick reference for auditors/buyers

## Compilation Instructions

### Prerequisites
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages: `booktabs`, `longtable`, `xcolor`, `colortbl`, `hyperref`, `listings`, `fancyhdr`

### Compile Commands

```bash
# For the comprehensive document
pdflatex regulatory_compliance_mapping.tex
pdflatex regulatory_compliance_mapping.tex  # Run twice for proper cross-references

# For the one-page assurance map
pdflatex assurance_map_one_page.tex
```

### Alternative: Online Compilation
Upload the .tex files to [Overleaf](https://www.overleaf.com) for online compilation.

## Document Contents

### Comprehensive Mapping (`regulatory_compliance_mapping.tex`)
- Executive summary of CIAF compliance achievements
- Detailed mapping of each core module to regulatory requirements
- Code examples and implementation details
- Complete regulatory crosswalk table
- Assurance map for auditors with line-number references

### One-Page Assurance Map (`assurance_map_one_page.tex`)
- Executive summary for decision makers
- Comprehensive regulatory coverage table
- Deployment readiness checklist
- Quick reference for compliance evidence

## Regulatory Coverage

Both documents cover compliance with:
- **EU AI Act** (Articles 9, 10, 12) - Risk management and record keeping
- **ISO/IEC 42001** - AI Management System
- **GDPR** (Articles 5, 25, 32) - Privacy and security
- **NIST AI RMF** - Measure and Manage functions
- **SOX/SEC** - Financial audit requirements
- **NIST 800-53** - Security controls (SC-12, SC-13)

## Key Compliance Mechanisms

1. **Canonicalization** → Required fields and tamper-evident logs
2. **Merkle Trees** → Immutable audit trails with inclusion proofs
3. **Cryptography** → AES-GCM encryption with context binding
4. **Deterministic Operations** → Reproducible audits and timestamps
5. **Policy Enforcement** → Risk assessment and governance hooks
6. **Key Management** → Production-ready signing infrastructure
7. **Pluggable Architecture** → Third-party verification support

## Usage for Compliance Teams

These documents provide:
- **Direct regulatory clause mappings** to specific CIAF functions
- **Evidence trails** that auditors can independently verify
- **Implementation status** showing production readiness
- **Code references** with line numbers for technical review

## Contact

For questions about regulatory compliance or implementation details, refer to the main CIAF documentation or the enhanced core demo (`enhanced_core_demo.py`).