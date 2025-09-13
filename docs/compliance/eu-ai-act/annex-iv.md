# EU AI Act Annex IV - Technical Documentation Mapping

*Version:* 1.0.0 · *Last updated:* 2025-09-12  
*Note:* This document maps CIAF artifacts to EU AI Act Annex IV technical documentation requirements. **It is not legal advice.**

## Annex IV Requirements → CIAF Artifacts

| Annex IV Requirement | CIAF Artifact | Implementation Status | Evidence/Location |
|---|---|---:|---|
| **(a) General description of the AI system** | | | |
| Purpose, function, general logic | Model metadata, training snapshots | ✅ | `ciaf/lcm/model_manager.py`, auto-generated docs |
| Persons/groups of persons on which system is intended to be used | Usage context metadata | 🔄 | `ciaf/metadata_integration.py` (extend for target population) |
| **(b) Description of the elements** | | | |
| Main algorithmic elements, techniques, logic | Model architecture fingerprints | ✅ | `model_anchor['architecture_fingerprint']` |
| Key design choices including rationale | Training parameters, design decisions | ✅ | `model_anchor['parameters_fingerprint']`, training snapshots |
| **(c) Description of the monitoring** | | | |
| Capabilities and limitations, expected performance | Performance metrics, uncertainty quantification | 🔄 | `ciaf/compliance/uncertainty_quantification.py` |
| Circumstances affecting performance | Context metadata, performance under different conditions | 🔄 | Extend metadata capture for operating conditions |
| **(d) Description of the data** | | | |
| Training, validation, testing datasets | Dataset anchors, provenance capsules | ✅ | `ciaf/anchoring/dataset_anchor.py`, capsule metadata |
| Information about data sources | Dataset metadata | ✅ | `dataset_metadata` in anchors |
| Data preparation and labeling procedures | Provenance chain, processing metadata | 🔄 | Extend capsules for labeling provenance |
| **(e) Assessment of risks** | | | |
| Risk management measures | Risk assessment patterns, bias detection | 🔄 | `ciaf/compliance/risk_assessment.py` |
| Residual risks and risk mitigation measures | Risk logs, corrective actions | 🔄 | `ciaf/compliance/corrective_action_log.py` |
| **(f) Validation and testing procedures** | | | |
| Description of testing procedures | Test snapshots, validation receipts | 🔄 | Extend training snapshots for test procedures |
| Testing results | Performance metrics, validation results | 🔄 | Capture validation outcomes in snapshots |
| **(g) Information about modifications** | | | |
| Changes made to the system | Version control, modification logs | ✅ | Training snapshots with version tracking |
| Evaluation of continued compliance | Compliance validation results | 🔄 | `ciaf/compliance/validators.py` |

## CIAF Artifact Generation for Annex IV

### Automated Documentation Export

> **Implementation note:** The `AnnexIVExporter` below is an example scaffold.
> If not present in the repository, treat it as planned work and adapt to your codebase.

```python
from ciaf.compliance import AnnexIVExporter

# Generate Annex IV compliant documentation
exporter = AnnexIVExporter()
doc = exporter.generate_technical_documentation(
    model_name="diagnostic_model",
    training_snapshot=snapshot,
    compliance_assessments=validation_results
)

# Export formats
doc.export_pdf("technical_documentation.pdf")
doc.export_html("technical_documentation.html")
```

### Required CIAF Metadata for Compliance

To support full Annex IV compliance, ensure your CIAF implementation captures:

```python
# Enhanced model metadata for Annex IV
model_metadata = {
    "purpose": "Medical diagnosis assistance",
    "target_population": "Healthcare professionals in radiology",
    "intended_use_cases": ["X-ray analysis", "CT scan review"],
    "operating_conditions": {
        "hardware_requirements": "GPU with 8GB+ VRAM",
        "expected_data_quality": "DICOM images, 512x512 minimum resolution",
        "performance_thresholds": {"accuracy": 0.95, "recall": 0.90}
    },
    "limitations": ["Not for emergency diagnosis", "Requires human oversight"],
    "risk_factors": ["False negative impact", "Bias in rare conditions"]
}

# Enhanced dataset metadata
dataset_metadata = {
    "source_description": "Multi-hospital radiology database",
    "collection_period": "2020-2024",
    "preprocessing_steps": ["DICOM normalization", "PHI removal", "Quality filtering"],
    "labeling_procedure": "Double-blind radiologist review",
    "data_quality_checks": ["Inter-rater agreement > 0.85"],
    "bias_mitigation": ["Demographic stratification", "Hospital diversity"]
}
```

## Implementation Roadmap

### Phase 1: Core Documentation (Current)
- ✅ Basic model and dataset metadata
- ✅ Training snapshots with provenance
- ✅ Version tracking and modification logs

### Phase 2: Enhanced Compliance (In Development)
- 🔄 Risk assessment integration
- 🔄 Performance monitoring expansion
- 🔄 Validation procedure documentation

### Phase 3: Full Annex IV Support (Planned)
- ❌ Automated compliance report generation
- ❌ Interactive documentation builder
- ❌ Legal review workflow integration

## Contributing

To improve Annex IV compliance:

1. **Identify gaps** in current artifact mapping
2. **Extend metadata schemas** to capture missing information
3. **Implement automated exporters** for required documentation formats
4. **Add validation rules** to ensure completeness
5. **Test with sample use cases** and regulatory scenarios

> **Legal Note**: This mapping is provided for technical guidance only. Consult legal counsel for regulatory compliance advice specific to your use case and jurisdiction.