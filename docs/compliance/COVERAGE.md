# CIAF Compliance Coverage Status
*Version:* 1.0.0 · *Scope:* OSS core (pyciaf) · *Last updated:* 2025-09-12  
*Note:* This document maps software artifacts to control intents. **It is not legal advice.** Implementation status reflects the current codebase; some controls also require org/process measures outside software.

## Legend
- ✅ Fully Implemented (code + docs + example/tests)
- 🔄 Partial / In Development (prototype or gaps remain)
- ❌ Not Implemented / Planned

---

## EU AI Act (High-Risk Requirements)

| Control | Status | How CIAF Helps (Evidence) | Gaps / Next Steps |
|---|---:|---|---|
| **Art. 9 – Risk Management** | 🔄 | Risk assessment patterns; snapshot/receipt evidence supports hazard analysis. *(Link: ciaf/compliance/risk_assessment.py, examples/risk_audit_example.py)* | Add structured risk register, pre/post-deployment checks, escalation workflow. |
| **Art. 10 – Data Governance** | ✅ | Dataset **anchors**, provenance **capsules**, Merkle roots; consent tags at capsule level. *(Link: ciaf/anchoring/, ciaf/provenance/capsules.py)* | Add dataset shift monitoring + lineage diffs over time. |
| **Art. 11 – Technical Documentation** | 🔄 | Auto-generated audit receipts and training snapshots; skeleton doc exporter. *(Link: ciaf/compliance/documentation.py, ciaf/compliance/reports.py)* | Map outputs to **Annex IV** items (purpose, data description, testing, PMM). |
| **Art. 12 – Record-Keeping (Logging)** | ✅ | Append-only (WORM) audit events; training snapshots; inference receipts. *(Link: ciaf/compliance/audit_trails.py, ciaf/inference/receipts.py)* | Harden retention policies + rollover strategy. |
| **Art. 13 – Transparency / Instructions** | 🔄 | Transparency metrics scaffolding; inference receipts. *(Link: ciaf/compliance/transparency_reports.py, ciaf/explainability/)* | Add model-card export + user-facing instruction templates. |
| **Art. 14 – Human Oversight** | ❌ | — | Provide reviewer UI hooks / override APIs / alerting. |
| **Art. 15 – Accuracy, Robustness, Cybersecurity** | 🔄 | Integrity via hashes/Merkle; optional AES-GCM; uncertainty metrics (prototype). *(Link: ciaf/core/crypto.py, ciaf/core/merkle.py, ciaf/compliance/uncertainty_quantification.py)* | Formal robustness tests, adversarial evaluation, crypto config self-tests. |

> **Annex IV mapping:** see [`eu-ai-act/annex-iv.md`](eu-ai-act/annex-iv.md).

---

## NIST AI RMF v1.0 (GOV / MAP / MEAS / MANAGE)

| Function | Status | How CIAF Helps (Evidence) | Gaps / Next Steps |
|---|---:|---|---|
| **GOV (Govern)** | 🔄 | System inventory via **model anchors**; append-only logs. *(Link: ciaf/lcm/model_manager.py, ciaf/api/framework.py)* | Define roles, RACI, policy bindings, change control. |
| **MAP (Map Context)** | 🔄 | Dataset lineage + consent tagging; usage context captured in metadata. *(Link: ciaf/anchoring/dataset_anchor.py, ciaf/metadata_integration.py)* | DPIA templates, stakeholder/context capture forms. |
| **MEAS (Measure)** | 🔄 | LCM performance metrics; fairness/uncertainty prototypes. *(Link: ciaf/compliance/validators.py, ciaf/metadata_integration.py)* | Calibrated metrics set, acceptance thresholds, periodic re-measurement jobs. |
| **MANAGE (Manage)** | 🔄 | Evidence chain enables CAPA auditing. *(Link: ciaf/compliance/corrective_action_log.py, ciaf/compliance/audit_trails.py)* | Risk register, mitigation tracking, risk comms playbooks. |

*Note:* Sub-category numbering (e.g., "MAP-X.Y") is omitted here; when you adopt it, mirror NIST labels exactly.

---

## GDPR / HIPAA (Data Protection)

| Control Theme | Status | How CIAF Helps (Evidence) | Gaps / Next Steps |
|---|---:|---|---|
| **Data Minimization (GDPR Art.5)** | ✅ | **LCM** materializes only necessary capsules; content-addressed storage. *(Link: ciaf/anchoring/lazy_manager.py, ciaf/anchoring/true_lazy_manager.py)* | Add redaction helpers + minimization lint checks. |
| **Lawful Basis & Consent (Art.6/7)** | 🔄 | Capsule-level consent flags & lineage. *(Link: ciaf/provenance/capsules.py, examples/basic_example.py)* | End-to-end consent workflow; revocation propagation. |
| **Data Subject Rights (Art.15–22)** | 🔄 | Content addressing + provenance speed DSAR lookup. *(Link: ciaf/anchoring/, ciaf/provenance/)* | Automated export/delete flows; audit of fulfillment. |
| **Security (Art.32) / HIPAA 164.312** | ✅ | AES-256-GCM (optional), HMAC-based anchors, integrity proofs. *(Link: ciaf/core/crypto.py, ciaf/core/base_anchor.py)* | Config scans, key/anchor rotation playbooks. |
| **Breach Detection/Proof** | ✅ | Tamper-evident logs; hash-linked receipts. *(Link: ciaf/compliance/audit_trails.py, ciaf/inference/receipts.py)* | Incident response runbooks + alert integrations. |

---

## SOX / ISO 27001 / PCI DSS (Selected)

| Control Theme | Status | How CIAF Helps (Evidence) | Gaps / Next Steps |
|---|---:|---|---|
| **Immutable Audit Trails** | ✅ | WORM audit events; snapshot lineage. *(Link: ciaf/compliance/audit_trails.py, ciaf/provenance/snapshots.py)* | External log shipping + retention policy tests. |
| **Documentation & Retention** | ✅ | Exportable receipts/snapshots; retention settings. *(Link: ciaf/compliance/reports.py, ciaf/metadata_storage.py)* | Evidence catalog; retention verification tests. |
| **Access Controls** | 🔄 | Role patterns in examples. *(Link: examples/*, ciaf/compliance/validators.py)* | Enforce RBAC in code paths; SoD checks; admin break-glass logs. |

---

## Contributing to Coverage

1. Identify a control → map to **CIAF artifact(s)** (anchor, capsule, snapshot, receipt).  
2. Add **evidence pointers** (code path, doc page, unit/integration test).  
3. Implement missing validation logic; add **automated tests**.  
4. Update status and gaps; open an issue with owner + target version.

> See also: `ciaf/SECURITY.md` for secure deployment guidance and vulnerability reporting.