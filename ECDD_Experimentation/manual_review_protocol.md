# Manual Review Protocol for ECDD Deepfake Detector

## Purpose
This document defines the policy for manual review of model predictions to ensure:
1. Privacy and consent compliance
2. Evaluation integrity
3. Feedback loop safety

---

## 1. Consent Requirements

**Rule**: Images are NEVER stored for manual review unless explicit user consent is obtained.

**Process**:
- Default behavior: Log only summary statistics (score, decision, reason code, metadata).
- Consent prompt: If user opts-in to "Help improve the system," images may be stored for review.
- Opt-out: User can revoke consent at any time; images are immediately deleted.

**Storage Location**: `AUDIT/manual_review/` (separate from TRAIN/TEST splits).

---

## 2. Retention Period

**Policy**: 
- Consented images: Retained for **90 days** maximum.
- After 90 days: Automatically purged unless specifically flagged for long-term dataset curation (requires separate ethics review).

**Audit Trail**: 
- Deletion events are logged with timestamps.
- Retention compliance checked monthly via automated script.

---

## 3. Stratification Strategy

**Goal**: Ensure manual review covers edge cases without biasing test sets.

**Sampling**:
- **False Positives** (Real flagged as Fake): Top priority for review.
- **False Negatives** (Fake flagged as Real): High priority.
- **Abstains** (Low confidence or quality gates triggered): Medium priority.
- **Correct predictions**: Low priority (only sampled for calibration checks).

**Quotas** (per review cycle):
- FP: 50 images minimum
- FN: 50 images minimum
- Abstains: 30 images
- Correct: 20 images (random sample)

---

## 4. Contamination Prevention

**Critical Rule**: Reviewed images are tagged as `AUDIT` and NEVER mixed into:
- `TEST` (evaluation sets)
- `TRAIN` (training data)
- `CALIBRATION` (threshold fitting)

**Enforcement**:
- Automated lineage check via `validate_lineage.py`.
- CI/CD gate: Deployment fails if any `AUDIT` image appears in TEST/TRAIN/CALIBRATION.

**Feedback Loop**:
- Labels from manual review can be used to:
  - Update guardrail thresholds (e.g., blur/compression).
  - Trigger recalibration (refit temperature scaling on fresh CALIBRATION set).
  - Identify new attack patterns for future dataset curation.
- Labels CANNOT be used to directly retrain the model without creating a new, properly split dataset.

---

## 5. Reviewer Guidelines

**Who**: 
- Internal team members with deepfake detection training.
- No crowdsourcing (to maintain quality and privacy).

**What to Review**:
- Is the prediction correct?
- If incorrect, why? (e.g., blur, occlusion, edge case, adversarial attack).
- Quality issues: compression, resolution, lighting.

**Labels**:
- `real` / `fake` (ground truth)
- `reason_code`: Why the model might have failed (e.g., "extreme_blur", "occlusion", "novel_generator").

---

## 6. Privacy and Security

**Data Handling**:
- Reviewed images encrypted at rest (AES-256).
- Access restricted to authorized personnel only (audit logged).
- No personally identifiable information (PII) extracted or stored.

**Anonymization**:
- If embeddings are logged, they must be aggregated (e.g., cluster centroids) before storage.
- Individual images not linked to user accounts in logs.

---

## 7. Compliance and Auditing

**Quarterly Review**:
- Check retention compliance.
- Verify contamination prevention (no AUDIT images in TEST/TRAIN).
- Review feedback incorporation process.

**Audit Log**:
- All manual review sessions logged with:
  - Reviewer ID
  - Timestamp
  - Images reviewed count
  - Labels assigned count

---

## Summary

This protocol ensures manual review improves the system without compromising user privacy, evaluation validity, or feedback loop integrity. Any violation of contamination rules invalidates all subsequent test results and triggers a mandatory dataset audit.
