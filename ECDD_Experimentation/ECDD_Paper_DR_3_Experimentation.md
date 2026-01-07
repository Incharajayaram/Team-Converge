Also, [incharajayaram2020@gmail.com](mailto:incharajayaram2020@gmail.com), Please add a separate branch in the git for this experimentation system.

Create a yaml/json to contain every policy knob that we plan to freeze, you can pick this list up directly from the architecture data flow tab, just simple list with topic - decision

Freeze a golden dataset for testing with 30 face images (balanced real/fake), 20 out-of-scope images (animals, cartoons, scenery, CGI, meme screenshots etc etc) and then, 30 edge cases (EXIF rotations, alpha PNG/WebP, very low JPEG quality, small faces, multi-face, blur drift, weird color profiles). Produce these however so you may choose to.

For every image in the golden sets, we store (and hash) these stage outputs:

S0 raw bytes hash (at browser and at server ingress).

S1 decoded RGB tensor hash (post-EXIF, post-alpha policy, post-gamma policy).

S2 face crop boxes and aligned crop tensors hash (one per face).

S3 resized 256x256 tensor hash.

S4 normalized tensor hash.

S5 patch-logit map (float) summary stats hash (shape, min/max/mean, and optionally a quantized checksum).

S6 pooled logit (float).

S7 calibrated logit and calibrated probability.

S8 decision label and reason codes.

Now, as for experiments,

Phase 1: Ingress and pixel pipeline equivalence (stop if any fails)

Goal: lock the entire pixel contract”(decoder, EXIF, color ordering, gamma, alpha, dtype/range, resizing, normalization) across browser → server → edge

E1.1 Byte-for-byte upload invariance test  
Procedure: upload the same file twice through your local-host website, capture server-ingress bytes. Compare SHA-256 of ingress bytes to the client-side file bytes.  
Pass: exact match for all tested images.  
Fail: any mismatch means your client pipeline is re-encoding or altering data; you must either disable client transforms or explicitly adopt them into the locked preprocessing.

E1.2 Format allowlist and corruption rejection test  
Procedure: try supported formats (JPEG/PNG/WebP), plus a corrupted file and a disguised extension (e.g., .jpg containing non-image).  
Pass: supported images accepted, corrupted/invalid rejected with explicit reason code; max size enforced.  
Fail: silent acceptance of corrupted or unexpected formats.

E1.3 Single decode-path enforcement test  
Procedure: implement a single “decode_image()” function used everywhere. Attempt to route inference through any alternate decode path (PIL/OpenCV mix, browser decode vs server decode) and ensure it is blocked or identical.  
Pass: only one decode path is executed, its output matches stored golden S1.  
Fail: any alternate path or output difference.

E1.4 EXIF orientation correctness test  
Procedure: on GFS-EdgeCases EXIF images, assert that decoded tensor after EXIF transpose exactly matches golden S1, and face detector sees upright faces.  
Pass: pixel-level agreement (or within a tiny tolerance if you store float).  
Fail: any rotation mismatch, shifted crops, or face detection failures attributable to orientation.

E1.5 RGB channel ordering and dtype/range invariants  
Procedure: insert assertions at the model boundary:  
channel order must be RGB, dtype must be exactly the contract dtype; range must be exactly the contract range (either 0–255 uint8 before normalization, or 0–1 float).  
Pass: all images satisfy invariants; any violation hard-fails with reason code.  
Fail: any silent conversion or mixed policy.

E1.6 Gamma policy invariance test (sRGB-only path)  
Procedure: feed images with potential embedded profiles and test two decodes: (a) your chosen behavior, (b) a deliberately different behavior. Verify your pipeline always uses (a) and matches golden S1.  
Pass: S1 stable and identical across runs; no implicit linearization.  
Fail: drift across environments.

E1.7 Alpha handling policy test  
Procedure: feed PNG/WebP with alpha; ensure your policy is deterministic and logged (reject, or composite over defined background).  
Pass: S1 stable and matches golden; reason codes for any rejects.  
Fail: different composites across environments.

E1.8 Fixed interpolation kernel test (resize determinism)  
Procedure: run resize step in server reference and (if any resizing occurs elsewhere) in edge path, compare S3 to golden.  
Pass: S3 matches within tolerance for all golden images.  
Fail: any mismatch means your resize kernel, rounding, or backend differs; fix and lock.

E1.9 Training normalization constants test  
Procedure: compute S4 normalized tensor and compare to golden; assert mean/std constants exactly match contract.  
Pass: S4 matches; invariants hold.  
Fail: normalization mismatch is a top cause of offline/online gaps; do not proceed.

Outputs of Phase 1 (must be finalized)  
Decoder/library, EXIF handling, gamma policy, alpha handling, dtype/range, resize kernel, rounding rules, normalization constants, and pixel-level tolerance thresholds (for any stage where exact matching is impossible).

C. Phase 2: Face detector routing and guardrails (stop if any fails)

Goal: guardrails should be deterministic, parameterized, and auditable. They should prevent nonsense inputs from reaching the detector, and they should emit explicit reason codes.

E2.1 Face detector version pin test  
Procedure: record face detector model name/version hash; run GFS-InScope and GFS-EdgeCases.  
Pass: face detection outputs (boxes/confidences) match stored golden S2 within tolerance.  
Fail: any version drift or nondeterministic outputs; pin versions and seeds or switch detectors.

E2.2 No-face abstain correctness test  
Procedure: run GFS-OOD-20.  
Pass: 100% of OOD images abstain with reason “No-face” (or “Out-of-scope”), and the student model is not called.  
Fail: any OOD image reaches model inference.

E2.3 Face confidence threshold selection sweep (policy definition experiment)  
Procedure: sweep face confidence threshold over a small grid (e.g., 0.2 to 0.9). Measure:  
in-scope pass-through rate,  
OOD false pass-through rate,  
end-to-end error rate at fixed detector threshold.  
Decision rule: pick the highest threshold that does not cause unacceptable in-scope abstain.  
Pass: threshold selected and written into contract; rerun tests to confirm.  
Fail: if no threshold works, your face detector is unstable for your domain; replace it or add fallback rules.

E2.4 Alignment/crop determinism test  
Procedure: for multi-run repeats on the same image, verify crop tensors identical (or within tolerance). Compare to golden S2.  
Pass: stable cropping.  
Fail: non-determinism makes audit impossible; fix.

E2.5 Multi-face policy conclusive test (max vs largest-face)  
Procedure: implement both policies behind a flag. Run GFS-MultiFace and measure:  
worst-case safety (missing any fake face),  
false positives on group photos,  
latency overhead (faces per image).  
Decision: choose one policy and freeze it. For deepfake safety, “max P-Fake across faces” is typically safer; but you must accept increased false positives in group images if any face is borderline.  
Pass: policy chosen and contract updated.  
Fail: if performance unacceptable, define UI semantics (“single-face only” upload requirement) and enforce it with abstain.

E2.6 Minimum face size threshold sweep  
Procedure: synthetically downscale faces or select small-face samples; sweep minimum face crop size. Measure performance and confidence calibration stability.  
Pass: choose threshold where predictions stop being reliable, and abstain below it.  
Fail: if model still outputs confident predictions on tiny faces, your calibration/abstain policy must be stricter.

E2.7 Blur metric selection and threshold sweep  
Procedure: pick 1 blur metric (e.g., variance of Laplacian) and sweep thresholds on blurred versions of golden faces. Track:  
abstain rate,  
false positive/negative at operating point,  
calibration error.  
Pass: lock blur metric and threshold.  
Fail: if blur metric does not correlate with errors, consider a second metric or a small quality classifier.

E2.8 Compression proxy definition and threshold sweep  
Procedure: define a compression proxy (e.g., estimated JPEG quality or DCT energy heuristics) and sweep thresholds across re-encoded images. Decide “abstain vs stricter threshold” behavior.  
Pass: lock proxy and behavior.  
Fail: if proxy unreliable, prefer abstain on low-resolution + blur jointly rather than compression alone.

Outputs of Phase 2  
Face detector model/version, confidence threshold, alignment/crop policy, multi-face policy, minimum face size threshold, blur metric and threshold, compression proxy and threshold, and precise abstain reason code taxonomy.

D. Phase 3: Patch grid, pooling, and heatmap correctness (stop if any fails)

Goal: lock the “patch-logit semantics”: shape, mapping, pooling, determinism, and quantization survivability.

E3.1 Patch grid shape contract test  
Procedure: run the student model on a canonical 256x256 input and assert patch-logit map shape is exactly expected (H x W), with deterministic ordering.  
Pass: shape matches contract; same across runs/environments.  
Fail: any mismatch breaks heatmap UI mapping and pooling consistency.

E3.2 Heatmap-to-image coordinate mapping test  
Procedure: render heatmap overlay for a few known images and verify cell-to-pixel alignment. You can do this deterministically: select a cell (i,j), highlight its mapped region, and assert it corresponds to the expected receptive field region.  
Pass: mapping is correct and stable.  
Fail: UI becomes misleading; lock stride/padding assumptions.

E3.3 Pooling choice A/B test (top-k vs attention) using only forward passes  
Procedure: run both pooling implementations on the same patch-logit maps (no retraining needed if pooling is post-hoc). Compare:  
separation between real/fake (simple score distributions),  
robustness to one noisy patch (inject a single high patch-logit and see if pooling overreacts),  
stability under transforms (JPEG/re-resize).  
Pass: pick pooling and freeze it.  
Fail: if both unstable, implement a robust pooling variant (e.g., trimmed mean over top percentile) and rerun.

E3.4 Top-k parameter selection sweep (if top-k chosen)  
Procedure: sweep r in K=ceil(r\*Npatches) (e.g., r in {0.01, 0.02, 0.05, 0.1}). Track operating-point metrics and calibration.  
Pass: choose r and define tie-breaking and numerical stability rules; freeze.  
Fail: if optimal r varies wildly across subsets, you likely have distribution shift and need better calibration/abstain.

E3.5 Attention pooling determinism test (if attention chosen)  
Procedure: run repeated inference on the same input across environments; compare attention weights and pooled logit.  
Pass: weights/logits stable within tolerance; no nondeterministic softmax behavior.  
Fail: revert to top-k or enforce float32 “islands” for attention computation with defined error bounds.

E3.6 Patch-logit sanity tests (conclusive invariants)  
Procedure: inject controlled perturbations:  
flip one patch region in the input (e.g., blur a small block),  
observe patch-logit changes localized to corresponding cells.  
Pass: local perturbation yields local response; global response consistent with pooling.  
Fail: if response is erratic, your receptive field is not behaving as expected or preprocessing mismatch remains.

Outputs of Phase 3  
Exact patch-logit map dimensions, stride/padding/mapping, pooling choice and parameters, determinism and quantization plan for pooling, and heatmap overlay correctness.

E. Phase 4: Calibration, thresholds, and abstain semantics (stop if any fails)

Goal: your deployed behavior is controlled by calibration and threshold policy more than by AUC. These must be versioned and tied to a deployment-like calibration set.

E4.1 Calibration set contract test  
Procedure: define a small “deployment calibration set” collected through the exact local-host pipeline. Ensure it is disjoint from training and test golden sets.  
Pass: dataset exists, minimum size set, sampling rules defined, versioned.  
Fail: you cannot claim calibration correctness without this.

E4.2 Temperature scaling fit and verification (low compute)  
Procedure: fit a single scalar temperature on the calibration set using logits and labels. Evaluate:  
reliability error (e.g., expected calibration error),  
threshold stability (how much threshold changes relative to raw).  
Pass: calibration improves reliability and stabilizes threshold selection; store T in bundle.  
Fail: if it worsens, your logits may be non-monotonic due to bugs; return to Phase 1–3.

E4.3 Platt scaling fit and verification (low compute)  
Procedure: fit logistic regression parameters (a,b) on calibration set using logits as input.  
Pass: improves reliability and meets operating-point constraints.  
Fail: if overfits (small set), prefer temperature scaling.

E4.4 Operating point selection test at fixed error budget  
Procedure: define primary constraint (example: FPR on real faces <= 5%) and compute threshold meeting it on calibration set. Then test that threshold on:  
GFS-InScope,  
GFS-EdgeCases,  
and a small “in-the-wild holdout” if available.  
Pass: constraints hold within acceptable slack; thresholds written to contract.  
Fail: if constraints do not transfer, your calibration set is not representative or you have drift; improve sampling or increase abstain.

E4.5 Abstain band design experiment (selective classification)  
Procedure: set two thresholds: t_real and t_fake with an abstain band in between. Sweep band width. Report:  
error rate on non-abstained,  
abstain rate,  
rate of abstain reason codes.  
Pass: choose band that meets product tolerance; define UI behavior explicitly (retry, manual review, block).  
Fail: if abstain rate required is too high, guardrails/pooling/calibration need revision.

E4.6 Guardrail-conditioned threshold policy test  
Procedure: define conditional behavior for low-quality inputs:  
Policy A: force abstain,  
Policy B: stricter fake threshold.  
Compare both on degraded samples.  
Pass: choose one and freeze it.  
Fail: if neither works, quality gate must be stricter or model needs robustness training.

Outputs of Phase 4  
Calibration method choice, parameter storage/versioning, calibration set contract, operating-point contract, threshold(s) and abstain semantics, and product-facing UI rules for abstain.

F. Phase 5: Quantization and float-to-TFLite parity (stop if any fails)

Goal: quantization is a common source of silent score distribution shifts. You need parity gates on intermediate tensors.

E5.1 Float vs TFLite probability parity test (end-to-end)  
Procedure: run golden sets through float reference and TFLite model. Compare final calibrated probabilities.  
Pass: max absolute difference <= epsilon_prob (define it) and rank correlation high.  
Fail: do not ship; proceed to more granular parity.

E5.2 Patch-logit map parity test (intermediate)  
Procedure: compare patch-logit maps between float and TFLite:  
shape equality,  
mean absolute difference,  
max absolute difference,  
argmax location stability (where the highest fake patch is).  
Pass: within tolerances.  
Fail: if patch logits drift, quantization is harming feature extraction; consider dynamic range vs full-int8, or keep parts in float.

E5.3 Pooled logit parity test  
Procedure: compare pooled logit float vs TFLite before calibration.  
Pass: within tolerance.  
Fail: likely pooling instability or quantization error accumulation.

E5.4 Post-quant calibration mandatory gate  
Procedure: refit calibration parameters (temperature or Platt) using the quantized model logits on the same calibration set; recompute thresholds.  
Pass: post-quant calibration restores operating point behavior.  
Fail: quantized logits may be too distorted; adjust quantization strategy or model architecture.

E5.5 Delegate and threading invariance test  
Procedure: run TFLite with different delegates (XNNPACK/NNAPI/GPU) and thread counts you might use. Compare outputs.  
Pass: output changes remain within tolerances; config is pinned in contract.  
Fail: you must lock the runtime config per device class or disable unstable delegates.

Outputs of Phase 5  
Quantization policy choice, representative set requirements (if applicable), parity tolerances at patch/pool/probability levels, mandatory post-quant calibration, and pinned TFLite runtime configuration.

G. Phase 6: “Realistic” evaluation battery (still low compute, but conclusive)

Goal: verify that performance claims are tied to your actual deployment conditions, not random offline splits.

E6.1 Source-based split stress test on your existing 2500 images  
Procedure: re-split by generator/source/device/platform/compression rather than random. Evaluate at fixed operating point.  
Pass: performance does not collapse beyond defined tolerance.  
Fail: you have real distribution shift; prioritize robustness augmentation and monitoring.

E6.2 Time-based split drift probe  
Procedure: sort by collection time (or proxies such as filename date or ingestion timestamp). Train/calibrate on earlier, test on later.  
Pass: stable operating-point metrics over time.  
Fail: drift is likely; monitoring and periodic recalibration become mandatory.

E6.3 Transform suite conclusive test (attack and pipeline simulation)  
Procedure: apply deterministic transforms to golden in-scope images:  
JPEG re-encode at Q in {95, 75, 50, 30},  
resize down-up chains,  
blur levels,  
screenshot-like resampling.  
Measure:  
operating-point metrics,  
score distribution shift,  
abstain triggers.  
Pass: bounded degradation; guardrails/abstain behave as designed.  
Fail: add robustness augmentation or tighten quality gates.

E6.4 Out-of-scope separation test (claims hygiene)  
Procedure: ensure OOD images always abstain (or are labeled out-of-scope), and are excluded from “accuracy on faces.”  
Pass: strict separation in reporting.  
Fail: your metrics become misleading; fix reporting policy.

Outputs of Phase 6  
Executable evaluation battery definition, transform suite parameters, metrics to report at operating point, and acceptance thresholds for release.

H. Phase 7: Monitoring, drift triggers, and auditability (low compute; mostly simulation)

Goal: prove you can detect and respond to drift without storing raw images by default.

E7.1 Logging schema sufficiency test (privacy-preserving)  
Procedure: log only scalars and reason codes from a small run. Confirm you can reconstruct:  
score histogram by day,  
abstain rate by reason,  
face confidence distribution,  
compression proxy distribution,  
model-version breakdown.  
Pass: dashboards possible without images.  
Fail: add minimal additional scalars (not images) or compute embeddings only under explicit consent and aggregation.

E7.2 Drift trigger simulation test  
Procedure: simulate drift by mixing in a batch of heavily compressed images or a new device class; run through pipeline; see if drift detectors trigger.  
Pass: drift triggers fire; response playbook invoked.  
Fail: thresholds too lax or features insufficient.

E7.3 Release/rollback rehearsal test  
Procedure: create two bundles (current and a deliberately degraded one) and run a canary comparator that checks operating-point metrics and abstain rates.  
Pass: rollback triggers function correctly.  
Fail: release policy is not enforceable; fix before production.

Outputs of Phase 7  
Logging policy, retention policy, drift trigger thresholds, mandatory responses, and release/rollback automation.

I. Phase 8: Dataset governance and feedback loop safety (still low compute)

Goal: prevent leakage and preserve validity of evaluation.

E8.1 Near-duplicate leakage scan  
Procedure: run perceptual hashing or embedding similarity on your 2500 images. Identify near-duplicates across train/calibration/test.  
Pass: duplicates are grouped and split-aware; leakage eliminated.  
Fail: offline AUC is inflated; rebuild splits.

E8.2 Manual review protocol test (small sample)  
Procedure: define “manual review” as an auditable process with consent/retention. Review only a small stratified set (false positives, false negatives, abstains).  
Pass: definitions are concrete, storage rules are clear, and feedback does not contaminate test sets.  
Fail: feedback loop risks invalidating evaluation.

E8.3 Data lineage enforcement test  
Procedure: implement tags: TRAIN, CALIBRATION, AUDIT, TEST-NEVER-TOUCH. CI fails if an item crosses boundaries.  
Pass: boundaries enforced.  
Fail: evaluation can no longer be trusted.

Outputs of Phase 8  
Dedup/leakage policy, label audit policy, feedback loop policy, and CI enforcement of data lineage.

J. What “conclusive” means in practice: mandatory gates you should implement

Gate G1 (Pixel Equivalence Gate): S1–S4 match golden (or within defined tolerances). If G1 fails, stop everything else.  
Gate G2 (Guardrail Gate): OOD abstain rate meets target (ideally 100% on your OOD golden set), and reason codes are correct.  
Gate G3 (Model Semantics Gate): patch grid shape and heatmap mapping are correct; pooling determinism holds.  
Gate G4 (Calibration Gate): calibrated model meets operating point on calibration set and does not catastrophically violate it on golden holdout.  
Gate G5 (Quantization Parity Gate): float vs TFLite parity holds at patch, pooled, and probability levels within tolerances; post-quant calibration completed.  
Gate G6 (Release Gate): evaluation battery (in-the-wild/time-split/cross-compression/transforms) meets minimum performance and max abstain rate.

K. Minimal parameter placeholders you should decide early

Choose and fill these in the contract file before you start coding tests:  

Which decode library is authoritative.  
Exact dtype/range convention.  
Exact resize kernel and rounding behavior.  
Normalization mean/std.  
Face detector model and initial confidence threshold.  
Multi-face policy (max vs largest) and face size threshold.  
Blur metric and threshold.  
Compression proxy and threshold.  
Pooling choice (top-k vs attention) and parameters.  
Calibration method (temperature vs Platt) and operating point constraint.  
Abstain band width and UI semantics.  
Quantization mode and TFLite delegate/thread config.  
Parity tolerances at each intermediate tensor.