# Verification history

## Weighted voting · 2026-10-15

- Active model: frozen layer-specific Weighted voting; SHA-256 `51fcba4aa45ba04c5d9e526332c10e21e7629e74cbc6702c6eccdef2263c88f8`.
- Rebuilt final fits use A development only. Existing A-development 10-fold OOF fixes voting weights, Platt parameters and maximum-Youden thresholds. The web application never fits or selects a model.
- Analysis/runtime parity: all 637 evaluation records, three layers, raw and calibrated probabilities (3,822 comparisons); maximum absolute difference 1.11 × 10⁻¹⁶.
- UI parity: all 634 records with at least one observed laboratory value; maximum absolute difference 1.11 × 10⁻¹⁶. Three all-blank records remain in batch analysis but are deliberately rejected by the interactive input validator.
- Completed features are rounded to 10 decimal places. Rotation Forest uses explicit fixed-order PCA contractions and rounds projections to 12 decimal places before the original fitted tree decisions, canonicalizing nullspace floating-point artefacts. All current analysis and deployment outputs were recomputed with this rule; no tree refit or clinical winsorization.
- Public test fixtures contain constructed examples only, never patient records.
- Automated regression tests: 13/13 passed in 13.348 seconds, including complete/sparse single/batch parity across batch sizes and Streamlit form-state tests. Fifteen constructed reference predictions matched exactly.
- Real Streamlit Cloud Linux deployment tested after reboot: one constructed single example and five constructed batch examples, all three raw/calibrated probabilities, thresholds, classes and batch order agree with local results. Maximum error 4.80 × 10⁻¹³ is due to the CSV `.12g` export precision. No patient data were uploaded.
- Earlier entries below describe historical model/interface versions, not the current model.

## Initial interface · 2026-09-26

- Automated tests: **11 / 11 passed**, final run 25.642 seconds; Python 3.13.9 / Streamlit 1.51.0.
- Source-project consistency: **15 predictions** across the three IMN layers matched saved expected values, maximum absolute error **1.11 × 10⁻¹⁶**. Patient fixtures remain outside this deployment folder.
- Integrity: the model, original inference helper and two original support modules match the recorded SHA-256 values.
- Runtime checks: complete and missing inputs; repeated/concurrent inference; single versus batch parity; fixed thresholds; composite formulae; aliases; CRP boundary parsing; empty, non-numeric, negative, infinite and malformed CSV rejection; spreadsheet-formula protection.
- Streamlit form checks: blank submission rejected, synthetic example predicts, results persist across reruns, invalid resubmission removes stale output, and clear resets the inputs.
- Browser checks: desktop hero, input form, main probability card, all three layer cards and six index displays inspected. A two-row synthetic CSV completed in the original row order, including one record with two missing fields and one censored CRP value; download control present.
- Narrow-layout check: the browser reported an effective viewport of **582 px** and a single-column result layout with no document-width overflow. The requested 390 px override did not take effect in the preview tool, so a true 390 px screenshot is **not** claimed.
- Local health endpoint: **OK**, bound to `127.0.0.1:8502`. No public deployment, cloud upload or remote Linux execution has been performed.
- Launcher: Windows PowerShell syntax validated. Fresh virtual-environment installation and macOS execution have not been exercised on this Windows host.

The original model, analysis outputs and figure folders were not modified.

## Interface 1.1 · 2026-09-27

- Renamed the interface to **IMN Insight**, including browser title, navigation badge, hero, footer, downloads, launcher and descriptive metadata.
- Replaced the blue sphere and platform with a local editable SVG: inflammatory cell, antibody shield and nutrient leaves / molecular motif. The three domains are labelled in English and linked with restrained lines; these are symbolic UI elements, not a biological pathway claim.
- The new graphic was inspected in the live desktop browser, and the SVG XML parsed successfully. The source contains no old sphere selectors or old product name.
- All **11 regression tests passed** in 24.340 seconds. The 15 reference predictions remain consistent to a maximum absolute error of **1.11 × 10⁻¹⁶**; no model, preprocessing or threshold changes.
- Updated deployment archive: `imn_insight_streamlit_deployment.zip` in the parent model-package folder. The previous archive is retained as a historical snapshot.
