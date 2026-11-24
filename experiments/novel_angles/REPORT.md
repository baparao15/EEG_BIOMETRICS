# What is novel enough to pursue in longitudinal EEG verification?

See the [plain-language summary](PLAIN_SUMMARY.md) and [claim-by-claim re-verification audit](CLAIM_AUDIT.md). The original analyses below are preserved; the audit distinguishes freshly reproduced scores from earlier preparation records and literature evidence.

## Prior results preserved

The [original-protocol rerun](https://app.devin.ai/attachments/b1d6a871-3dd9-4cec-a85f-bf932736fad6/eeg-cloud-feasibility.md) on ds007176 v1.0.1 reported **17.10% cross-session SVM EER** and **9.92% leave-one-session-out rank-average EER**. The [equal-budget eye-state enrollment pivot](https://app.devin.ai/attachments/14c8705e-4442-437a-a5b1-dd16db729107/eeg-cross-condition-results.md) reported a **19.4 pp** mismatch penalty in FRR, reduced to **5.5 pp** by mixed enrollment, and **2.4–2.6%** late-visit FAR versus the frozen threshold's **1%** calibration target. The [prior novelty review](https://app.devin.ai/attachments/2d51faa3-e223-4d40-aa67-2a439a48c98d/eeg-literature-novelty-check.md) noted that the mismatch itself was already known.

The Round 1 pilot used a lightweight preprocessing path because the verified notebook was not yet available. It uses checksum-verified v1.0.1 raw inputs and cosine templates, not the prior classifiers. Its median log-band feature RMSE against the 40 available recipe FIFs was 0.75 on shared channels. The following Round 2 notebook validation is separate from that comparison.

## Round 2: preprocessing validation and baseline checks

- **Near-equivalent against the committed reference.** Running the supplied, hash-pinned notebook with MNE 1.12.1, PyPREP 0.7.1, and PyWavelets 1.8.0 reproduced the channel order and retained events for all 40 recordings. Event-set Jaccard was 1.0 in all 40. Waveform correlation ranged from 0.9999999999999909 to 1.0. Waveform RMSE in standardized units was 8.06 × 10⁻¹⁰ median and 1.11 × 10⁻⁸ maximum; waveforms were not bitwise identical. Retained epochs agree exactly; the reference FIFs do not provide an independent pre-interpolation channel list against which to validate the notebook's interpolation log.
- **Cross-condition anchor exactly reproduced.** All [232 ds007176 FIFs](loop2-ds007176-preprocessing.json) passed preprocessing and have at least 45 retained epochs. The unmodified cross-condition kit scored all 116 claimant/role splits. With condition-aware frozen thresholds, the full-montage RBF SVM mismatch penalty was **19.4253 pp** FRR [13.4480, 25.7859] versus matched enrollment; mixed enrollment reduced the penalty to **5.5172 pp** [2.1262, 8.9751]. Late mixed-enrollment FAR was **2.3755%** on CE and **2.6068%** on OE. The generated contrast and policy CSVs matched the expected kit CSVs byte for byte; only line endings in the committed [contrast](loop2-cross-anchor-paired-contrasts.csv) and [policy](loop2-cross-anchor-policy-metrics.csv) copies were normalized.
- **Original scorer check, five-subject committed subset.** Notebook 07's full-montage RBF SVM reproduced all eight expected CS and LOSO fold EERs to six decimal places. This does not reproduce the 29-subject 17.10%/9.92% figures or the all-model suite. On the pinned CPU evaluation environment the all-model runner aborted on nonfinite output from 1D-CNN and BiLSTM-Attention despite finite model inputs. No scorer or data were modified to bypass that failure.
- **External cohort and eye-state order.** RestCog ds004148 v1.0.1 retained all 60 subjects with three complete CE/OE pairs after preprocessing ([360/360 FIFs](loop2-ds004148-preprocessing.json), at least 41 retained epochs per recording). The dataset specifies 50 Hz line frequency, used in the locally adapted notebook run. No RestCog acquisition timestamps were found in BIDS scans tables or in BrainVision new-segment markers. For the 29 ds007176 subjects, 44 of 116 paired visits have comparable acquisition timestamps: 43 CE-first and one OE-first, the latter at V1. Twelve late visits are CE-first and 17 have unknown order; no late OE-first subgroup can identify an order effect.
- **Dortmund feasibility.** The [ds005385 v1.0.3 documentation](https://github.com/OpenNeuroDatasets/ds005385/blob/6a558cd5852503e66df9ac7fdac2f3a7f4ed5f12/README.md) describes CE/OE before and after a cognitive block, 608 baseline participants, and 208 roughly five-year follow-ups. It has only two chronological visits and fixes CE before OE. The within-visit repeats could test fatigue or repeatability, but cannot supply a third calibration visit or counterbalance order; we did not use them as a substitute for the locked protocol.

### Matched probes on the verified ds007176 FIFs

Each arm consumed the same 24 retained windows per eye state, with 24 attempts per arm and 10 usable EEG seconds per attempt. Claimants used identical 160 s mixed enrollment across probe policies. The SVM used Notebook 07's feature bank and model but the new 160 s and 10 s budgets, not the anchor's budget. Differences are paired minus single, in percentage points, with 95% identity-bootstrap intervals. All FRR and FAR use per-claimant cutoffs frozen on separate calibration impostors.

| Scorer | Single → paired EER | Δ EER (pp) | Δ FRR (pp) | Δ FAR (pp) | Paired held-out FAR |
|---|---:|---:|---:|---:|---:|
| Cosine | 32.30% → 27.60% | −4.70 [−7.49, −2.23] | −13.58 [−20.64, −6.30] | +1.47 [+0.15, +3.09] | 4.42% [2.60%, 6.87%] |
| RBF SVM | 23.91% → 21.49% | −2.42 [−5.34, +0.60] | −5.06 [−10.24, +1.26] | +0.61 [−0.28, +1.78] | 3.53% [2.20%, 5.46%] |

The cosine scorer met the directional EER criterion on the verified inputs; the SVM did not. Both met the +2 pp FRR margin, but neither met the +0.5 pp FAR margin or the ≤1% held-out FAR upper bound. **No deployment go.** Paired cosine FAR was already 4.46% on unseen V2 impostors (late minus V2: −0.04 pp [−1.61, +1.41]); SVM showed the same pattern (3.97% at V2; late minus V2: −0.45 pp [−1.55, +0.61]). This is evidence of calibration-identity transfer failure without needing a later visit, not evidence that elapsed time caused it.

**Controls.** With cosine scoring, the paired EER gain persisted after removing alpha and renormalizing (−4.29 pp [−6.52, −2.53]) and on both six-channel panels (frontal −5.26 pp [−7.50, −2.59]; posterior −4.53 pp [−7.57, −2.06]). Neither SVM panel confirmed a directional gain. The signed CE−OE contrast beat the pooled within-state repeat contrast by 9.89 pp EER [5.31, 16.23], but is not a measured eye-onset or liveness effect. The cosine EER estimate favored pairing in the 12 known CE-first late visits (−5.95 pp) and the 17 with unknown order (−4.02 pp); this is descriptive, not a test of reversed order.

**Alignment.** Against 32.91% late EER for unaligned mixed enrollment and single-state probes, historical/same-visit CE alignment scored 32.54%/32.26%; both paired CIs for the difference cross zero. Their FARs were 4.50%/2.10%, respectively, and their FRRs were 69.83%/77.55% versus 72.84% unaligned. Same-visit OE reduced FAR by 2.06 pp [0.29, 4.43], but increased FRR by 12.10 pp [6.50, 17.82] and EER by 4.34 pp [0.06, 9.48]. The results do not establish that single-state alignment can replace mixed enrollment. The complete [aggregate results](loop2-ds007176.json) include paired alignment probes, reverse contrasts, and transfer endpoints.

### External replication on RestCog

The [locked external run](loop2-ds004148.json) used session 1 for enrollment (one session rather than two), session 2 for calibration, and session 3 for testing, with the same duration and identity separation. Differences are paired minus single; intervals and units are as above.

| Scorer | Single → paired EER | Δ EER (pp) | Δ FRR (pp) | Δ FAR (pp) | Paired held-out FAR |
|---|---:|---:|---:|---:|---:|
| Cosine | 33.28% → 28.14% | −5.14 [−6.65, −3.80] | −9.31 [−12.92, −6.00] | +0.35 [−0.15, +0.91] | 2.69% [1.83%, 3.92%] |
| RBF SVM | 26.37% → 24.64% | −1.73 [−3.46, −0.22] | −2.48 [−5.64, +0.29] | +0.45 [−0.09, +1.08] | 2.44% [1.78%, 3.35%] |

Both scorers met the directional EER and +2 pp FRR criteria. Both FAR point differences were below +0.5 pp, but neither interval's upper bound met that margin; neither held-out FAR upper bound met 1%. **No deployment go.** The held-out calibration-visit FAR already exceeded 1% for both paired scorers (2.51% cosine, 2.35% SVM). The cosine EER gain replicated across cohorts; the SVM estimate favored pairing in both but excluded zero only on RestCog.

**Controls and alignment.** The cosine EER gain persisted without alpha (−3.52 pp [−5.09, −2.03]) and on frontal (−3.14 pp [−4.72, −1.82]) and posterior (−4.62 pp [−6.29, −2.78]) panels, but not on either SVM panel. The signed CE−OE contrast beat the pooled repeat control by 10.69 pp EER [7.99, 13.76]. No ordering comparison was identifiable. Historical/same-visit CE alignment had 36.14%/36.64% late EER versus 37.91% for unaligned mixed enrollment; both EER contrast intervals include zero. Their FRRs were 81.48%/83.11% versus 88.23%, and FARs were 1.29%/1.45% versus 1.11%. OE alignment raised EER with both donor rules (43.34%/41.15%). The internal same-visit OE FAR reduction did not replicate: the external difference was +0.94 pp [+0.36, +1.60].

## Round 1 study design

29 subjects, 232 recordings. V0/V1 enrollment; V2 threshold fit on a different impostor identity group; compare held-out impostors at V2 and a later visit. The latter is **V3 for 24 subjects** and **V4 for five**, not two later measurements per person. Across four rotations, each subject serves once as claimant, reference cohort, threshold impostor, and scoring impostor; the roles are disjoint within each rotation. Default enrollment is 160 usable EEG seconds; attempts use 10 seconds. See [protocol and limitations](README.md#protocol). The sample supports exploratory contrasts, not a claim of absolute 1% risk control.

All rates are percentages. Brackets are 95% subject-bootstrap intervals conditional on the fitted thresholds, reference and split. Complete tables: [metric rows](results.csv), [paired contrasts](results-contrasts.csv), [follow-up contrasts](follow-up-contrasts.csv) and [diagnostics](diagnostics.json). FRR is measured at the frozen 1%-target cutoff, **not** at EER.

## Results: which interventions hold up?

| Late-visit method, 160 s enrollment | FRR % [95% CI] | FAR % [95% CI] | EER % [95% CI] |
|---|---:|---:|---:|
| Spectral mixed enrollment, single-state probes | 80.32 [70.11, 88.94] | 1.61 [0.37, 3.35] | 31.47 [26.01, 37.10] |
| Same, Z-norm | 84.48 [77.37, 92.60] | 1.59 [0.48, 3.33] | 30.75 [24.43, 36.77] |
| Same, T-norm | 86.93 [81.39, 92.17] | 2.58 [0.78, 5.33] | 31.85 [26.14, 37.73] |
| Same, S-norm | 80.75 [72.27, 88.44] | 2.14 [0.61, 4.46] | 30.46 [23.93, 36.49] |
| Same, separate CE/OE cutoffs | 79.45 [69.46, 88.22] | 1.71 [0.38, 3.55] | 31.47 [26.01, 37.10] |
| Covariance tangent, mixed | 86.06 [78.88, 92.89] | 1.03 [0.04, 2.95] | 34.64 [28.66, 39.92] |
| Covariance tangent, enroll CE, historical state alignment | 86.64 [79.89, 93.10] | 0.36 [0.00, 1.11] | 36.06 [29.60, 41.95] |
| Covariance tangent, enroll OE, historical state alignment | 89.37 [82.25, 95.69] | 1.29 [0.09, 3.62] | 36.78 [30.32, 41.96] |
| Covariance tangent, enroll CE, same-visit state alignment (partial coverage) | 91.82 [83.33, 97.92] | 1.25 [0.00, 4.28] | 32.56 [26.63, 38.13] |
| Spectral paired mean, one 5 s window per state | 76.15 [64.15, 87.65] | 1.53 [0.19, 3.39] | 27.52 [21.16, 34.49] |
| Spectral paired CE–OE contrast only | 93.39 [85.19, 98.56] | 0.58 [0.02, 1.88] | 38.12 [33.19, 43.68] |
| Spectral paired mean + contrast | 80.17 [68.60, 90.96] | 1.69 [0.43, 3.45] | 30.46 [24.13, 37.86] |
| Spectral same-state repeat contrast negative control | 97.13 [94.25, 99.71] | 1.15 [0.29, 2.21] | 50.00 [45.69, 53.74] |
| Approximate spectral slope, mixed | 88.36 [81.25, 94.97] | 1.33 [0.21, 3.43] | 41.37 [34.33, 48.34] |
| Imaginary alpha coherency, mixed | 88.36 [83.33, 93.04] | 0.60 [0.20, 1.11] | 30.60 [27.12, 33.47] |
| Imaginary alpha coherency, paired mean | 85.92 [80.46, 91.67] | 0.40 [0.08, 0.98] | 28.71 [24.72, 32.69] |

- **Cross-condition enrollment:** With single-state enrollment, late FRR increased **31.32 pp [18.24, 45.98]** for CE→OE relative to CE→CE, and **17.53 pp [4.89, 31.76]** for OE→CE relative to OE→OE. Mixed enrollment nearly equalized CE and OE FRR (79.02% vs. 81.61%, a 2.59 pp gap), but its pooled FRR was worse than CE-only in this cosine pilot. These directional comparisons share each enrollment method's pooled V2 threshold; they do not compare FRR at equal realized FAR.
- **Threshold transfer:** Mixed spectral FAR was **1.92% at V2 on unseen identities** and **1.61% later** (paired change −0.32 pp [−2.54, 1.34]); FRR increased 4.89 pp [−1.44, 10.42]. An operating-point miss was present before the later visit, so this run does not isolate temporal drift as its cause or reproduce worsening FAR with visit. No score normalization convincingly restored 1% at lower FRR. Per-state thresholds changed pooled FAR by just +0.10 pp [−0.35, 0.55] for mixed enrollment. Removing one V2 calibration impostor changed late **fold-specific** FAR by −0.17 to +2.21 pp. **15/29** held-out impostor identities had at least one false acceptance across repeated probes against multiple accounts; that descriptive event count is not a single-attempt FAR. The median claimant FRR was 95.83% (interquartile range 66.67–100%).
- **Alignment:** Historical CE alignment vs. unaligned mixed covariance gave +0.57 pp FRR [−6.11, 8.55] and +1.42 pp EER [−3.00, 6.37]; OE alignment gave +3.30 pp FRR [−1.15, 7.76] and +2.14 pp EER [−2.12, 6.63]. No equivalence or noninferiority margin was set. The corrected contemporaneous arm uses V3 donors for V3 probes and V4 donors for V4 probes, so it requires an independent reference acquisition at the same nominal visit. One rotation has no V4 donor; we did not score those probes (two genuine identities and one impostor identity; coverage 27/29 genuine and 28/29 impostor identities). The earlier apparent reduction in CE FRR does not hold under this correction, and partial coverage prevents a direct comparison with the full-cohort rows. Same-visit OE enrollment yielded 95.68% FRR and 35.49% EER.
- **Reactivity versus averaging:** The spectral CE–OE contrast beat the equal-duration CE–CE repeat-difference control by **11.88 pp EER [5.24, 18.54]**. It still had 93.39% FRR at the frozen cutoff, and adding it to the pair mean did not help (joint minus mean EER +2.94 pp [−2.16, 7.43]). The slope contrast also beat its repeat control (−8.05 pp [−15.69, −1.52]); the imaginary-coherency interval included zero. These exploratory contrasts match duration per attempt, not total attempt counts: the cross-state arm has 24 attempts and the CE-only repeat arm has 12. They show information in a resting-state contrast, not a measured eye-onset response, a source-localized trait or liveness. Eye-state, order, posture and artifact differences may contribute. An OE repeat control and independent, counterbalanced repeats are needed.
- **Update versus freeze:** On the held-back half of the late visit, updating after accepted early attempts cut FRR from 80.17% to 75.29% (**−4.89 pp [−8.77, −1.72]**) but raised FAR from 1.90% to 3.37% (**+1.47 pp [0.21, 3.00]**); EER was essentially unchanged (+0.10 pp [−0.35, 0.86]). This is a genuine-versus-security trade-off under ordinary impostors, not evidence of targeted poisoning or multi-visit adaptation. Do not recommend unsupervised updates at this threshold.
- **Stability and demographics:** V0–V1 same-person template agreement correlated with later claimant FRR (Spearman ρ = −0.45, nominal unadjusted p = .014; n = 29). This is a hypothesis for identifying people who need more enrollment or an alternative factor, not a validated risk score. The metadata include age and sex, but 29 participants cannot support low-FAR fairness claims. Only three participants have both V2 and late-visit timestamps and only one has all four visits stamped; ordinal visits cannot identify a month-by-month aging curve.

### Enrollment budgets and channel panels

Each cell is late FRR / FAR / EER (%), for mixed enrollment and single-state trials. The full CSV includes confidence intervals and the CE/OE breakdowns.

| Channels | 40 s | 80 s | 160 s |
|---|---:|---:|---:|
| Four frontal | 87.36 / 1.45 / 37.07 | 89.37 / 1.25 / 36.06 | 87.79 / 1.85 / 34.88 |
| Eight frontal | 88.07 / 1.45 / 37.07 | 87.79 / 1.27 / 36.64 | 86.78 / 1.59 / 35.04 |
| 16 spread | 86.06 / 1.83 / 34.84 | 83.76 / 1.73 / 34.13 | 83.05 / 1.59 / 33.08 |
| All | 84.91 / 1.43 / 33.91 | 82.76 / 1.39 / 32.96 | 80.32 / 1.61 / 31.47 |

Full-channel longer enrollment helped descriptively, but more frontal channels were not consistently better; the panel grid does not simulate new hardware or establish a minimum viable budget.

### Follow-up loop: what made paired probes promising?

The single-state and paired-state spectral methods used the same mixed enrollment and spent 10 s of usable EEG per attempt. The original **5.71 pp** EER improvement compared paired probes from only the first 12 retained windows per state with single-state probes from the first 24. In the corrected comparison, both arms consume the exact same first 24 retained windows per state once each (24 attempts per arm). Across-state averaging (one 5 s window per state) reduced pooled EER from **31.47% to 27.52%**, a **3.95 pp improvement [0.84, 7.45]**. Frozen-threshold changes were −4.17 pp FRR [−8.84, 0.22] and −0.08 pp FAR [−1.01, 1.02]; both intervals include zero. The gap to OE alone (28.74% EER) is smaller and was not separately tested. State acquisition and switching overhead are **not** matched.

Removing the alpha feature **and re-normalizing power without alpha** changed EER by +0.21 pp [−5.21, 5.87]. This does not establish the mechanism; the effect need not come from alpha alone. Paired-probe Z, T and S normalization did not provide a clear gain over raw scores. The main shortlist therefore favors **paired-state averaging as a reproducible control** over a new normalization algorithm or an alpha-reactivity-only authenticator.

## Round 3: independent replication search

**No third cohort was verified to meet the locked design.** BIDS session labels alone are not evidence of separate recording days: the completed RestCog replication has three recording sessions but its enrollment and calibration were [90 minutes apart](https://doi.org/10.1093/cercor/bhaf034). It remains an independent dataset replication of the scoring protocol, not a three-day replication. The locked probe analysis needs at least 24 retained nonoverlapping 5 s windows (120 usable seconds) **per state** at calibration and test; one-session enrollment needs 32 per state for all controls (160 usable seconds). Another day's windows, task periods, or overlapping crops cannot make up the deficit.

| Candidate | What was checked | Why it was not a locked third replication |
|---|---|---|
| [WBCIC-SHU 2C](https://doi.org/10.25452/figshare.plus.22671172.v5) | ZIP directory: 51 subjects, three sessions each, one signal and one event BDF per session. All 153 event BDFs were read from Figshare byte ranges and passed ZIP CRC checks; 42 subjects have three distinct calendar dates in their event headers, and 151 sessions start with consecutive codes 7 and 8 before motor-imagery markers. Signal prefixes were read for the modified test below. | The [recording protocol](https://doi.org/10.1038/s41597-025-04826-y) allocates just 60 s per eye state per day. Longer inter-marker gaps do not establish longer valid rest; prescribed rest is half the locked probe minimum even before rejection. |
| [Dortmund ds005385](https://openneuro.org/datasets/ds005385/versions/1.0.3) | Pinned file inventory: 208 subjects with CE/OE before and after the task block at both sessions. All 1,248 selected recordings were processed or classified as failures. | Only two chronological visits, approximately five years apart. Used only for the two-day sensitivity analysis below. |
| [L-FAME](https://huggingface.co/datasets/L-FAME-Dataset-Benchmark/L-FAME) | Repository tree: 74 subject directories; 44 have both pre- and post-meditation session directories and 30 have pre only. Sample EEG/marker files listed. | At most two visits. |
| [COG-BCI](https://doi.org/10.5281/zenodo.6923638) | Publication and public archive layout. | Three weekly dates, but only 60 s per eye state at each pre/post block; pooling both would change the within-session protocol and still cannot meet the 160 s one-session enrollment minimum. Signals not audited. |
| [BMT_EEG](https://doi.org/10.21227/ayfc-hc17), [BED](https://doi.org/10.5281/zenodo.4309472), [CSTE](https://ieee-dataport.org/), [TDBRAIN](https://doi.org/10.1038/s41597-022-01409-z) | Access pages and study methods; no signal audit. | BMT_EEG is by request; the preprint (23 participants vs. the archive description's 20) describes three dates but only 60 s per rest state. BED is access-restricted despite reported three dates and both states. CSTE lists only two sessions and is login-gated. TDBRAIN requires registered access; three complete EO/EC days per person were not verified. |
| [Roma Tre longitudinal study](https://doi.org/10.1109/TIFS.2017.2778010), [Höller et al.](https://doi.org/10.1049/bme2/5158377), [12-month nine-subject study](https://doi.org/10.1038/s41598-025-13614-y) | Publications and access searches; no signal audit. | Roma Tre reports 45 participants with ≥5 dates and 4 min per eye state, but no public raw release was located. Höller reports three days of cognitive tests and no public release. The nine-subject study does not release raw data. |
| [PhysioNet auditory EEG](https://physionet.org/content/auditory-eeg/1.0.0/), [EEG Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/), [ds007987](https://openneuro.org/datasets/ds007987), Jülich repeated EEG and TMS/TBS studies | File listings where available and acquisition methods; no complete-pair audit. | PhysioNet's repeated trials/runs are not three separate eye-state visits; ds007987 labels conditions within one recording visit. Screened Jülich and stimulation datasets did not establish both resting states on three separate days. |

### Dortmund: two-day sensitivity analysis with locked scoring

Enrollment used baseline pre-task rest, calibration used baseline post-task rest, and testing used follow-up pre-task rest approximately five years later. CE preceded OE. The Loop 2 budgets, scorers, bootstrap and identity separation are unchanged, but enrollment and calibration share a date, and calibration follows a task. This is not a three-day replication. The hash-pinned notebook ran unchanged after EDF loading (dropping Status and applying standard 10–20 channel locations), with 50 Hz line frequency. The adapter aligns the order of identical electrode sets by name before feature extraction.

**128 of 208 participants were eligible.** Of 1,248 recordings, 1,196 passed notebook preprocessing and 52 failed (all single-component ICA rejections under the notebook's 0.99 criterion; 40 participants affected). A separate 57 readable recordings were below the retained-window minima (53 enrollment, zero calibration, four test). That excluded 40 additional participants; 12 of those with an ICA failure also had a short block. The scorer uses the 768 recordings from participants with all six eligible blocks.

Later-visit results, in % for arm rates and percentage points (pp) for paired-minus-single differences. Intervals are 95% subject-bootstrap CIs. [Aggregate output](loop3-dortmund.json).

| Scorer | EER single → paired (%) | Δ EER (pp) | FRR single → paired (%) | Δ FRR (pp) | FAR single → paired (%) | Δ FAR (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Cosine | 32.27 → 24.54 | −7.73 [−9.21, −6.40] | 83.36 → 71.79 | −11.57 [−15.31, −7.70] | 1.32 → 1.84 | +0.52 [+0.23, +0.82] |
| RBF SVM | 24.93 → 23.00 | −1.93 [−3.20, −0.42] | 71.93 → 68.03 | −3.91 [−6.96, −0.65] | 1.69 → 2.08 | +0.39 [+0.08, +0.73] |

Both scorers met the directional EER and FRR criteria, but neither met the FAR margin or the ≤1% held-out FAR upper bound (paired upper bounds: 2.30% cosine, 2.69% SVM). Paired-probe FAR on unseen identities was already 1.64% cosine and 1.63% SVM within the baseline date. The result extends the comparison to a long test interval but measures only the recipe-eligible subset.

### WBCIC-SHU: modified-duration three-day test

The [protocol was prespecified](WBCIC_PROTOCOL.md) before scoring. The first 60 s per eye state were processed with the hash-pinned notebook. The probe and enrollment window counts were both scaled by one quarter: six retained 5 s windows per state at calibration/test and eight total for enrollment (four of each state for the mixed template). Both arms use the same probe windows in six 10 s attempts; acquisition was OE-first and elapsed time between eye states was not matched. The three chronological dates serve as enrollment, calibration, and test; the identity splits, frozen thresholds, bootstrap and decision criteria otherwise follow Loop 2.

**10 of 51 participants were eligible.** Preparation generated 146 readable blocks, including 60 for the eligible cohort. The exclusions were 31 participants with signal/event header timestamp mismatches (50 sessions), eight with fewer than three distinct dates after marker screening, and two with a missing or duplicate eye-state marker (two sessions). The prespecified exclusions were not relaxed. All six eligible blocks per participant met their retained-window minima.

The original run checked event-member CRCs but did not checksum the signal prefixes. A fresh rerun checked full ZIP member size and CRC for all 123 signal BDFs from participants who passed event-marker and date screening, including all 30 signal BDFs used in scoring. It reproduced the preparation manifest and aggregate results exactly.

Later-date results, in % for arm rates and percentage points (pp) for paired-minus-single differences. Intervals are 95% subject-bootstrap CIs. [Aggregate output](loop3-wbcic.json).

| Scorer | EER single → paired (%) | Δ EER (pp) | FRR single → paired (%) | Δ FRR (pp) | FAR single → paired (%) | Δ FAR (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Cosine | 37.29 → 30.83 | −6.46 [−9.28, −3.21] | 64.58 → 50.00 | −14.58 [−28.97, −3.95] | 11.67 → 13.75 | +2.08 [−3.97, +8.87] |
| RBF SVM | 40.00 → 43.33 | +3.33 [−5.22, +13.01] | 64.58 → 67.50 | +2.92 [−4.58, +10.42] | 10.83 → 10.21 | −0.63 [−7.18, +6.16] |

Cosine met the directional EER and FRR criteria; SVM met neither. Neither met the FAR margin or ≤1% held-out FAR upper bound (paired upper bounds: 25.24% cosine, 23.19% SVM). On the calibration date, paired-probe FAR on identities not used to set thresholds was already 15.63% cosine and 16.25% SVM. These are sensitivity results on a small selected subset with only two calibration identities per rotation, not an unchanged replication or evidence that the durations are interchangeable.

## Fresh novelty recheck (24 November 2026)

**Partially covered; the full benchmark was not found in the sources screened.** The [standalone literature review](NOVELTY_REVIEW.md) records the component verdicts, closest papers and DOI links, exact search queries, citation trails, and limitations from an independent pass.

- **Published:** pooling EO and EC samples (Lai); combining both states as an EC−EO representation (Choi); multi-paradigm, cross-session authentication (Rakshe); and later-session verification (Maiorana). The Rakshe code checked stacks separate task epochs; it does not construct one paired EO+EC probe from the same windows used by the comparison arm.
- **Published in open-set identification; partial precedent for our 1:1 endpoint:** [the 2026 M3CV study](https://doi.org/10.3390/s26175471) calibrates rejection on development unknowns and measures realized false-positive identification on different final unknowns. It also analyzes session transfer. Its max-score rejection across enrolled identities is not per-claimant verification FAR. [The cross-session spectral study](https://doi.org/10.3390/s26154811) instead sweeps blind-test scores retrospectively for verification and explicitly does not fix an operational cutoff.
- **Not found in the screened material:** later-session 1:1 verification directly comparing EO+EC averaging to EO+EO/EC+EC averages from the **same retained windows**, with the same usable EEG time and attempt count, and reporting EER plus FRR/FAR at frozen thresholds calibrated on disjoint identities. The benchmark is the claim; state fusion, fixed window lengths, and threshold calibration are not.

The review keeps paired probes ranked first as a benchmark, not as a new fusion technique or a proven authenticator. Search coverage was incomplete (including abstracts only for some conference work and sparse indexing of recent preprints). The full-budget external three-day replication gap and unmet FAR criteria remain.

## Round 3: original novelty screen

**Verdict: related work is published; the exact endpoint was not found.** Combining, pooling, or contrasting EO and EC in biometric recognition is not new. Neither is testing multi-paradigm verification. In the material screened, we did not find an earlier comparison that simultaneously (1) averages one EO and one EC segment in each **verification attempt**, (2) compares against single-state attempts made from the **same windows and with the same total usable duration and attempt count**, and (3) tests later recordings with thresholds frozen on separate calibration identities and unseen test impostors. That is a differentiated **benchmark**, not a priority claim about EO/EC fusion or evidence that this design is absent from unindexed work.

| Closest publication | What it does and what differs |
|---|---|
| Lai et al. (2019), [10.1155/2019/7895924](https://doi.org/10.1155/2019/7895924) | Closed-set CNN identification. The combined REO+REC dataset pools one-second samples and doubles test samples (6,540 vs. 3,270), rather than fusing segments in each attempt; no external impostor group. |
| Choi et al. (2019), [10.1109/IWW-BCI.2019.8737254](https://doi.org/10.1109/IWW-BCI.2019.8737254) | Combines states as an EC−EO PSD difference and compares to EC and EO in repeated 10 s state recordings; per-subject 5×5 cross-validation, not an equal-budget paired **mean** and frozen later-visit threshold. |
| Kim and Kim (2019), [10.1109/ACCESS.2019.2917918](https://doi.org/10.1109/ACCESS.2019.2917918) | Concatenates quadrant connectivity features for verification across EO, EC and waiting conditions. The concatenation spans scalp regions rather than resting states. |
| Zhong et al. (2020), [10.1145/3444884.3444889](https://doi.org/10.1145/3444884.3444889) | EO/EC identification across three runs with at least two weeks between runs; trains on two runs and reports held-out identification, not matched 1:1 verification at a frozen operating point. |
| Maiorana (2021), [10.1016/j.patrec.2021.01.004](https://doi.org/10.1016/j.patrec.2021.01.004) | Open-set, single-/multi-protocol enrollment and cross-task verification including CE/OE; expressly uses one 5 s frame per probe and fuses **electrode** scores, not states within the attempt. |
| Rakshe et al. (2026), [10.1186/s40708-026-00337-7](https://doi.org/10.1186/s40708-026-00337-7) | Cross-session single- vs multi-paradigm authentication including EO and EC among six paradigms, using overlapping 2 s epochs. Published overlap with the broader claim; accessible methods/abstract evidence did not establish the exact same-window, same-duration two-state probe comparison. |
| Goldstein, phase/coherency preprint (2026), [10.64898/2026.10.26.761764](https://doi.org/10.64898/2026.10.26.761764) | Same-state versus cross-state enrollment–probe matching, band fusion and duration comparisons. EO and EC are conditions in Dortmund rather than fused states in a probe; the study reports rank/AUC. |
| Goldstein, time-odd geometry preprint (2026), [10.64898/2026.09.08.749160](https://doi.org/10.64898/2026.09.08.749160) | Longitudinal identity matching including ds007176 and fixed alpha–beta fusion; frequency-band fusion is not within-attempt eye-state fusion. |
| Cross-session controls | Multiple-training-session spectral verification [10.3390/s23042057](https://doi.org/10.3390/s23042057) is EO-only; target-session calibration [10.3390/s26154811](https://doi.org/10.3390/s26154811) consumes target-session data; M3CV [10.1016/j.neuroimage.2022.119666](https://doi.org/10.1016/j.neuroimage.2022.119666) offers multi-task/session challenges; large-scale multisession evaluation [10.1109/TBIOM.2026.3666044](https://doi.org/10.1109/TBIOM.2026.3666044) studies test size, intervals and acquisition device. None established this budget-matched paired-eye-state endpoint in the material checked. |

Targeted web and domain searches (23 November 2026) were followed by DOI/methods checks, selected Semantic Scholar forward citations from Maiorana (DOI `10.1016/j.patrec.2021.01.004`), and reference/title chasing from both 2026 preprints. OpenAlex returned HTTP 429; Google Scholar domain results were sparse. This was not systematic screening, and the full protocol was not available for every hit. Exact search strings used to screen the central question and candidate datasets (domain filters shown separately):

```text
EEG biometrics combining eyes-open eyes-closed "test" multi-state fingerprint feature fusion Maiorana Eigenbrains
EEG biometric authentication "eyes closed" "eyes open" probe "equal" fusion
EEG identification "eyes open" "eyes closed" "combined" "both" "test sample" biometric
EEG biometric eye state "feature fusion" "closed" "open" concatenate resting probe
multi-task EEG biometrics fusion multiple tasks one probe equal time identification arxiv biorxiv
EEG biometrics "eyes open" "eyes closed" "concatenated" identification trial cross session
"EEG" "eyes open" "eyes closed" pairing mixed probe same duration biometric identification
EEG biometric "eyes open" "eyes closed" "fusion" verification  [scholar.google.com]
EEG biometrics "eyes open" "eyes closed" combined probe authentication  [ieeexplore.ieee.org]
EEG biometric verification eyes open eyes closed paired probes equal duration  [arxiv.org]
EEG identification eyes open eyes closed fused probe equal duration  [biorxiv.org]
"Biometrics Based on Single-Trial EEG" 10 s difference EC EO author full text
"Arrangements of Resting State Electroencephalography" "REO + REC" pooled segments concatenated 2.2
site:pmc.ncbi.nlm.nih.gov/articles/PMC6589242 "REO + REC" "samples" "seconds"
"Learning deep features for task-independent EEG-based biometric verification" "averaging" "probe" Maiorana
site:iris.uniroma3.it "PRL_Deep_EEG_2021_Online.pdf" "enrolment" "verification" different tasks
"Polar Geometry of Time-Odd EEG Dynamics" "Longitudinal EEG identification" references biometric pairing
"s40708-026-00337-7" "multi-paradigm" "feature vectors" session evaluation
"10.1145/3444884.3444889" eyes open eyes closed runs identification "combined"
open EEG dataset repeated sessions longitudinal "eyes open" "eyes closed" resting 3 4 visits OpenNeuro Zenodo
longitudinal EEG "eyes open" "eyes closed" six sessions data public EEG cognitive training control
EEG test retest "Eyes Closed" "Eyes Open" "3 visits" data "sessions" electroencephalography dataset
EEG "3 minutes" eyes open eyes closed "three sessions" open dataset BrainVision
EEG biometrics "eyes-open" "eyes-closed" "five separate recording days" dataset Abdullah
"BMT_EEG" "eyes closed" "minutes" sessions
"CSTE" EEG "eyes" sessions minutes 10 subjects
"TDBRAIN" "three sessions" raw EEG dataset session
"COG-BCI" "eyes closed" "eyes open" resting 3 weeks 3 minutes EEG dataset data file
site:pmc.ncbi.nlm.nih.gov/articles/PMC11930978 "eye-opening" "marker" 7 8 BDF
Maiorana Campisi longitudinal EEG biometric database 45 subjects five sessions download raw public Roma Tre EEG database
"Resting State EEG-Based Biometric System Using Concatenation of Quadrantal Functional Networks" eyes open eyes closed test 10.1109/ACCESS.2019.2917918
"BED: Biometric EEG dataset" "eyes closed" "eyes open" resting separate sessions times duration
"Test-retest reliability of resting-state EEG intrinsic neural timescales" eyes closed eyes open multiple dates dataset
```

## Ranked novelty and prior-art screen

Scores reflect the specific claim written in each row, not the broad topic. Novelty: **1** established, **2** incremental, **3** specific extension, **4** plausible differentiated journal question, **5** no direct precedent after a comprehensive review. Evidence: **1** concept or metadata only, **2** exploratory/indeterminate, **3** within-cohort result with an informative control, **4** replicated independently, **5** prospective validation. This was a targeted search, not a systematic review; no angle earns 5 for novelty or evidence. Publishability is a judgment about the next paper, not a prediction of acceptance.

| Rank | Specific angle | Novelty / 5 | Evidence / 5 | Replication | Publishability and closest work |
|---:|---|---:|---:|---|---|
| 1 | **Equal-budget paired probes, separating state averaging from the state contrast under a frozen threshold.** Combine mixed enrollment, single- vs dual-state probes, and contrast-only and repeat controls. | 4 | 4 | Cosine: ds007176, RestCog, and both modified protocols. SVM: RestCog and Dortmund only. | **Strongest journal direction as a benchmark.** The cosine EER gain is consistent; the SVM evidence is mixed, and no FAR criteria were met. [Maiorana 2021](https://doi.org/10.1016/j.patrec.2021.01.004), [Eigenbrains and Eigentensorbrains](https://doi.org/10.1016/j.neucom.2015.07.005), [EC vs visual stimulation](https://doi.org/10.1186/s40708-021-00142-4), and [five-year eye-state sensitivity](https://doi.org/10.3389/fnagi.2026.1885392) narrow novelty to the combination of budget, probes, controls and operational protocol. EO/EC pooling and subtraction are published; no match was found for the full benchmark (see Round 3), and an unchanged external three-day replication remains outstanding. |
| 2 | **Separate identity-transfer error from time-transfer error** using the same unseen impostors at calibration and later; audit dependence on calibration identities and report identity-level burden. | 4 | 4 | Both scorers on ds007176, RestCog, and both modified protocols | **Strong companion result.** All four cohorts missed the FAR target on unseen identities at calibration; aging is not necessary to explain the miss. [Identities, Not Pairs](https://arxiv.org/abs/2610.35277) studies dependence in impostor pairs outside this EEG case; [cross-session EEG calibration](https://pmc.ncbi.nlm.nih.gov/articles/PMC13469576/) uses limited target-session calibration, a different protocol. |
| 3 | **Equal-budget mixed-enrollment benchmark**, including state-specific outcomes under frozen thresholds and never-seen impostors. | 3 | 3 | ds007176 anchor only | **Medium–high as the comparator, not a new phenomenon.** The anchor exactly reproduced the earlier FRR penalties. Mismatch and task-invariant representations are prior art, including [Maiorana 2021](https://doi.org/10.1016/j.patrec.2021.01.004) and the [cross-state phase/coherency preprint](https://doi.org/10.64898/2026.10.26.761764); [time-odd geometry](https://doi.org/10.64898/2026.09.08.749160) explicitly includes ds007176. |
| 4 | **Enrollment stability as a triage signal**, crossed with budget and an alternative factor rather than unconditional FRR optimization. | 3 | 2 | Round 1 ds007176 only | **Medium.** Template agreement–FRR association is useful for designing prospective abstention, not yet a rule. [EEG artifact/stability analysis](https://doi.org/10.1049/bme2/5158377) and [age, sex and pathology stability study](https://doi.org/10.1109/TIFS.2018.2854728) are nearby. Test an enrollment-only reliability rule on fresh subjects; audit access impact. |
| 5 | **Cross-condition edge deployment**: enrollment seconds × montage × trial state, with an external real-headset and artifact control. | 3 | 2 | Round 1 ds007176 only | **Medium as validation.** [Analysis of effective channel placement](https://doi.org/10.1109/IECBES.2010.5742249) already tests sparse CE/OE, and [large-scale multisession evaluation](https://arxiv.org/html/2501.17866v2) reduces to consumer-style montages. Our grid adds frozen operating-point behavior, not a headset claim. |
| 6 | **Accepted-probe update** under frozen thresholds, evaluated for natural FAR inflation and a potential verified-update fallback. | 3 | 3 | Round 1 ds007176 only | **Medium as a negative control.** FRR/FAR trade-off survives paired bootstrap, but only within one later visit. [EEG model adaptation](https://doi.org/10.1109/TPAMI.2007.1012) and [long-run template reinforcement](https://arxiv.org/html/2501.17866v2) predate it. Next: independent authenticated updates across multiple timepoints, not further unchecked self-updates. |
| 7 | **Historical cohort normalization and known-state thresholds**; combine with fixed enrollment budgets, include negative results. | 3 | 2 | Round 1 ds007176 only | **Low as a standalone paper.** Z/T/S are conventional in [speaker biometrics](https://www.isca-archive.org/interspeech_2019/nautsch19b_interspeech.pdf); the channel/eye-state transfer test is the extension. Neither this nor known-condition cutoffs reliably fixed operating-point error. |
| 8 | **Single-condition covariance alignment** using historical independent donors vs. extra contemporaneous acquisition. | 2 | 2 | ds007176 + RestCog; inconsistent | **Low standalone.** [Session-wise Riemannian brainprint alignment](https://doi.org/10.21203/rs.3.rs-8802883/v1) and [tangent-space transfer](https://doi.org/10.3389/fnhum.2022.1049985) cover the core idea. Results here were inconsistent and didn't establish that single-state can replace mixed. |
| 9 | **Separate neural versus measurement signatures** via preprocessing, channel location and reference stress tests. | 2 | 2 | 40 ds007176 reference FIFs; no causal test | **Medium as a safeguard, not a claim.** Verified vs. lightweight processing changed features in the 40-recording overlap; we did not measure the causal contribution of ocular, muscle or amplifier artifacts to identity. [Noise reduction vs. signature preservation](https://doi.org/10.1038/s41598-026-36840-4) and [artifact/within vs. cross-session work](https://doi.org/10.1049/bme2/5158377) already address related questions. |
| 10 | **Cancelable paired templates**: utility, cross-key linkability and revocation under state shift. | 3 | 1 | Not tested | **Medium only if a new threat-model contribution is demonstrated.** [Cancellable EEG template designs](https://arxiv.org/abs/2203.16730) and [deep cancellable EEG biometrics](https://doi.org/10.1109/TIFS.2024.3369405) already exist. No transformation, attack or privacy guarantee was evaluated here. |
| 11 | **Randomized eye-state liveness challenge** vs. equal-time resting controls. | 3 | 1 | Not tested | **Not supported here.** Prior work covers [blink-based EEG authentication](https://www.mdpi.com/1424-8220/18/2/335) and [eye closure plus visual challenge](https://doi.org/10.1186/s40708-021-00142-4). This dataset has separate rest recordings and no liveness ground truth. A prospective, randomized, timed consent-based study would be a separate task. |
| 12 | **Calendar aging and demographic heterogeneity.** | 2 | 1 | Round 1 ds007176 only | **Not supported here.** Ordinal aging was inconclusive, timing completeness is poor and subgroup tails would be too sparse. [Longitudinal Evaluation of EEG-Based Biometric Recognition](https://doi.org/10.1109/TIFS.2017.2778010) and [demographic stability effects](https://doi.org/10.1109/TIFS.2018.2854728) already cover the general questions. |
| 13 | **1/f traits or sensor-level phase/coherency as the main novelty.** | 1 | 2 | Round 1 ds007176 only | **Low.** Cross-eye-state aperiodic fingerprinting is explicit in [Demuru and Fraschini](https://doi.org/10.1016/j.compbiomed.2020.103748). [The phase/coherency preprint](https://doi.org/10.64898/2026.10.26.761764) and [time-odd geometry](https://doi.org/10.64898/2026.09.08.749160) cover longitudinal representations closely related to ours. Useful comparators; the short-window approximations here are not direct replications of either approach. |

## Round 4: reducing false accepts at the same EEG budget

The [protocol](ROUND4_PROTOCOL.md) was committed before new later-visit scores were examined. This follow-up uses the same enrollment windows, the same 10 usable seconds per attempt, the same identity rotations and the same held-out visits as Round 2; only scoring or thresholding changes. These are new analyses on previously used cohorts, not new participants or prospective validation. The rerun reproduced both cosine baselines' EER, FRR and FAR point estimates and intervals exactly at both visits in both cohorts. Aggregate results: [ds007176](round4-ds007176.json) (29 people) and [RestCog](round4-ds004148.json) (60 people).

| Cohort / method | Later EER | Later FRR | Later FAR | Upper 95% FAR bound | FAR change vs paired mean, pp [95% interval] | Unseen FAR at calibration visit |
|---|---:|---:|---:|---:|---:|---:|
| ds007176 / single-state mix | 32.30% | 81.57% | 2.95% | 4.50% | — | 3.52% |
| ds007176 / paired mean | 27.60% | 68.00% | 4.42% | 6.87% | Reference | 4.46% |
| ds007176 / worst-state agreement | 29.72% | 73.60% | 3.37% | 5.35% | −1.05 [−2.84, +0.55] | 4.00% |
| ds007176 / reactivity penalty | 28.16% | 75.29% | 3.26% | 5.20% | −1.16 [−2.68, +0.00] | 3.85% |
| ds007176 / repeatability weights | 27.67% | 66.99% | 4.66% | 7.31% | +0.24 [−0.15, +0.75] | 4.68% |
| ds007176 / identity-maximum cutoff | 27.60% | 74.78% | 2.75% | 4.40% | **−1.67 [−3.18, −0.85]** | 2.81% |
| RestCog / single-state mix | 33.28% | 85.61% | 2.34% | 3.28% | — | 2.20% |
| RestCog / paired mean | 28.14% | 76.30% | 2.69% | 3.92% | Reference | 2.51% |
| RestCog / worst-state agreement | 30.73% | 80.38% | 2.00% | 2.60% | −0.68 [−1.59, +0.07] | 2.12% |
| RestCog / reactivity penalty | 28.04% | 80.64% | 2.22% | 3.35% | −0.47 [−1.15, +0.10] | 2.21% |
| RestCog / repeatability weights | 28.18% | 76.35% | 2.77% | 4.03% | +0.08 [−0.02, +0.21] | 2.51% |
| RestCog / identity-maximum cutoff | 28.14% | 84.72% | 1.44% | 2.35% | **−1.24 [−1.75, −0.90]** | 1.22% |

**Best new result: calibrating on each impostor's highest score reduced false accepts in both cohorts, at the cost of rejecting more genuine attempts.** Compared with paired mean, FRR rose by **6.79 pp [4.97, 8.84]** in ds007176 and **8.42 pp [6.67, 10.40]** in RestCog. EER did not change because the score is identical; only its acceptance cutoff changes. With this small calibration pool the rule puts the cutoff just above the largest observed calibration score for each claimant. This shows how the operating point responds to stricter calibration, not that this cutoff is uniquely better than any equally strict cutoff, nor that it guarantees 1% FAR. The decrease in FAR was the only one with an upper change bound below zero in both cohorts. The ds007176 reactivity interval ends slightly above zero before rounding. Repeatability weights did not reduce FAR in either cohort at the point estimate.

**None met the ≤1% held-out FAR upper-bound gate; no variant qualified for Dortmund/WBCIC-SHU replication.** Those earlier results are unchanged. Every variant also exceeded 1% on unseen people at the calibration visit. The study is still a benchmark for discrimination and the limits of frozen thresholds, not a deployable biometric or a liveness test.

### Reranked Round 4 screen

| Rank | Idea | Novelty / 5 | Evidence / 5 | Journal value after Round 4 |
|---:|---|---:|---:|---|
| 1 | Distinct-identity risk control with abstention | 3 | 1 | Best next design question: determine when there are too few independent impostors for the intended claim. Untested. |
| 2 | Identity-maximum calibration | 3 | 4 | Strongest new result as a companion analysis: replicated FAR/FRR trade-off, not a new authenticator. |
| 3 | Cross-dataset frozen-cutoff transport | 3 | 1 | Distinct operational endpoint; start with a compatible montage and source-only scoring scale. Untested. |
| 4 | Worst-state agreement | 2 | 2 | Useful comparator, but the FAR change was inconclusive. |
| 5 | Enrollment repeatability weighting | 2 | 2 | No observed FAR gain; low standalone value. |
| 6 | Posterior alpha reactivity penalty | 2 | 4 | FRR worsened versus paired mean in both cohorts; no conclusive FAR gain. Prior EEG reactivity authentication narrows novelty further. |
| 7 | Sequential decisions at a fixed time budget | 2 | 1 | Useful study if preplanned at transaction level with dependent attempts, not a window-level FAR substitute. Untested. |
| 8 | Per-window quality weighting | 2 | 1 | Incremental; needs a preplanned quality measure not based on identity. Untested. |
| 9 | Wrong-person competition with independent reference profiles | 2 | 1 | Incremental alternative to historical score normalization, with clear speaker-biometric precedents. Untested. |
| 10 | Repeatability-aware left/right EEG contrasts | 1 | 1 | Existing asymmetry biometrics; potential control only. Untested. |

The [DOI-backed screen](ROUND4_LITERATURE.md) maps the closest methods to each question. In particular, [Choi et al. 2018](https://doi.org/10.9718/JBER.2018.39.1.43) already used EO/EC alpha changes for EEG authentication. Round 4 does not support a new claim about reactivity-based identity or a new first-ever algorithm. **Overall ranking after Round 4:** the controlled paired-probe benchmark and identity-vs-time comparison remain first and second; identity-maximum calibration belongs with the second as a new paired sensitivity analysis. The original cross-condition comparator remains third. Nothing in the new screen supersedes those directions; the best next addition is an independent-identity risk and abstention design, not tuning these cohorts again.

## Round 5: cross-session model comparison

Cross-session verification is the relevant test for recognizing an enrolled person on a different recording occasion. The [fixed model roster](MODELS.md) includes all 12 original-kit models and ten non-kit comparators. The [new protocol](ROUND5_PROTOCOL.md) preserves the matched single/paired EEG windows and separates the claimant, historical training reference, threshold-setting impostors and held-out impostor identities in each rotation.

### Original kit: historical comparison, not the Round 5 operating point

These are mean fold EERs from the supplied 29-person expected CSV: use configuration `ALL`, or `Spatial_Fusion` for that row. “CS” trains on one session and tests the other three in turn; “LOSO” uses three to train and holds out one. Neither setup is the prospective enrollment/calibration/held-out-impostor design above, so do not interpret its lower EERs as direct improvements on Round 5.

For context, rerunning the kit's **WS-labelled** chronological 80:20 split on the same 29 people yielded SVM EER **10.85%**, compared with **17.10%** in the verified original cross-session folds (**+6.25 percentage points**). The WS-labelled split cuts the pooled, chronologically concatenated recordings, not each visit separately. This is a protocol comparison, not a controlled same-day versus different-day effect.

| Original kit model | CS expected EER | LOSO expected EER | Fresh original-protocol check |
|---|---:|---:|---|
| SVM_RBF | 17.10% | 10.02% | All four CS EER folds matched; 29-person LOSO not checked |
| ExtraTrees | 19.67% | 12.97% | Expected only |
| LightGBM | 22.25% | 12.99% | Expected only |
| XGBoost | 23.19% | 13.58% | Expected only |
| Res-MLP | 18.33% | 12.50% | Expected only |
| 1D-CNN | 34.00% | 30.23% | Expected only |
| BiLSTM-Attention | 30.04% | 27.85% | Expected only |
| SpecFormer | 27.67% | 25.70% | Expected only |
| Spatial_Fusion | 17.68% | 11.10% | Expected only |
| Ens_RankAvg | 17.11% | 9.92% | All four LOSO EER folds matched; CS not checked |
| Ens_SoftVote | 17.43% | 10.05% | Expected only |
| Ens_Stacker | 40.81% | 10.26% | Expected only |

### Round 5 results: all four cohorts

The [comparison tables](ROUND5_RESULTS.md) show both arms, 95% subject-bootstrap intervals and fit/inference times for all 22 models in each completed cohort. The aggregates also report calibration-unseen results, paired differences against the cosine and RBF SVM anchors, and the neural and ensemble seed summaries. These compare end-to-end pipelines, not architectures with identical inputs and fit strategies.

| Cohort | Participants | Lowest paired EER | Paired EER / FRR / FAR | Paired minus single EER (pp, 95% CI) |
|---|---:|---|---:|---:|
| ds007176 | 29 | Tangent-SVM | 15.06% / 36.93% / 7.15% | −4.23 [−7.09, −1.84] |
| RestCog | 60 | Logistic | 22.45% / 80.97% / 2.46% | −2.10 [−3.13, −0.95] |
| Dortmund — two-day sensitivity analysis | 128 | Logistic | 19.86% / 83.36% / 1.25% | −3.83 [−4.85, −2.88] |
| WBCIC-SHU — modified-duration three-day | 10 | ExtraTrees | 27.20% / 62.92% / 10.21% | −0.94 [−4.11, +4.08] |

**Pairing helped some models but not uniformly.** The paired-minus-single EER interval was entirely below zero for ds007176 13/22, RestCog 11/22, Dortmund 18/22, WBCIC-SHU 1/22. Models with that result in every completed cohort: Cosine. These intervals are exploratory and are not adjusted for multiple comparisons.

Logistic regression remained below RBF SVM on paired EER in both main cohorts (Logistic minus RBF SVM, pp with 95% intervals: ds007176 −2.92 [−5.16, −0.49]; RestCog −2.19 [−3.68, −0.49]).

**Operating-point result:** Only BiLSTM-Attention passed all four prespecified gates, in Dortmund's two-day sensitivity analysis. Its paired EER fell by 4.96 pp (95% CI 3.87 to 6.07 pp lower), but at the frozen cutoff it still rejected 95.76% of genuine paired attempts. The FRR gate limits worsening relative to the single-state arm; it does not require low absolute FRR. This is not a usable operating point or a replicated gate pass.

Models that met the 1% FAR upper-bound gate:

| Cohort | Model | Paired FRR | Paired FAR | 95% upper FAR bound |
|---|---|---:|---:|---:|
| RestCog | kNN | 99.01% | 0.35% | 0.80% |
| Dortmund | BiLSTM-Attention | 95.76% | 0.51% | 0.66% |
| Dortmund | kNN | 99.11% | 0.07% | 0.14% |

The [small-fit LightGBM correction](ROUND5_PROTOCOL.md) changed the WBCIC-SHU standalone row from **50.00% / 100.00% / 0.00%** to **31.25% / 76.67% / 7.08%** (EER / FRR / FAR). The corrected standalone model does not meet the 1% FAR upper-bound gate. The three ensembles use the corrected LightGBM fit; the original-kit notebook and independent reproduction remain unmodified.

The independent original 29-person all-model rerun was stopped at the user's request to prioritize the Round 5 comparison. No complete original-kit CSV was produced. Apart from the previously checked SVM CS and rank-average LOSO headlines, the original expected model values were not independently re-verified. The original-wrapper NaN symptom did not reproduce for either neural model on the first original CS fold; its cause is unresolved. Round 5 uses stabilized wrappers without changing the notebook.

See [the evidence audit](CLAIM_AUDIT.md#round-5-evidence).

## Round 6: robustness without a new cohort

[Round 6 results](ROUND6_RESULTS.md) and the [claim audit](CLAIM_AUDIT.md#round-6-evidence) separate what survived follow-up from what still needs new data.

- **Pairing's strongest result remains cosine.** Across 88 Round 5 contrasts, 30 passed approximate Holm correction and 43 passed approximate Benjamini–Hochberg correction. Cosine's EER improved in all four cohorts under both corrections and at every tested probe budget. The descriptive pooled difference was −6.10 percentage points (95% interval −8.69 to −3.51; I² 62.34%). This pools previously seen cohorts with different schedules and durations, not independent replications of the full protocol.
- **The improvement did not depend on a single identity.** At the largest budgets, cosine's individual EER improved for 27/29 ds007176, 52/60 RestCog, 122/128 Dortmund and 10/10 WBCIC-SHU claimants. Dropping any one identity from both claimant and impostor roles left the pooled EER contrast negative in every cohort. All 80 reconstructed rotations passed the disjoint-role audit; perturbing late scores in the targeted rerun left calibration scores and frozen cutoffs unchanged.
- **Discrimination still does not establish a usable threshold.** The one-sided clustered upper FAR bounds for paired cosine were 6.51%, 3.70%, 2.21% and 23.11%, respectively. Even a zero-event result on 128 independent impostors would have a 2.31% binomial upper bound for the identity-level ever-accepted event. The best-case 1% claim takes 299 independent trials, not 299 repeated windows or claimant comparisons. None of cosine, logistic regression or Tangent-SVM met all original gates at any tested budget.
- **Fusion is a comparator, not an improvement over the strongest constituent.** After correcting the fused scores to preserve equal rank sums as ties, calibration-percentile cosine/logistic fusion still improved EER compared with cosine alone in the three larger cohorts, but an EER improvement over logistic was not established in any cohort. Its frozen operating points still exceeded 1% FAR in every cohort. Deep model seed results are descriptive: two seeds do not establish stability.

**Publication priorities after Round 6:** (1) equal-usable-time eye-state pairing as a longitudinal discrimination benchmark, with the limits on false-accept inference integral to the result; (2) a prospective validation with fully separate days, counterbalanced probe order and independent impostor recruitment; (3) threshold and fusion comparisons as secondary results, not as new authentication algorithms. Round 6 strengthens the first direction but does not establish that the design is unprecedented.

## Recommendation and stop/go criteria

**Research go; deployment no-go.** Pursue a methods benchmark: *Eye-state pairing for longitudinal EEG verification: improved discrimination without reliable false-accept control*. At equal usable EEG duration, paired probes lowered cosine EER in both Loop 2 cohorts and both sensitivity analyses; neither scorer controlled FAR. Present Dortmund and WBCIC-SHU as checks on separate design limits, not additional locked replications. Combine #1 and #2; keep the original study and mixed-enrollment pivot as separate comparators. The SVM disagreement, nulls, and alignment trade-offs belong in the paper. This is a credible journal direction, not proof of better authentication, equal elapsed time, alpha-specific physiology, or liveness.

1. **Close the full-budget three-day and ordering gaps.** Seek access to a compatible external multi-day cohort or collect separate enrollment, calibration and test days with sufficiently long counterbalanced CE/OE rest, timed instructions and transitions, and controls for reference/ICA sensitivity. Dortmund supplies the long interval, and WBCIC-SHU supplies three distinct dates, but neither supplies both under the locked design. [ds007987](https://openneuro.org/datasets/ds007987) can test order/repeat controls but not chronological transfer.
2. **Power calibration and test cohorts around people, not windows.** Prespecify independent recruitment, report identity-level burden, and include a fallback factor. A one-sided 95% upper bound under zero errors reaches 1% only after 299 truly independent Bernoulli attempts; repeated epoch × claimant pairs are not that sample. Do not tune on either held-out cohort or call the system a 1%-FAR authenticator.
