# Round 6: robustness checks and new ideas

Retrospective analysis of the existing cohorts; no new participants. See the [frozen analysis plan](ROUND6_PROTOCOL.md) and the [claim audit](CLAIM_AUDIT.md#round-6-evidence). The findings are not a deployment or security claim.

## Multiple testing

Among 88 two-sided pairing-EER contrasts, 30 passed Holm and 43 passed BH at .05. These tests approximate standard errors from Round 5 percentile intervals; raw bootstrap draws for all models were not saved. The adjusted values are approximate, particularly in the tails. BH additionally assumes a dependence structure that we have not established. The complete ordered-by-cohort results are in [the aggregate](round6-aggregate.json).

| Cohort | Cosine pairing Δ EER, pp (95% unadjusted CI) | Holm p | BH q |
| --- | --- | --- | --- |
| ds007176 | −4.70 [−7.49, −2.23] | 0.02728 | 0.001356 |
| RestCog | −5.14 [−6.65, −3.80] | 1.274e-10 | 1.42e-11 |
| Dortmund | −7.73 [−9.21, −6.40] | 3.24e-25 | 1.638e-25 |
| WBCIC-SHU | −6.46 [−9.28, −3.21] | 0.002054 | 0.0001184 |

Holm passes (negative Δ favors pairing; positive Δ favors single-state):

- **ds007176:** BiLSTM-Attention (lower), Cosine (lower), MDM (lower), ShallowConvNet-inspired (lower), Siamese (lower)
- **RestCog:** BiLSTM-Attention (lower), Cosine (lower), DeepConvNet-inspired (lower), EEGNet-inspired (lower), Logistic (lower), ShallowConvNet-inspired (lower), Shrinkage-LDA (lower), Siamese (lower), Tangent-SVM (lower), kNN (higher)
- **Dortmund:** 1D-CNN (lower), BiLSTM-Attention (lower), Cosine (lower), DeepConvNet-inspired (lower), EEGNet-inspired (lower), Ens_SoftVote (lower), Logistic (lower), Res-MLP (lower), ShallowConvNet-inspired (lower), Shrinkage-LDA (lower), Siamese (lower), SpecFormer (lower), Tangent-SVM (lower), XGBoost (lower)
- **WBCIC-SHU:** Cosine (lower)

Additional BH-only passes:

- **ds007176:** DeepConvNet-inspired (lower), EEGNet-inspired (lower), Ens_SoftVote (lower), Logistic (lower), Shrinkage-LDA (lower), SpecFormer (lower), Tangent-SVM (lower)
- **RestCog:** Res-MLP (lower)
- **Dortmund:** Ens_RankAvg (lower), LightGBM (lower), MDM (lower), SVM_RBF (lower), kNN (higher)
- **WBCIC-SHU:** none

## Leakage and split audit

| Cohort | Identities | Disjoint rotations | Visit labels | Decision interventions |
| --- | --- | --- | --- | --- |
| ds007176 | 29 | 20 | V0/V1/V2/V3; V0/V1/V2/V4 | 380 |
| RestCog | 60 | 20 | V0/V2/V3 | 380 |
| Dortmund | 128 | 20 | V0/V2/V3 | 380 |
| WBCIC-SHU | 10 | 20 | V0/V2/V3 | 380 |

An identity is in only one role per rotation; roles rotate across fits. Source tracing: `round5.py:compare_rotation,enrollment,geometry_scores,embedding_scores` restrict fitted data to early visits and claimant/reference identities. `loop2.py:training_samples,standardized` use early visits and reference-only normalization for cosine. `round5_models.py:claimant_scores` fits scalers, base models and ensemble stacking on enrollment data, including out-of-fold enrollment predictions. `loop2.py:recording_arrays` extracts features from each recording independently. `round4.py:variant_rows` reads only calibration-impostor scores at the calibration visit to set cutoffs. In the targeted rerun, changing late scores was required not to change calibration ledgers or late cutoffs. Unit tests perturb probe visits and non-fit identities before enrollment and normalization. This checks the code and the targeted execution, not a retained execution trace from every original batch fit.

## How much evidence is available for a 1% false-accept claim?

| Cohort | Paired cosine accepts / comparisons | Attempt CP upper | Clustered FAR upper | Impostors ever accepted / distinct | Identity-event CP upper | Identity CP upper if no events | Trials needed at fixed errors: attempts / identities |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ds007176 | 713/16,128 | 4.70% | 6.51% | 29/29 | 100.00% | 9.81% | 75,829 / 3,949 |
| RestCog | 1,858/69,120 | 2.79% | 3.70% | 60/60 | 100.00% | 4.87% | 193,013 / 7,432 |
| Dortmund | 5,784/314,544 | 1.88% | 2.21% | 127/128 | 99.96% | 2.31% | 591,004 / 14,706 |
| WBCIC-SHU | 66/480 | 16.61% | 23.11% | 9/10 | 99.49% | 25.89% | 8,094 / 1,568 |

Bounds are one-sided 95%. Even with no errors, 299 genuinely independent trials are needed for a binomial upper bound of at most 1%. All cohorts contain enough *rows* under the unrealistic zero-error/independent-attempt calculation, but no cohort contains that many independent impostor identities even in the zero-event best case. Reusing a person, claimant, model or threshold creates dependence. The clustered column resamples both identity roles and estimates per-attempt FAR, conditional on fitted models and cutoffs. Identity events instead mean at least one acceptance anywhere in the protocol, so they cannot be used as interchangeable per-attempt bounds. Holding observed errors fixed is arithmetic, not a sample-size forecast.

## Probe-window sensitivity

Values are paired-minus-single EER in percentage points with unadjusted 95% bootstrap intervals. Enrollment and its transformations stay fixed. At each budget the cutoff is calibrated again on disjoint calibration identities, then held fixed. At the largest budget all three rerun models' EER, FRR, FAR and pairing metric objects matched the saved Round 5 objects exactly in each cohort.

| Cohort | Windows per state | Cosine Δ EER | Logistic Δ EER | Tangent-SVM Δ EER | Pass all original gates |
| --- | --- | --- | --- | --- | --- |
| ds007176 | 8 | −5.17 [−8.42, −1.77] | −4.73 [−7.27, −1.77] | −3.99 [−7.01, −1.16] | None |
| ds007176 | 16 | −4.80 [−7.62, −1.79] | −3.99 [−6.60, −1.49] | −4.20 [−7.36, −1.62] | None |
| ds007176 | 24 | −4.70 [−7.49, −2.23] | −3.70 [−6.22, −1.51] | −4.23 [−7.09, −1.84] | None |
| RestCog | 8 | −6.04 [−8.22, −4.20] | −1.84 [−3.64, −0.32] | −1.76 [−3.23, −0.84] | None |
| RestCog | 16 | −5.78 [−7.43, −4.18] | −2.09 [−3.38, −0.85] | −2.02 [−3.19, −1.11] | None |
| RestCog | 24 | −5.14 [−6.65, −3.80] | −2.10 [−3.13, −0.95] | −2.04 [−3.14, −1.22] | None |
| Dortmund | 8 | −7.45 [−9.35, −5.76] | −4.00 [−5.38, −2.81] | −2.93 [−3.90, −1.65] | None |
| Dortmund | 16 | −7.84 [−9.42, −6.31] | −4.07 [−5.19, −3.05] | −2.62 [−3.69, −1.47] | None |
| Dortmund | 24 | −7.73 [−9.21, −6.40] | −3.83 [−4.85, −2.88] | −2.42 [−3.56, −1.37] | None |
| WBCIC-SHU | 2 | −11.25 [−17.94, −3.81] | −3.75 [−11.65, +4.13] | −0.63 [−6.73, +4.94] | None |
| WBCIC-SHU | 4 | −7.50 [−12.10, −3.54] | −0.31 [−7.86, +4.30] | −0.31 [−4.84, +5.14] | None |
| WBCIC-SHU | 6 | −6.46 [−9.28, −3.21] | −0.83 [−5.90, +3.81] | −0.42 [−4.50, +4.18] | None |

## Who benefited?

The counts refer to per-claimant EER against that claimant's held-out impostors, and genuine false rejects at that claimant's frozen cutoff. The last column instead removes one identity from both roles and recalculates the pooled EER contrast, without refitting.

| Cohort | Model | EER improved/tied/worse | FRR improved/tied/worse | Pooled Δ EER range dropping one, pp |
| --- | --- | --- | --- | --- |
| ds007176 | Cosine | 27/0/2 | 22/1/6 | −5.06 to −4.08 |
| ds007176 | Logistic | 26/1/2 | 24/1/4 | −3.99 to −3.00 |
| ds007176 | Tangent-SVM | 28/0/1 | 28/0/1 | −4.46 to −3.41 |
| RestCog | Cosine | 52/0/8 | 40/6/14 | −5.31 to −4.89 |
| RestCog | Logistic | 53/0/7 | 48/6/6 | −2.35 to −1.95 |
| RestCog | Tangent-SVM | 54/1/5 | 45/10/5 | −2.28 to −1.87 |
| Dortmund | Cosine | 122/1/5 | 82/9/37 | −7.88 to −7.56 |
| Dortmund | Logistic | 123/0/5 | 92/15/21 | −4.01 to −3.67 |
| Dortmund | Tangent-SVM | 106/1/21 | 51/56/21 | −2.57 to −2.26 |
| WBCIC-SHU | Cosine | 10/0/0 | 8/1/1 | −7.26 to −5.47 |
| WBCIC-SHU | Logistic | 5/0/5 | 7/0/3 | −2.03 to +1.50 |
| WBCIC-SHU | Tangent-SVM | 3/2/5 | 5/2/3 | −2.34 to +1.04 |

## Deep learning and ensemble seed sensitivity

Paired EER, seed 42 → 43 (sample SD in percentage points). The largest observed two-seed spread was 5.06 pp for EEGNet-inspired in ds007176. Two seeds do not establish stability.

| Model | ds007176 | RestCog | Dortmund | WBCIC-SHU |
| --- | --- | --- | --- | --- |
| 1D-CNN | 33.38% → 32.33% (0.75) | 37.36% → 35.99% (0.97) | 29.12% → 29.64% (0.37) | 33.75% → 35.42% (1.18) |
| BiLSTM-Attention | 30.51% → 29.96% (0.39) | 31.18% → 32.17% (0.70) | 26.93% → 28.72% (1.26) | 33.33% → 32.92% (0.29) |
| DeepConvNet-inspired | 29.13% → 29.31% (0.13) | 34.70% → 36.28% (1.12) | 30.54% → 29.70% (0.59) | 40.42% → 39.58% (0.59) |
| EEGNet-inspired | 39.83% → 34.77% (3.58) | 39.63% → 37.34% (1.62) | 36.73% → 35.00% (1.22) | 48.96% → 47.92% (0.74) |
| Ens_RankAvg | 21.85% → 21.74% (0.08) | 27.46% → 27.56% (0.07) | 25.24% → 25.17% (0.05) | 31.57% → 31.70% (0.10) |
| Ens_SoftVote | 19.72% → 19.72% (0.00) | 24.18% → 24.11% (0.05) | 20.69% → 20.62% (0.05) | 30.83% → 32.29% (1.03) |
| Ens_Stacker | 21.12% → 21.35% (0.16) | 37.07% → 37.36% (0.21) | 39.73% → 39.80% (0.05) | 39.79% → 39.17% (0.44) |
| Res-MLP | 20.40% → 20.05% (0.25) | 24.24% → 23.75% (0.34) | 20.05% → 19.86% (0.14) | 35.62% → 34.79% (0.59) |
| ShallowConvNet-inspired | 27.73% → 25.65% (1.48) | 34.55% → 33.52% (0.73) | 28.59% → 27.82% (0.55) | 41.25% → 38.12% (2.21) |
| Siamese | 25.36% → 24.52% (0.59) | 30.90% → 30.99% (0.06) | 26.20% → 25.13% (0.75) | 32.29% → 32.50% (0.15) |
| SpecFormer | 25.83% → 26.22% (0.28) | 28.40% → 27.77% (0.45) | 24.21% → 23.83% (0.27) | 32.92% → 33.75% (0.59) |

## New ideas

**Score fusion.** Add the two integer calibration ranks before dividing by twice the common calibration count, so equal rank sums stay tied. Set the frozen cutoff on the same calibration impostors used to fit the percentile maps. Negative differences favor fusion. The repeated score count is rows in excess of distinct fused scores, pooled across fits.

| Cohort | Fused EER / FRR / FAR | EER Δ vs cosine, pp | EER Δ vs logistic, pp | Repeated score rows / total |
| --- | --- | --- | --- | --- |
| ds007176 | 21.74% / 66.99% / 3.16% | −5.86 [−8.11, −3.53] | +3.17 [+0.42, +6.37] | 18,431/18,912 |
| RestCog | 23.96% / 75.80% / 2.24% | −4.18 [−5.81, −2.38] | +1.51 [−0.31, +3.27] | 74,303/74,880 |
| Dortmund | 19.79% / 72.47% / 1.59% | −4.75 [−5.96, −3.36] | −0.08 [−1.35, +1.17] | 324,431/326,832 |
| WBCIC-SHU | 31.45% / 72.08% / 6.04% | +0.62 [−6.12, +7.25] | +0.82 [−5.48, +7.14] | 695/720 |

| Cohort | Constituent | Fusion − constituent FRR, pp | Fusion − constituent FAR, pp |
| --- | --- | --- | --- |
| ds007176 | Cosine | −1.01 [−4.53, +2.39] | −1.26 [−3.55, +0.38] |
| ds007176 | Logistic | +4.27 [−2.34, +11.21] | −0.26 [−1.63, +1.39] |
| RestCog | Cosine | −0.50 [−4.55, +3.48] | −0.44 [−1.14, +0.02] |
| RestCog | Logistic | −5.17 [−9.31, −1.45] | −0.22 [−0.96, +0.44] |
| Dortmund | Cosine | +0.68 [−2.06, +4.27] | −0.25 [−0.51, −0.00] |
| Dortmund | Logistic | −10.89 [−14.38, −7.81] | +0.34 [+0.02, +0.63] |
| WBCIC-SHU | Cosine | +22.08 [+12.08, +32.08] | −7.71 [−17.76, +0.68] |
| WBCIC-SHU | Logistic | +12.50 [+0.00, +25.42] | −5.21 [−16.67, +1.43] |

**Pooling across cohorts.** Cosine random-effects Δ EER: −6.10 [−8.69, −3.51] pp; I² 62.34%, Q 7.97. This uses different durations and visit schedules and approximate standard errors, and there are few cohorts. The pooled value summarizes the direction and scale of what was already observed, not an external replication.

Score-level fusion has clear precedent ([Ross & Jain, *Information fusion in biometrics*](https://doi.org/10.1016/s0167-8655(03)00079-5)); this is a low-cost comparator, not a novelty claim. The pooled analysis uses established methods ([DerSimonian & Laird](https://doi.org/10.1016/0197-2456(86)90046-2); [modified Hartung–Knapp overview](https://doi.org/10.1186/s12874-015-0091-1)).
