# Is the EEG verification split too restrictive?

**The benchmark makes 1-to-1 decisions with separate thresholds per claimant.** The critique identifies a real coverage gap: the original rotation never pairs two different people from the same block as claimant and impostor. Limited cutoff and test identities also make per-person false-accept estimates noisy. Neither issue is evidence that the original procedure used a global threshold or that a supervised model memorized its donors.

## What was actually run in Rounds 5 and 6

Five identity blocks rotate through four roles; two blocks are pooled as the reference set. [`loop2.py:114–124`](loop2.py#L114-L124) constructs four rotations for each claimant block. [`round6.py:77–89`](round6.py#L77-L89) independently checks that every identity is a claimant, calibration impostor, and test impostor in four rotations each, and a reference identity in eight. It is 1-to-1 scoring, not a vote between groups.

| Cohort | Claimant | Reference/training | Cutoff impostors | Final-test impostors |
| --- | ---: | ---: | ---: | ---: |
| ds007176 | 5–6 | 11–12 | 5–6 | 5–6 |
| RestCog | 12 | 24 | 12 | 12 |
| Dortmund (two-day sensitivity) | 25–26 | 50–52 | 25–26 | 25–26 |
| WBCIC-SHU (modified duration) | 2 | 4 | 2 | 2 |

[`round5.py:190–204`](round5.py#L190-L204) and [`round6.py:198–206`](round6.py#L198-L206) use the kit's final-visit stratification in ds007176 and seeded five-way partitioning elsewhere. The ds007176 split therefore matches the critique's approximate percentages and small calibration/test groups, **not** its assumed visit numbers. [`loop2.py:80–89`](loop2.py#L80-L89) and [`round5.py:36–42`](round5.py#L36-L42) imply that ds007176 enrolls on V0 and V1, calibrates on V2, then tests on V3 for 24 people and V4 for five. RestCog and WBCIC-SHU use V0/V2/V3; in Dortmund those labels mean baseline pre-task, baseline post-task, and follow-up pre-task ([`loop3_dortmund.py:16–17`](loop3_dortmund.py#L16-L17)).

1. **Who can act as an impostor?** All people become claimants, but [`round4.py:48–60`](round4.py#L48-L60) keeps only each claimant's own genuine trials and the designated test impostors. Other claimants in the same block never act as that user's impostors. This is a limitation for an insider/all-others threat model, not a leakage error for the held-out-identity question.
2. **Is the cutoff global?** No. The same block supplies the calibration impostors, but [`round4.py:48–60`](round4.py#L48-L60) computes a separate cutoff for each target score column, each model and each arm (the arm is selected in [`round5.py:144–150`](round5.py#L144-L150)). [`benchmark.py:111–113`](benchmark.py#L111-L113) places the cutoff just above the relevant calibration-impostor order statistic to target at most 1% **on those calibration comparisons**. A low test FAR alone does not imply an unusably high FRR; report both. User-to-user and cohort-to-cohort variability are legitimate concerns even with tailored cutoffs.
3. **Can the verifier overfit to known negatives?** Yes in principle. [`round5.py:45–46,102–113`](round5.py#L45-L46) samples reference enrollment windows in deterministic donor slots; [`round5_models.py:112–121`](round5_models.py#L112-L121) fits scaling and the binary classifier to those negative rows plus the claimant's positive rows. Final-test impostors are not those donors in the same fit. MDM already builds its scoring centroid from the claimant only ([`round5.py:65–74`](round5.py#L65-L74)); cosine's template is claimant-only but its normalization uses reference data ([`round5.py:118–122`](round5.py#L118-L122), [`loop2.py:206–211`](loop2.py#L206-L211)); Round 5 kNN is a supervised classifier, not claimant-only cosine kNN ([`round5_models.py:83–89`](round5_models.py#L83-L89)).

**Referenced implementations:** Neither `models_new/01` nor `regional_and_band_analysis` was found in merged main, fetched remote branches, reachable Git history, the separately attached preprocessing notebook, or the three attached ZIPs (paper archive, cross-condition kit, reproduction kit). We cannot infer their fit or cutoff behavior or rule out leakage without those files. The new experiments below are independent implementations, not a reproduction of those scripts.

## Follow-up designs

This is a descriptive follow-up on the already seen cohorts. Reuse the prepared recordings, spectral features, enrollment windows and paired late attempts from Round 5; do not retune on late probes. Compare fixed One-Class SVM (RBF, `nu=.1`, claimant-only scaling), Isolation Forest (100 trees, seed 42, claimant-only scaling), and mean similarity to the three closest claimant enrollment vectors by cosine distance. Logistic and kit RBF SVM use the original binary training design.

- **Matched rotation:** Compare all five scorers on identical claimants, probes, and disjoint roles. Contrast the original per-claimant impostor cutoff with one score cutoff per model and rotation, estimated by pooling the same earlier calibration-impostor scores. Compute each claimant's test FAR separately in each of their four rotations. For supervised variants, both the training negatives (scored on their **later** recordings) and unseen test impostors use the same per-fit cutoff. The membership contrast is descriptive: identities and their weighting differ across the two pools, and membership is not isolated from other fit differences.
- **All others with genuine-only cutoffs:** Fit each one-class model once per claimant from enrollment alone. Compare every other identity against that claimant at the genuine-only cutoff calibrated on the claimant's earlier probe visit (empirical target at most 10% genuine rejection, **not** an FAR target). Also pool all claimants' genuine calibration scores for one global cutoff per model. No impostor informs the per-claimant fit or cutoff, but in the pooled control an identity can contribute as a genuine claimant to the global cutoff and later be an impostor for someone else. In either case the late scores are never used to fit or set the cutoff.
- **Expanded disjoint test coverage:** Use the same claimant-only scores and per-claimant impostor cutoffs as the matched rotation for every original trial. Add the missing within-block impostor pairs using each claimant's first rotation cutoff, calibrated on a different block. Every ordered non-self pair is tested once somewhere, and no test impostor calibrates its own decision in that split. This cross-fitted check does not give a claimant one cutoff calibrated on some impostors **and** tested independently on all others. That would require independent external calibration or a genuine-only cutoff.

Comparisons are pooled descriptive metrics, not independent trials or confidence bounds. Different cutoff policies can change FRR/FAR but not EER on a fixed score ledger. Differences across the three designs can reflect their different impostor mixtures.

## Results

Results are EER / FRR / FAR (%). All operating points use frozen cutoffs.

| Design | Model | ds007176 | RestCog | Dortmund |
| --- | --- | ---: | ---: | ---: |
| Rotation | One-Class SVM | 29.86 / 70.55 / 4.17 | 38.06 / 81.77 / 2.82 | 35.78 / 75.50 / 2.58 |
| Rotation | Isolation Forest | 25.86 / 69.43 / 4.35 | 31.32 / 78.12 / 3.00 | 28.22 / 74.96 / 2.58 |
| Rotation | Cosine kNN | 40.40 / 83.55 / 4.76 | 42.91 / 92.85 / 1.57 | 40.38 / 90.45 / 1.92 |
| Rotation | Logistic | 18.57 / 62.72 / 3.42 | 22.45 / 80.97 / 2.46 | 19.86 / 83.36 / 1.25 |
| Rotation | SVM_RBF | 21.49 / 62.03 / 3.53 | 24.64 / 69.76 / 2.44 | 23.00 / 68.03 / 2.08 |
| Rotation, pooled cutoff | One-Class SVM | 29.86 / 84.05 / 1.74 | 38.06 / 89.17 / 1.14 | 35.78 / 86.47 / 1.42 |
| Rotation, pooled cutoff | Isolation Forest | 25.86 / 85.56 / 1.66 | 31.32 / 87.85 / 1.65 | 28.22 / 88.82 / 1.47 |
| Rotation, pooled cutoff | Cosine kNN | 40.40 / 85.34 / 4.03 | 42.91 / 94.24 / 1.01 | 40.38 / 91.31 / 1.44 |
| Rotation, pooled cutoff | Logistic | 18.57 / 76.87 / 1.71 | 22.45 / 93.00 / 1.13 | 19.86 / 92.78 / 0.68 |
| Rotation, pooled cutoff | SVM_RBF | 21.49 / 76.69 / 1.17 | 24.64 / 80.75 / 1.17 | 23.00 / 79.35 / 0.91 |
| Expanded disjoint | One-Class SVM | 29.74 / 70.55 / 4.13 | 38.06 / 81.77 / 3.06 | 35.84 / 75.50 / 2.61 |
| Expanded disjoint | Isolation Forest | 25.86 / 69.43 / 4.73 | 31.44 / 78.12 / 3.07 | 28.22 / 74.96 / 2.67 |
| Expanded disjoint | Cosine kNN | 40.37 / 83.55 / 5.43 | 42.92 / 92.85 / 1.61 | 40.27 / 90.45 / 1.87 |
| All others, genuine-only | One-Class SVM | 29.74 / 17.82 / 43.24 | 38.06 / 29.58 / 47.49 | 35.84 / 20.05 / 56.73 |
| All others, genuine-only | Isolation Forest | 25.86 / 12.79 / 44.79 | 31.44 / 22.01 / 43.75 | 28.22 / 10.45 / 53.83 |
| All others, genuine-only | Cosine kNN | 40.37 / 14.37 / 58.16 | 42.92 / 38.40 / 49.08 | 40.27 / 18.03 / 61.00 |
| All others, pooled genuine | One-Class SVM | 29.74 / 11.21 / 59.69 | 38.06 / 10.62 / 80.52 | 35.84 / 8.72 / 86.72 |
| All others, pooled genuine | Isolation Forest | 25.86 / 6.90 / 52.91 | 31.44 / 12.99 / 58.05 | 28.22 / 4.65 / 68.59 |
| All others, pooled genuine | Cosine kNN | 40.37 / 7.76 / 75.76 | 42.92 / 13.96 / 79.55 | 40.27 / 8.40 / 77.80 |

- **Claimant-only training did not beat the binary models on matched EER in these cohorts.** Isolation Forest had the lowest EER among the three tested one-class models; logistic had the lowest EER in the matched comparison. This does not rank all possible one-class methods. Under genuine-only individualized cutoffs, FRR fell but all-other FAR was high (43.24%–61.00%). A genuine-only cutoff does not protect against impostors without evidence from impostor tests.
- **Removing the same-block blind spot added ordered identity pairs** (ds007176 672 → 812; RestCog 2,880 → 3,540; Dortmund 13,106 → 16,256). On these three one-class models, the extra pairs changed EER by at most 0.12 pp in these cohorts. The original protocol missed plausible impostors, but adding them did not reverse the observed discrimination results.

### Negative-set membership

Test FAR on **late** attempts with the same per-fit cutoff. A person counted as seen only if that person's enrollment actually supplied a negative fit row.

| Cohort | Model | Seen donors | Unseen test impostors |
| --- | --- | ---: | ---: |
| ds007176 | Logistic | 2.29% (739/32,256) | 3.42% (551/16,128) |
| ds007176 | SVM_RBF | 2.67% (860/32,256) | 3.53% (569/16,128) |
| RestCog | Logistic | 1.82% (2,514/138,240) | 2.46% (1,703/69,120) |
| RestCog | SVM_RBF | 2.01% (2,785/138,240) | 2.44% (1,689/69,120) |
| Dortmund | Logistic | 1.05% (4,145/393,216) | 1.25% (3,934/314,544) |
| Dortmund | SVM_RBF | 1.73% (6,807/393,216) | 2.08% (6,538/314,544) |

Unseen FAR was higher in each comparison. This is consistent with poorer transfer to unseen impostors; it does not establish memorization. The groups differ, seen donors can be scored in more fits, and the analysis did not randomize an individual into or out of the fit while holding everything else constant.

### Did a user's FAR swing from zero to 15% or more?

Each cell gives the median and maximum **within-claimant** FAR spread over the four test groups, then the number of claimants who had both a zero-FAR group and a group at or above 15%. Training and calibration groups rotate too, so this measures variability of the entire rotation protocol, not the isolated effect of swapping an impostor group.

| Cohort | Model | Per-claimant cutoff | Pooled cutoff |
| --- | --- | ---: | ---: |
| ds007176 | Logistic | 7.50 / 28.47 pp; 4/29 | 4.17 / 15.97 pp; 1/29 |
| ds007176 | SVM_RBF | 5.56 / 37.50 pp; 4/29 | 2.08 / 13.89 pp; 0/29 |
| RestCog | Logistic | 4.86 / 29.17 pp; 3/60 | 2.60 / 11.11 pp; 0/60 |
| RestCog | SVM_RBF | 5.03 / 34.72 pp; 2/60 | 1.74 / 15.62 pp; 1/60 |
| Dortmund | Logistic | 2.13 / 15.00 pp; 0/128 | 0.96 / 8.17 pp; 0/128 |
| Dortmund | SVM_RBF | 3.19 / 19.71 pp; 0/128 | 1.52 / 15.71 pp; 0/128 |

The swing occurred even with per-claimant cutoffs in both main cohorts. Pooled impostor cutoffs reduced median spreads for these two models in all three cohorts, but also raised genuine rejection. Less variation alone did not make the pooled policy better. The 1% target was calibration-only, not a test FAR guarantee.

**Recommendation:** Keep the held-out-identity rotation for comparisons requiring negative training or impostor calibration. Add the all-pair disjoint sensitivity check when reporting an insider/all-others threat model. For one-class devices with no impostor calibration available, report the genuine-only operating point as such and validate it prospectively before making any security claim.

## Reproduction

The score script is [`split_critique.py`](split_critique.py). Aggregate-only outputs: [ds007176](split-critique-ds007176.json), [RestCog](split-critique-ds004148.json), [Dortmund](split-critique-ds005385.json). These use the existing Round 5 Python environment and prepared FIF inputs. Example from this directory:

```sh
python -B split_critique.py ds007176 split-critique-ds007176.json --fifs "$DS007176_FIFS" --kit "$CROSS_CONDITION_KIT"
```

Use `ds004148` or `ds005385` and the matching FIF root for the other cohorts. The supervised matched-rotation EER/FRR/FAR points reproduce the saved Round 5 paired results in all three cohorts.
