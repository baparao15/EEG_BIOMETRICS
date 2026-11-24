# Round 4: rules locked before new later-visit scores

This is exploratory follow-up on the already reported cohorts, not a prospective validation. We know the earlier baseline results but do not use later-visit outcomes to fit, select, or revise any variant.

Use the exact Loop 2 inputs, retained windows, duration and identities: 24 matched CE/OE pairs per probe visit, 32 mixed enrollment windows across eligible enrollment visits; disjoint claimant, reference, threshold-setting impostor and final-test impostor roles; 20 fixed rotations per cohort. Run ds007176 and RestCog first. Use the locked cosine scorer as the common reference, with the existing single-state arm as the study reference and paired mean as the direct comparator. All four variants are cosine-only follow-ups; do not claim an SVM result. No additional probe recordings or attempts are allowed.

## Four variants

All spectral variables are the existing five log-relative-power bands. Reference-population standardization is the same as Loop 2.

1. **Worst state:** score CE and OE against the claimant's respective same-state enrollment means; take the smaller cosine. Recalibrate this score. This is a two-state agreement requirement, not an eye-state classifier or a liveness test.
2. **Reactivity consistency:** subtract `0.25 × min(|probe reactivity − claimant enrollment reactivity|, 2)` from the original paired cosine. Reactivity is mean log-relative alpha (8–13 Hz) across the original six posterior electrodes in CE minus OE. Compute the claimant mean on the very same enrollment windows as the baseline. Constants are fixed heuristics, not optimized. This tests a consistency penalty on a paired attempt, not the previously evaluated standalone CE−OE identity feature or joint feature vector.
3. **Enrollment repeatability weighting:** for each standardized spectral coordinate, calculate the between-reference-person variance of their state-specific enrollment means and the mean within-person variance over their enrollment windows separately by state, averaging both quantities across states. The weight is between/(between+within), clipped to [0.25, 1]; use 0.25 when both terms are zero. Multiply each feature by the square root of its weight on both enrollment and probe. This weights coordinates based on a separate reference cohort, not a claimant triage rule or access to test quality.
4. **Identity-maximum calibration:** leave paired cosine scores unchanged. For each claimant, compute the maximum score across the fixed probe attempts of each calibration impostor. Apply the unchanged empirical cutoff rule to these per-person maxima instead of treating all person-window scores as calibration items. At these group sizes the cutoff is strictly above the largest calibration score. This is a conservative sensitivity test, not a finite-sample 1% guarantee or conformal claim.

Fit the first three variant cutoffs with the original pooled calibration-attempt rule per claimant and method. Score both the held-out calibration visit and later visit with all thresholds frozen; reuse the exact same held-out identities across visits. Include all variants regardless of result. Primary comparative endpoint: later-visit FAR change versus paired cosine at the respective frozen thresholds; report its point estimate and 95% paired identity-bootstrap interval. Report EER and FRR for both arms and their deltas as well, and apply the original Loop 2 gate against the duration-matched **single-state** reference: directional EER improvement (upper bound <0), FRR noninferiority (+2 pp), FAR noninferiority (+0.5 pp), and paired FAR upper bound ≤1%. Use the original identity-role bootstrap with 500 draws, seed 177. Report bootstrap-based absolute FAR bounds as descriptive where no false accepts occur; zero observed events do not establish zero uncertainty or qualify as a win.

A replication candidate must satisfy all original gates with observed nonzero accepted impostor attempts, *and* have a strictly negative upper bound for FAR difference against paired cosine in both locked cohorts. If multiple qualify, test each on Dortmund and WBCIC-SHU with the existing two-day and reduced-duration rules, not as full replications. Do not substitute a near miss, tune constants or select individual cohorts. Among non-winners, report trade-offs rather than calling a reduction a fix.

## Remaining screen (not scored in this round)

- Temporal agreement: require concordance on multiple paired attempts at a prespecified enrolled-user transaction budget.
- Per-window quality weighting from acquisition/cleaning indicators, rather than weighting spectral coordinates from reference repeatability.
- Transport an independently frozen threshold between datasets with a common montage, keeping cohort-specific enrollment separate.
- Distinct-identity finite-sample risk bounds, including abstention when calibration is too small.
- Demote potential wrong-person matches by competition with independent reference profiles, without trying to identify the person.
- Repeatability-aware left/right channel contrasts as a representation rather than another montage subset.
