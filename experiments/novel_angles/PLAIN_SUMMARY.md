# EEG verification: what we tested and what we found

**Bottom line:** A test that combines five seconds with eyes open and five seconds with eyes closed usually separated the right person from the wrong person better than an equally long test using only one eye state. The improvement was most consistent with the simple cosine scorer. **It did not make verification reliable enough to use:** with decision cutoffs set for a 1% false-accept rate, the paired test still let through too many attempts from people not used to set the cutoffs.

## Was the preprocessing the same as my friend's?

**Yes for the main ds007176 experiments.** We ran the cleaning code in your friend's actual notebook, not a rewrite. The attachment and the notebook copy we used have the same SHA-256 file fingerprint. It resamples to 250 Hz, cleans and filters the EEG, denoises artifact-related components, and cuts, screens and standardizes five-second segments. We used the pinned versions MNE 1.12.1, PyPREP 0.7.1 and PyWavelets 1.8.0. We reprocessed all 40 reference recordings: electrode order and retained segment starts matched in all 40. The largest per-recording root-mean-square waveform difference was about 0.000000011 in standardized units: numerically almost identical, not bit-for-bit identical. The comparison checks the saved outputs, not a separate log of which channels were interpolated.

We used the same notebook's cleaning steps for the other cohorts, but adapted the inputs to those datasets: RestCog, Dortmund and WBCIC-SHU use 50 Hz power-line settings instead of 60 Hz; Dortmund's EDF files and WBCIC-SHU's BDF files need format-specific loading, and WBCIC-SHU has shorter rest blocks. The **early Round 1 pilot** used a lighter pipeline because the notebook was not yet available. Its numbers are exploratory and should not be compared with the notebook results as if only the verification method changed.

## Which recordings did we use?

| Dataset | What was used | What this can tell us |
|---|---|---|
| **ds007176** (main study) | 29 people, four visits each, 232 eyes-open or eyes-closed recordings. | The original study and enrollment test, plus the main equal-time test using earlier visits for enrollment, the next visit for calibration and the final available visit for testing. |
| **RestCog** (independent dataset) | 60 people, three complete pairs of recordings each, 360 recordings. | An independent dataset check under the same scoring rules. The first two sessions were 90 minutes apart, not on separate days. |
| **Dortmund** (two-day check) | 128 of 208 people qualified. Of 1,248 selected recordings, 52 failed the notebook's artifact-cleaning criterion; 57 other recordings were too short after cleaning. | Assesses testing about five years later, but enrollment and calibration were before and after a task on the same day. |
| **WBCIC-SHU** (three-day, shorter-recording check) | 10 of 51 people qualified. Of the others, 31 had mismatched signal and event timestamps, eight lacked three distinct dates after screening, and two lacked an unambiguous eye-state marker. | Assesses three separate days, but with one-quarter of the full enrollment and probe data budget. All 30 signal files used in scoring came from ZIP members verified for full size and checksum. |

## How the main test worked

1. **Enroll people.** Build a reference for each person from earlier recordings, using both eyes-open and eyes-closed EEG. The main design used 160 seconds of usable data per person. Both test methods use the same enrollment data.
2. **Set a cutoff without looking at the final test.** Use a separate group of people who were not the claimants or the people used to train the scoring model. Have these recordings stand in for people trying to pass as someone else. Set a cutoff for each claimant and probe method so no more than 1% of its calibration attempts are accepted. Then **freeze** those cutoffs.
3. **Make two kinds of later test attempt.** Each attempt contains exactly 10 usable seconds. The paired version averages the features from one five-second eyes-open segment and one five-second eyes-closed segment. The single-state version averages two eyes-open or two eyes-closed segments, half of the attempts of each kind. In the main design, both methods use the exact same 24 five-second segments per state, once each, and make 24 attempts. Only the grouping changes. This does not match real-world elapsed time, which includes switching eye states.
4. **Check the right people and new wrong people.** Score each claimant's later recording and score attempts from a new group of people whose EEG was not used to set that claimant's cutoff. Nobody fills more than one role in a split; repeat with people rotating through roles. WBCIC-SHU keeps these rules but uses six attempts and 40 seconds of mixed enrollment.
5. **Read two kinds of result.** EER (equal-error rate) measures how well scores separate right and wrong people as the cutoff varies; lower is better. At the actual frozen cutoff, count how often the right person is rejected (FRR) and the wrong person is accepted (FAR). Report uncertainty from resampling people and keeping the fitted models and cutoffs fixed, rather than treating hundreds of EEG windows from one person as independent people.

## What the numbers mean

**Original experiment and enrollment follow-up:** The original ds007176 result was **17.10% EER** for the SVM across sessions and **9.92% EER** for rank averaging when holding out one session at a time. That was a different protocol, so those numbers should not be compared directly with the table below. The enrollment follow-up found that enrolling with one eye state and testing with the other raised false rejections by **19.43 percentage points**. Enrolling with both eye states cut that penalty to **5.52 points**. It still let through about **2.38%** of eyes-closed and **2.61%** of eyes-open wrong-person attempts at the later visit, against a 1% calibration target.

**Equal-time test:** The arrows show **single state → paired eye states**. The last column is the false-accept rate for the paired test at the frozen cutoff, not the rate where we measured EER. The [technical report](REPORT.md) gives the changes in false rejections and false accepts, confidence intervals, and the other controls.

| Dataset and scorer | EER: single → paired | Change in EER (95% interval), percentage points | Paired false accepts at the frozen cutoff |
|---|---:|---:|---:|
| ds007176, cosine | 32.30% → 27.60% | −4.70 [−7.49, −2.23] | 4.42% |
| ds007176, RBF SVM | 23.91% → 21.49% | −2.42 [−5.34, +0.60] | 3.53% |
| RestCog, cosine | 33.28% → 28.14% | −5.14 [−6.65, −3.80] | 2.69% |
| RestCog, RBF SVM | 26.37% → 24.64% | −1.73 [−3.46, −0.22] | 2.44% |
| Dortmund, cosine | 32.27% → 24.54% | −7.73 [−9.21, −6.40] | 1.84% |
| Dortmund, RBF SVM | 24.93% → 23.00% | −1.93 [−3.20, −0.42] | 2.08% |
| WBCIC-SHU, cosine | 37.29% → 30.83% | −6.46 [−9.28, −3.21] | 13.75% |
| WBCIC-SHU, RBF SVM | 40.00% → 43.33% | +3.33 [−5.22, +13.01] | 10.21% |

- **What improved:** Pairing lowered cosine EER in all four datasets. The cosine test also rejected fewer genuine attempts: by about 14, 9, 12 and 15 percentage points, respectively. The SVM evidence was mixed: an EER improvement was supported in RestCog and Dortmund, was uncertain on the main cohort, and was not demonstrated in WBCIC-SHU.
- **What failed:** Neither false-accept requirement was met by any scorer on any cohort. The upper uncertainty bound for the paired-minus-single FAR difference had to be at most 0.5 percentage points, and the upper bound for paired FAR had to be at most 1%. Even the paired point estimates were all above 1%. On the **calibration visit**, people who were *not* used to set the cutoff were already accepted too often: for example, 4.46% for cosine on ds007176 and 2.51% on RestCog. A later visit was not necessary for the miss.
- **What the extra checks showed:** In the main study and RestCog, cosine still improved after removing the alpha frequency band, and with either six frontal or six rear electrodes from the full cap. That does not show why pairing helps, or prove that a six-electrode headset would work. Shifting single-state recordings onto a common reference did not reliably replace enrollment using both states. In the exploratory pilot, updating a person's reference after accepting earlier attempts reduced later false rejections but raised false accepts (1.90% → 3.37%); score normalization did not reliably fix the false-accept problem. The corrected pilot paired comparison lowered EER from 31.47% to 27.52%; the earlier 5.71-point headline used different windows between arms and was superseded.

## What did the re-verification check?

We repeated the calculations on the same recordings, using the same rules. That checks repeatability; it does not add new people or provide a new external replication. We regenerated the 40 preprocessing reference recordings from the raw EEG, then reused previously prepared recordings for the larger scoring runs. Both original 29-person headline scores, the original five-person SVM scores, the enrollment follow-up, the corrected pilot and all four paired-probe analyses matched their saved results. This does not mean the original suite of all models finished: we reproduced the two reported headline scores in a separate run of their scorers. The [claim audit](CLAIM_AUDIT.md#rerun-outcomes) gives the evidence for each comparison.

## What is new, and what would make this ready for a journal?

Other groups have already combined eye states, tested EEG across sessions, and calibrated on one set of unknown people before evaluating another. The [fresh literature review](NOVELTY_REVIEW.md) did not find the *full* controlled comparison in the sources checked: same retained data and time in both conditions, paired versus single-state attempts, later recordings and frozen cutoffs tested on new impostors. That is a defensible research contribution, not proof that nobody has tried it.

The practical next test is an independent cohort with enough eyes-open and eyes-closed rest on **three separate dates** for the full protocol. Alternate which eye state comes first across people and visits, time the state changes, and recruit enough independent people to evaluate rare false accepts. Keep eligibility rules, the time budget and decision margins fixed before seeing test results. Until then, the evidence supports a journal paper about the benchmark and its failure to control false accepts, not a ready-to-deploy authenticator.

## Short glossary

- **EEG:** Electrical activity measured by sensors on the scalp.
- **Preprocessing:** Cleaning, filtering and cutting the raw recordings into usable segments before calculating scores.
- **Enrollment / calibration / test:** Build a person's reference / set the decision cutoff / measure it on data not used to set it.
- **Cosine / RBF SVM:** Two ways of scoring a match: compare the direction of the EEG feature patterns, or use a trained classifier.
- **FRR / FAR:** False-reject rate (the right person is turned away) / false-accept rate (the wrong person gets through).
- **EER:** The error rate where false rejects and false accepts are equal on a score curve. It is not the error rate at the frozen cutoff.
- **Percentage point:** The direct difference between two percentages: 4% to 5% is one percentage point.
