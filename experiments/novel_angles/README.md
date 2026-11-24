# EEG biometric follow-up: chronological verification under eye-state shift

See [REPORT.md](REPORT.md) for the original benchmark, cross-condition pivot, and two rounds of exploratory work. The Round 1 pilot uses exact v1.0.1 raw inputs and lightweight preprocessing; it does not share the original model, features, scores, or numerical baseline. Round 2 uses the supplied preprocessing notebook and rerun kits. Do not compare numbers across pipelines as method deltas.

## Round 2: verified preprocessing

Use the supplied reproduction kit's pinned preprocessing environment (MNE 1.12.1, PyPREP 0.7.1, PyWavelets 1.8.0). Put the supplied notebook next to the extracted kit scripts; `recipe.py` validates the notebook hash and delegates preprocessing to the kit's notebook loader. Use an output directory outside the checkout and keep it private; only aggregate results belong in Git.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
umask 077
# Run from experiments/novel_angles with the kit's pinned preprocessing Python.
python -B fetch_raw.py ~/ds007176-v1.0.1
python -B recipe.py "$KIT" ~/ds007176-v1.0.1 ~/eeg-verified-loop2 \
  --reference ../../preprocessed_dataset --reference-only
python -B recipe.py "$KIT" ~/ds007176-v1.0.1 ~/eeg-verified-loop2 \
  --reference ../../preprocessed_dataset
python -B fetch_raw.py ~/ds004148-v1.0.1 --dataset ds004148
python -B recipe.py "$KIT" ~/ds004148-v1.0.1 ~/eeg-verified-restcog --restcog
```

Both raw checkouts must be at the exact v1.0.1 revisions validated by `fetch_raw.py`. The recipe runner emits aggregate validation and private per-recording logs; rerunning reuses completed FIFs. The original evaluation kit reads the generated FIFs and JSON logs, and the cross-condition kit accepts the generated FIF root as its dataset argument. Follow the kits' evaluation instructions separately. See [LOOP2_PROTOCOL.md](LOOP2_PROTOCOL.md) for the prespecified follow-up controls.

With the cross-condition kit's **observed** aggregate `paired-contrasts.csv` at `$ANCHOR`, run the locked analysis using the kit's evaluation Python:

```sh
python -B loop2.py ds007176 ~/eeg-verified-loop2 ~/ds007176-v1.0.1 \
  "$CROSS_CONDITION_KIT" "$ANCHOR" loop2-ds007176.json
python -B loop2.py ds004148 ~/eeg-verified-restcog ~/ds004148-v1.0.1 \
  "$CROSS_CONDITION_KIT" "$ANCHOR" loop2-ds004148.json
```

The runner refuses to score either cohort unless the observed SVM mismatch and mixed-enrollment FRR contrasts round to 19.4 and 5.5 percentage points. The output contains only aggregate rates, subject-bootstrap intervals, and criterion decisions; rates are fractions unless a field is labeled `pp`.

## Round 4: paired-probe FAR follow-up

See [ROUND4_PROTOCOL.md](ROUND4_PROTOCOL.md) for the locked cosine variants
and decision criteria, and [ROUND4_LITERATURE.md](ROUND4_LITERATURE.md) for the
prior-art screen. Reuse the verified FIFs, evaluation Python, and observed
cross-condition anchor from Round 2:

```sh
python -B round4.py ds007176 ~/eeg-verified-loop2 \
  "$CROSS_CONDITION_KIT" "$ANCHOR" round4-ds007176.json
python -B round4.py ds004148 ~/eeg-verified-restcog \
  "$CROSS_CONDITION_KIT" "$ANCHOR" round4-ds004148.json
```

These JSONs contain aggregate rates and signed differences as fractions,
with `point`, `low`, and `high` fields. No participant scores are written.

## Round 6: robustness and follow-ups

See the [protocol](ROUND6_PROTOCOL.md) and [results](ROUND6_RESULTS.md).
With the Round 5 environment and prepared FIFs, including the staged
`scoring-inputs/` for Dortmund and WBCIC-SHU:

```sh
python -B round6.py aggregate round6-aggregate.json
python -B round6.py ds007176 round6-ds007176.json --fifs ~/eeg-verified-loop2 --kit "$CROSS_CONDITION_KIT"
```

Repeat the second command with each cohort's own name, staged inputs and output
path. The scorer writes only aggregate objects, checks the original full-budget
metrics before writing, and does not rerun the other Round 5 models.
Then run `python -B render_round6.py` to regenerate the results tables.

## Round 5: model comparison

Use the pinned evaluation Python (CPU PyTorch) with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. See
[MODELS.md](MODELS.md) and the preregistered [protocol](ROUND5_PROTOCOL.md).
With the same `$CROSS_CONDITION_KIT`, `$ANCHOR`, and prepared FIFs as above:

```sh
python -B round5.py ds007176 ~/eeg-verified-loop2 \
  "$CROSS_CONDITION_KIT" "$ANCHOR" round5-ds007176.json
python -B round5.py ds004148 ~/eeg-verified-restcog \
  "$CROSS_CONDITION_KIT" "$ANCHOR" round5-ds004148.json
```

For Dortmund (`ds005385`) and WBCIC-SHU (`wbcic-shu`), pass the existing
dataset's `scoring-inputs/` path produced by its Round 3 scorer. Do not pass
the unfiltered preparation directories. JSON fractions report both arms and
visits, paired bootstrap differences to the two anchors, and model-specific
timing and neural-seed summaries. The original kit's broader historical
evaluation runs separately.

The fixed 12-model core CPU milestone uses the same fits, arms and decisions.
To reproduce it, add `--core` and use a different output such as
`round5-core-ds007176.json`; it must not replace a complete-roster aggregate.

## Round 3: Dortmund two-day sensitivity analysis

This is **not** a three-day replication. Use a raw git-annex checkout at
`6a558cd5852503e66df9ac7fdac2f3a7f4ed5f12` (OpenNeuro ds005385 v1.0.3)
and a fresh private output folder. Preparation uses the recipe Python; score
uses the cross-condition kit's evaluation Python. As above, limit BLAS threads
per worker and set `$ANCHOR` to the observed cross-condition CSV.

```sh
umask 077
python -B loop3_dortmund.py prepare "$DORTMUND_RAW" "$DORTMUND_FIFS" "$KIT"
python -B loop3_dortmund.py score "$DORTMUND_RAW" "$DORTMUND_FIFS" \
  "$CROSS_CONDITION_KIT" --anchor "$ANCHOR" --output loop3-dortmund.json
```

The adapter fetches and checks each selected EDF against its annex key, loads
the scalp channels, and delegates the remaining steps to the hash-pinned
notebook at 50 Hz line frequency. It reuses completed FIFs when resumed. All
six expected blocks must finish preparation or have the notebook's
single-component ICA rejection; other errors stop scoring. Eligibility uses
the locked window minima before any scoring. The adapter brings recordings
with identical electrode sets into the same name order without changing
the original FIFs or the locked scorer. No raw EDF, FIF, per-recording
log or failure manifest belongs in Git. See [Round 3 in the report](REPORT.md#round-3-independent-replication-search)
for the candidate audit and current analysis status.

## Round 3: WBCIC-SHU modified-duration three-day analysis

Follow the [prespecified WBCIC protocol](WBCIC_PROTOCOL.md). Use a fresh
private output folder. The runner streams signal members from the pinned
Figshare ZIP, checks full-member size and CRC before preprocessing, and
keeps only the rest prefix. It stages outputs under `crc-checked/` rather
than reusing FIFs generated before signal checksum verification was added.
Use the recipe Python for preparation and the kit evaluation Python for
scoring, with the same BLAS limits and cross-condition anchor as above.

```sh
umask 077
python -B loop3_wbcic.py prepare "$WBCIC_FIFS" "$KIT"
python -B loop3_wbcic.py score "$WBCIC_FIFS" "$CROSS_CONDITION_KIT" \
  --anchor "$ANCHOR" --output loop3-wbcic.json
```

Only the score JSON is aggregate. Keep the preparation manifest, source
recordings, FIFs and per-recording logs private.

## Reproduce the Round 1 pilot

Tested using Python 3.12, MNE 1.13.2, NumPy 2.5.5, SciPy 1.18.1 and scikit-learn 1.9.1. Use a clean Python environment with those versions. Set BLAS thread counts to avoid oversubscription; allow about 18 GiB for raw files and under 1 GiB for the feature cache. Run from this directory:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
git clone --depth 1 --branch 1.0.1 https://github.com/OpenNeuroDatasets/ds007176 ~/ds007176-v1.0.1
python -B fetch_raw.py ~/ds007176-v1.0.1
python -B extract.py ~/ds007176-v1.0.1 ~/eeg-novel-raw --source raw --workers 6
python -B benchmark.py ~/eeg-novel-raw results.csv
python -B refine.py ~/eeg-novel-raw follow-up-contrasts.csv
python -B extract.py ../../preprocessed_dataset ~/eeg-novel-recipe --source epochs
python -B diagnose.py ~/eeg-novel-raw ~/ds007176-v1.0.1 --recipe-cache ~/eeg-novel-recipe > diagnostics.json
python -B -m unittest discover -s . -v
```

The downloader checks newly fetched bytes against the pinned git-annex SHA-256 keys, using public OpenNeuro S3. Use a fresh cache when changing the source or preprocessing. The cache files contain subject-level EEG features and are not tracked; output tables and diagnostics are aggregate only. If you have the verified full-cohort FIFs, point `extract.py --source epochs` at them and use a separate cache and output filename. The five included subjects do not support the four-role verification benchmark.

## Protocol

- Require V0, V1, V2 and exactly one of V3/V4, each with CE and OE. Raw pilot: 250 Hz, common average reference, 1–45 Hz, non-overlapping 5 s windows, recording-local log-RMS MAD rejection, then epoch-by-channel standardization. Features: five log relative powers (2–44 Hz), shrinkage correlation tangent vectors, approximate log-log spectral slopes excluding the alpha band, or signed imaginary alpha coherency. The slope estimate is **not** a formal periodic/aperiodic decomposition, and sensor-level coherency is **not** source connectivity.
- Four seeded rotations of disjoint identity groups: claimant, reference cohort, V2 calibrating impostor, unseen scoring impostor. Each identity acts in each role once; another enrolled claimant never serves as an impostor for that rotation. The V2-unseen row uses withheld impostor identities at the calibration visit; the late row uses these same unseen impostor identities at the later visit. Genuine comparisons use actual claimant attempts, not impostor-group identities.
- Fit templates at V0/V1 using 40/80/160 clean seconds total. The default 160 s spends 80 s/visit, split equally across states for mixed/paired enrollment. Attempt duration is 10 s of usable EEG (two within-state 5 s windows, or one from each condition for paired probes). Both the paired-mean and pooled single-state arms use every one of the same first 24 retained windows per state exactly once, for 24 attempts per arm. Paired probes combine equal retained-window indices from **separate resting recordings**, not eye-onset transitions; task switching time is not counted. Rejected raw windows can leave gaps between retained segments.
- Set one threshold per fold and configuration at at-most 1% V2 empirical impostor FAR; compare pooled with two thresholds requiring an eye-state label. Later thresholds never move. Z uses old background scores against claimants; T uses current probe against old background templates, and S averages both. The background never includes V2 calibration or scored subjects in the same fold. Historical condition alignment uses V0/V1 reference subjects only. Contemporaneous visit/condition alignment uses independent donors from the exact nominal visit (V3 and V4 kept separate); if a fold has no donor at the probe visit, that probe is not scored in this arm. This requires extra contemporaneous acquisition and is not an on-device subject-only method.
- FRR and realized FAR are rates at the frozen decision point; EER summarizes the pooled ROC independently of that threshold and is not a forecast of deployment error. Report percentile intervals from 500 global subject resamples, with claimant weights on genuine attempts and claimant × impostor weights on impostor attempts. Models, background, role partition and already-fitted thresholds remain fixed in the bootstrap; intervals are conditional and do **not** cover full calibration/model-selection uncertainty. Paired contrasts use shared bootstrap draws. The additional leave-one-calibrator diagnostic stress-tests sensitivity not captured by these intervals.
- Adaptation scores prespecified early late-visit windows before allowing accepted scores to update templates; evaluation uses the other half of the visit. Each state has a separate copy of the originally mixed template for this pilot. The simulation uses zero-effort impostors, not synthetic or targeted attacks, and provides no security guarantee.
- Small channel panels are post-hoc channel subsets of full-cap preprocessing, not authentic reduced-electrode hardware experiments. FRR/FAR trade-offs should be validated on raw subsets with appropriate re-referencing and on real devices.
