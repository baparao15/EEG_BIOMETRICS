# WBCIC-SHU modified-duration three-day analysis

Prespecified before any WBCIC identity scores. Use only Figshare v5's two-class
cohort (archive file 51001884). Require one signal/event BDF pair at each of
the three nominal sessions, matching signal and event header timestamps, and
three distinct calendar days. Use the earliest, middle and latest dated
sessions for enrollment, calibration and test, respectively. Require unique,
ordered eye-opening and eye-closing markers (7 then 8, before the first
motor-imagery marker) and a full 60 seconds of each state without crossing
into the next task. Use the first 60 seconds after each marker; round marker
positions to the nearest sample. Do not count waiting gaps as eye-state time.

Apply the hash-pinned notebook unchanged after loading the 59 scalp EEG
channels with standard montage coordinates, dropping ECG/EOG, and setting
50 Hz line frequency. Interpret the Neuracle signal header's `?V` fields
as microvolts when its physical extrema are ±375000, consistent with the
[manufacturer's ±375 mV range](https://www.neuracle.cn/productinfo/148636.html).
Do not modify ICA or rejection. Exclude an entire
participant if any of the six blocks cannot be identified or preprocessed,
or does not meet the retained-window floor. Report exclusions by reason
without accessing verification scores; network or archive integrity failures
stop the analysis rather than acting as signal-quality exclusions.

- Scale both Loop 2 budgets by **one quarter**: 24 → **6** retained
  five-second windows per state at calibration/test; 32 → **8** enrollment
  windows total (40 usable seconds), four per state for mixed enrollment.
  Require eight retained epochs per state at enrollment to also support
  the single-state controls. One quarter is the largest common exact
  multiplier making both window counts even (for within-state pairs and
  balanced mixed enrollment) while fitting 12 raw windows per state.
- Both probe arms spend 10 usable seconds per attempt and use the same
  first six retained windows per state, each once: six CE+OE attempts
  versus three CE+CE and three OE+OE attempts. Scale reference fitting and
  contrast/covariance controls by the same factors as their corresponding
  Loop 2 probe or enrollment counts. Acquisition order is fixed OE-first.
- Everything else follows [Loop 2](LOOP2_PROTOCOL.md): cosine and RBF SVM,
  five seed-42 identity groups with disjoint roles, 500 shared subject
  bootstrap draws (seed 177), frozen per-claimant thresholds, endpoints and
  go criteria. Do not change budgets, preprocessing, identities, or decision
  criteria in response to results.

This is a **modified-duration** external three-day test, not an unchanged
Loop 2 replication. Equal usable EEG time does not match elapsed time.
