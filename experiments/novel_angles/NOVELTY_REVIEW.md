# Literature recheck: equal-budget eye-state pairing in longitudinal EEG verification

**Search date: 24 November 2026. Full-claim verdict: not found in the sources screened; individual components are published or partially covered.** Published work already combines EO (eyes open) and EC (eyes closed) in a biometric input, pools both states, compares multi-paradigm with single-paradigm authentication, and evaluates later sessions and impostor transfer. The defensible contribution is the *controlled comparison*, not introducing eye-state fusion or frozen thresholds.

## Claim screened

With identical mixed-state enrollment, compare a verification attempt averaging features from one 5 s EO window and one 5 s EC window against an equal mixture of attempts averaging two EO or two EC windows. Use the **same retained windows in the two arms**, once per arm, with the same number of attempts and 10 s of usable EEG per attempt. Evaluate later recordings; use EER to assess discrimination and FRR/FAR at per-claimant thresholds set on separate calibration identities and frozen before evaluating unseen test impostors. No claim of equal elapsed time is made. This is the full-budget [Loop 2 design](LOOP2_PROTOCOL.md); Dortmund and WBCIC-SHU change timing and duration, respectively.

| Component | Verdict | Closest precedent or gap |
|---|---|---|
| EO and EC as biometric inputs; combining them in one representation | **Published** | Lai pools EO and EC input samples. Choi uses an EC−EO spectral representation in a binary authentication comparison. Neither result makes paired inputs in general new. |
| Mean of one EO and one EC feature vector *within a verification attempt* | **Partially covered; exact mean not found** | Choi combines states by subtraction, not averaging; Rakshe aggregates epochs across paradigms in a dataset, not into a joint test epoch in the implementation checked. |
| Matched usable time and attempt count, reusing identical retained windows in EO+EC versus EO+EO/EC+EC | **Not found** | I did not find that control in the close papers. Fixed windows, duration sweeps, sample balancing, or a fixed training-epoch budget alone do not establish it. |
| Later-session / longitudinal EEG verification | **Published** | Maiorana, the long-term Roma Tre study, and large-scale multisession work establish the verification setting. The Goldstein preprints also evaluate longitudinal identity matching. |
| Threshold calibration before final evaluation on different unknown identities | **Published in open-set EEG identification; partially covered for the specific 1:1 longitudinal endpoint** | The 2026 M3CV open-set paper separates development and final-test unknown identities, reports realized FPIR, and studies source-only cross-session transfer. Its max-score open-set rejection endpoint is not per-claimant verification FAR. FRR at a nominal FAR read off a test ROC is not evidence of a frozen cutoff. Do not claim the threshold split itself is new. |
| Full comparison including the matched-window control and frozen 1:1 operating point | **Not found in the screened literature** | The closest precedents split across the above components. This is a targeted literature check, not an exhaustive or systematic review or a claim of priority. |

## Closest work: what was actually compared

“Not shown” means the examined methods, abstract or code did not establish the element, not that an inaccessible source cannot contain it. Identification assigns an identity; 1:1 verification tests a claimed identity. FPIR for max-score open-set identification is not per-claimant FAR.

| Source and evidence | What composes one test input? | Same-window and equal-budget control? | Later sessions and threshold / impostor policy |
|---|---|---|---|
| **Choi et al. (2019)**, [10.1109/IWW-BCI.2019.8737254](https://doi.org/10.1109/IWW-BCI.2019.8737254); IEEE abstract | Repeated 10 s EO and EC blocks; compares EO, EC and EC−EO with 3–20 Hz PSD features and LDA. Closest within-input combination; a difference, not a mean. | The reported difference uses both recorded states; the abstract does not describe a one-state two-window equal-duration or same-window control. | 5×5-fold binary CV per person; no later-visit, disjoint-calibration frozen operating point established in the abstract. |
| **Lai et al. (2019)**, [10.1155/2019/7895924](https://doi.org/10.1155/2019/7895924); [§2.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC6589242/) | REO+REC pools separate 1 s samples for closed-set CNN classification. | Combined arm doubles each split (6,540 test samples versus 3,270); no pair formed per test sample. | Splits of the recordings, not held-out visits; §2.2 explicitly says no additional impostor subjects. |
| **Rakshe et al. (2026)**, [10.1186/s40708-026-00337-7](https://doi.org/10.1186/s40708-026-00337-7); article abstract and [public implementation tree](https://github.com/chetanrakshe2510/MultiParadigm-EEG-Biometrics/tree/e8b6d6e439fbc7c32cd964f14c03501cd852db85) | Six paradigms, including EO and EC, with overlapping 2 s epochs. Core scripts `00_merge_features.py` and `01_train_classic_ml.py` stack task epochs into a unified dataset and load one feature row per epoch. | The restricted comparison samples training epochs per subject; it does not regroup identical test windows into two-state versus same-state attempts. | Session 1→2 evaluation is prior art. Core classic-ML script computes ROC from test identity scores; the inspected scripts do not establish calibration on separate identities followed by a frozen per-claimant cutoff. |
| **Maiorana (2021)**, [10.1016/j.patrec.2021.01.004](https://doi.org/10.1016/j.patrec.2021.01.004); [author methods PDF](https://iris.uniroma3.it/retrieve/e397d80e-a2e9-b0de-e053-6605fe0a1c76/PRL_Deep_EEG_2021_Online.pdf) | One 5 s frame is the verification sample. Scores fused across electrodes, not across EO/EC frames. Studies multistate enrollment and cross-task verification. | Standardizes the length of individual frames, not the same-window paired-probe comparison. | Open-set verification and short/long session gaps (about 15 months for the long comparison) are established. The identical disjoint per-claimant calibration/test-impostor procedure was not verified. |
| **EEG Subject Identification and Open-Set Rejection Across Paradigms (2026)**, [10.3390/s26175471](https://doi.org/10.3390/s26175471); [methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC13567940/) | M3CV task-matched probes in the primary cross-session experiment; resting EO and EC also analyzed separately in task comparisons. | Task quotas and non-overlapping temporal blocks; no regrouping of the same EO/EC windows into paired and single-state verification attempts shown. | Within-session unknown rejection calibrates on 15 development unknowns and tests on 20 different unknowns, reporting realized FPIR. The two-session experiment uses 60 enrolled / 8 development unknown / 8 final unknown and reports source-only transfer and separate target-calibration sensitivity. Closest identity-split precedent; max-score rejection rather than per-claimant FAR. |
| **Interpretable Spectral Features (2026)**, [10.3390/s26154811](https://doi.org/10.3390/s26154811); [evaluation methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC13469576/) | Spectral cross-session identity scores; no same-window paired-eye-state control shown. | Trial-level isolation; not a paired-probe budget comparison. | Target-session calibration is a model analysis. Verification sweeps blind-test scores retrospectively; methods explicitly say no operational blind-test threshold was fixed. |
| **Zhong et al. (2020)**, [10.1145/3444884.3444889](https://doi.org/10.1145/3444884.3444889); dataset and experiment | EO and EC recordings, 150 s per state; hidden Markov model identification. | No matched dual-/single-state 1:1 attempt comparison found. | Three runs with at least two weeks between runs; held-out-run identification, not the claimed frozen verification endpoint. |
| **Maiorana and Campisi (2018)**, [10.1109/TIFS.2017.2778010](https://doi.org/10.1109/TIFS.2017.2778010); [author PDF](https://biomedia4n6.uniroma3.it/papers/IEEETIFS_LongitudinalEEG_website.pdf) | Five-session dataset includes 4 min each of EC and EO; HMM verification on probe epochs with channels combined. | Does not establish the paired-eye-state matched-window comparison. | A strong longitudinal verification precedent, including thresholding; the exact calibration/test identity split was not verified. |
| **How Time Window Influences Biometrics Performance (2024)**, [10.3390/signals5030033](https://doi.org/10.3390/signals5030033); feature and evaluation methods | Compares PLI and PLV connectivity features by recording condition and duration, with pairwise similarity scores. | Duration is a tested factor, but the curves are by condition, band, and window length; not matched sets of paired and single-state attempts. | Reports EER and ROC AUC from genuine and impostor score distributions; the checked methods do not establish an independent frozen later-visit cutoff. |
| **Goldstein, time-odd geometry (2026 preprint, v2)**, [10.64898/2026.09.08.749160](https://doi.org/10.64898/2026.09.08.749160); dataset and methods | Uses ds007176 V0 CE→V3 CE, 27 participants, four 50 s windows; fixed alpha–beta weights combine bands, not eye states. | Fixed analysis windows are not matched state-pair controls. | Same data and cross-visit identity matching already appear in prior work. Fixed fusion weights are not independently calibrated FAR thresholds. |
| **Goldstein, phase/coherency (2026 preprint, v2)**, [10.64898/2026.10.26.761764](https://doi.org/10.64898/2026.10.26.761764); abstract and methods | Direct matching on RestCog, L-FAME and Dortmund; tests switching between EO and EC in Dortmund. Reported fusion combines alpha and beta bands. | Same-state vs switched-state matching, not two states in one matched probe. | Longitudinal rank-1 and AUC; not the frozen per-claimant FAR comparison. |
| **Henao Isaza et al. (2026)**, [10.1016/j.bspc.2026.109484](https://doi.org/10.1016/j.bspc.2026.109484), [dataset v1.0.1 DOI](https://doi.org/10.18112/openneuro.ds007176.v1.0.1); dataset metadata and abstract | Relative power across components and scalp regions, emphasizing ICC reliability in 43 participants over four sessions. | Test–retest consistency, not a biometric probe intervention. | No biometric verification/threshold endpoint in the dataset paper. |

Other close controls: Mota et al. ([10.7717/peerj-cs.549](https://doi.org/10.7717/peerj-cs.549)) compare cross-task recognition with segmented test epochs; Rehman et al. ([10.1007/s40747-025-02012-6](https://doi.org/10.1007/s40747-025-02012-6)) study combined and cross-task identification on PhysioNet; Garau et al. ([10.1109/ICB.2016.7550080](https://doi.org/10.1109/ICB.2016.7550080)) fuse two connectivity methods at feature and score level; Kim and Kim ([10.1109/ACCESS.2019.2917918](https://doi.org/10.1109/ACCESS.2019.2917918)) concatenate scalp-region networks. None of the checked descriptions establishes the full comparison. Time-window sweeps ([10.3390/signals5030033](https://doi.org/10.3390/signals5030033)) already quantify the duration sensitivity of EEG fingerprinting by condition. Large-scale multisession authentication ([10.1109/TBIOM.2026.3666044](https://doi.org/10.1109/TBIOM.2026.3666044), [arXiv v3](https://arxiv.org/html/2501.17866v3)) and the NeuroShield preprint ([arXiv:2606.20673](https://arxiv.org/html/2606.20673)) report EER and FRR at low FAR operating points. The RSVP-based longitudinal study [10.1109/UEMCON70256.2026.11775524](https://doi.org/10.1109/UEMCON70256.2026.11775524) evaluates days 80 and 200, not paired resting probes.

## Search and citation trail

The fresh pass used web queries with database/domain filters (Google Scholar, Semantic Scholar, arXiv, medRxiv, bioRxiv, IEEE Xplore and PubMed), direct PubMed ESearch plus ESummary for the first 20 hits (91 indexed hits), publisher methods, author copies, and public code. The Scholar-filtered results were largely author profiles, not a native Scholar export. The medRxiv-filtered pass yielded clinical or unrelated records, not a close comparison. This is a relevance-screened sample, not a database census.

For citation chasing, queried OpenAlex by seed DOI, then screened up to 25 forward and 25 backward titles per seed (default relevance order; forward `filter=cites:<work ID>`, backward `filter=cited_by:<work ID>`). Followed relevant titles to methods, abstracts or author copies. Also screened 20 Semantic Scholar forward citations to Maiorana (`paper/DOI:10.1016/j.patrec.2021.01.004/citations?limit=20&fields=title,externalIds`); this surfaced the Rehman and NeuroShield comparisons. Examples of paths:

| Seed (DOI) | OpenAlex items screened forward / backward | Follow-up |
|---|---:|---|
| Lai, `10.1155/2019/7895924` | 25 / 23 | Checked the original data split and followed stress/longitudinal and resting-fingerprint leads (including 2025 EO default-mode identification, `10.1109/APSIPAASC65261.2025.11249025`). |
| Choi, `10.1109/IWW-BCI.2019.8737254` | 4 / 22 | Screened the later segment-length paper and the older spectral-signature/authentication references. |
| Maiorana, `10.1016/j.patrec.2021.01.004` | 25 / 25 | Followed M3CV (`10.1016/j.neuroimage.2022.119666`), the 2018 longitudinal study, and multi-task verification; also ran the Semantic Scholar branch. |
| Henao Isaza, `10.1016/j.bspc.2026.109484` | 8 / 25 | Dataset releases and methods/reliability references; separate ds007176 title/identifier search located Goldstein v2. |
| Goldstein phase / time-odd preprints, `10.64898/2026.10.26.761764` and `10.64898/2026.09.08.749160` | 0 / 20 and 0 / 18 | Backward to Fraschini (`10.1016/j.patrec.2019.03.025`), Maiorana permanence, RestCog and ds007176. Forward DOI/title searches found no verified relevant citing paper; recent preprints may not yet be indexed as citations. |
| Reviews: `10.1109/ACCESS.2023.3253026`, `10.1109/ACCESS.2024.3421264`, `10.1109/ACCESS.2025.3605614`, `10.3390/s26134045` | 25 / 25; 11 / 25; 17 / 25; 1 / 25 | Followed eye-state and verification leads, including duration sweeps, the 2026 open-set M3CV paper, and the 200-day RSVP study. Reviews help locate precedents; their silence is not evidence of absence. |

### Exact discovery queries

```text
site:scholar.google.com EEG biometric fusing eyes open closed probe concatenation equal recording duration
site:semanticscholar.org EEG biometric verification multi task fused probe multiple eye states longitudinal
site:arxiv.org EEG verification biometric multi paradigm fusion eyes open closed impostor duration
site:medrxiv.org EEG biometric fingerprint paired eyes open closed cross-session identification
site:biorxiv.org EEG biometric longitudinal resting eyes open closed fusion fingerprint time odd geometry
site:ieeexplore.ieee.org EEG authentication eyes open eyes closed feature score fusion probe comparison
site:pubmed.ncbi.nlm.nih.gov EEG biometrics multi-paradigm authentication multi-session eyes open closed fusion
EEG biometric verification multi-task probe fusion “same duration” fixed threshold unseen impostors ROC Rakshe
EEG biometric eye open close paired EEG verification concatenate test EEG "fusion" both durations recognition
EEG biometrics survey 2023 2024 2025 2026 multi task multi paradigm authentication EEG review
EEG biometrics simultaneous eyes-open eyes-closed fusion test sample feature averaging verification Maiorana Choi Lai
"ds007176" EEG EEG biometrics dataset paper "eyes closed" cross session
"Arrangements of Resting State Electroencephalography" cited by EEG biometric combined eyes
"Biometrics Based on Single-Trial EEG" cited by EEG authentication
"Learning deep features for task-independent EEG-based biometric verification" cited by deep descriptor multi-task
EEG biometrics average probe multiple tasks matching identical time enrollment test M3CV impostor
EEG biometrics "threshold" "disjoint" impostors independent calibration cross-session
"10.64898/2026.10.26.761764" OR "10.64898/2026.09.08.749160" "cited by" EEG
EEG biometrics two paradigms "eyes open" "eyes closed" combined moments recognition PSD fusion verification
EEG multiple states score fusion equal number test epochs identity authentication eye open closed cross session
EEG fingerprinting authentication ("eyes open" OR "eyes-open") ("eyes closed" OR "eyes-closed") "feature averaging" pooled independent test
EEG biometric verification "mean" "eyes open and eyes closed" probe per sample false rejection threshold
biométrie EEG "yeux ouverts" "yeux fermés" fusion authentification identification intersession durée
biometría EEG "ojos abiertos" "ojos cerrados" fusión verificación duración sesiones
```

Direct PubMed ESearch term (`retmax=20`):

```text
(EEG OR electroencephalography) AND (biometric OR authentication) AND (fusion OR longitudinal OR cross-session OR eyes-closed)
```

Selected exact methods follow-ups:

```text
"s40708-026-00337-7" "combined" "epochs" "feature vectors" subjects threshold multi-paradigm authentication training test split
"Biometrics Based on Single-Trial EEG" Choi PDF 5×5 10 seconds eyes closed eye open subtraction
site:pmc.ncbi.nlm.nih.gov/articles/PMC6589242 "6540" "3270" "REO + REC"
"Learning deep features for task-independent EEG-based biometric verification" "probe" "frame" "5" single
"s26175471" "cross-session" "threshold calibration" final test unknown sessions
"10.3390/s26154811" EEG threshold calibration fixed FAR "impostor" verification
"10.64898/2026.09.08.749160" "ds007176" "centroids" alpha beta cross-visit
"10.64898/2026.10.26.761764" EEG "cross-state" "closed" "rank"
"Longitudinal test-retest reliability of quantitative EEG in healthy individuals" "ICC" eyes Henao
"Analysis of the EEG Signal for a Practical Biometric System" "eyes" separate results Abdullah EEG combination DOI
"10.1007/s40747-025-02012-6" "combined" five tasks test subject segments
site:pmc.ncbi.nlm.nih.gov/articles/PMC13567940 "Cross-session" "OSCR" "same" "five paradigms" "pooled"
```

## Risks and submission wording

- **High risk for broad novelty.** Choi already tested an EO/EC biometric contrast within one input; Rakshe and Rehman already compared or trained with multiple paradigms; the 2026 M3CV paper already calibrates/rejects unknown identities using disjoint groups. Cite these explicitly. Do not say “first to fuse EO/EC,” “first multi-state EEG authentication,” or “first to test unseen impostors.” EER and frozen FAR measure different things.
- **Residual risk for the narrow contribution.** Choi was checked at abstract level; not all conference papers, theses, supplementary protocols or very recent preprints were accessible or indexed. Rakshe's public classic-model scripts clarify pooling and scoring but are not evidence about uninspected analyses. Neither an empty citation list nor lack of an exact phrase is proof of novelty.
- **Evidence limits are distinct from novelty.** Equal usable duration does not establish equal elapsed time or an order-independent effect. WBCIC-SHU uses a shorter budget on 10 eligible people; Dortmund shares the enrollment and calibration date; RestCog is not a distinct-day enrollment/calibration replication. Pairing improved cosine EER across the cohorts tested, but the SVM evidence was mixed and realized FAR did not meet the deployment criteria. Do not present a production authenticator, biological mechanism, or full-budget third-day replication.

**Suggested manuscript wording:** “Prior EEG biometrics work has combined or contrasted resting eye states, compared multiple paradigms, and evaluated later sessions and unknown identities. In the literature we screened, we did not find a within-attempt EO/EC averaging benchmark that reused identical windows in duration- and attempt-matched single-state controls and measured later-session error at thresholds frozen on disjoint calibration identities. We evaluate that comparison and distinguish improved discrimination from false-accept control.”
