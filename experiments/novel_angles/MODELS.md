# Round 5 model selection

CPU PyTorch 2.10.0, CUDA unavailable, eight logical CPUs and 31 GiB RAM visible at selection. Use the installed evaluation environment and compact PyTorch implementations, no new dependency. The twelve original-kit rows are the primary comparison set. Sources below motivate families; none establish this dataset's outcome.

| Model | Why compare it / implementation | Source |
|---|---|---|
| SVM_RBF | Locked nonlinear spectral anchor; original kit SVC, same balanced claimant/donor slots | [Cortes & Vapnik](https://doi.org/10.1007/BF00994018), Notebook 07 |
| ExtraTrees | Random splits, nonlinear feature interactions; fulfills the forest comparator without adding a redundant random forest | [Geurts et al.](https://doi.org/10.1007/s10994-006-6226-1) |
| LightGBM | Kit leaf-wise histogram booster with a Round 5 small-fit minimum leaf size; original-kit runner unchanged | [Ke et al., NeurIPS](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) |
| XGBoost | Kit regularized tree boosting | [Chen & Guestrin](https://doi.org/10.1145/2939672.2939785) |
| Res-MLP | Lightweight nonlinear spectral model and constituent of all three kit ensembles | Notebook 07; residual design precedent [He et al.](https://doi.org/10.1109/CVPR.2016.90) |
| 1D-CNN | Tests kit convolutions over the flattened spectral feature bank, **not** over raw time | Notebook 07; convolution precedent [LeCun et al.](https://doi.org/10.1109/5.726791) |
| BiLSTM-Attention | Kit recurrent model over ordered channel vectors, **not** over recording time | Notebook 07; [bidirectional LSTM](https://doi.org/10.1016/j.neunet.2005.06.042) and [attention](https://doi.org/10.48550/arXiv.1409.0473) |
| SpecFormer | Kit channel-token transformer over spectral input | Notebook 07; [Vaswani et al.](https://doi.org/10.48550/arXiv.1706.03762) |
| Spatial_Fusion | Average scores from four separately fitted regional RBF SVMs as in the kit; regional available-channel intersections differ across datasets | Notebook 07; [Kittler et al.](https://doi.org/10.1109/34.667881) |
| Ens_SoftVote | Direct probability combination, contrasts with rank and stacking | Notebook 07; [Kittler et al.](https://doi.org/10.1109/34.667881) |
| Ens_RankAvg | Enrollment OOF empirical ranks reduce sensitivity to constituent score scale | Notebook 07; rank aggregation precedent [Dwork et al.](https://doi.org/10.1145/371920.372165) |
| Ens_Stacker | Enrollment OOF logistic combination tests whether data-driven weights help | Notebook 07; [Wolpert](https://doi.org/10.1016/S0893-6080(05)80023-1) |
| Cosine | Low-cost locked log-relative-bandpower template anchor | [Maiorana](https://doi.org/10.1016/j.patrec.2021.01.004) for cross-session EEG verification context; Loop 2 for this exact scorer |
| Shrinkage LDA | Strongly regularized linear generative baseline for limited enrollment | [Friedman](https://doi.org/10.1080/01621459.1989.10478752) |
| Logistic regression | Simple linear discriminative comparator to SVM and shrinkage LDA | [Cox](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x) |
| kNN | Local nonparametric baseline exposing sensitivity to distances in high dimension | [Cover & Hart](https://doi.org/10.1109/TIT.1967.1053964) |
| Tangent linear SVM | Tests a linear decision on covariance geometry rather than bandpower | [Barachant et al.](https://doi.org/10.1109/TBME.2011.2172211), [log-Euclidean metric](https://doi.org/10.1002/mrm.20965) |
| MDM | Log-Euclidean minimum-distance centroid adapted to claimed-identity verification | [Barachant et al.](https://doi.org/10.1109/TBME.2011.2172211), [Arsigny et al.](https://doi.org/10.1002/mrm.20965) |
| EEGNet-inspired | Separable temporal/spatial operations reduce CPU cost and parameter count | [Lawhern et al.](https://doi.org/10.1088/1741-2552/aace8c) |
| ShallowConvNet-inspired | Square/pool/log temporal-spatial features provide a raw counterpart to power features | [Schirrmeister et al.](https://doi.org/10.1002/hbm.23730) |
| DeepConvNet-inspired | Stacked temporal convolutions test additional hierarchical capacity; narrow CPU implementation | [Schirrmeister et al.](https://doi.org/10.1002/hbm.23730) |
| Siamese spectral embedding | Trains similarity on reference identities; accepts templates for identities never used to train the encoder | [Supervised contrastive loss](https://doi.org/10.48550/arXiv.2004.11362); contrastive Siamese precedent [Hadsell et al.](https://doi.org/10.1109/CVPR.2006.100) |

Round 5 reuses the notebook's four ensemble constituents and its three combination rules. Unlike the original notebook's shared training scaler, each Round 5 out-of-fold fit scales only its own enrollment training fold. The matched attempts, donor pool and calibration also differ from the original kit's protocol.

The kit's LightGBM minimum of 20 samples per leaf prevents a split with fewer than 40 training windows. Round 5 scales that minimum by the fit's window count relative to the full 64-row fit, rounding down, with a floor of 2 and a cap of the kit's minimum. This gives 20/10 samples at 64/32 windows and 5/2 samples at 16/8 windows. The same rule applies to the standalone model and its ensemble fits; it is not used in the historical reproduction.

The root cause of the earlier original-run NaN is not assumed from finite input alone. A diagnostic refit the original wrappers on enrollment-only data and checked their original first cross-session fold, without using its outcomes for fitting or model selection. Neither model returned nonfinite predictions there; the historical failure was not reproduced in that checked fold. The stabilized variants in Round 5 do not count as reruns of the unmodified original kit. No exact Braindecode implementation, ArcFace or additional forest is entered because the compact families already test those design choices at this CPU budget. The 50 Hz waveform encoders also cannot represent frequencies above 25 Hz, so comparisons to full-band spectral models compare pipelines, not architectures in isolation.
