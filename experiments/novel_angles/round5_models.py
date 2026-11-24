"""Fixed model implementations for the Round 5 verification comparison."""
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
from torch import nn
from torch.nn import functional as F


KIT_BASE = ("SVM_RBF", "ExtraTrees", "LightGBM", "XGBoost", "Res-MLP", "1D-CNN",
            "BiLSTM-Attention", "SpecFormer")
BASE4 = ("SVM_RBF", "Res-MLP", "ExtraTrees", "LightGBM")
FUSION = ("Ens_SoftVote", "Ens_RankAvg", "Ens_Stacker")
KIT_DL = KIT_BASE[4:]
RAW_MODELS = ("EEGNet-inspired", "ShallowConvNet-inspired", "DeepConvNet-inspired")
SIAMESE = "Siamese"
STOCHASTIC = (*KIT_DL, *FUSION, *RAW_MODELS, SIAMESE)
SPECTRAL_ML = ("Shrinkage-LDA", "Logistic", "kNN")


def measured(operation, times, model, phase):
    start = perf_counter()
    value = operation()
    times[model][phase] += perf_counter() - start
    return value


class ShrinkageLDA:
    def fit(self, x, y):
        positive, negative = x[y == 1], x[y == 0]
        pos_mean, neg_mean = positive.mean(axis=0), negative.mean(axis=0)
        var = .5 * (positive.var(axis=0) + negative.var(axis=0))
        self.direction = (pos_mean - neg_mean) / (.5 * (var + var.mean())).clip(1e-8)
        self.center = .5 * (pos_mean + neg_mean)
        return self

    def decision_function(self, x):
        return (x - self.center) @ self.direction


class StableBinary:
    def __init__(self, constructor, seed):
        self.constructor, self.seed = constructor, seed

    @staticmethod
    def tensor(x):
        value = torch.as_tensor(x, dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite standardized input")
        return value.clamp(-12, 12)

    def fit(self, x, y):
        torch.manual_seed(self.seed)
        self.model = self.constructor(x.shape[1])
        x = self.tensor(x)
        y = torch.as_tensor(y, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        for _ in range(15):
            optimizer.zero_grad()
            logits = self.model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training logits or loss")
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        return self

    def predict_proba(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = torch.cat([self.model(batch) for batch in self.tensor(x).split(128)])
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite inference logits")
            p = logits.sigmoid().numpy()
        return np.column_stack((1 - p, p))


def estimator(name, n_features, seed, ns, fit_rows=None):
    if name == "Shrinkage-LDA":
        return ShrinkageLDA()
    if name == "Logistic":
        return LogisticRegression(C=1., max_iter=500, random_state=seed)
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    if name == "Tangent-SVM":
        return LinearSVC(C=1., max_iter=3000, random_state=seed)
    if name in ("1D-CNN", "BiLSTM-Attention"):
        return StableBinary(ns[{"1D-CNN": "CNN1DNet", "BiLSTM-Attention": "BiLSTMAttentionNet"}[name]], seed)
    model = ns["get_model_pipeline"](name, n_features)
    if name in KIT_DL:
        model.random_state = seed
    if name in ("ExtraTrees", "LightGBM", "XGBoost"):
        model.set_params(n_jobs=1)
    if name == "LightGBM" and fit_rows is not None:
        original = model.get_params()["min_child_samples"]
        model.set_params(min_child_samples=min(original, max(2, original * fit_rows // 64)))
    return model


def prediction(model, x):
    score = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x)
    if not np.isfinite(score).all():
        raise FloatingPointError(f"Nonfinite scores: {type(model).__name__}")
    return score


def claimant_scores(x, y, probes, regions, ns, seed, times, mode="complete"):
    def predict(name, train, labels, query, timing=None):
        if timing is None:
            timing = name
        def fit():
            scaler = StandardScaler().fit(train)
            return scaler, estimator(name, train.shape[1], seed, ns, len(train)).fit(scaler.transform(train), labels)

        scaler, model = measured(fit, times, timing, "fit")
        return measured(lambda: prediction(model, scaler.transform(query)), times, timing, "infer")

    if mode == "core":
        names = (*KIT_BASE[:5], *SPECTRAL_ML) if seed == 42 else ("Res-MLP",)
        return {name: predict(name, x, y, probes) for name in names}
    names = BASE4 if mode == "lightgbm-review" else ((*KIT_BASE, *SPECTRAL_ML) if seed == 42 else KIT_DL)
    scores = {name: predict(name, x, y, probes) for name in names}
    if seed == 42 and mode == "complete":
        regional = [predict("SVM_RBF", x[:, idx], y, probes[:, idx], "Spatial_Fusion") for idx in regions]
        scores["Spatial_Fusion"] = measured(lambda: np.mean(regional, axis=0), times,
                                             "Spatial_Fusion", "infer")
    else:
        for name in BASE4:
            if name not in scores:
                scores[name] = predict(name, x, y, probes)
    fold = np.tile((np.arange(len(y) // 2) >= len(y) // 4).astype(int), 2)
    oof = {}
    for name in BASE4:
        oof[name] = np.empty(len(y))
        for held in (0, 1):
            keep = fold == held
            oof[name][keep] = predict(name, x[~keep], y[~keep], x[keep], f"OOF/{name}")
    columns = np.column_stack([scores[name] for name in BASE4])
    scores["Ens_SoftVote"] = measured(lambda: columns.mean(axis=1), times, "Ens_SoftVote", "infer")
    scores["Ens_RankAvg"] = measured(
        lambda: np.mean([np.searchsorted(np.sort(oof[name]), scores[name]) / len(y) for name in BASE4], axis=0),
        times, "Ens_RankAvg", "infer")
    meta = measured(lambda: LogisticRegression(C=1., random_state=seed).fit(
        np.column_stack([oof[name] for name in BASE4]), y), times, "Ens_Stacker", "fit")
    scores["Ens_Stacker"] = measured(lambda: prediction(meta, columns), times, "Ens_Stacker", "infer")
    if mode == "lightgbm-review":
        names = ("SVM_RBF", "LightGBM", *FUSION) if seed == 42 else FUSION
        return {name: scores[name] for name in names}
    return scores if seed == 42 else {name: scores[name] for name in (*KIT_DL, *FUSION)}


class SquareLogPool(nn.Module):
    def forward(self, x):
        return torch.log(F.avg_pool1d(x.square(), 15, stride=5).clamp_min(1e-6))


def encoder(name, shape):
    if name == SIAMESE:
        return nn.Sequential(nn.Linear(shape[0], 64), nn.LayerNorm(64), nn.ELU(), nn.Linear(64, 32))
    channels = shape[0]
    if name == "EEGNet-inspired":
        body = [nn.Unflatten(1, (1, channels)), nn.Conv2d(1, 4, (1, 15), padding=(0, 7), bias=False),
                nn.BatchNorm2d(4), nn.Conv2d(4, 8, (channels, 1), groups=4, bias=False),
                nn.Flatten(2, 3), nn.BatchNorm1d(8), nn.ELU(), nn.AvgPool1d(4),
                nn.Conv1d(8, 8, 7, padding=3, groups=8, bias=False), nn.Conv1d(8, 16, 1),
                nn.BatchNorm1d(16), nn.ELU()]
    elif name == "ShallowConvNet-inspired":
        body = [nn.Conv1d(channels, 24, 15, padding=7, bias=False),
                nn.BatchNorm1d(24), SquareLogPool()]
    elif name == "DeepConvNet-inspired":
        body = []
        width = channels
        for n in (8, 16, 24, 32):
            body.extend((nn.Conv1d(width, n, 5, padding=2), nn.BatchNorm1d(n), nn.ELU(),
                         nn.MaxPool1d(2)))
            width = n
    else:
        raise ValueError(name)
    pool = nn.AdaptiveAvgPool1d(4)
    depth = {"EEGNet-inspired": 16, "ShallowConvNet-inspired": 24, "DeepConvNet-inspired": 32}[name]
    return nn.Sequential(*body, pool, nn.Flatten(), nn.Linear(depth * 4, 32))


def contrastive(z, labels):
    sim = F.normalize(z) @ F.normalize(z).T / .1
    diagonal = torch.eye(len(z), dtype=torch.bool)
    positives = labels[:, None].eq(labels) & ~diagonal
    log_prob = sim - torch.logsumexp(sim.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True)
    count = positives.sum(dim=1)
    valid = count > 0
    return -(log_prob.masked_fill(~positives, 0).sum(dim=1)[valid] / count[valid]).mean()


def fit_encoder(name, enrollment, seed):
    torch.manual_seed(seed)
    dims = enrollment.shape[2:]
    net = encoder(name, dims)
    head = nn.Linear(32, len(enrollment))
    optimizer = torch.optim.Adam([*net.parameters(), *head.parameters()], lr=.001)
    x = torch.from_numpy(np.ascontiguousarray(enrollment.reshape(-1, *dims), dtype=np.float32))
    y = torch.arange(len(enrollment)).repeat_interleave(enrollment.shape[1])
    for _ in range(8):
        for idx in torch.randperm(len(y)).split(64):
            if len(idx) < 2:
                continue
            if name == SIAMESE and not torch.unique(y[idx], return_counts=True)[1].max() > 1:
                continue
            optimizer.zero_grad()
            z = net(x[idx])
            loss = contrastive(z, y[idx]) if name == SIAMESE else F.cross_entropy(head(z), y[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite reference loss: {name}")
            loss.backward()
            optimizer.step()
    return net.eval()


def embed(net, x):
    shape = x.shape[1:]
    with torch.inference_mode():
        value = torch.cat([F.normalize(net(batch)) for batch in
                           torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).split(64)])
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite embedding")
        return value.numpy().reshape(len(x), -1)
