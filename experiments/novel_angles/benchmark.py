"""Chronological verification with disjoint subject roles and frozen thresholds."""
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv

import numpy as np
from sklearn.metrics import roc_curve


FAR = 0.01
SEED = 176
VISITS = ("V0", "V1", "V2", "LATE")
TASKS = ("CE", "OE")
BLOCKS = 12
MIN_EPOCHS = 24
SUBSETS = {
    "4-frontal": "FP1 FP2 F7 F8",
    "8-frontal": "FP1 FP2 F7 F8 AF3 AF4 F3 F4",
    "16-spread": "FP1 FP2 F7 F8 AF3 AF4 F3 F4 C3 C4 T7 T8 P3 P4 O1 O2",
}


def load(root):
    files = {}
    late_visits = {}
    for path in sorted(root.glob("sub-*.npz")):
        person, visit, task = path.stem.split("_")
        if visit in ("V3", "V4"):
            if late_visits.setdefault(person, visit) != visit:
                raise ValueError(f"Expected exactly one later visit for {person}")
            visit = "LATE"
        key = person, visit, task
        if key in files:
            raise ValueError(f"Duplicate recording {key}")
        files[key] = path
    people = sorted({key[0] for key in files})
    expected = [(s, v, t) for s in people for v in VISITS for t in TASKS]
    if len(people) < 8 or set(expected) != files.keys():
        raise ValueError("Need at least eight subjects with V0/V1/V2 and one V3/V4, each paired")
    with np.load(files[expected[0]]) as data:
        channels = data["channels"].tolist()
    arrays = {name: [] for name in ("spectral", "slope", "cov", "imag")}
    for key in expected:
        with np.load(files[key]) as data:
            if set(data["channels"]) != set(channels):
                raise ValueError(f"Channel inventory mismatch: {files[key]}")
            indices = [data["channels"].tolist().index(c) for c in channels]
            for name in arrays:
                x = data[name][:MIN_EPOCHS, indices]
                if name in ("cov", "imag"):
                    x = x[..., indices]
                arrays[name].append(x)
    features = {
        name: np.stack(items).reshape(len(people), 4, 2, MIN_EPOCHS, *items[0].shape[1:])
        for name, items in arrays.items()
    }
    return people, channels, {**features, "late_visits": np.array([late_visits[s] for s in people])}


def roles(n):
    groups = np.array_split(np.random.default_rng(SEED).permutation(n), 4)
    for i in range(4):
        yield tuple(groups[(i + j) % 4] for j in range(4))


def matrix_function(matrix, function):
    value, vector = np.linalg.eigh(matrix.astype(np.float64))
    return (vector * function(value.clip(1e-8))[..., None, :]) @ vector.swapaxes(-1, -2)


def upper(x, diagonal):
    i, j = np.triu_indices(x.shape[-1], k=0 if diagonal else 1)
    return x[..., i, j] * np.where(i == j, 1, np.sqrt(2))


def tangent(cov, reference):
    inverse_root = matrix_function(reference, lambda x: x ** -0.5)
    return upper(matrix_function(inverse_root @ cov @ inverse_root, np.log), diagonal=True)


def cosine(probe, templates):
    probe = probe / np.linalg.norm(probe, axis=-1, keepdims=True).clip(1e-9)
    templates = templates / np.linalg.norm(templates, axis=-1, keepdims=True).clip(1e-9)
    return probe @ templates.T


def probe_features(x, mode):
    if mode in TASKS:
        return x[:, :, TASKS.index(mode)].reshape(*x.shape[:2], BLOCKS, 2, -1).mean(axis=3)
    ce, oe = x[:, :, 0], x[:, :, 1]
    if mode == "pair-mean":
        return (ce + oe) / 2
    if mode == "pair-delta":
        return ce - oe
    if mode == "pair-joint":
        return np.concatenate([probe_features(x, "pair-mean"), probe_features(x, "pair-delta")], axis=-1)
    if mode == "ce-repeat-delta":
        return ce[:, :, ::2] - ce[:, :, 1::2]
    raise ValueError(mode)


def enrollment(x, mode, epochs):
    if mode in TASKS:
        return x[:, :2, TASKS.index(mode), :epochs].mean(axis=(1, 2))
    if mode == "mixed":
        return x[:, :2, :, :epochs // 2].mean(axis=(1, 2, 3))
    return probe_features(x, mode)[:, :2, :epochs // 2].mean(axis=(1, 2))


def threshold(impostor):
    ordered = np.sort(impostor.astype(np.float64).ravel())[::-1]
    return np.nextafter(ordered[int(np.floor(FAR * len(ordered)))], np.inf)


def score_transform(raw, z_background, t_background, method):
    if method == "raw":
        return raw
    z = (raw - z_background.mean(axis=(0, 1))) / z_background.std(axis=(0, 1)).clip(1e-8)
    t = ((raw - t_background.mean(axis=-1, keepdims=True))
         / t_background.std(axis=-1, keepdims=True).clip(1e-8))
    return {"Z": z, "T": t, "S": (z + t) / 2}[method]


@dataclass(frozen=True)
class Setting:
    feature: str
    enroll: str
    score: str = "raw"
    channels: str = "all"
    epochs: int = 16
    alignment: str = "none"
    probes: tuple = TASKS
    adaptation: bool = False
    calibration: str = "pooled"


def representations(arrays, channels, ref, feature, subset, alignment):
    select = [channels.index(c) for c in SUBSETS[subset].split()] if subset in SUBSETS else slice(None)
    if feature == "cov":
        cov = arrays["cov"][..., select, :][..., select]
        if alignment == "condition":
            reference = cov[ref, :2].mean(axis=(0, 1, 3), keepdims=True)
        elif alignment == "visit-condition":
            historical = tangent(cov[:, :3], cov[ref, :3].mean(axis=(0, 3), keepdims=True))
            late = np.full_like(historical[:, :1], np.nan)
            visit = arrays["late_visits"]
            for label in np.unique(visit[ref]):
                donors = ref[visit[ref] == label]
                reference = cov[donors, 3:4].mean(axis=(0, 3), keepdims=True)
                late[visit == label] = tangent(cov[visit == label, 3:4], reference)
            return np.concatenate([historical, late], axis=1)
        else:
            reference = cov[ref, :2].mean(axis=(0, 1, 2, 3))
        return tangent(cov, reference)
    x = arrays[feature]
    if feature == "imag":
        x = upper(x[..., select, :][..., select], diagonal=False)
    elif feature in ("spectral", "slope"):
        x = x[..., select, :] if feature == "spectral" else x[..., select]
        x = x.reshape(*x.shape[:4], -1)
    center = x[ref, :2].mean(axis=(0, 1, 2, 3))
    scale = x[ref, :2].std(axis=(0, 1, 2, 3)).clip(0.05)
    return (x - center) / scale


def metric(rows, counts):
    score, genuine, target, speaker, cutoff = rows.T
    genuine = genuine.astype(bool)
    weight = counts[target.astype(int)]
    weight = weight * np.where(genuine, 1, counts[speaker.astype(int)])
    accept = score >= cutoff
    ga, ia = weight[genuine].sum(), weight[~genuine].sum()
    if not (ga and ia):
        return np.full(3, np.nan)
    fpr, tpr, _ = roc_curve(genuine, score, sample_weight=weight, drop_intermediate=False)
    fnr = 1 - tpr
    ix = np.searchsorted(fpr - fnr, 0)
    if ix > 0:
        fraction = (fnr[ix - 1] - fpr[ix - 1]) / (fpr[ix] - fnr[ix] - fpr[ix - 1] + fnr[ix - 1])
        eer = fpr[ix - 1] + fraction * (fpr[ix] - fpr[ix - 1])
    else:
        eer = fpr[ix]
    return np.array([weight[genuine & ~accept].sum() / ga, weight[~genuine & accept].sum() / ia, eer])


def summarize(rows, draws):
    rows = np.concatenate(rows)
    point = metric(rows, np.ones(draws.shape[1], dtype=int))
    interval = np.nanquantile([metric(rows, draw) for draw in draws], (0.025, 0.975), axis=0)
    return np.concatenate([point, interval.ravel()]), rows


def ledger(scores, population, targets, cutoff):
    shape = scores.shape
    subject = np.broadcast_to(population[:, None, None], shape)
    target = np.broadcast_to(targets, shape)
    genuine = subject == target
    eligible = genuine | ~np.isin(subject, targets)
    return np.stack([scores, genuine, target, subject, np.full(shape, cutoff)], axis=-1)[eligible]


def evaluate(x, split, setting, mode):
    targets, ref, _, _ = split
    all_templates = enrollment(x, setting.enroll, setting.epochs)
    templates = all_templates[targets]
    probe = probe_features(x, mode)
    if mode in TASKS:
        background = np.concatenate([probe_features(x, task)[ref, :2] for task in TASKS], axis=2)
        background = background.reshape(len(ref), -1, probe.shape[-1])
    else:
        background = probe[ref, :2].reshape(len(ref), -1, probe.shape[-1])
    z = cosine(background, templates)

    def score(x, templates):
        return score_transform(cosine(x, templates), z, cosine(x, all_templates[ref]), setting.score)

    return probe, templates, score


def run(people, channels, arrays, settings, draws):
    output, paired = [], {}
    cache = {}
    for setting in settings:
        rows = {}
        for i, split in enumerate(roles(len(people))):
            targets, ref, cal, test = split
            key = (i, setting.feature, setting.channels, setting.alignment)
            if key not in cache:
                cache[key] = representations(arrays, channels, ref, *key[1:])
            x = cache[key]
            evaluated = [evaluate(x, split, setting, mode) for mode in setting.probes]
            calibration_scores = [score(probe[cal, 2], templates).ravel()
                                  for probe, templates, score in evaluated]
            pooled_cutoff = threshold(np.concatenate(calibration_scores))
            for mode, (probe, templates, score), calibration in zip(
                setting.probes, evaluated, calibration_scores
            ):
                cutoff = threshold(calibration) if setting.calibration == "condition" else pooled_cutoff
                for label, visit, attempts in (("V2-unseen", 2, slice(None)),
                                                ("late", 3, slice(None)),
                                                ("late-tail", 3, slice(BLOCKS // 2, None))):
                    if label == "late-tail" and setting not in (
                        Setting("spectral", "mixed"), Setting("spectral", "mixed", adaptation=True)
                    ):
                        continue
                    if setting.adaptation and label != "late-tail":
                        continue
                    population = np.concatenate([targets, test])
                    if visit == 3 and setting.alignment == "visit-condition":
                        visits = arrays["late_visits"]
                        population = population[np.isin(visits[population], visits[ref])]
                    current = templates
                    if setting.adaptation:
                        current = templates.copy()
                        early = probe[population, visit, :BLOCKS // 2]
                        for window in range(early.shape[1]):
                            vectors = early[:, window]
                            allowed = ((population[:, None] == targets) |
                                       ~np.isin(population[:, None], targets))
                            accept = (score(vectors, current).astype(np.float64) >= cutoff) & allowed
                            accepted = accept.sum(axis=0)
                            means = (accept.T @ vectors) / accepted[:, None].clip(1)
                            current[accepted > 0] = 0.9 * current[accepted > 0] + 0.1 * means[accepted > 0]
                    scores = score(probe[population, visit, attempts], current)
                    item = ledger(scores, population, targets, cutoff)
                    rows.setdefault((label, mode), []).append(item)
                    rows.setdefault((label, "pooled"), []).append(item)
        for (visit, mode), items in sorted(rows.items()):
            metrics, scores = summarize(items, draws)
            row = (setting.feature, setting.enroll, setting.score, setting.channels, setting.epochs * 10,
                   setting.alignment, setting.adaptation, setting.calibration, visit, mode)
            output.append([*row, *metrics])
            paired[row] = scores
        if setting.feature == "cov" and not any(s.feature == "cov" and s.alignment == setting.alignment
                                               for s in settings[settings.index(setting) + 1:]):
            for key in list(cache):
                if key[1] == "cov" and key[3] == setting.alignment:
                    del cache[key]
        print(f"Evaluated {setting}", flush=True)
    return output, paired


def settings():
    result = [Setting("spectral", enroll, score) for enroll in (*TASKS, "mixed")
              for score in ("raw", "Z", "T", "S")]
    result += [Setting("cov", enroll, alignment=alignment)
               for alignment in ("none", "condition", "visit-condition") for enroll in (*TASKS, "mixed")]
    result += [Setting("spectral", "mixed", channels=subset, epochs=epochs)
               for subset in (*SUBSETS, "all") for epochs in (4, 8, 16)
               if (subset, epochs) != ("all", 16)]
    result += [Setting("spectral", "mixed", adaptation=True)]
    result += [Setting("spectral", enroll, calibration="condition") for enroll in (*TASKS, "mixed")]
    result += [Setting(feat, "mixed") for feat in ("slope", "imag")]
    result += [Setting(feat, mode, probes=(mode,)) for feat in ("spectral", "slope", "imag")
               for mode in ("pair-mean", "pair-delta", "pair-joint", "ce-repeat-delta")]
    return result


def contrasts(ledgers, draws):
    def key(feature, enroll, probe="pooled", score="raw", alignment="none", update=False,
            visit="late", calibration="pooled"):
        return feature, enroll, score, "all", 160, alignment, update, calibration, visit, probe

    pairs = [
        ("spectral CE→OE minus CE→CE", key("spectral", "CE", "OE"), key("spectral", "CE", "CE")),
        ("spectral OE→CE minus OE→OE", key("spectral", "OE", "CE"), key("spectral", "OE", "OE")),
        ("spectral mixed S minus raw", key("spectral", "mixed", score="S"), key("spectral", "mixed")),
        ("spectral mixed late minus V2", key("spectral", "mixed"), key("spectral", "mixed", visit="V2-unseen")),
        ("spectral mixed Z minus raw", key("spectral", "mixed", score="Z"), key("spectral", "mixed")),
        ("spectral mixed T minus raw", key("spectral", "mixed", score="T"), key("spectral", "mixed")),
        ("spectral mixed condition thresholds minus pooled", key("spectral", "mixed", calibration="condition"),
         key("spectral", "mixed")),
        ("cov CE condition align minus mixed", key("cov", "CE", alignment="condition"), key("cov", "mixed")),
        ("cov OE condition align minus mixed", key("cov", "OE", alignment="condition"), key("cov", "mixed")),
        ("spectral accepted update minus frozen", key("spectral", "mixed", update=True, visit="late-tail"),
         key("spectral", "mixed", visit="late-tail")),
    ]
    for feature in ("spectral", "slope", "imag"):
        pairs.extend([
            (feature + " paired joint minus mean", key(feature, "pair-joint"), key(feature, "pair-mean")),
            (feature + " paired delta minus repeat delta", key(feature, "pair-delta"),
             key(feature, "ce-repeat-delta")),
        ])
    for name, a, b in pairs:
        first, second = ledgers[a], ledgers[b]
        ones = np.ones(draws.shape[1], dtype=int)
        point = metric(first, ones) - metric(second, ones)
        interval = np.nanquantile([metric(first, w) - metric(second, w) for w in draws], (0.025, 0.975), axis=0)
        yield [name, *point, *interval.ravel()]


METRICS = ["FRR", "FAR", "EER", "FRR_low", "FAR_low", "EER_low", "FRR_high", "FAR_high", "EER_high"]


def write(path, fields, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow([*fields, *METRICS])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()
    people, channels, arrays = load(args.cache)
    rng = np.random.default_rng(SEED + 1)
    draws = rng.multinomial(len(people), np.full(len(people), 1 / len(people)), size=args.draws)
    results, ledgers = run(people, channels, arrays, settings(), draws)
    write(args.output, ["feature", "enroll", "normalization", "channels", "enroll_seconds", "alignment",
                        "update", "calibration", "visit", "probe"], results)
    write(args.output.with_name(args.output.stem + "-contrasts.csv"), ["comparison"], contrasts(ledgers, draws))
    print(f"Subjects: {len(people)}; recorded {len(results)} metric rows and paired contrasts")


if __name__ == "__main__":
    main()
