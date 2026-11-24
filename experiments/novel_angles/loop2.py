"""Run the locked matched-window and covariance verification analysis."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
import json
from multiprocessing import get_context
from pathlib import Path
import re

import numpy as np

from benchmark import cosine, tangent, threshold
from extract import features


WINDOWS = 24
ENROLLMENT_WINDOWS = 32
STATES = ("CE", "OE")
PANELS = {"frontal": "FP1 FP2 F3 F4 F7 F8", "posterior": "P3 P4 P7 P8 O1 O2"}
ANCHORS = {"mismatch": 19.4, "mixed": 5.5}
METRICS = ("frr", "far", "eer")


def check_anchor(path):
    with path.open(newline="") as file:
        result = {r["enrollment"]: float(r["frr_delta_pp"]) for r in csv.DictReader(file)
                  if (r["threshold_policy"], r["model"], r["probe"], r["reference"])
                  == ("condition-aware", "SVM_RBF", "both", "matched")}
    if set(ANCHORS) - result.keys() or any(round(result[k], 1) != v for k, v in ANCHORS.items()):
        raise ValueError(f"Cross-condition anchor not reproduced: {result}")
    return {k: result[k] for k in ANCHORS}


def kit_module(path):
    spec = spec_from_file_location("cross_condition_kit", path / "run_cross_condition.py")
    kit = module_from_spec(spec)
    spec.loader.exec_module(kit)
    return kit


def initialize_features(kit_path, notebook):
    global FEATURE_NS
    FEATURE_NS = kit_module(kit_path).notebook_env(notebook, extract=True)


def recording_features(job):
    return recording_arrays(job, False)


def recording_arrays(job, waveform):
    path, cap = job
    ep = FEATURE_NS["mne"].read_epochs(path, preload=True, verbose="ERROR")
    if len(ep) < cap or ep.info["sfreq"] != 250 or len(ep.times) != 1251:
        raise ValueError(f"Expected at least {cap} recipe epochs: {path} (got {len(ep)})")
    x = ep.get_data()[:cap]
    spectral, _, cov, _ = features(x[..., :1250])
    svm = FEATURE_NS["extract_spectral_features_tensor"](x, sfreq=250)
    arrays = {"spectral": spectral.astype(np.float32), "cov": cov.astype(np.float32), "svm": svm}
    if waveform:
        from scipy.signal import resample_poly

        arrays["waveform"] = resample_poly(x[..., :1250].astype(np.float32), 1, 5, axis=-1)
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError(f"Nonfinite features: {path}")
    return [c.upper() for c in ep.ch_names], len(ep), arrays


def load_features(root, cohort, kit, notebook, workers, expected_people=None,
                  windows=WINDOWS, enrollment_windows=ENROLLMENT_WINDOWS, reader=recording_features):
    files = {}
    for path in sorted(root.rglob("*_desc-preproc_epo.fif")):
        match = re.fullmatch(r"(sub-[^_]+)_ses-(V\d+)_task-(CE|OE)_desc-preproc_epo", path.stem)
        if match is None or match.groups() in files:
            raise ValueError(f"Unexpected or duplicate recipe file: {path}")
        files[match.groups()] = path
    people = sorted({person for person, _, _ in files})
    rest = cohort in ("ds004148", "ds005385", "wbcic-shu")
    prefix = ("V0", "V2") if rest else ("V0", "V1", "V2")
    visits = [tuple(sorted({v for person, v, _ in files if person == s})) for s in people]
    if expected_people is None:
        expected_people = {"ds004148": 60, "ds007176": 29}[cohort]
    if len(people) != expected_people or any(
        vs[:-1] != prefix or vs[-1] not in (("V3",) if rest else ("V3", "V4")) for vs in visits
    ):
        raise ValueError(f"Incomplete cohort: {len(people)} subjects, {len(files)} recordings")
    expected = [(s, v, state) for s, vs in zip(people, visits, strict=True) for v in vs for state in STATES]
    if set(expected) != files.keys():
        raise ValueError("FIFs do not form complete CE/OE pairs")
    cap = max(enrollment_windows, windows) if rest else windows
    jobs = [(files[k], cap if k[1] == "V0" and rest else windows) for k in expected]
    counts = []
    channels = None
    arrays = {}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=initialize_features,
                             initargs=(kit, notebook)) as pool:
        for i, (names, n_epochs, features_by_name) in enumerate(pool.map(reader, jobs)):
            if channels is None:
                channels = names
            if channels != names or len(set(names)) != len(names):
                raise ValueError(f"Unexpected channel inventory: {expected[i]}")
            counts.append(n_epochs)
            for key, data in features_by_name.items():
                if key not in arrays:
                    arrays[key] = np.zeros((len(people), len(prefix) + 1, 2, cap, *data.shape[1:]),
                                           dtype=data.dtype)
                index = np.unravel_index(i, arrays[key].shape[:3])
                arrays[key][index][:len(data)] = data
    return people, channels, np.array(visits), arrays, min(counts)


def rotations(groups):
    population = set().union(*map(set, groups))
    for group_index, claim in enumerate(groups):
        others = [g for i, g in enumerate(groups) if i != group_index]
        for shift in range(4):
            cal, test, first, second = (others[(shift + i) % 4] for i in range(4))
            train = sorted((*first, *second))
            packed = (*claim, *train, *cal, *test)
            if len(packed) != len(population) or set(packed) != population:
                raise ValueError("Identity groups must be disjoint")
            yield tuple(np.array(ids, dtype=int) for ids in (claim, train, cal, test))


def mixed_probes(x, windows=WINDOWS):
    ce, oe = x[..., 0, :windows, :], x[..., 1, :windows, :]
    ce_repeat = (ce[..., ::2, :] + ce[..., 1::2, :]) / 2
    oe_repeat = (oe[..., ::2, :] + oe[..., 1::2, :]) / 2
    return {"ce": ce_repeat, "oe": oe_repeat, "pair": (ce + oe) / 2}


def contrasts(x):
    ce, oe = x[..., 0, :, :], x[..., 1, :, :]
    return {"cross": ce - oe, "ce-repeat": ce[..., ::2, :] - ce[..., 1::2, :],
            "oe-repeat": oe[..., ::2, :] - oe[..., 1::2, :]}


def training_samples(x, n_visits, enrollment, enrollment_windows=ENROLLMENT_WINDOWS):
    budget = enrollment_windows // n_visits
    if enrollment == "mixed":
        return x[:, :n_visits, :, :budget // 2].reshape(len(x), enrollment_windows, -1)
    return x[:, :n_visits, STATES.index(enrollment), :budget].reshape(len(x), enrollment_windows, -1)


def cosine_scores(train, probes, claim):
    return cosine(probes, train[claim].mean(axis=1))


def svm_scores(train, probes, claim, reference, ns):
    shape = probes.shape[:-1]
    flat = probes.reshape(-1, probes.shape[-1])
    scores = np.empty((*shape, len(claim)))
    labels = np.repeat([1, 0], train.shape[1])
    for i, target in enumerate(claim):
        slots = np.arange(train.shape[1])
        # Rotate slot-to-donor assignment without using calibration or probe outcomes.
        donors = reference[(slots + target) % len(reference)]
        x = np.vstack([train[target], train[donors, slots]])
        scale = ns["StandardScaler"]().fit(x)
        model = ns["get_model_pipeline"]("SVM_RBF", x.shape[1]).fit(scale.transform(x), labels)
        scores[..., i] = model.predict_proba(scale.transform(flat))[:, 1].reshape(shape)
    return scores


def rows(scores, subjects, targets, cutoffs):
    subject = np.broadcast_to(subjects[:, None, None], scores.shape)
    target = np.broadcast_to(targets, scores.shape)
    return np.stack([scores, subject == target, target, subject, np.broadcast_to(cutoffs, scores.shape)],
                    axis=-1).reshape(-1, 5)


def score_modes(trains, probes, split, sources, scorer, pools):
    claim, ref, cal, test = split
    subjects = np.r_[cal, claim, test]
    final_subjects = np.r_[claim, test]
    final_idx = np.arange(len(cal), len(subjects))
    keep = (final_subjects[:, None] == claim) | ~np.isin(final_subjects[:, None], claim)
    modes_by_source = defaultdict(list)
    for mode, source in sources.items():
        modes_by_source[source].append(mode)
    out = {}
    for source, modes in modes_by_source.items():
        batches = [probes[m][subjects, -2:] for m in modes]
        scored = scorer(trains[source], np.concatenate(batches, axis=2), claim, ref)
        parts = np.split(scored, np.cumsum([x.shape[2] for x in batches])[:-1], axis=2)
        out.update(zip(modes, parts, strict=True))
    result = {}
    for pool, modes in pools.items():
        cal_score = np.concatenate([out[m][:len(cal), 0] for m in modes], axis=1)
        cutoff = np.array([threshold(cal_score[..., j]) for j in range(len(claim))])
        for j, visit in enumerate(("cal-unseen", "late")):
            parts = []
            for mode in modes:
                value = out[mode][final_idx, j]
                take = np.broadcast_to(keep[:, None, :], value.shape)
                item = rows(value, final_subjects, claim, cutoff)[take.ravel()]
                parts.append(item)
                if len(modes) > 1:
                    result[visit, mode] = item
            result[visit, pool] = np.concatenate(parts)
    return result


def standardized(x, reference, n_visits, windows=WINDOWS):
    x = x.reshape(*x.shape[:4], -1)
    old = x[reference, :n_visits, :, :windows]
    center = old.mean(axis=(0, 1, 2, 3))
    scale = old.std(axis=(0, 1, 2, 3)).clip(.05)
    return (x - center) / scale


def align(cov, reference, visits, n_visits, mode, windows=WINDOWS):
    donors = cov[reference, ..., :windows, :, :]
    if mode == "none":
        return tangent(cov, donors[:, :n_visits].mean(axis=(0, 1, 2, 3)))
    if mode == "condition":
        return tangent(cov, donors[:, :n_visits].mean(axis=(0, 1, 3), keepdims=True))
    out = np.empty((*cov.shape[:-2], cov.shape[-1] * (cov.shape[-1] + 1) // 2))
    for idx in range(visits.shape[1]):
        for visit in np.unique(visits[:, idx]):
            source = reference[visits[reference, idx] == visit]
            if not len(source):
                raise ValueError(f"No same-visit reference donors at {visit}")
            target = visits[:, idx] == visit
            mean = cov[source, idx:idx+1, :, :windows].mean(axis=(0, 3), keepdims=True)
            out[target, idx:idx+1] = tangent(cov[target, idx:idx+1], mean)
    return out


def spectral_data(arrays, channels):
    x = arrays["spectral"]
    non_alpha = x[..., [0, 1, 3, 4]]
    non_alpha = non_alpha - np.logaddexp.reduce(non_alpha, axis=-1, keepdims=True)
    yield "cosine", x
    yield "svm", arrays["svm"]
    yield "cosine-no-alpha", non_alpha
    for panel, names in PANELS.items():
        idx = [channels.index(c) for c in names.split()]
        yield f"cosine-{panel}", x[..., idx, :]
        yield f"svm-{panel}", arrays["svm"][..., idx, :]


def ledgers(arrays, channels, visits, groups, ns, windows=WINDOWS, enrollment_windows=ENROLLMENT_WINDOWS):
    n_visits = visits.shape[1] - 2
    result = defaultdict(list)
    mixed_pools = {"single": ("ce", "oe"), "pair": ("pair",)}
    to_mixed = {m: "mixed" for m in ("ce", "oe", "pair")}

    def similarity(train, probe, claim, ref):
        return cosine_scores(train, probe, claim)

    def svm(train, probe, claim, ref):
        return svm_scores(train, probe, claim, ref, ns)

    def add(kind, data):
        for key, value in data.items():
            result[(kind, *key)].append(value)

    spectral = list(spectral_data(arrays, channels))
    for i, split in enumerate(rotations(groups), 1):
        reference = split[1]
        for kind, data in spectral:
            svc = kind.startswith("svm")
            x = data.reshape(*data.shape[:4], -1) if svc else standardized(data, reference, n_visits, windows)
            trains = {"mixed": training_samples(x, n_visits, "mixed", enrollment_windows)}
            add(kind, score_modes(trains, mixed_probes(x, windows), split, to_mixed,
                                  svm if svc else similarity, mixed_pools))
            if kind == "cosine":
                diff = contrasts(x)
                cross_budget = enrollment_windows // 2
                cross_count = cross_budget // n_visits
                trains = {m: data[:, :n_visits, :cross_count].reshape(len(x), cross_budget, -1)
                          for m, data in diff.items()}
                probes = contrasts(x[..., :windows, :])
                add("cosine-contrast", score_modes(
                    trains, probes, split, {m: m for m in probes}, similarity,
                    {"within": ("ce-repeat", "oe-repeat"), "cross": ("cross",)},
                ))
        for mode in ("none", "condition", "visit-condition"):
            x = align(arrays["cov"], reference, visits, n_visits, mode, windows)
            for enroll in (("mixed",) if mode == "none" else STATES):
                trains = {"mixed": training_samples(x, n_visits, enroll, enrollment_windows)}
                add(f"cov-{mode}-{enroll}", score_modes(trains, mixed_probes(x, windows), split, to_mixed,
                                                       similarity, mixed_pools))
        print(f"Scored {i}/20 identity rotations", flush=True)
    return {key: np.concatenate(parts) for key, parts in result.items()}


def boot_metrics(data, draws):
    order = np.argsort(-data[:, 0], kind="stable")
    score, genuine, target, subject, cutoff = data[order].T
    genuine = genuine.astype(bool)
    target, subject = target.astype(int), subject.astype(int)
    reject, accept = (score < cutoff) & genuine, (score >= cutoff) & ~genuine
    bounds = np.r_[np.flatnonzero(score[1:] != score[:-1]), len(score) - 1]
    rates = np.full((len(draws), len(METRICS)), np.nan)
    for i, draw in enumerate(draws):
        weight = draw[target] * np.where(genuine, 1, draw[subject])
        pos, neg = weight[genuine].sum(), weight[~genuine].sum()
        if not (pos and neg):
            continue
        fpr = np.r_[0., np.cumsum(weight * ~genuine)[bounds] / neg]
        fnr = 1 - np.r_[0., np.cumsum(weight * genuine)[bounds] / pos]
        k = np.searchsorted(fpr - fnr, 0)
        slope = fpr[k] - fnr[k] - fpr[k - 1] + fnr[k - 1]
        eer = fpr[k - 1] + (fnr[k - 1] - fpr[k - 1]) / slope * (fpr[k] - fpr[k - 1])
        rates[i] = weight[reject].sum() / pos, weight[accept].sum() / neg, eer
    return rates


def bounds(values):
    low, high = np.nanquantile(values[1:], (.025, .975), axis=0)
    return {k: {"point": float(p), "low": float(l), "high": float(h)}
            for k, p, l, h in zip(METRICS, values[0], low, high, strict=True)}


def late_order(raw, people, visits):
    order = []
    for person, visit in zip(people, visits[:, -1], strict=True):
        path = raw / person / f"ses-{visit}" / f"{person}_ses-{visit}_scans.tsv"
        stamps = {}
        if path.is_file():
            with path.open(newline="") as file:
                for row in csv.DictReader(file, delimiter="\t"):
                    for label in STATES:
                        stamp = row.get("acq_time")
                        if f"_task-{label}_eeg.vhdr" in row["filename"] and stamp not in ("n/a", "", None):
                            stamps[label] = datetime.fromisoformat(stamp)
        if len(stamps) != 2 or stamps["CE"] == stamps["OE"]:
            order.append("unknown")
        else:
            order.append("ce-first" if stamps["CE"] < stamps["OE"] else "oe-first")
    return np.array(order)


def summarize(data, subjects, draws, order):
    rng = np.random.default_rng(177)
    weights = np.vstack([np.ones(subjects, int), rng.multinomial(subjects, np.full(subjects, 1 / subjects),
                                                                size=draws)])
    values = {key: boot_metrics(value, weights) for key, value in data.items()}
    output = {"metrics": {"/".join(key): bounds(v) for key, v in values.items()}, "contrasts": {}}
    delta = output["contrasts"]
    for kind in sorted({kind for kind, _, _ in values}):
        if kind == "cosine-contrast":
            for mode in ("within", "ce-repeat", "oe-repeat"):
                delta[f"{kind}/cross-minus-{mode}"] = bounds(
                    values[kind, "late", "cross"] - values[kind, "late", mode])
            continue
        for mode in ("single", "ce", "oe"):
            delta[f"{kind}/pair-minus-{mode}"] = bounds(
                values[kind, "late", "pair"] - values[kind, "late", mode])
        for mode in ("pair", "single"):
            delta[f"{kind}/late-minus-cal-unseen/{mode}"] = bounds(
                values[kind, "late", mode] - values[kind, "cal-unseen", mode])
        if kind.startswith("cov-") and kind != "cov-none-mixed":
            delta[f"{kind}/minus-unaligned-mixed"] = bounds(
                values[kind, "late", "single"] - values["cov-none-mixed", "late", "single"])
    output["order"] = {}
    for label in np.unique(order):
        who = np.flatnonzero(order == label)
        if len(who) < 2:
            continue
        report = {"probe_identities": len(who)}
        for scorer in ("cosine", "svm"):
            series = []
            for mode in ("pair", "single"):
                row = data[scorer, "late", mode]
                series.append(boot_metrics(row[np.isin(row[:, 3], who)], weights))
            report[scorer] = bounds(series[0] - series[1])
        output["order"][label] = report
    output["criteria"] = {}
    for kind in ("cosine", "svm"):
        main = delta[f"{kind}/pair-minus-single"]
        far = output["metrics"][f"{kind}/late/pair"]["far"]
        gates = {"directional_eer_gain": main["eer"]["high"] < 0,
                 "frr_margin": main["frr"]["high"] <= .02,
                 "far_margin": main["far"]["high"] <= .005,
                 "far_control": far["high"] <= .01}
        output["criteria"][kind] = {**gates, "deployment_go": all(gates.values())}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("ds007176", "ds004148"))
    parser.add_argument("fifs", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("kit", type=Path)
    parser.add_argument("anchor", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()
    anchor = check_anchor(args.anchor)
    kit = kit_module(args.kit)
    notebook = Path(__file__).resolve().parents[2] / "07_complete_results_all_models_all_sessions.ipynb"
    people, channels, visits, arrays, min_epochs = load_features(
        args.fifs, args.cohort, args.kit, notebook, args.workers,
    )
    groups = kit.identity_groups([int(v[-1]) for v in visits[:, -1]]) if args.cohort == "ds007176" else [
        tuple(sorted(g)) for g in np.array_split(np.random.default_rng(42).permutation(len(people)), 5)
    ]
    model_ns = kit.notebook_env(notebook)
    result = summarize(ledgers(arrays, channels, visits, groups, model_ns), len(people), args.draws,
                       late_order(args.raw, people, visits))
    result.update(dataset=args.cohort, subjects=len(people), shortest_recording=min_epochs,
                  draws=args.draws, anchor_frr_penalties_pp=anchor)
    partial = args.output.with_suffix(".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    partial.replace(args.output)
    for scorer in ("cosine", "svm"):
        diff = result["contrasts"][f"{scorer}/pair-minus-single"]["eer"]
        print(f"{scorer}: EER difference {diff['point'] * 100:.2f} pp "
              f"[{diff['low'] * 100:.2f}, {diff['high'] * 100:.2f}]", flush=True)


if __name__ == "__main__":
    main()
