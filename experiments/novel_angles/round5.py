"""Cross-session EEG verification on the locked equal-budget attempts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark import cosine, matrix_function, upper
from loop2 import (ENROLLMENT_WINDOWS, WINDOWS, boot_metrics, bounds, check_anchor, kit_module,
                   load_features, mixed_probes, recording_arrays, recording_features, rotations, standardized,
                   training_samples)
from round4 import variant_rows
from round5_models import (RAW_MODELS, SIAMESE, STOCHASTIC, claimant_scores, embed, estimator,
                           fit_encoder, measured, prediction)


COHORTS = {"ds007176": (29, WINDOWS, ENROLLMENT_WINDOWS),
           "ds004148": (60, WINDOWS, ENROLLMENT_WINDOWS),
           "ds005385": (128, WINDOWS, ENROLLMENT_WINDOWS),
           "wbcic-shu": (10, 6, 8)}
ARMS = ("ce", "oe", "pair")
KIT_SPATIAL = (
    "FP1 FPZ FP2 AF3 AF4 F7 F5 F3 F1 FZ F2 F4 F6 F8",
    "FC5 FC3 FC1 FCZ FC2 FC4 FC6 C5 C3 C1 CZ C2 C4 C6",
    "CP3 CP1 CPZ CP2 CP4 P7 P5 P3 P1 PZ P2 P4 P6 P8",
    "PO7 PO5 PO3 POZ PO4 PO6 PO8 O1 OZ O2",
)


def recording_with_waveforms(job):
    return recording_arrays(job, True)


def enrollment(x, windows):
    return training_samples(x, x.shape[1] - 2, "mixed", windows)


def average_probes(x, subjects, windows):
    batches = mixed_probes(x[subjects, -2:], windows)
    return np.concatenate([batches[arm] for arm in ARMS], axis=2)


def donor_slots(reference, target, budget):
    return reference[(np.arange(budget) + target) % len(reference)], np.arange(budget)


def labels(budget):
    return np.repeat((1, 0), budget)


def logs_of_covariance(cov):
    n = cov.shape[-1]
    result = np.empty((*cov.shape[:-2], n * (n + 1) // 2), dtype=np.float32)
    flat = result.reshape(-1, result.shape[-1])
    source = cov.reshape(-1, n, n)
    for i in range(0, len(flat), 128):
        block = source[i:i + 128] + np.float32(1e-5) * np.eye(n, dtype=np.float32)
        block /= np.trace(block, axis1=-2, axis2=-1)[..., None, None]
        flat[i:i + 128] = upper(matrix_function(block, np.log), diagonal=True)
    return result


def geometry_scores(cov, split, subjects, windows, budget, times):
    claim, reference, _, _ = split
    train = enrollment(cov, budget)
    query = average_probes(cov, subjects, windows)
    means = measured(lambda: train[claim].mean(axis=1).astype(np.float64), times, "MDM", "fit")
    def distances():
        p = query.astype(np.float64)
        return -(np.sum(p * p, axis=-1, keepdims=True) + np.sum(means * means, axis=-1)
                 - 2 * (p @ means.T))
    mdm = measured(distances, times, "MDM", "infer")
    center = measured(lambda: train[reference].mean(axis=(0, 1)), times, "Tangent-SVM", "fit")
    flat_query = query.reshape(-1, query.shape[-1]) - center
    result = []
    for target in claim:
        donor = train[donor_slots(reference, target, budget)]
        x = np.vstack([train[target], donor]) - center
        model = measured(lambda: estimator("Tangent-SVM", len(center), 42, None).fit(x, labels(budget)),
                         times, "Tangent-SVM", "fit")
        result.append(measured(lambda: prediction(model, flat_query), times, "Tangent-SVM", "infer"))
    return {"MDM": mdm, "Tangent-SVM": np.stack(result, axis=-1).reshape(mdm.shape)}


def embedding_scores(name, enrolled, query, split, windows, seed, times):
    claim, reference, _, _ = split
    net = measured(lambda: fit_encoder(name, enrolled[reference], seed), times, name, "fit")
    def encode_templates():
        x = enrolled[claim]
        return embed(net, x.reshape(-1, *x.shape[2:])).reshape(len(claim), x.shape[1], -1).mean(axis=1)
    templates = measured(encode_templates, times, name, "fit")
    def score():
        shape = query.shape
        z = embed(net, query.reshape(-1, *shape[4:])).reshape(*shape[:4], -1)
        probes = mixed_probes(z, windows)
        return cosine(np.concatenate([probes[arm] for arm in ARMS], axis=2), templates)
    return measured(score, times, name, "infer")


def compare_rotation(arrays, groups, split, windows, budget, ns, regions, seed, times, mode="complete"):
    claim, reference, cal, test = split
    subjects = np.r_[cal, claim, test]
    spectral = arrays["svm"].reshape(*arrays["svm"].shape[:4], -1)
    data = enrollment(spectral, budget)
    query = average_probes(spectral, subjects, windows)
    all_scores = defaultdict(list)
    for target in claim:
        donor = data[donor_slots(reference, target, budget)]
        input_x = np.vstack((data[target], donor))
        scores = claimant_scores(input_x, labels(budget), query.reshape(-1, query.shape[-1]),
                                 regions, ns, seed, times, mode=mode)
        for name, score in scores.items():
            all_scores[name].append(score.reshape(query.shape[:-1]))
    out = {name: np.stack(scores, axis=-1) for name, scores in all_scores.items()}
    if seed == 42:
        x = measured(lambda: standardized(arrays["spectral"], reference, groups.shape[1] - 2, windows),
                     times, "Cosine", "fit")
        template = measured(lambda: enrollment(x, budget)[claim].mean(axis=1), times, "Cosine", "fit")
        out["Cosine"] = measured(lambda: cosine(average_probes(x, subjects, windows), template),
                                 times, "Cosine", "infer")
        if mode != "lightgbm-review":
            out.update(geometry_scores(arrays["logcov"], split, subjects, windows, budget, times))
    if mode == "lightgbm-review":
        return out
    if mode == "complete":
        wave = arrays["waveform"]
        enrolled = enrollment(wave.reshape(*wave.shape[:4], -1), budget)
        enrolled = enrolled.reshape(len(wave), budget, *wave.shape[4:])
        raw_query = wave[subjects, -2:, :, :windows]
        for name in RAW_MODELS:
            out[name] = embedding_scores(name, enrolled, raw_query, split, windows, seed, times)
    def normalize_siamese():
        old = data[reference]
        mean = old.mean(axis=(0, 1))
        scale = old.std(axis=(0, 1)).clip(.05)
        return (data - mean) / scale, (spectral[subjects, -2:, :, :windows] - mean) / scale
    train, probes = measured(normalize_siamese, times, SIAMESE, "fit")
    out[SIAMESE] = embedding_scores(SIAMESE, train, probes, split, windows, seed, times)
    return out


def append_rotation(output, scores, split, windows):
    for name, values in scores.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite rotation scores: {name}")
        for arm, attempts in (("single", values[:, :, :windows]), ("pair", values[:, :, windows:])):
            for visit, ledger in variant_rows(attempts, split, "attempt").items():
                output[name, arm, visit].append(ledger)


def evaluate(ledgers, weights, pairwise=True):
    samples = {key: boot_metrics(np.concatenate(parts), weights) for key, parts in ledgers.items()}
    metrics = {"/".join(key): bounds(value) for key, value in samples.items()}
    comparisons, gates = {}, {}
    for model in {k[0] for k in samples}:
        delta = bounds(samples[model, "pair", "late"] - samples[model, "single", "late"])
        comparisons[f"{model}/pair-minus-single"] = delta
        gates[model] = {
            "directional_eer_gain": delta["eer"]["high"] < 0,
            "frr_margin": delta["frr"]["high"] <= .02,
            "far_margin": delta["far"]["high"] <= .005,
            "far_control": metrics[f"{model}/pair/late"]["far"]["high"] <= .01,
        }
        gates[model]["all"] = all(gates[model].values())
        if pairwise:
            for anchor in ("Cosine", "SVM_RBF"):
                if model != anchor:
                    for arm in ("single", "pair"):
                        comparisons[f"{model}/minus-{anchor}/{arm}"] = bounds(
                            samples[model, arm, "late"] - samples[anchor, arm, "late"])
    return {"metrics": dict(sorted(metrics.items())), "comparisons": dict(sorted(comparisons.items())),
            "criteria": dict(sorted(gates.items()))}


def spatial_regions(channels):
    def indices(names):
        return [20 * channels.index(ch.upper()) + k for ch in names if ch.upper() in channels
                for k in range(20)]
    selected = [indices(region.split()) for region in KIT_SPATIAL]
    if not all(selected):
        raise ValueError("Spatial fusion requires coverage of all four regions")
    return selected


def run(args):
    start = perf_counter()
    mode = "lightgbm-review" if args.lightgbm_review else "core" if args.core else "complete"
    n, windows, budget = COHORTS[args.cohort]
    anchor = check_anchor(args.anchor)
    notebook = Path(__file__).resolve().parents[2] / "07_complete_results_all_models_all_sessions.ipynb"
    kit = kit_module(args.kit)
    _, channels, visits, arrays, shortest = load_features(
        args.fifs, args.cohort, args.kit, notebook, args.workers, expected_people=n, windows=windows,
        enrollment_windows=budget, reader=recording_with_waveforms if mode == "complete" else recording_features)
    cov = arrays.pop("cov")
    if mode != "lightgbm-review":
        arrays["logcov"] = logs_of_covariance(cov)
    del cov
    seconds = perf_counter() - start
    print(f"Prepared scoring features for {n} participants in {seconds:.1f}s", flush=True)
    grouping = (kit.identity_groups([int(v[-1]) for v in visits[:, -1]]) if args.cohort == "ds007176"
                else [tuple(sorted(g)) for g in np.array_split(np.random.default_rng(42).permutation(n), 5)])
    ns = kit.notebook_env(notebook)
    regions = spatial_regions(channels)
    rng = np.random.default_rng(177)
    weights = np.vstack([np.ones(n, int), rng.multinomial(n, np.full(n, 1 / n), size=500)])
    ledgers = {seed: defaultdict(list) for seed in (42, 43)}
    times = {seed: defaultdict(lambda: dict(fit=0., infer=0.)) for seed in ledgers}
    for i, split in enumerate(rotations(grouping), 1):
        for seed in ledgers:
            scores = compare_rotation(arrays, visits, split, windows, budget, ns, regions, seed, times[seed],
                                      mode=mode)
            append_rotation(ledgers[seed], scores, split, windows)
        print(f"Scored rotation {i}/20", flush=True)
    first = evaluate(ledgers[42], weights)
    second = evaluate(ledgers[43], weights, pairwise=False)
    seed_summary = {}
    for model in STOCHASTIC:
        if model not in second["criteria"]:
            continue
        for arm in ("single", "pair"):
            key = f"{model}/{arm}/late"
            seed_summary[key] = {}
            for metric in ("eer", "frr", "far"):
                values = [obj["metrics"][key][metric]["point"] for obj in (first, second)]
                seed_summary[key][metric] = {"values": values, "sd": float(np.std(values, ddof=1))}
    first.update(dataset=args.cohort, subjects=n, shortest_recording=shortest, draws=500,
                 probe_windows_per_state=windows, enrollment_windows=budget, anchor_frr_penalties_pp=anchor,
                 roster=mode,
                 feature_seconds=seconds, timing_seconds=dict(sorted(times[42].items())), seeds=[42, 43],
                 seed_summary=seed_summary, regional_feature_counts=[len(r) for r in regions])
    partial = args.output.with_suffix(".partial")
    partial.write_text(json.dumps(first, indent=2, allow_nan=False) + "\n")
    partial.replace(args.output)
    print(f"Completed {len(first['criteria'])} model rows: {args.output.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=tuple(COHORTS))
    for label in ("fifs", "kit", "anchor", "output"):
        parser.add_argument(label, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--core", action="store_true", help="Score only the fixed fast ML/spectral subset")
    modes.add_argument("--lightgbm-review", action="store_true",
                       help="Score LightGBM and its three ensembles plus the two comparison anchors")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
