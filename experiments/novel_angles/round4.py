"""Prespecified cosine follow-ups using the locked paired-probe budget."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from benchmark import cosine, threshold
from loop2 import (ENROLLMENT_WINDOWS, PANELS, WINDOWS, boot_metrics, bounds, check_anchor,
                   kit_module, load_features, mixed_probes, rotations, rows, score_modes,
                   standardized, training_samples)


VARIANTS = ("worst-state", "reactivity", "repeatability", "identity-maximum")


def repeatability_weights(enrollment):
    between = enrollment.mean(axis=2).var(axis=0).mean(axis=0)
    within = enrollment.var(axis=2).mean(axis=(0, 1))
    total = between + within
    return np.divide(between, total, out=np.full_like(total, .25), where=total > 0).clip(.25, 1.)


def variant_scores(x, alpha, split, windows=WINDOWS, enrollment_windows=ENROLLMENT_WINDOWS):
    claim, reference, cal, test = split
    subjects = np.r_[cal, claim, test]
    n_visits = x.shape[1] - 2
    per_state = enrollment_windows // (2 * n_visits)
    enrolled = x[:, :n_visits, :, :per_state].transpose(0, 2, 1, 3, 4)
    enrolled = enrolled.reshape(len(x), 2, -1, x.shape[-1])
    probes = x[subjects, -2:, :, :windows]
    pair = (probes[:, :, 0] + probes[:, :, 1]) / 2
    mixed = training_samples(x, n_visits, "mixed", enrollment_windows)
    templates = mixed[claim].mean(axis=1)
    pair_scores = cosine(pair, templates)
    worst_state = np.minimum(*(cosine(probes[:, :, state], enrolled[claim, state].mean(axis=1))
                               for state in range(2)))
    reactivity = alpha[:, :, 0] - alpha[:, :, 1]
    expected = reactivity[claim, :n_visits, :per_state].mean(axis=(1, 2))
    observed = reactivity[subjects, -2:, :windows]
    penalty = .25 * np.abs(observed[..., None] - expected).clip(max=2.)
    root = np.sqrt(repeatability_weights(enrolled[reference]))
    return {"worst-state": worst_state, "reactivity": pair_scores - penalty,
            "repeatability": cosine(pair * root, templates * root), "identity-maximum": pair_scores}


def variant_rows(scores, split, kind):
    claim, _, cal, test = split
    calibration = scores[:len(cal), 0]
    if kind == "identity-maximum":
        calibration = calibration.max(axis=1)
    cutoffs = np.array([threshold(calibration[..., j]) for j in range(len(claim))])
    subjects = np.r_[claim, test]
    keep = (subjects[:, None] == claim) | ~np.isin(subjects[:, None], claim)
    result = {}
    for j, visit in enumerate(("cal-unseen", "late")):
        values = scores[len(cal):, j]
        take = np.broadcast_to(keep[:, None, :], values.shape).ravel()
        result[visit] = rows(values, subjects, claim, cutoffs)[take]
    return result


def ledgers(data, channels, visits, groups, windows=WINDOWS, enrollment_windows=ENROLLMENT_WINDOWS):
    packed = defaultdict(list)
    posterior = [channels.index(c) for c in PANELS["posterior"].split()]
    alpha = data[..., posterior, 2].mean(axis=-1)
    n_visits = visits.shape[1] - 2
    for i, split in enumerate(rotations(groups), 1):
        x = standardized(data, split[1], n_visits, windows)
        base = score_modes(
            {"mixed": training_samples(x, n_visits, "mixed", enrollment_windows)},
            mixed_probes(x, windows), split, dict.fromkeys(("ce", "oe", "pair"), "mixed"),
            lambda train, probe, claim, _: cosine(probe, train[claim].mean(axis=1)),
            {"single": ("ce", "oe"), "pair": ("pair",)},
        )
        for (visit, kind), value in base.items():
            if kind in ("pair", "single"):
                packed[kind, visit].append(value)
        for name, values in variant_scores(x, alpha, split, windows, enrollment_windows).items():
            for visit, ledger in variant_rows(values, split, name).items():
                packed[name, visit].append(ledger)
        print(f"Scored rotation {i}/20", flush=True)
    return {key: np.concatenate(value) for key, value in packed.items()}


def gates(vs_single, vs_pair, far):
    tests = {"directional_eer_gain": vs_single["eer"]["high"] < 0,
             "frr_margin": vs_single["frr"]["high"] <= .02,
             "far_margin": vs_single["far"]["high"] <= .005,
             "far_control": far["high"] <= .01,
             "far_gain_vs_pair": vs_pair["far"]["high"] < 0,
             "nonzero_observed_far": far["point"] > 0}
    return {**tests, "replication_candidate": all(tests.values())}


def summarize(data, n):
    rng = np.random.default_rng(177)
    draws = np.vstack([np.ones(n, int), rng.multinomial(n, np.full(n, 1 / n), size=500)])
    values = {key: boot_metrics(ledger, draws) for key, ledger in data.items()}
    metrics = {"/".join(key): bounds(value) for key, value in values.items()}
    contrasts = {}
    criteria = {}
    for variant in VARIANTS:
        late = values[variant, "late"]
        vs_single = bounds(late - values["single", "late"])
        vs_pair = bounds(late - values["pair", "late"])
        contrasts[f"{variant}/minus-single"] = vs_single
        contrasts[f"{variant}/minus-pair"] = vs_pair
        criteria[variant] = gates(vs_single, vs_pair, metrics[f"{variant}/late"]["far"])
    return {"metrics": metrics, "contrasts": contrasts, "criteria": criteria}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("ds007176", "ds004148"))
    parser.add_argument("fifs", type=Path)
    parser.add_argument("kit", type=Path)
    parser.add_argument("anchor", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    anchor = check_anchor(args.anchor)
    notebook = Path(__file__).resolve().parents[2] / "07_complete_results_all_models_all_sessions.ipynb"
    subjects, channels, visits, arrays, shortest = load_features(
        args.fifs, args.cohort, args.kit, notebook, args.workers)
    n = len(subjects)
    groups = (kit_module(args.kit).identity_groups([int(v[-1]) for v in visits[:, -1]])
              if args.cohort == "ds007176" else
              [tuple(sorted(g)) for g in np.array_split(np.random.default_rng(42).permutation(n), 5)])
    result = summarize(ledgers(arrays["spectral"], channels, visits, groups), n)
    result.update(dataset=args.cohort, subjects=n, shortest_recording=shortest, draws=500,
                  anchor_frr_penalties_pp=anchor)
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    for variant in VARIANTS:
        far = result["metrics"][f"{variant}/late"]["far"]
        print(f"{variant}: later FAR {100 * far['point']:.2f}%, upper {100 * far['high']:.2f}%; "
              f"candidate: {result['criteria'][variant]['replication_candidate']}")


if __name__ == "__main__":
    main()
