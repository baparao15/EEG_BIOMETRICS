"""Retrospective paired-probe checks of claimant and impostor role choices."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import OneClassSVM

from benchmark import threshold
from loop2 import METRICS, boot_metrics, kit_module, load_features, rows
from round5 import COHORTS, average_probes, donor_slots, enrollment, labels
from round5_models import estimator, prediction
from round6 import audited_rotations


ONE_CLASS = ("One-Class SVM", "Isolation Forest", "Cosine kNN")
SUPERVISED = ("Logistic", "SVM_RBF")


def claimant_only(train, probes):
    scale = StandardScaler().fit(train)
    x, q = scale.transform(train), scale.transform(probes)
    neighbors = normalize(probes) @ normalize(train).T
    return {
        "One-Class SVM": prediction(OneClassSVM(nu=.1).fit(x), q),
        "Isolation Forest": prediction(IsolationForest(n_estimators=100, random_state=42).fit(x), q),
        "Cosine kNN": np.partition(neighbors, -3, axis=-1)[:, -3:].mean(axis=-1),
    }


def genuine_cutoff(scores):
    ordered = np.sort(scores.ravel())
    return ordered[int(.1 * len(ordered))]


def ledger(scores, target, people, cutoff):
    return rows(scores[people, 1, :, None], people, np.array([target]), np.array([cutoff]))


def metrics(ledgers, n):
    data = np.concatenate(ledgers)
    values = boot_metrics(data, np.ones((1, n), dtype=int))[0]
    return dict(zip(METRICS, map(float, values), strict=True))


def group_range(fars):
    bounds = np.array([(min(v), max(v)) for v in fars.values()])
    return {
        "group_far_min": float(bounds[:, 0].min()),
        "group_far_max": float(bounds[:, 1].max()),
        "median_within_claimant_span": float(np.median(bounds[:, 1] - bounds[:, 0])),
        "max_within_claimant_span": float(np.max(bounds[:, 1] - bounds[:, 0])),
        "claimants_with_zero_and_at_least_15pct": int(((bounds[:, 0] == 0) & (bounds[:, 1] >= .15)).sum()),
    }


def compare(train, probes, groups, ns):
    n, budget, features = train.shape
    flat = probes.reshape(-1, features)
    shape = probes.shape[:-1]
    everyone = np.arange(n)
    one = [{m: v.reshape(shape) for m, v in claimant_only(x, flat).items()} for x in train]
    splits = audited_rotations(groups, n)
    ledgers = defaultdict(list)
    group_fars = defaultdict(lambda: defaultdict(list))
    pairs = defaultdict(list)
    donor_outcomes = defaultdict(lambda: np.zeros((2, 2), dtype=int))
    for claim, reference, cal, test in splits:
        fitted = []
        for target in claim:
            donor_ids, slots = donor_slots(reference, target, budget)
            x = np.vstack((train[target], train[donor_ids, slots]))
            scale = StandardScaler().fit(x)
            x, q = scale.transform(x), scale.transform(flat)
            binary = {m: prediction(estimator(m, features, 42, ns).fit(x, labels(budget)), q).reshape(shape)
                      for m in SUPERVISED}
            fitted.append((target, np.unique(donor_ids), one[target] | binary))
        global_cutoffs = {m: threshold(np.stack([scores[m][cal, 0] for _, _, scores in fitted]))
                          for m in (*ONE_CLASS, *SUPERVISED)}
        for target, donors, models in fitted:
            people = np.r_[target, test]
            pairs["rotation"].extend((int(target), int(other)) for other in test)
            for m, scores in models.items():
                per_user = threshold(scores[cal, 0])
                for policy, cutoff in (("per_claimant", per_user), ("pooled", global_cutoffs[m])):
                    ledgers["rotation", m, policy].append(ledger(scores, target, people, cutoff))
                    group_fars[m, policy][target].append(float(np.mean(scores[test, 1] >= cutoff)))
                if m in SUPERVISED:
                    for i, ids in enumerate((donors, test)):
                        hits = scores[ids, 1] >= per_user
                        donor_outcomes[m][i] += (hits.sum(), hits.size)

    # One claimant per fit: preserve the same out-of-block thresholds and add the
    # missing within-block pairs under that claimant's first disjoint cutoff.
    pairs["crossfit"] = pairs["rotation"].copy()
    for m in ONE_CLASS:
        ledgers["crossfit", m, "per_claimant"] = ledgers["rotation", m, "per_claimant"].copy()
    for group_idx, group in enumerate(groups):
        cal = np.asarray(groups[0 if group_idx else 1])
        assert set(group).isdisjoint(cal)
        for target in group:
            peers = np.asarray([subject for subject in group if subject != target], dtype=int)
            pairs["crossfit"].extend((int(target), int(peer)) for peer in peers)
            # Baseline genuine attempts have four weights, one per rotation;
            # add only the missing impostor rows so the genuine side stays fixed.
            for m, scores in one[target].items():
                ledgers["crossfit", m, "per_claimant"].append(
                    ledger(scores, target, peers, threshold(scores[cal, 0])))

    genuine_pool = {m: genuine_cutoff(np.stack([one[i][m][i, 0] for i in everyone])) for m in ONE_CLASS}
    for target in everyone:
        for m, scores in one[target].items():
            for policy, cutoff in (("per_claimant", genuine_cutoff(scores[target, 0])),
                                   ("pooled", genuine_pool[m])):
                ledgers["all_other_genuine_cal", m, policy].append(ledger(scores, target, everyone, cutoff))

    assert len(set(pairs["rotation"])) == len(pairs["rotation"])
    assert sorted(pairs["crossfit"]) == [(a, b) for a in range(n) for b in range(n) if a != b]
    results = defaultdict(lambda: defaultdict(dict))
    for (design, model, policy), data in ledgers.items():
        value = metrics(data, n)
        if design == "rotation":
            value["across_test_groups"] = group_range(group_fars[model, policy])
        results[design][model][policy] = value
    membership = {
        model: {name: {"accepts": int(hits), "attempts": int(total), "far": float(hits / total)}
                for name, (hits, total) in zip(("seen_donors", "unseen_test"), counts, strict=True)}
        for model, counts in donor_outcomes.items()
    }
    return {
        "role_sizes": [{"claim": a, "reference": b, "calibration": c, "test": d, "rotations": count}
                       for (a, b, c, d), count in sorted(Counter(tuple(map(len, s)) for s in splits).items())],
        "identity_pairs": {k: len(v) for k, v in pairs.items()},
        "results": results,
        "supervised_membership": membership,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("ds007176", "ds004148", "ds005385"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--fifs", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    n, windows, budget = COHORTS[args.cohort]
    notebook = Path(__file__).resolve().parents[2] / "07_complete_results_all_models_all_sessions.ipynb"
    _, _, visits, arrays, _ = load_features(args.fifs, args.cohort, args.kit, notebook, args.workers,
                                           expected_people=n, windows=windows, enrollment_windows=budget)
    kit = kit_module(args.kit)
    groups = (kit.identity_groups([int(v[-1]) for v in visits[:, -1]]) if args.cohort == "ds007176"
              else [tuple(sorted(g)) for g in np.array_split(np.random.default_rng(42).permutation(n), 5)])
    spectral = arrays["svm"].reshape(*arrays["svm"].shape[:4], -1)
    train = enrollment(spectral, budget)
    probes = average_probes(spectral, np.arange(n), windows)[:, :, windows:]
    result = compare(train, probes, groups, kit.notebook_env(notebook))
    result.update(cohort=args.cohort, subjects=n, retained_probe_windows_per_state=windows,
                  enrollment_windows=budget, test_visits=dict(sorted(Counter(visits[:, -1]).items())))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output.name}")


if __name__ == "__main__":
    main()
