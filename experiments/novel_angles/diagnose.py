"""Audit person-level uncertainty, metadata coverage, and preprocessing sensitivity."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from benchmark import TASKS, Setting, cosine, evaluate, ledger, load, metric, representations, roles, threshold
from fetch_raw import inventory


def calibration_stress(people, channels, arrays):
    setting = Setting("spectral", "mixed")
    far_counterfactuals, stability, rejects = [], [], []
    person_frr = np.empty(len(people))
    impostor_accepts = np.empty(len(people))
    for split in roles(len(people)):
        targets, ref, cal, test = split
        x = representations(arrays, channels, ref, "spectral", "all", "none")
        paths = [evaluate(x, split, setting, task) for task in TASKS]
        cal_scores = np.concatenate(
            [score(probe[cal, 2], templates).reshape(len(cal), -1)
             for probe, templates, score in paths], axis=1)
        cut = threshold(cal_scores)
        subjects = np.concatenate((targets, test))
        scores = [(score(probe[subjects, 3], templates), task) for (probe, templates, score), task
                  in zip(paths, TASKS)]
        scores_by_target = np.concatenate(
            [a[np.arange(len(targets)), :, np.arange(len(targets))] for a, _ in scores], axis=1)
        person_frr[targets] = (scores_by_target < cut).mean(axis=1)
        scores_by_impostor = np.concatenate([a[len(targets):].reshape(len(test), -1) for a, _ in scores], axis=1)
        impostor_accepts[test] = (scores_by_impostor >= cut).mean(axis=1)
        first, second = x[targets, :2, :, :8].mean(axis=(2, 3)).transpose(1, 0, 2)
        stability.extend(np.diag(cosine(first, second)))
        rejects.extend(person_frr[targets])
        fixed = metric(np.concatenate([ledger(a, subjects, targets, cut) for a, _ in scores]),
                       np.ones(len(people)))[1]
        for ignore in range(len(cal)):
            loo = threshold(np.delete(cal_scores, ignore, axis=0))
            new = metric(np.concatenate([ledger(a, subjects, targets, loo) for a, _ in scores]),
                         np.ones(len(people)))[1]
            far_counterfactuals.append(float(new - fixed))
    rho, p = spearmanr(stability, rejects)
    return {
        "leave_one_calibrator_out_delta_far_range": [min(far_counterfactuals), max(far_counterfactuals)],
        "target_frr_quartiles": np.quantile(person_frr, [.25, .5, .75]).tolist(),
        "impostors_with_any_false_accept": int(np.count_nonzero(impostor_accepts)),
        "stability_vs_frr_spearman": {"rho": float(rho), "nominal_p": float(p)},
    }


def dates(path):
    with path.open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    return [dt.datetime.fromisoformat(r["acq_time"]) for r in rows
            if r.get("acq_time") not in (None, "", "n/a")]


def metadata(root):
    subjects = inventory(root)
    duration = []
    complete = 0
    for person, visits in subjects.items():
        known = {visit: dates(root / person / ("ses-" + visit) /
                              f"{person}_ses-{visit}_scans.tsv") for visit in visits}
        complete += all(known.values())
        late = next(v for v in visits if v in ("V3", "V4"))
        if known["V2"] and known[late]:
            duration.append((known[late][0] - known["V2"][0]).days)
    return {
        "complete_timing": complete,
        "V2_late_timing": len(duration),
        "late_visit_counts": {v: sum(v in visits for visits in subjects.values()) for v in ("V3", "V4")},
    }


def recipe_sensitivity(raw_dir, recipe_dir):
    raw, recipe = [], []
    for path in sorted(recipe_dir.glob("sub-*.npz")):
        with np.load(path) as a, np.load(raw_dir / path.name) as b:
            common = sorted(set(a["channels"]) & set(b["channels"]))
            spectra = [x["spectral"][:24, [x["channels"].tolist().index(c) for c in common]].mean(axis=0)
                       for x in (a, b)]
        raw.append(spectra[1].ravel())
        recipe.append(spectra[0].ravel())
    distance = np.linalg.norm(np.array(raw) - np.array(recipe), axis=1)
    return {"shared_recordings": len(raw), "median_log_band_feature_rmse": float(
        np.median(distance) / np.sqrt(len(raw[0])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("raw_checkout", type=Path)
    parser.add_argument("--recipe-cache", type=Path)
    args = parser.parse_args()
    report = {"calibration": calibration_stress(*load(args.cache)), "metadata": metadata(args.raw_checkout)}
    if args.recipe_cache:
        report["preprocessing"] = recipe_sensitivity(args.cache, args.recipe_cache)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
