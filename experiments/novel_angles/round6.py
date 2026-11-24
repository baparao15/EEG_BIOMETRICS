"""Retrospective robustness checks with aggregate-only output."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, chi2, norm, t
from sklearn.preprocessing import StandardScaler

from benchmark import FAR, cosine
from loop2 import boot_metrics, bounds, kit_module, load_features, rotations, standardized
from round4 import variant_rows
from round5 import COHORTS, average_probes, donor_slots, enrollment, geometry_scores, labels, logs_of_covariance
from round5_models import estimator, prediction


ROOT = Path(__file__).resolve().parent
MODELS = ("Cosine", "Logistic", "Tangent-SVM")
FUSED = "Cosine+Logistic"


def previous(slug):
    suffix = "-parallel" if slug == "ds005385" else ""
    return json.loads((ROOT / f"round5-{slug}{suffix}.json").read_text())


def standard_error(interval):
    se = (interval["high"] - interval["low"]) / (2 * norm.ppf(.975))
    if se <= 0:
        raise ValueError("Zero-width EER interval cannot support normal approximation")
    return se


def adjustments(p):
    p = np.asarray(p)
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, len(p) + 1)
    values = p[order]
    holm = np.maximum.accumulate(values * (len(p) + 1 - ranks)).clip(max=1)
    bh = np.minimum.accumulate((values * len(p) / ranks)[::-1])[::-1].clip(max=1)
    return holm[np.argsort(order)], bh[np.argsort(order)]


def meta_analysis(effects):
    points = np.array([d["point"] for d in effects])
    variances = np.array([standard_error(d)**2 for d in effects])
    fixed = 1 / variances
    q = float(np.sum(fixed * (points - np.average(points, weights=fixed))**2))
    dof = len(points) - 1
    tau2 = max(0., (q - dof) / (fixed.sum() - np.sum(fixed**2) / fixed.sum()))
    weights = 1 / (variances + tau2)
    point = np.average(points, weights=weights)
    scale = max(1., np.sum(weights * (points - point)**2) / dof)
    span = t.ppf(.975, dof) * np.sqrt(scale / weights.sum())
    return dict(point=float(point), low=float(point-span), high=float(point+span), q=q, q_p=float(chi2.sf(q, dof)),
                i2=float(max(0., 1 - dof/q)) if q else 0., tau2=float(tau2))


def aggregate():
    previous_runs = {slug: previous(slug) for slug in COHORTS}
    effects = [(slug, m, d["comparisons"][f"{m}/pair-minus-single"]["eer"])
               for slug, d in previous_runs.items() for m in d["criteria"]]
    p = [2 * norm.sf(abs(e["point"]) / standard_error(e)) for _, _, e in effects]
    holm, bh = adjustments(p)
    return {
        "tests": {f"{slug}/{m}": dict(delta=e, p=float(raw), holm=float(h), bh=float(b))
                  for (slug, m, e), raw, h, b in zip(effects, p, holm, bh, strict=True)},
        "seeds": {slug: {key.split("/")[0]: dict(zip(d["seeds"], metrics["eer"]["values"], strict=True))
                        for key, metrics in d["seed_summary"].items() if key.endswith("/pair/late")}
                  for slug, d in previous_runs.items()},
        "cosine_meta": meta_analysis([e for _, m, e in effects if m == "Cosine"]),
    }


def audited_rotations(groups, n):
    if sorted(i for group in groups for i in group) != list(range(n)):
        raise ValueError("Identity groups must partition participants")
    counts = np.zeros((4, n), dtype=int)
    splits = list(rotations(groups))
    for split in splits:
        combined = np.concatenate(split)
        if not np.array_equal(np.sort(combined), np.arange(n)):
            raise ValueError("Rotation roles overlap or omit identities")
        for role, ids in enumerate(split):
            counts[role, ids] += 1
    assert len(splits) == 20
    np.testing.assert_array_equal(counts, np.repeat([[4], [8], [4], [4]], n, axis=1))
    return splits


def logistic_fit_data(spectral, split, budget):
    claim, reference, _, _ = split
    data = enrollment(spectral, budget)
    return [np.vstack((data[target], data[donor_slots(reference, target, budget)])) for target in claim]


def targeted_scores(arrays, split, widths, full, budget):
    claim, reference, cal, test = split
    subjects = np.r_[cal, claim, test]
    spectral = arrays["svm"].reshape(*arrays["svm"].shape[:4], -1)
    cosine_x = standardized(arrays["spectral"], reference, spectral.shape[1] - 2, full)
    template = enrollment(cosine_x, budget)[claim].mean(axis=1)
    fitted = []
    for train in logistic_fit_data(spectral, split, budget):
        scaler = StandardScaler().fit(train)
        model = estimator("Logistic", train.shape[1], 42, None).fit(scaler.transform(train), labels(budget))
        fitted.append((scaler, model))
    for width in widths:
        query = average_probes(spectral, subjects, width)
        flat = query.reshape(-1, query.shape[-1])
        logistic = np.stack([prediction(model, scaler.transform(flat)).reshape(query.shape[:-1])
                             for scaler, model in fitted], axis=-1)
        clocks = defaultdict(lambda: dict(fit=0., infer=0.))
        geometry = geometry_scores(arrays["logcov"], split, subjects, width, budget, clocks)
        yield width, {"Cosine": cosine(average_probes(cosine_x, subjects, width), template),
                      "Logistic": logistic, "Tangent-SVM": geometry["Tangent-SVM"]}


def calibration_ranks(scores, ncal):
    return np.stack([np.searchsorted(np.sort(scores[:ncal, 0, :, j].ravel()), scores[..., j], side="right")
                     for j in range(scores.shape[-1])], axis=-1)


def fused_calibration_score(cosine_scores, logistic_scores, ncal):
    rank_sum = calibration_ranks(cosine_scores, ncal) + calibration_ranks(logistic_scores, ncal)
    return rank_sum / (2 * ncal * cosine_scores.shape[2])


def audited_rows(scores, split):
    visits = variant_rows(scores, split, "attempt")
    changed = scores.copy()
    changed[:, 1] = -changed[:, 1] + 17
    other = variant_rows(changed, split, "attempt")
    np.testing.assert_array_equal(visits["cal-unseen"], other["cal-unseen"])
    np.testing.assert_array_equal(visits["late"][:, 4], other["late"][:, 4])
    return visits


def cp_upper(errors, n):
    return float(beta.ppf(.95, errors + 1, n - errors)) if errors < n else 1.


def minimum_n(errors):
    lo, hi = errors, max(1, errors + 1)
    while cp_upper(errors, hi) > FAR:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cp_upper(errors, mid) <= FAR:
            hi = mid
        else:
            lo = mid
    return hi


def far_power(ledger, draws):
    negatives = ledger[ledger[:, 1] == 0]
    hit = negatives[:, 0] >= negatives[:, 4]
    n = draws.shape[1]
    target, subject = negatives[:, 2:4].astype(int).T
    total = np.zeros((n, n), int)
    accepted = np.zeros_like(total)
    np.add.at(total, (target, subject), 1)
    np.add.at(accepted, (target, subject), hit)
    rate = lambda matrix: np.einsum("bi,bi->b", draws @ matrix, draws)
    den = rate(total)
    clustered = rate(accepted)[den > 0] / den[den > 0]
    identities, accepting = len(np.unique(subject)), len(np.unique(subject[hit]))
    errors = int(hit.sum())
    return dict(attempts=len(negatives), accepted=errors, attempts_cp=cp_upper(errors, len(negatives)),
                impostor_identities=identities, identities_ever_accepted=accepting,
                identities_cp=cp_upper(accepting, identities), clustered_upper=float(np.quantile(clustered[1:], .95)),
                zero_error_minimum=minimum_n(0), minimum_attempts_at_fixed_errors=minimum_n(errors),
                minimum_identities_at_fixed_events=minimum_n(accepting),
                zero_event_identity_upper=cp_upper(0, identities))


def summarize_heterogeneity(single, pair, ones):
    changes = []
    for target in range(ones.size):
        a = boot_metrics(single[single[:, 2] == target], ones[None, :])[0]
        b = boot_metrics(pair[pair[:, 2] == target], ones[None, :])[0]
        changes.append(b - a)
    changes = np.array(changes)
    leave_one = ones - np.eye(ones.size, dtype=int)
    jack = (boot_metrics(pair, leave_one) - boot_metrics(single, leave_one))[:, 2]
    counts = {}
    for col, name in ((0, "frr"), (2, "eer")):
        x = changes[:, col]
        assert np.isfinite(x).all()
        counts[name] = dict(improved=int((x < -1e-12).sum()), tied=int((abs(x) <= 1e-12).sum()),
                            worsened=int((x > 1e-12).sum()))
    assert np.isfinite(jack).all()
    return dict(counts, pooled_eer_leave_one_out=[float(jack.min()), float(jack.max())])


def run(args):
    n, full, budget = COHORTS[args.ds]
    widths = (8, 16, 24) if full == 24 else (2, 4, 6)
    notebook = ROOT.parents[1] / "07_complete_results_all_models_all_sessions.ipynb"
    _, _, visits, arrays, _ = load_features(args.fifs, args.ds, args.kit, notebook, args.workers,
                                           expected_people=n, windows=full, enrollment_windows=budget)
    groups = (kit_module(args.kit).identity_groups([int(v[-1]) for v in visits[:, -1]]) if args.ds == "ds007176"
              else [tuple(sorted(g)) for g in np.array_split(np.random.default_rng(42).permutation(n), 5)])
    splits = audited_rotations(groups, n)
    print(f"Audited {len(splits)} disjoint rotations over {n} identities; visits: {sorted(set(map(tuple, visits)))}",
          flush=True)
    arrays["logcov"] = logs_of_covariance(arrays.pop("cov"))
    ledgers = defaultdict(list)
    for i, split in enumerate(splits, 1):
        for width, scores in targeted_scores(arrays, split, widths, full, budget):
            for arm, index in (("single", slice(None, width)), ("pair", slice(width, None))):
                arm_scores = {m: v[:, :, index] for m, v in scores.items()}
                if width == full and arm == "pair":
                    arm_scores[FUSED] = fused_calibration_score(
                        arm_scores["Cosine"], arm_scores["Logistic"], len(split[2]))
                for model, score in arm_scores.items():
                    for visit, ledger in audited_rows(score, split).items():
                        if visit == "late":
                            ledgers[width, model, arm].append(ledger)
        print(f"Scored audited rotation {i}/{len(splits)}", flush=True)
    ones = np.ones(n, int)
    draws = np.vstack([ones, np.random.default_rng(177).multinomial(n, np.full(n, 1/n), size=500)])
    anchor = np.concatenate(ledgers[full, "Cosine", "pair"])
    power = far_power(anchor, draws)
    print("Cosine FAR precision:", json.dumps(power), flush=True)
    established = previous(args.ds)
    samples = {key: boot_metrics(np.concatenate(ledger), draws) for key, ledger in ledgers.items()}
    sensitivity = {}
    heterogeneity = {}
    for width in widths:
        sensitivity[str(width)] = {}
        for m in MODELS:
            single, pair = samples[width, m, "single"], samples[width, m, "pair"]
            delta = bounds(pair - single)
            paired = bounds(pair)
            directional = delta["eer"]["high"] < 0
            frr = delta["frr"]["high"] <= .02
            far = delta["far"]["high"] <= .005
            control = paired["far"]["high"] <= FAR
            sensitivity[str(width)][m] = dict(single=bounds(single), pair=paired, delta=delta,
                                              gates=dict(directional_eer_gain=directional, frr_margin=frr,
                                                         far_margin=far, far_control=control,
                                                         all=directional and frr and far and control))
            if width == full:
                assert paired == established["metrics"][f"{m}/pair/late"], (m, "pair")
                assert bounds(single) == established["metrics"][f"{m}/single/late"], (m, "single")
                assert delta == established["comparisons"][f"{m}/pair-minus-single"], (m, "delta")
                heterogeneity[m] = summarize_heterogeneity(np.concatenate(ledgers[full, m, "single"]),
                                                           np.concatenate(ledgers[full, m, "pair"]), ones)
    fused = samples[full, FUSED, "pair"]
    ledger = np.concatenate(ledgers[full, FUSED, "pair"])
    ties = len(ledger) - len(np.unique(ledger[:, 0]))
    return dict(dataset=args.ds, subjects=n, split_audit=dict(rotations=len(splits),
                visit_patterns=sorted(set(map(tuple, visits))), decision_interventions=sum(map(len, ledgers.values()))),
                power=power, sensitivity=sensitivity, heterogeneity=heterogeneity,
                fusion=dict(metrics=bounds(fused), versus={m: bounds(fused - samples[full, m, "pair"])
                             for m in ("Cosine", "Logistic")}, repeat_score_rows=ties, score_rows=len(ledger)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ds", choices=("aggregate", *COHORTS))
    parser.add_argument("output", type=Path)
    parser.add_argument("--fifs", type=Path)
    parser.add_argument("--kit", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.ds != "aggregate" and (args.fifs is None or args.kit is None):
        parser.error("cohort scoring requires --fifs and --kit")
    result = aggregate() if args.ds == "aggregate" else run(args)
    temp = args.output.with_suffix(".partial")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    print(f"Wrote {args.output.name}", flush=True)


if __name__ == "__main__":
    main()
