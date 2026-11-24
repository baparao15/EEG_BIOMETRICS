import json
import math
from pathlib import Path

repo = Path(__file__).resolve().parent
inputs = (("ds007176","ds007176"), ("ds004148","RestCog"), ("ds005385","Dortmund"), ("wbcic-shu","WBCIC-SHU"))
title = dict(inputs)
models = ("Cosine", "Logistic", "Tangent-SVM")


def load(slug):
    return json.loads((repo / f"round6-{slug}.json").read_text())


base = load("aggregate")
cohorts = [(name, load(slug)) for slug, name in inputs]
tests = base["tests"]


def table(header, rows):
    def line(values):
        return "| " + " | ".join(map(str, values)) + " |"
    return "\n".join([line(header), line(["---"] * len(header)), *(line(r) for r in rows)])


def p(x):
    return f"{x:.4g}"


def pct(x):
    return f"{100*x:.2f}%"


def pp(x):
    return f"{100*x:+.2f}".replace("-", "−")


def delta(d):
    return f"{pp(d['point'])} [{pp(d['low'])}, {pp(d['high'])}]"


def block(*parts):
    return "\n\n".join(parts) + "\n"


passes = {k: [name for name, d in tests.items() if d[k] <= .05] for k in ("holm","bh")}
meta = base["cosine_meta"]
seed_models = list(next(iter(base["seeds"].values())))
seed_spread = [(abs(row["43"] - row["42"]), title[slug], model)
               for slug, models_ in base["seeds"].items() for model, row in models_.items()]
spread, spread_cohort, spread_model = max(seed_spread)
power_rows = []
split_rows = []
sensitivity_rows = []
hetero_rows = []
fusion_rows = []
fusion_operating_rows = []
for name, d in cohorts:
    power = d["power"]
    split = d["split_audit"]
    power_rows.append((name, f"{power['accepted']:,}/{power['attempts']:,}", pct(power["attempts_cp"]),
                       pct(power["clustered_upper"]), f"{power['identities_ever_accepted']}/{power['impostor_identities']}",
                       pct(power["identities_cp"]), pct(power["zero_event_identity_upper"]),
                       f"{power['minimum_attempts_at_fixed_errors']:,} / {power['minimum_identities_at_fixed_events']:,}"))
    split_rows.append((name, d["subjects"], split["rotations"], "; ".join("/".join(v) for v in split["visit_patterns"]),
                       split["decision_interventions"]))
    for width, scores in d["sensitivity"].items():
        sensitivity_rows.append((name, width, *(delta(scores[m]["delta"]["eer"]) for m in models),
                                 ", ".join(m for m in models if scores[m]["gates"]["all"]) or "None"))
    for m in models:
        h = d["heterogeneity"][m]
        triple = lambda k: "/".join(str(h[k][s]) for s in ("improved", "tied", "worsened"))
        hetero_rows.append((name, m, triple("eer"), triple("frr"), " to ".join(pp(x) for x in h["pooled_eer_leave_one_out"])))
    f = d["fusion"]
    fusion_rows.append((name, " / ".join(pct(f["metrics"][k]["point"]) for k in ("eer","frr","far")),
                        *(delta(f["versus"][m]["eer"]) for m in ("Cosine", "Logistic")),
                        f"{f['repeat_score_rows']:,}/{f['score_rows']:,}"))
    for m in ("Cosine", "Logistic"):
        fusion_operating_rows.append((name, m, *(delta(f["versus"][m][k]) for k in ("frr", "far"))))


text = block(
    "# Round 6: robustness checks and new ideas",
    "Retrospective analysis of the existing cohorts; no new participants. See the [frozen analysis plan](ROUND6_PROTOCOL.md) and the [claim audit](CLAIM_AUDIT.md#round-6-evidence). The findings are not a deployment or security claim.",
    "## Multiple testing",
    f"Among {len(tests)} two-sided pairing-EER contrasts, {len(passes['holm'])} passed Holm and {len(passes['bh'])} passed BH at .05. These tests approximate standard errors from Round 5 percentile intervals; raw bootstrap draws for all models were not saved. The adjusted values are approximate, particularly in the tails. BH additionally assumes a dependence structure that we have not established. The complete ordered-by-cohort results are in [the aggregate](round6-aggregate.json).",
    table(("Cohort","Cosine pairing Δ EER, pp (95% unadjusted CI)","Holm p","BH q"),
          [(name, delta(tests[f"{slug}/Cosine"]["delta"]), p(tests[f"{slug}/Cosine"]["holm"]), p(tests[f"{slug}/Cosine"]["bh"])) for slug,name in inputs]),
    "Holm passes (negative Δ favors pairing; positive Δ favors single-state):",
    "\n".join(f"- **{name}:** " + ", ".join(f"{k.split('/')[1]} ({'lower' if tests[k]['delta']['point'] < 0 else 'higher'})"
                                            for k in passes['holm'] if k.startswith(slug+'/')) for slug,name in inputs),
    "Additional BH-only passes:",
    "\n".join(f"- **{name}:** " + (", ".join(f"{k.split('/')[1]} ({'lower' if tests[k]['delta']['point'] < 0 else 'higher'})"
                                             for k in passes['bh'] if k not in passes['holm'] and k.startswith(slug+'/')) or "none")
              for slug,name in inputs),
    "## Leakage and split audit",
    table(("Cohort","Identities","Disjoint rotations","Visit labels","Decision interventions"),split_rows),
    "An identity is in only one role per rotation; roles rotate across fits. Source tracing: `round5.py:compare_rotation,enrollment,geometry_scores,embedding_scores` restrict fitted data to early visits and claimant/reference identities. `loop2.py:training_samples,standardized` use early visits and reference-only normalization for cosine. `round5_models.py:claimant_scores` fits scalers, base models and ensemble stacking on enrollment data, including out-of-fold enrollment predictions. `loop2.py:recording_arrays` extracts features from each recording independently. `round4.py:variant_rows` reads only calibration-impostor scores at the calibration visit to set cutoffs. In the targeted rerun, changing late scores was required not to change calibration ledgers or late cutoffs. Unit tests perturb probe visits and non-fit identities before enrollment and normalization. This checks the code and the targeted execution, not a retained execution trace from every original batch fit.",
    "## How much evidence is available for a 1% false-accept claim?",
    table(("Cohort","Paired cosine accepts / comparisons","Attempt CP upper","Clustered FAR upper","Impostors ever accepted / distinct","Identity-event CP upper","Identity CP upper if no events","Trials needed at fixed errors: attempts / identities"),power_rows),
    f"Bounds are one-sided 95%. Even with no errors, {cohorts[0][1]['power']['zero_error_minimum']} genuinely independent trials are needed for a binomial upper bound of at most 1%. All cohorts contain enough *rows* under the unrealistic zero-error/independent-attempt calculation, but no cohort contains that many independent impostor identities even in the zero-event best case. Reusing a person, claimant, model or threshold creates dependence. The clustered column resamples both identity roles and estimates per-attempt FAR, conditional on fitted models and cutoffs. Identity events instead mean at least one acceptance anywhere in the protocol, so they cannot be used as interchangeable per-attempt bounds. Holding observed errors fixed is arithmetic, not a sample-size forecast.",
    "## Probe-window sensitivity",
    "Values are paired-minus-single EER in percentage points with unadjusted 95% bootstrap intervals. Enrollment and its transformations stay fixed. At each budget the cutoff is calibrated again on disjoint calibration identities, then held fixed. At the largest budget all three rerun models' EER, FRR, FAR and pairing metric objects matched the saved Round 5 objects exactly in each cohort.",
    table(("Cohort","Windows per state",*(f"{m} Δ EER" for m in models),"Pass all original gates"),sensitivity_rows),
    "## Who benefited?",
    "The counts refer to per-claimant EER against that claimant's held-out impostors, and genuine false rejects at that claimant's frozen cutoff. The last column instead removes one identity from both roles and recalculates the pooled EER contrast, without refitting.",
    table(("Cohort","Model","EER improved/tied/worse","FRR improved/tied/worse","Pooled Δ EER range dropping one, pp"),hetero_rows),
    "## Deep learning and ensemble seed sensitivity",
    f"Paired EER, seed 42 → 43 (sample SD in percentage points). The largest observed two-seed spread was {100*spread:.2f} pp for {spread_model} in {spread_cohort}. Two seeds do not establish stability.",
    table(("Model", *(name for _,name in inputs)),
          [(m, *[f"{pct(d['42'])} → {pct(d['43'])} ({abs(100*(d['43']-d['42']))/math.sqrt(2):.2f})"
                 for d in (base["seeds"][slug][m] for slug,_ in inputs)]) for m in seed_models]),
    "## New ideas",
    "**Score fusion.** Add the two integer calibration ranks before dividing by twice the common calibration count, so equal rank sums stay tied. Set the frozen cutoff on the same calibration impostors used to fit the percentile maps. Negative differences favor fusion. The repeated score count is rows in excess of distinct fused scores, pooled across fits.",
    table(("Cohort","Fused EER / FRR / FAR","EER Δ vs cosine, pp","EER Δ vs logistic, pp","Repeated score rows / total"),fusion_rows),
    table(("Cohort","Constituent","Fusion − constituent FRR, pp","Fusion − constituent FAR, pp"),fusion_operating_rows),
    f"**Pooling across cohorts.** Cosine random-effects Δ EER: {delta(meta)} pp; I² {pct(meta['i2'])}, Q {meta['q']:.2f}. This uses different durations and visit schedules and approximate standard errors, and there are few cohorts. The pooled value summarizes the direction and scale of what was already observed, not an external replication.",
    "Score-level fusion has clear precedent ([Ross & Jain, *Information fusion in biometrics*](https://doi.org/10.1016/s0167-8655(03)00079-5)); this is a low-cost comparator, not a novelty claim. The pooled analysis uses established methods ([DerSimonian & Laird](https://doi.org/10.1016/0197-2456(86)90046-2); [modified Hartung–Knapp overview](https://doi.org/10.1186/s12874-015-0091-1)).",
)
(repo / "ROUND6_RESULTS.md").write_text(text)
print(f"Rendered {len(tests)} adjusted tests, {len(sensitivity_rows)} budget rows, {len(hetero_rows)} heterogeneity rows, {len(seed_models)} seed rows and {len(fusion_rows)} fusion rows from aggregate JSON.")
