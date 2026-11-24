"""Two-day Dortmund sensitivity analysis, with same-day post-task calibration."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess

import mne
import numpy as np

from fetch_raw import fetch
from loop2 import check_anchor, kit_module, ledgers, load_features, summarize
from recipe import NOTEBOOK_SHA, execute, file_sha


REVISION = "6a558cd5852503e66df9ac7fdac2f3a7f4ed5f12"
BLOCKS = {"V0": (1, "pre"), "V2": (1, "post"), "V3": (2, "pre")}
STATE_NAMES = {"CE": "Closed", "OE": "Open"}
PROTOCOL = "two-day: baseline pre enrollment, baseline post calibration, follow-up pre test"


def inventory(root):
    people = {}
    for subject in sorted(root.glob("sub-*")):
        blocks = {
            (visit, state): subject / f"ses-{day}" / "eeg" /
            f"{subject.name}_ses-{day}_task-Eyes{name}_acq-{block}_eeg.edf"
            for visit, (day, block) in BLOCKS.items() for state, name in STATE_NAMES.items()
        }
        if all(path.is_symlink() for path in blocks.values()):
            people[subject.name] = blocks
    return people


def setup_edf(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.drop_channels(["Status"])
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    return raw


def configure_edf(ns):
    ns["step1_channel_setup"] = setup_edf


def prepare_one(job):
    root, kit, output, key, path = job
    if epoch_file(output, *key).is_file():
        return execute((kit, output, None, key, path, 50.0), configure_edf)
    target = path.resolve()
    already_present = target.is_file()
    try:
        fetch(root, path, "ds005385")
        return execute((kit, output, None, key, path, 50.0), configure_edf)
    finally:
        if not already_present:
            target.unlink(missing_ok=True)


def epoch_file(root, person, visit, state):
    return root / person / f"ses-{visit}" / "eeg" / f"{person}_ses-{visit}_task-{state}_desc-preproc_epo.fif"


def stage_channels(source, target, canonical):
    epochs = mne.read_epochs(source, preload=False, verbose="ERROR")
    names = {name.upper(): name for name in epochs.ch_names}
    if len(names) != len(epochs.ch_names) or set(names) != set(canonical):
        raise ValueError(f"Dortmund electrode set does not match the canonical recording: {source.name}")
    if tuple(names) == canonical:
        if not target.exists():
            target.symlink_to(source.resolve())
        return False
    partial = target.with_name(f"partial-{target.name}")
    epochs.load_data().reorder_channels([names[name] for name in canonical])
    epochs.save(partial, fmt="double", overwrite=True, verbose="ERROR")
    partial.replace(target)
    return True


def eligibility(population, counts, failures):
    known_failures = {r["recording"] for r in failures
                      if r["error"].startswith("RuntimeError('One PCA component captures most")}
    if len(known_failures) != len(failures):
        raise ValueError("Unexpected preprocessing failures; see private failures.json")
    eligible = []
    for person, blocks in population.items():
        enough = True
        for visit, state in blocks:
            key = "_".join((person, visit, state))
            if key in known_failures:
                enough = False
            elif key not in counts:
                raise ValueError("Incomplete preparation; missing recording summary")
            elif counts[key] < (32 if visit == "V0" else 24):
                enough = False
        if enough:
            eligible.append(person)
    return eligible


def prepare(args, population):
    if file_sha(args.kit / "eeg_preprocessing_pipeline.ipynb") != NOTEBOOK_SHA:
        raise ValueError("Unexpected preprocessing notebook")
    (args.fifs / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [(args.raw, args.kit, args.fifs, (s, *block), path)
            for s, blocks in population.items() for block, path in blocks.items()]
    failures, counts = [], {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(prepare_one, jobs, chunksize=1), 1):
            if "error" in result:
                failures.append(result)
            else:
                key = jobs[i - 1][3]
                counts["_".join(key)] = result["epochs"]
            if i % 30 == 0 or i == len(jobs):
                print(f"Prepared {i}/{len(jobs)} recordings ({len(failures)} failed)", flush=True)
    (args.fifs / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    eligible = eligibility(population, counts, failures)
    print(json.dumps({"inventory_subjects": len(population), "generated": len(counts),
                      "eligible_subjects": len(eligible),
                      "minimum_epochs": min(counts.values())}, indent=2), flush=True)


def score(args, population):
    anchor = check_anchor(args.anchor)
    failures = json.loads((args.fifs / "failures.json").read_text())
    counts = {}
    for person, blocks in population.items():
        for visit, state in blocks:
            key = "_".join((person, visit, state))
            summary = args.fifs / "logs" / f"{key}.json"
            if summary.is_file() and epoch_file(args.fifs, person, visit, state).is_file():
                counts[key] = json.loads(summary.read_text())["epochs"]
    eligible = eligibility(population, counts, failures)
    if len(eligible) < 5:
        raise ValueError("Fewer than five eligible participants")
    inputs = args.fifs / "scoring-inputs"
    inputs.mkdir(exist_ok=True)
    if any(path.name not in eligible for path in inputs.iterdir()):
        raise ValueError("Stale scoring inputs; remove the private scoring-inputs directory before retrying")
    reference = epoch_file(args.fifs, eligible[0], "V0", "CE")
    canonical = tuple(name.upper() for name in mne.read_epochs(reference, preload=False, verbose="ERROR").ch_names)
    reordered = 0
    for person in eligible:
        target = inputs / person
        target.mkdir(exist_ok=True)
        for visit, state in population[person]:
            path = epoch_file(args.fifs, person, visit, state)
            reordered += stage_channels(path, target / path.name, canonical)
    print(f"Staged {sum(len(population[person]) for person in eligible)} recordings; "
          f"reordered {reordered} by electrode name", flush=True)
    kit = kit_module(args.kit)
    notebook = Path(__file__).resolve().parents[2] / "07_complete_results_all_models_all_sessions.ipynb"
    people, channels, visits, arrays, min_epochs = load_features(
        inputs, "ds005385", args.kit, notebook, args.workers, expected_people=len(eligible),
    )
    groups = [tuple(sorted(g)) for g in np.array_split(np.random.default_rng(42).permutation(len(people)), 5)]
    order = np.repeat("fixed-ce-first", len(people))
    result = summarize(ledgers(arrays, channels, visits, groups, kit.notebook_env(notebook)),
                       len(people), 500, order)
    result.update(dataset="ds005385", protocol=PROTOCOL, subjects=len(people),
                  inventory_subjects=len(population), generated_recordings=len(counts),
                  preprocessing_failures=len(failures), shortest_recording=min_epochs,
                  reordered_recordings=reordered, draws=500, anchor_frr_penalties_pp=anchor)
    partial = args.output.with_suffix(".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    partial.replace(args.output)
    for scorer in ("cosine", "svm"):
        vals = result["contrasts"][f"{scorer}/pair-minus-single"]["eer"]
        print(f"{scorer}: EER difference {100 * vals['point']:.2f} pp "
              f"[{100 * vals['low']:.2f}, {100 * vals['high']:.2f}]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("raw", type=Path)
    parser.add_argument("fifs", type=Path)
    parser.add_argument("kit", type=Path)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "score" and (args.anchor is None or args.output is None):
        parser.error("score requires --anchor and --output")
    head = subprocess.check_output(["git", "-C", args.raw, "rev-parse", "HEAD"], text=True).strip()
    population = inventory(args.raw)
    if head != REVISION or len(population) != 208:
        parser.error(f"Expected pinned revision and 208 complete inventories; got {head}, {len(population)}")
    if args.stage == "prepare":
        prepare(args, population)
    else:
        score(args, population)


if __name__ == "__main__":
    main()
