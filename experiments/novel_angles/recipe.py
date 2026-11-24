"""Run the hash-pinned reproduction kit's notebook using locally verified raw files."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from hashlib import file_digest
import importlib.util
import json
from pathlib import Path
import subprocess

import mne
import numpy as np

from fetch_raw import RESTCOG_REVISION, REVISION, inventory, restcog_inventory


NOTEBOOK_SHA = "1e8fb8ae57e23698e537dbebde577d3691a06135087f2ca3696ca2bb34978d0e"


def file_sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def pipeline(kit, output):
    spec = importlib.util.spec_from_file_location("preprocessing_kit", kit / "run_preprocessing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if file_sha(module.NOTEBOOK) != NOTEBOOK_SHA:
        raise ValueError("Unexpected preprocessing notebook")
    module.OUTPUT = output
    return module.load_pipeline()


def compare(actual, expected):
    if expected.ch_names != actual.ch_names or actual.info["sfreq"] != expected.info["sfreq"]:
        raise ValueError("Mismatched reference channels or sample rates")
    e = expected.events[:, 0]
    a = actual.events[:, 0]
    common, ei, ai = np.intersect1d(e, a, return_indices=True)
    if not len(common):
        raise ValueError("No overlapping event starts")
    x = expected.get_data()[ei].ravel()
    y = actual.get_data()[ai].ravel()
    return {
        "reference_epochs": len(expected),
        "common_epochs": len(common),
        "same_channel_order": True,
        "event_jaccard": len(common) / (len(e) + len(a) - len(common)),
        "waveform_correlation": float(np.corrcoef(x, y)[0, 1]),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "same_events": bool(np.array_equal(expected.events, actual.events)),
        "same_values": bool(np.array_equal(x, y) and len(e) == len(a)),
    }


def execute(job, configure=None):
    kit, output, reference, key, path, line_freq = job
    person, visit, task = key
    folder = Path(person) / f"ses-{visit}" / "eeg"
    target = output / folder / f"{person}_ses-{visit}_task-{task}_desc-preproc_epo.fif"
    log_path = output / "logs" / ("_".join(key) + ".log")
    summary_path = output / "logs" / ("_".join(key) + ".json")
    partial = target.with_name(f"partial-{target.name}")
    try:
        detail = {}
        generated = not target.exists()
        if generated:
            with log_path.open("w") as log, redirect_stdout(log), redirect_stderr(log):
                ns = pipeline(kit, output)
                ns["LINE_FREQ"] = line_freq
                if configure is not None:
                    configure(ns)
                detail = ns["run_pipeline"](path, partial)
        elif summary_path.is_file():
            previous = json.loads(summary_path.read_text())
            detail = {k: previous[k] for k in ("interpolated_channels", "epochs_kept", "seconds")
                      if k in previous}
        actual = mne.read_epochs(partial if generated else target, preload=True, verbose="ERROR")
        result = dict(detail, recording=target.name.removesuffix("_desc-preproc_epo.fif"), status="ok",
                      epochs=len(actual), channels=len(actual.ch_names), channel_names=actual.ch_names,
                      sfreq=actual.info["sfreq"], samples=len(actual.times),
                      finite=bool(np.isfinite(actual.get_data()).all()))
        if result["sfreq"] != 250 or result["samples"] != 1251 or not result["finite"]:
            raise ValueError("Malformed output epoch data")
        if reference and (ref := reference / folder / target.name).exists():
            result.update(compare(actual, mne.read_epochs(ref, preload=True, verbose="ERROR")))
        if generated:
            partial.replace(target)
        summary_path.write_text(json.dumps(result) + "\n")
        return result
    except Exception as exc:
        partial.unlink(missing_ok=True)
        return {"recording": "_".join(key), "error": repr(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kit", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--restcog", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.reference_only and not args.reference:
        parser.error("--reference-only requires --reference")
    if file_sha(args.kit / "eeg_preprocessing_pipeline.ipynb") != NOTEBOOK_SHA:
        parser.error("Unexpected preprocessing notebook")
    head = subprocess.check_output(["git", "-C", args.raw, "rev-parse", "HEAD"], text=True).strip()
    if head != (RESTCOG_REVISION if args.restcog else REVISION):
        parser.error(f"Unexpected dataset revision: {head}")
    cohort = restcog_inventory(args.raw) if args.restcog else inventory(args.raw)
    line_freq = 50.0 if args.restcog else 60.0
    jobs = [(args.kit, args.output, args.reference, (person, visit, task), path, line_freq)
            for person, visits in cohort.items() for visit, tasks in visits.items() for task, path in tasks.items()
            if not args.reference_only or (args.reference / person).is_dir()]
    expected = 40 if args.reference_only else 360 if args.restcog else 232
    if len(jobs) != expected:
        parser.error(f"Expected {expected} recordings, found {len(jobs)}")
    for _, _, _, _, path, _ in jobs:
        if not all(path.with_suffix(ext).is_file() for ext in (".eeg", ".vhdr", ".vmrk")):
            parser.error(f"Missing raw files for {path}")
    (args.output / "logs").mkdir(parents=True, exist_ok=True)
    records, errors = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, record in enumerate(pool.map(execute, jobs), 1):
            if "error" in record:
                errors.append(record)
                print(record, flush=True)
            else:
                records.append(record)
            if i % 10 == 0 or i == len(jobs):
                print(f"Read {i}/{len(jobs)} results ({len(errors)} failed)", flush=True)
    matched = [row for row in records if "event_jaccard" in row]
    residual = {key: {"median": float(np.median(items)), "min": min(items), "max": max(items)}
                for key in ("event_jaccard", "waveform_correlation", "rmse")
                if (items := [row[key] for row in matched])}
    print(json.dumps({"generated": len(records), "failed": len(errors), "references": len(matched),
                      "exact_events": sum(row["same_events"] for row in matched),
                      "exact_values": sum(row["same_values"] for row in matched), "residual": residual,
                      "same_epoch_count": sum(row["epochs"] == row["reference_epochs"] for row in matched),
                      "minimum_epochs": min((r["epochs"] for r in records), default=None)}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
