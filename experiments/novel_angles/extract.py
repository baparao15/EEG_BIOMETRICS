"""Extract per-epoch features from pinned raw BrainVision data or existing recipe FIFs."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import argparse
import re

import mne
import numpy as np
from scipy import signal

from fetch_raw import inventory


SFREQ = 250
EPOCH_SECONDS = 5
CAP = 48
BANDS = [(2, 4), (4, 8), (8, 13), (13, 30), (30, 44)]


def from_raw(path):
    raw = mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR").pick("eeg")
    raw.resample(SFREQ, verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    raw.filter(1, 45, verbose="ERROR")
    x = raw.get_data()
    size = SFREQ * EPOCH_SECONDS
    x = x[:, :x.shape[1] // size * size].reshape(len(x), -1, size).transpose(1, 0, 2)
    rms = np.log(np.sqrt(np.mean(x ** 2, axis=-1)).clip(1e-15))
    med = np.median(rms, axis=0)
    mad = np.median(np.abs(rms - med), axis=0).clip(0.02)
    good = np.quantile(np.abs(rms - med) / mad, 0.95, axis=1) < 6
    return x[good], raw.ch_names


def from_epochs(path):
    epochs = mne.read_epochs(path, preload=True, verbose="ERROR").pick("eeg")
    samples = SFREQ * EPOCH_SECONDS
    if epochs.info["sfreq"] != SFREQ or len(epochs.times) not in (samples, samples + 1):
        raise ValueError(f"Unexpected FIF epoch sampling: {path}")
    return epochs.get_data()[..., :samples], epochs.ch_names


def features(x):
    x = x[:CAP].astype(np.float64, copy=True)
    x -= x.mean(axis=-1, keepdims=True)
    x /= x.std(axis=-1, keepdims=True).clip(1e-12)
    f, p = signal.welch(x, fs=SFREQ, nperseg=500, noverlap=250, axis=-1)
    powers = [p[..., (f >= lo) & (f < hi)].sum(axis=-1) for lo, hi in BANDS]
    power = np.stack(powers, axis=-1)
    spectral = np.log((power / power.sum(axis=-1, keepdims=True).clip(1e-12)).clip(1e-12))
    log_p = np.log(p.clip(1e-12))
    use = ((f >= 3) & (f < 7)) | ((f >= 15) & (f <= 35))
    log_f = np.log(f[use])
    weights = (log_f - log_f.mean()) / np.sum((log_f - log_f.mean()) ** 2)
    slope = log_p[..., use] @ weights
    cov = x @ x.swapaxes(-1, -2) / x.shape[-1]
    cov = 0.95 * cov + 0.05 * np.eye(x.shape[1])
    segments = np.lib.stride_tricks.sliding_window_view(x, 500, axis=-1)[..., ::250, :]
    z = np.fft.rfft((segments - segments.mean(axis=-1, keepdims=True)) * signal.windows.hann(500),
                    axis=-1)[..., (f >= 8) & (f < 13)]
    z = z.reshape(len(x), x.shape[1], -1)
    cross = z @ z.conj().swapaxes(-1, -2)
    auto = np.sum(np.abs(z) ** 2, axis=-1).clip(1e-12)
    imag = cross.imag / np.sqrt(auto[..., :, None] * auto[..., None, :])
    return spectral, slope, cov, imag


def extract(job, output, raw):
    key, path = job
    out = output / ("_".join(key) + ".npz")
    if out.exists():
        return len(np.load(out)["spectral"])
    x, channels = from_raw(path) if raw else from_epochs(path)
    spectral, slope, cov, imag = features(x)
    arrays = [spectral, slope, cov, imag]
    if len(spectral) < 24 or len(set(channels)) != len(channels) or not all(
        np.isfinite(array).all() for array in arrays
    ):
        raise ValueError(f"Invalid epoch data ({len(spectral)} epochs): {path}")
    temporary = out.with_suffix(".partial")
    with temporary.open("wb") as file:
        np.savez(file, channels=np.char.upper(channels), **dict(
            zip(("spectral", "slope", "cov", "imag"), (a.astype(np.float32) for a in arrays))))
    temporary.replace(out)
    return len(spectral)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", choices=("raw", "epochs"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.source == "raw":
        jobs = [((s, v, t), p) for s, visits in inventory(args.input).items()
                for v, tasks in visits.items() for t, p in tasks.items()]
    else:
        pattern = re.compile(r"(sub-[^_]+)_ses-(V\d+)_task-(CE|OE)_desc-preproc_epo")
        jobs = [(m.groups(), p) for p in sorted(args.input.rglob("*_desc-preproc_epo.fif"))
                if (m := pattern.fullmatch(p.stem))]
    if not jobs:
        parser.error("No matching EEG files")
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        counts = []
        for done, count in enumerate(pool.map(partial(
            extract, output=args.output, raw=args.source == "raw"), jobs), start=1
        ):
            counts.append(count)
            if done % 25 == 0 or done == len(jobs):
                print(f"Extracted {done}/{len(jobs)} recordings ({min(counts)}–{max(counts)} epochs)",
                      flush=True)


if __name__ == "__main__":
    main()
