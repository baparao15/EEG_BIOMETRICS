"""Follow up paired-state averaging with score normalization and alpha ablation."""
from pathlib import Path
import argparse

import numpy as np

from benchmark import SEED, Setting, load, metric, run, write


def pooled(ledgers, mode, norm="raw"):
    return ledgers["spectral", mode, norm, "all", 160, "none", False, "pooled", "late", "pooled"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()
    people, channels, arrays = load(args.cache)
    n = len(people)
    draws = np.random.default_rng(SEED + 1).multinomial(n, np.full(n, 1 / n), size=args.draws)
    pair = "pair-mean"
    settings = [Setting("spectral", "mixed")] + [
        Setting("spectral", pair, norm, probes=(pair,)) for norm in ("raw", "Z", "T", "S")
    ]
    _, full = run(people, channels, arrays, settings, draws)
    remaining = np.delete(arrays["spectral"], 2, axis=-1)
    no_alpha = {**arrays, "spectral": remaining - np.logaddexp.reduce(remaining, axis=-1, keepdims=True)}
    _, dropped = run(people, channels, no_alpha, [settings[1]], draws)
    pairs = [
        ("paired mean minus mixed enrollment with single-state probes", pooled(full, pair), pooled(full, "mixed")),
        ("paired mean no alpha feature minus full paired mean", pooled(dropped, pair), pooled(full, pair)),
        *((f"paired mean {norm} minus raw", pooled(full, pair, norm), pooled(full, pair))
          for norm in ("Z", "T", "S")),
    ]
    ones = np.ones(n)
    rows = []
    for name, first, second in pairs:
        point = metric(first, ones) - metric(second, ones)
        diff = np.nanquantile([metric(first, w) - metric(second, w) for w in draws], (.025, .975), axis=0)
        rows.append([name, *point, *diff.ravel()])
    write(args.output, ["comparison"], rows)
    print(f"Recorded {len(rows)} follow-up contrasts")


if __name__ == "__main__":
    main()
