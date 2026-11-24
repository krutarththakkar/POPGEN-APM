"""Sweep N, h and q, estimate the tipping point for each, write a CSV and a plot.

    python sweep.py                                  # default grid
    python sweep.py --N 500 5000 --h 2 20 --q 0.1 0.3 0.5 0.7 0.9
    python sweep.py --N 200 --h 0.5 2 8 --exact      # exact Markov chain, no noise
"""

import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np

from plot import plot_tipping
from sim import exact_tipping_point, q_critical, tipping_point


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", default=[100, 1000, 10000], help="population sizes")
    ap.add_argument("--h", type=float, nargs="+", default=[1, 3, 10, 30], help="h values (the handsome guy has weight N*h)")
    ap.add_argument("--q", type=float, nargs="+", default=list(np.round(np.linspace(0.05, 0.95, 37), 4)), help="handsome guy probabilities")
    ap.add_argument("--reps", type=int, default=2000, help="replicate populations per starting white count")
    ap.add_argument("--max-cycles", type=int, default=2000, help="give up on runs still polymorphic after this many cycles")
    ap.add_argument("--exact", action="store_true", help="solve the Markov chain instead of simulating (N up to ~2000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    args = ap.parse_args()

    configs = list(itertools.product(args.N, args.h, args.q))
    seeds = np.random.SeedSequence(args.seed).spawn(len(configs))

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "tipping.csv"
    start = time.time()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "h", "q", "q_critical", "tipping_white_frac", "unresolved"])
        for i, ((N, h, q), seed) in enumerate(zip(configs, seeds), 1):
            if args.exact:
                tip, unresolved = exact_tipping_point(N, h, q), 0.0
            else:
                tip, unresolved = tipping_point(N, h, q, args.reps, np.random.default_rng(seed), max_cycles=args.max_cycles)
            writer.writerow([N, f"{h:g}", f"{q:g}", f"{q_critical(N, h):.6g}", f"{tip:.6g}", f"{unresolved:.4g}"])
            f.flush()
            print(f"[{i}/{len(configs)}] N={N} h={h:g} q={q:g}  tipping={tip:.4f}"
                  + (f"  (unresolved {unresolved:.2%})" if unresolved > 0 else ""))

    png_path = plot_tipping(csv_path)
    print(f"\n{len(configs)} configs in {time.time() - start:.0f}s -> {csv_path}, {png_path}")


if __name__ == "__main__":
    main()
