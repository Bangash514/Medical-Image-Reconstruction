# BEN 2026-11-01
# Change aim: Command-line entry: make a simulated sweep, or reconstruct one from a data folder and print the metrics.
import argparse
import os
import numpy as np
from recon_lib import load_data, reconstruct, evaluate, format_summary
from phantom import make_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", default="", help="write a simulated sweep to this folder first")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="python")
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--sigma-t", type=float, default=0.3)
    ap.add_argument("--sigma-r", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.make:
        make_sweep(a.make, a.spacing, a.sigma_t, a.sigma_r, a.seed)
    m, slices, poses, gt = load_data(a.data)
    vol, filled, direct, used = reconstruct(m, slices, poses)
    r = evaluate(m, vol, filled, gt, direct)
    text = format_summary(m, used, r)
    print(text, end="")
    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/summary_{a.tag}.txt", "w").write(text)
    vol.astype(np.float32).tofile(f"{a.out}/recon_{a.tag}.bin")


if __name__ == "__main__":
    main()
