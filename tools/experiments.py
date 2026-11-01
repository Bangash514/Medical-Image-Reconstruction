# BEN 2026-11-01
# Change aim: robustness experiments (pose noise, slice spacing, 5 seeds) and figures for the README.
# File: experiments.py
# SIMULATED data only. Run from the repo root: python3 tools/experiments.py
import os, sys, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import recon_lib as rl
import phantom

SCEN = [("1 mm, noise 0", 1.0, 0.0, 0.0), ("1 mm, noise 0.3", 1.0, 0.3, 0.3), ("1 mm, noise 0.6", 1.0, 0.6, 0.6),
        ("1 mm, noise 1.0", 1.0, 1.0, 1.0), ("2 mm, noise 0", 2.0, 0.0, 0.0), ("2 mm, noise 0.6", 2.0, 0.6, 0.6)]
KEYS = ["l_err", "l_dice", "ncc", "p_vol"]
SEEDS = [1, 2, 3, 4, 5]


def main():
    res = {}
    lines = ["scenario | lesion centroid error (mm) | lesion Dice | NCC | prostate volume error (%)  [mean +- std, 5 seeds]"]
    for name, sp, st, sr in SCEN:
        vals = {k: [] for k in KEYS}
        for sd in SEEDS:
            with tempfile.TemporaryDirectory() as d:
                phantom.make_sweep(d, sp, st, sr, sd)
                m, sl, po, gt = rl.load_data(d)
                vol, filled, direct, used = rl.reconstruct(m, sl, po)
                r = rl.evaluate(m, vol, filled, gt, direct)
            for k in KEYS:
                vals[k].append(abs(r[k]) if k == "p_vol" else r[k])
        res[name] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in vals.items()}
        lines.append(f"{name} | " + " | ".join(f"{res[name][k][0]:.3f} +- {res[name][k][1]:.3f}" for k in KEYS))
    os.makedirs("results", exist_ok=True)
    open("results/experiments.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    names = [s[0] for s in SCEN]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, k, t in zip(ax, ["l_err", "l_dice"], ["Lesion centroid error (mm)", "Lesion Dice"]):
        a.bar(range(len(names)), [res[n][k][0] for n in names], yerr=[res[n][k][1] for n in names], color="#3b6ea5", capsize=3)
        a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=30, ha="right", fontsize=8); a.set_title(t)
    fig.suptitle("SIMULATED: effect of pose noise (mm and deg) and slice spacing, 5 seeds")
    fig.tight_layout(); fig.savefig("results/experiments.png", dpi=130); plt.close(fig)

    # montage: raw slice + recon vs ground truth
    m, sl, po, gt = rl.load_data("demo_data")
    vol = np.fromfile("results/recon_python.bin", np.float32).reshape(gt.shape)
    n = int(m["N"]); lc = [int(np.floor(m[k] / m["dx"] + n / 2.0)) for k in ("lx", "ly", "lz")]
    fig, ax = plt.subplots(2, 4, figsize=(11, 5.6))
    ax[0, 0].imshow(sl[len(sl) // 2], cmap="gray", vmin=0, vmax=0.9); ax[0, 0].set_title("raw ultrasound slice")
    ax[1, 0].axis("off")
    views = [("axial", lambda v: v[lc[2]]), ("coronal", lambda v: v[:, lc[1], :]), ("sagittal", lambda v: v[:, :, lc[0]])]
    for c, (t, f) in enumerate(views, 1):
        ax[0, c].imshow(f(vol), cmap="gray", vmin=0, vmax=0.9); ax[0, c].set_title("recon " + t)
        ax[1, c].imshow(f(gt), cmap="gray", vmin=0, vmax=0.9); ax[1, c].set_title("ground truth " + t)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle("SIMULATED 3D ultrasound reconstruction (no patient data)")
    fig.tight_layout(); fig.savefig("results/montage.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
