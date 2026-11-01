# BEN 2026-11-01
# Change aim: Freehand 3D ultrasound reconstruction (pixel binning + hole filling) and evaluation. Same definitions as cpp/reconstruct.cpp.
"""SIMULATED data only. Units: millimetres. Voxel (i, j, k) has its centre at ((i + 0.5 - N/2) * dx, ...) and the volume array is indexed [k, j, i] (z, y, x)."""
import numpy as np

PROSTATE_THR = 0.475   # midway between background (0.35) and prostate (0.60)
LESION_THR = 0.425     # midway between lesion (0.25) and prostate (0.60)
ROI_HALF = (28.0, 24.0, 20.0)
LESION_ROI_RADIUS = 6.0   # sphere around the planned (approximate) target; stays inside the prostate


def load_meta(d):
    meta = {}
    for line in open(f"{d}/meta.txt"):
        if "=" in line:
            k, v = line.strip().split("=")
            meta[k] = float(v)
    return meta


def load_data(d):
    m = load_meta(d)
    n, s, h, w = int(m["N"]), int(m["S"]), int(m["H"]), int(m["W"])
    slices = np.fromfile(f"{d}/slices.bin", dtype=np.float32).reshape(s, h, w).astype(np.float64)
    gt = np.fromfile(f"{d}/gt_volume.bin", dtype=np.float32).reshape(n, n, n).astype(np.float64)
    poses = np.loadtxt(f"{d}/poses_est.csv", delimiter=",", skiprows=1)
    return m, slices, poses, gt


def box_sum(a):
    p = np.pad(a, 1)
    out = np.zeros_like(a)
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                out += p[dz:dz + a.shape[0], dy:dy + a.shape[1], dx:dx + a.shape[2]]
    return out


def reconstruct(m, slices, poses, passes=3):
    n, h, w, dx, pdx = int(m["N"]), int(m["H"]), int(m["W"]), m["dx"], m["pdx"]
    total = np.zeros(n ** 3)
    count = np.zeros(n ** 3)
    x = np.tile((np.arange(w) + 0.5 - w / 2.0) * pdx, h)
    y = np.repeat((np.arange(h) + 0.5 - h / 2.0) * pdx, w)
    used = 0
    for i in range(len(slices)):
        r, t = poses[i, :9].reshape(3, 3), poses[i, 9:12]
        px = x * r[0, 0] + y * r[0, 1] + t[0]
        py = x * r[1, 0] + y * r[1, 1] + t[1]
        pz = x * r[2, 0] + y * r[2, 1] + t[2]
        ii = np.floor(px / dx + n / 2.0).astype(np.int64)
        jj = np.floor(py / dx + n / 2.0).astype(np.int64)
        kk = np.floor(pz / dx + n / 2.0).astype(np.int64)
        ok = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n) & (kk >= 0) & (kk < n)
        flat = (kk[ok] * n + jj[ok]) * n + ii[ok]
        total += np.bincount(flat, weights=slices[i].ravel()[ok], minlength=n ** 3)
        count += np.bincount(flat, minlength=n ** 3)
        used += int(ok.sum())
    filled = count > 0
    vol = np.where(filled, total / np.maximum(count, 1), 0.0).reshape(n, n, n)
    filled = filled.reshape(n, n, n)
    direct = filled.mean()
    for _ in range(passes):
        s3, c3 = box_sum(np.where(filled, vol, 0.0)), box_sum(filled.astype(np.float64))
        new = (~filled) & (c3 > 0)
        vol = np.where(new, s3 / np.maximum(c3, 1), vol)
        filled = filled | new
    return vol, filled, direct, used


def smooth(vol):
    return box_sum(vol) / box_sum(np.ones_like(vol))


def evaluate(m, vol, filled, gt, direct):
    n, dx = int(m["N"]), m["dx"]
    c = (np.arange(n) + 0.5 - n / 2.0) * dx
    zz, yy, xx = np.meshgrid(c, c, c, indexing="ij")
    roi = (np.abs(xx) <= ROI_HALF[0]) & (np.abs(yy) <= ROI_HALF[1]) & (np.abs(zz) <= ROI_HALF[2])
    a, b = vol[roi], gt[roi]
    ncc = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    sm = smooth(vol)
    pr, pg = sm > PROSTATE_THR, gt > PROSTATE_THR
    p_dice = 2.0 * (pr & pg).sum() / (pr.sum() + pg.sum())
    p_vol = 100.0 * (pr.sum() - pg.sum()) / pg.sum()
    lc = np.array([m["lx"], m["ly"], m["lz"]])
    nom = lc + np.array([m["nx"], m["ny"], m["nz"]])
    box = (xx - nom[0]) ** 2 + (yy - nom[1]) ** 2 + (zz - nom[2]) ** 2 <= LESION_ROI_RADIUS ** 2
    lr, lg = (sm < LESION_THR) & box & filled, (gt < LESION_THR) & box
    l_dice = 2.0 * (lr & lg).sum() / max(lr.sum() + lg.sum(), 1)
    if lr.any():
        cen = np.array([xx[lr].mean(), yy[lr].mean(), zz[lr].mean()])
        l_err = float(np.linalg.norm(cen - lc))
    else:
        l_err = float("nan")
    return {"direct": 100.0 * direct, "filled": 100.0 * filled.mean(), "ncc": ncc, "rmse": rmse, "p_dice": float(p_dice),
            "p_vol": float(p_vol), "l_dice": float(l_dice), "l_err": l_err}


def format_summary(m, used, r):
    return (f"slices: {int(m['S'])}, pixels used: {used}\n"
            f"voxels filled directly: {r['direct']:.2f} %, after hole filling: {r['filled']:.2f} %\n"
            f"recon vs ground truth in ROI: NCC {r['ncc']:.4f}, RMSE {r['rmse']:.4f}\n"
            f"prostate segmentation: Dice {r['p_dice']:.4f}, volume error {r['p_vol']:.2f} %\n"
            f"lesion: Dice {r['l_dice']:.4f}, centroid error {r['l_err']:.3f} mm\n")
