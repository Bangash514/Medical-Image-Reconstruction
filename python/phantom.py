# BEN 2026-11-01
# Change aim: Simulated prostate phantom and a simulated tracked 2D ultrasound sweep (speckle, tilt, pose noise).
"""SIMULATED data only (no patient images). Echogenicity values: background 0.35, prostate 0.60, urethra 0.20, lesion 0.25."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

N, DX, H, W, PDX = 96, 0.5, 96, 96, 0.5
LESION = np.array([8.0, -5.0, 4.0])
LESION_R = 4.0
NOMINAL_OFFSET = np.array([1.5, -1.0, 1.0])


def make_gt():
    c = (np.arange(N) + 0.5 - N / 2.0) * DX
    zz, yy, xx = np.meshgrid(c, c, c, indexing="ij")
    vol = np.full((N, N, N), 0.35)
    vol[(xx / 22.0) ** 2 + (yy / 18.0) ** 2 + (zz / 16.0) ** 2 < 1] = 0.60
    vol[(xx ** 2 + (yy + 4.0) ** 2 < 2.0 ** 2) & (np.abs(zz) < 16)] = 0.20
    vol[(xx - LESION[0]) ** 2 + (yy - LESION[1]) ** 2 + (zz - LESION[2]) ** 2 < LESION_R ** 2] = 0.25
    return vol


def rot(ax_deg, ay_deg):
    a, b = np.radians(ax_deg), np.radians(ay_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return rx @ ry


def small_rotation(rng, sigma_deg):
    v = np.radians(sigma_deg) * rng.standard_normal(3)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    kk = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * kk + (1 - np.cos(th)) * kk @ kk


def speckle(rng, shape):
    a = gaussian_filter(rng.standard_normal(shape), 0.8)
    b = gaussian_filter(rng.standard_normal(shape), 0.8)
    s = np.sqrt(a * a + b * b)
    return s / s.mean()


def make_sweep(out_dir, spacing=1.0, sigma_t=0.3, sigma_r=0.3, seed=1):
    import os
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    gt = make_gt()
    n_slices = int(round(47.0 / spacing))
    z0 = -23.5
    x = (np.arange(W) + 0.5 - W / 2.0) * PDX
    y = (np.arange(H) + 0.5 - H / 2.0) * PDX
    xg, yg = np.meshgrid(x, y)
    slices, est, true = [], [], []
    for i in range(n_slices):
        r = rot(rng.normal(0, 1.5), rng.normal(0, 1.5))
        t = np.array([0.0, 0.0, z0 + i * spacing])
        p = np.stack([xg, yg, np.zeros_like(xg)], axis=-1) @ r.T + t
        idx = np.stack([p[..., 2], p[..., 1], p[..., 0]]) / DX + N / 2.0 - 0.5
        img = map_coordinates(gt, idx, order=1, mode="nearest")
        img = img * speckle(rng, img.shape) + 0.02 * rng.standard_normal(img.shape)
        slices.append(np.clip(img, 0, None).astype(np.float32))
        true.append(np.concatenate([r.ravel(), t]))
        re = small_rotation(rng, sigma_r) @ r
        te = t + sigma_t * rng.standard_normal(3)
        est.append(np.concatenate([re.ravel(), te]))
    np.array(slices, dtype=np.float32).tofile(f"{out_dir}/slices.bin")
    gt.astype(np.float32).tofile(f"{out_dir}/gt_volume.bin")
    hdr = "r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz"
    np.savetxt(f"{out_dir}/poses_est.csv", est, delimiter=",", header=hdr, comments="", fmt="%.12f")
    np.savetxt(f"{out_dir}/poses_true.csv", true, delimiter=",", header=hdr, comments="", fmt="%.12f")
    with open(f"{out_dir}/meta.txt", "w") as f:
        f.write(f"N={N}\nS={n_slices}\nH={H}\nW={W}\ndx={DX}\npdx={PDX}\n")
        f.write(f"lx={LESION[0]}\nly={LESION[1]}\nlz={LESION[2]}\nnx={NOMINAL_OFFSET[0]}\nny={NOMINAL_OFFSET[1]}\nnz={NOMINAL_OFFSET[2]}\n")
        f.write(f"spacing={spacing}\nsigma_t={sigma_t}\nsigma_r={sigma_r}\nseed={seed}\n")
