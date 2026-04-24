import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

SYNTHETIC_PATH    = "../SYNTHETIC DATA/output/engine_3d_tuned/volume_0000.npz"
EXPERIMENTAL_PATH = "../DATA/2D TFM Data/Cu Pure 7.5MHz Ex 15042026 Filtered/11_filtered_3D_TFM.npy"

# Volumes are stored in dB (peak = 0 dB, lower = darker).
syn = np.load(SYNTHETIC_PATH)['img_db'].ravel()
exp = np.load(EXPERIMENTAL_PATH).ravel()

res = ks_2samp(syn, exp)
syn_s, exp_s = np.sort(syn), np.sort(exp)
grid = np.sort(np.concatenate([syn_s, exp_s]))
cdf_s = np.searchsorted(syn_s, grid, side="right") / syn.size
cdf_e = np.searchsorted(exp_s, grid, side="right") / exp.size
x_gap = grid[np.argmax(np.abs(cdf_s - cdf_e))]

D, p = res.statistic, res.pvalue
print(f"KS D = {D:.3f}   p = {p:.2e}   max-gap @ {x_gap:+.2f} dB")

fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(12, 5))

ax_h.hist(syn, bins=200, alpha=0.5, label="synthetic", density=True)
ax_h.hist(exp, bins=200, alpha=0.5, label="experimental", density=True)
ax_h.set_yscale("log")
ax_h.set_xlabel("intensity (dB)")
ax_h.set_ylabel("density")
ax_h.set_title(f"Intensity distribution (KS D = {D:.3f})")
ax_h.legend()

ax_c.plot(syn_s, np.linspace(0, 1, syn.size), label="synthetic")
ax_c.plot(exp_s, np.linspace(0, 1, exp.size), label="experimental")
ax_c.set_xlabel("intensity (dB)")
ax_c.set_ylabel("CDF")
ax_c.set_title("CDF comparison")
ax_c.legend()

plt.tight_layout()
plt.show()
