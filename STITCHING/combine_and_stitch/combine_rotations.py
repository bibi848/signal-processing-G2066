#%%
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
from scipy.ndimage import rotate, gaussian_filter, sobel
import matplotlib.pyplot as plt


ROTATIONS = {
    1: 0,
    2: -90,
    3: -180,
    4: -270,
}


class RotationCombiner:
    def __init__(
        self,
        input_dir,
        output_dir,
        method="median",
        crop=40,
        gradient_sigma=1.0,
        save_mip=True,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.method = method
        self.crop = crop
        self.gradient_sigma = gradient_sigma
        self.save_mip = save_mip

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.method not in ("mean", "median", "gradient", "max"):
            raise ValueError("method must be 'mean', 'median', 'gradient', or 'max'")

    def parse_filename(self, name):
        match = re.match(r"(\d+)_filtered_3D_TFM\.npy", name)
        if not match:
            return None

        prefix = match.group(1)
        if len(prefix) < 2:
            return None

        position = int(prefix[:-1])
        rotation_idx = int(prefix[-1])

        if rotation_idx not in (1, 2, 3, 4):
            return None

        return position, rotation_idx

    def load_grouped_files(self):
        grouped = {}

        for file_path in self.input_dir.glob("*_filtered_3D_TFM.npy"):
            parsed = self.parse_filename(file_path.name)
            if parsed is None:
                continue

            position, rotation_idx = parsed
            grouped.setdefault(position, {})
            grouped[position][rotation_idx] = file_path

        return grouped

    def rotate_volume_to_reference(self, vol, rotation_idx):
        return rotate(
            vol,
            angle=ROTATIONS[rotation_idx],
            axes=(1, 2),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
        ).astype(np.float32)

    def crop_xy(self, vol):
        if self.crop <= 0:
            return vol
        return vol[:, self.crop:-self.crop, self.crop:-self.crop]

    @staticmethod
    def normalise(vol):
        vmin = np.min(vol)
        vmax = np.max(vol)
        if vmax > vmin:
            return (vol - vmin) / (vmax - vmin)
        return np.zeros_like(vol, dtype=np.float32)

    @staticmethod
    def robust_normalise(vol, q_low=1, q_high=99):
        lo, hi = np.percentile(vol, [q_low, q_high])
        if hi <= lo:
            return np.zeros_like(vol, dtype=np.float32)
        out = (vol - lo) / (hi - lo)
        return np.clip(out, 0, 1).astype(np.float32)

    def gradient_mag(self, vol):
        vol_s = gaussian_filter(vol, sigma=self.gradient_sigma)
        gz = sobel(vol_s, axis=0)
        gx = sobel(vol_s, axis=1)
        gy = sobel(vol_s, axis=2)
        return np.sqrt(gz**2 + gx**2 + gy**2).astype(np.float32)

    def fuse_mean(self, volumes):
        return np.mean(np.stack(volumes, axis=0), axis=0).astype(np.float32)

    def fuse_median(self, volumes):
        return np.median(np.stack(volumes, axis=0), axis=0).astype(np.float32)

    def fuse_max(self, volumes):
        return np.max(np.stack(volumes, axis=0), axis=0).astype(np.float32)

    def fuse_gradient(self, volumes, eps=1e-6):
        volumes_norm = [self.robust_normalise(v) for v in volumes]
        weights = [self.gradient_mag(v) + eps for v in volumes_norm]

        num = np.zeros_like(volumes_norm[0], dtype=np.float32)
        den = np.zeros_like(volumes_norm[0], dtype=np.float32)

        for v, w in zip(volumes_norm, weights):
            num += v * w
            den += w

        return (num / den).astype(np.float32)

    def fuse_volumes(self, volumes):
        if self.method == "mean":
            return self.fuse_mean(volumes)
        if self.method == "median":
            return self.fuse_median(volumes)
        if self.method == "gradient":
            return self.fuse_gradient(volumes)
        if self.method == "max":
            return self.fuse_max(volumes)
        raise ValueError(f"Unknown method: {self.method}")

    def save_mip_image(self, volume, out_path, title=""):
        mip = np.max(self.robust_normalise(volume), axis=0)
        plt.figure(figsize=(5, 5))
        plt.imshow(mip, cmap="gray", origin="lower")
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    def combine_position(self, position, files):
        aligned_volumes = []

        for rotation_idx in [1, 2, 3, 4]:
            vol = np.load(files[rotation_idx]).astype(np.float32)
            vol = self.rotate_volume_to_reference(vol, rotation_idx)
            vol = self.crop_xy(vol)
            aligned_volumes.append(vol)

        return self.fuse_volumes(aligned_volumes)

    def combine_all(self, positions=None):
        grouped = self.load_grouped_files()

        if positions is not None:
            grouped = {k: grouped[k] for k in positions if k in grouped}

        if not grouped:
            raise RuntimeError("No valid scan files found.")

        fused_paths = {}

        for position in sorted(grouped.keys()):
            files = grouped[position]
            missing = [r for r in [1, 2, 3, 4] if r not in files]
            if missing:
                print(f"Skipping position {position}: missing rotations {missing}")
                continue

            fused_volume = self.combine_position(position, files)

            out_npy = self.output_dir / f"position_{position}_fused_{self.method}.npy"
            np.save(out_npy, fused_volume)
            fused_paths[position] = out_npy

            print(f"Saved fused position {position}: {out_npy.name}")

            if self.save_mip:
                out_png = self.output_dir / f"position_{position}_fused_{self.method}_MIP.png"
                self.save_mip_image(
                    fused_volume,
                    out_png,
                    title=f"Position {position} - {self.method}"
                )

        return fused_paths