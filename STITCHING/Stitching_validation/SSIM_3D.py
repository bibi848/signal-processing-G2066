from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compare_volumes_ssim(volume_a_path: str | Path, volume_b_path: str | Path) -> float:
    """
    Compare two 3D .npy volumes using SSIM.
    Returns a score where 1.0 means identical.
    """

    volume_a_path = Path(volume_a_path)
    volume_b_path = Path(volume_b_path)

    if not volume_a_path.exists():
        raise FileNotFoundError(f"File not found: {volume_a_path}")
    if not volume_b_path.exists():
        raise FileNotFoundError(f"File not found: {volume_b_path}")

    vol1 = np.load(volume_a_path)
    vol2 = np.load(volume_b_path)

    if vol1.shape != vol2.shape:
        raise ValueError(f"Volumes must have the same shape. Got {vol1.shape} vs {vol2.shape}")

    # Convert to float for stable comparison
    vol1 = vol1.astype(np.float64)
    vol2 = vol2.astype(np.float64)

    # Important for float arrays
    data_min = min(vol1.min(), vol2.min())
    data_max = max(vol1.max(), vol2.max())
    data_range = data_max - data_min

    if data_range == 0:
        return 1.0 if np.array_equal(vol1, vol2) else 0.0

    score = ssim(vol1, vol2, data_range=data_range)
    return score


if __name__ == "__main__":
    in_dir = Path.cwd() / "DATA" / "2D TFM Data" / "Cu Pure 7.5MHz Ex 11032026 Filtered"
    volume1 = in_dir / "A5_filtered_3D_TFM.npy"
    volume2 = in_dir / "C5_filtered_3D_TFM.npy"

    score = compare_volumes_ssim(volume1, volume2)
    print(f"SSIM score: {score:.4f}")

    if score > 0.95:
        print("Volumes are very similar.")
    elif score > 0.80:
        print("Volumes are somewhat similar.")
    else:
        print("Volumes are quite different.")