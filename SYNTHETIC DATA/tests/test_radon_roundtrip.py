"""
Radon Transform Roundtrip Test
==============================

Proves the Radon math works by:
  1. Creating a simple 2D test image (circle + rectangle)
  2. Forward Radon transform -> sinogram
  3. Inverse Radon transform -> reconstructed image
  4. Comparing original vs reconstruction

No simulation or FMC data needed — pure image-level validation.

Usage:
    python test_radon_roundtrip.py
    python test_radon_roundtrip.py --n-angles 32 64 128 256
    python test_radon_roundtrip.py --filter Shepp-Logan --image phantom
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom
from skimage.metrics import structural_similarity as ssim


def make_test_image(size: int = 256) -> np.ndarray:
    """Create a simple test image: circle + off-centre rectangle."""
    img = np.zeros((size, size), dtype=np.float64)

    # Circle at centre, radius = size/6
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    r = size // 6
    img[(y - cy)**2 + (x - cx)**2 <= r**2] = 1.0

    # Small rectangle, off-centre
    rx, ry = size // 3, size // 4
    rw, rh = size // 10, size // 8
    img[ry:ry+rh, rx:rx+rw] = 0.6

    return img


def make_phantom(size: int = 256) -> np.ndarray:
    """Shepp-Logan phantom resized to requested size."""
    from skimage.transform import resize
    phantom = shepp_logan_phantom()
    return resize(phantom, (size, size), anti_aliasing=True)


def roundtrip(
    image: np.ndarray,
    n_angles: int,
    filter_name: str = 'shepp-logan',
    title: str = '',
) -> dict:
    """
    Forward Radon -> inverse Radon on a 2D image.

    Args:
        image:       (N, N) float64 input image
        n_angles:    number of projection angles (evenly spaced over 180 deg)
        filter_name: FBP filter for iradon
        title:       label for printing

    Returns:
        dict with sinogram, reconstruction, and metrics
    """
    theta = np.linspace(0, 180, n_angles, endpoint=False)

    # Forward Radon
    sinogram = radon(image, theta=theta, circle=True)

    # Inverse Radon
    recon = iradon(sinogram, theta=theta, filter_name=filter_name,
                   circle=True, output_size=image.shape[0])

    # Metrics (within inscribed circle)
    size = image.shape[0]
    centre = (size - 1) / 2.0
    yy, xx = np.ogrid[:size, :size]
    mask = (yy - centre)**2 + (xx - centre)**2 <= (size / 2.0)**2

    img_masked = image.copy()
    rec_masked = recon.copy()
    img_masked[~mask] = 0
    rec_masked[~mask] = 0

    ssim_val = ssim(img_masked, rec_masked, data_range=image.max())
    nrmse = np.sqrt(np.mean((img_masked[mask] - rec_masked[mask])**2))
    peak_err = np.max(np.abs(img_masked[mask] - rec_masked[mask]))

    if title:
        print(f"  {title:<30s}  SSIM={ssim_val:.4f}  "
              f"NRMSE={nrmse:.4f}  peak_err={peak_err:.4f}")

    return {
        'sinogram': sinogram,
        'recon': recon,
        'theta': theta,
        'ssim': ssim_val,
        'nrmse': nrmse,
        'peak_err': peak_err,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Radon roundtrip: forward -> inverse on a 2D image')
    parser.add_argument('--image', choices=['simple', 'phantom'],
                        default='simple',
                        help='Test image: "simple" (circle+rect) or '
                             '"phantom" (Shepp-Logan)')
    parser.add_argument('--size', type=int, default=256,
                        help='Image size in pixels (default: 256)')
    parser.add_argument('--n-angles', type=int, nargs='+',
                        default=[8, 16, 32, 64, 180],
                        help='Number of projection angles to test')
    parser.add_argument('--filter', type=str, default='shepp-logan',
                        help='FBP filter (default: shepp-logan)')
    parser.add_argument('--output', type=str, default='output/plots/radon_roundtrip.png',
                        help='Output figure path')
    args = parser.parse_args()

    # Build test image
    if args.image == 'phantom':
        image = make_phantom(args.size)
        img_label = 'Shepp-Logan phantom'
    else:
        image = make_test_image(args.size)
        img_label = 'Circle + rectangle'

    print(f"\nRadon roundtrip test")
    print(f"  Image: {img_label} ({args.size}x{args.size})")
    print(f"  Filter: {args.filter}")
    print(f"  Angles: {args.n_angles}\n")

    # Run roundtrips
    results = []
    for n in sorted(args.n_angles):
        r = roundtrip(image, n, filter_name=args.filter,
                      title=f'{n} angles')
        r['n_angles'] = n
        results.append(r)

    # Plot
    n_results = len(results)
    fig, axes = plt.subplots(3, n_results + 1, figsize=(4 * (n_results + 1), 11))

    # Top-left: original image
    axes[0, 0].imshow(image, cmap='gray', vmin=0)
    axes[0, 0].set_title(f'Original\n({img_label})')
    axes[0, 0].axis('off')

    # Middle-left: blank (no sinogram for original)
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, f'Filter: {args.filter}',
                    ha='center', va='center', fontsize=11,
                    transform=axes[1, 0].transAxes)

    # Bottom-left: metric summary
    axes[2, 0].axis('off')
    lines = [f"{'N angles':>10s}  {'SSIM':>8s}  {'NRMSE':>8s}"]
    lines.append(f"{'─'*10}  {'─'*8}  {'─'*8}")
    for r in results:
        lines.append(f"{r['n_angles']:>10d}  {r['ssim']:>8.4f}  {r['nrmse']:>8.4f}")
    axes[2, 0].text(0.1, 0.5, '\n'.join(lines),
                    ha='left', va='center', fontsize=9, family='monospace',
                    transform=axes[2, 0].transAxes)

    for col, r in enumerate(results, start=1):
        n = r['n_angles']

        # Row 0: reconstruction
        axes[0, col].imshow(r['recon'], cmap='gray', vmin=0,
                            vmax=image.max())
        axes[0, col].set_title(f'Recon (N={n})\n'
                               f'SSIM={r["ssim"]:.3f}')
        axes[0, col].axis('off')

        # Row 1: sinogram
        axes[1, col].imshow(r['sinogram'], cmap='gray', aspect='auto',
                            extent=[r['theta'][0], r['theta'][-1],
                                    r['sinogram'].shape[0], 0])
        axes[1, col].set_title(f'Sinogram (N={n})')
        axes[1, col].set_xlabel('Angle (deg)')
        if col == 1:
            axes[1, col].set_ylabel('Detector')

        # Row 2: |difference|
        diff = np.abs(image - r['recon'])
        axes[2, col].imshow(diff, cmap='hot', vmin=0,
                            vmax=np.percentile(diff, 99))
        axes[2, col].set_title(f'|Difference| (N={n})')
        axes[2, col].axis('off')

    fig.suptitle('Radon Transform Roundtrip: forward radon -> inverse radon',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {args.output}")


if __name__ == '__main__':
    main()
