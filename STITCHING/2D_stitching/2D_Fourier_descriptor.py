import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import napari
from scipy import ndimage as ndi
import imageio.v3 as iio


def load_image_any(path):
    """
    Load either:
    - a NumPy array file (.npy, .npz)
    - an image file (.png, .tif, .tiff, .jpg, etc.)

    Then ensure the result is a 2D grayscale float32 image.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        img = np.load(path)

    elif suffix == ".npz":
        data = np.load(path)
        if len(data.files) == 0:
            raise ValueError(f"No arrays found in {path}")
        img = data[data.files[0]]

    else:
        img = iio.imread(path)

    img = np.asarray(img)

    # Already grayscale
    if img.ndim == 2:
        return img.astype(np.float32)

    # RGB or RGBA -> grayscale
    if img.ndim == 3:
        if img.shape[2] >= 3:
            rgb = img[..., :3].astype(np.float32)
            gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
            return gray.astype(np.float32)

    raise ValueError(f"Unsupported image shape for grayscale conversion: {img.shape}")

def normalize_image_01(img):
    """
    Normalize a 2D image to the range [0,1].
    """
    v = img.astype(np.float32)
    v_min = np.min(v)
    v_max = np.max(v)

    if v_max == v_min:
        return np.zeros_like(v)

    return (v - v_min) / (v_max - v_min)


def apply_binary_cutoff(img, threshold, use_abs=True, min_size=0, fill_holes=False):
    """
    Convert grayscale 2D image to binary 0/1 image.
    """
    v = np.abs(img) if use_abs else img
    v_bin = (v >= threshold)

    if fill_holes:
        v_bin = ndi.binary_fill_holes(v_bin)

    if min_size > 0:
        labels, num = ndi.label(v_bin)
        if num > 0:
            counts = np.bincount(labels.ravel())
            keep = counts >= min_size
            keep[0] = False
            v_bin = keep[labels]

    sparsity = (np.count_nonzero(~v_bin) / v_bin.size) * 100.0
    return v_bin.astype(np.uint8), sparsity


# ==========================================
# 2. COMPONENT EXTRACTION
# ==========================================

def extract_components_2d(img_bin, min_size=20):
    """
    Find connected regions of 1s in a binary 2D image.
    Returns a list of dicts, one per component.
    """
    structure = np.ones((3, 3), dtype=np.uint8)  # 8-connectivity
    labels, num = ndi.label(img_bin, structure=structure)

    components = []

    if num == 0:
        return components

    objs = ndi.find_objects(labels)

    for label_id, slc in enumerate(objs, start=1):
        if slc is None:
            continue

        comp_mask = (labels[slc] == label_id)
        size = int(np.count_nonzero(comp_mask))

        if size < min_size:
            continue

        coords_local = np.argwhere(comp_mask)
        y0, x0 = slc[0].start, slc[1].start
        coords_global = coords_local + np.array([y0, x0])

        centroid = coords_global.mean(axis=0)  # [y, x]

        components.append({
            "label": label_id,
            "slice": slc,
            "mask_local": comp_mask,
            "coords_global": coords_global,
            "centroid": centroid,
            "size": size,
        })

    return components


def extract_boundary_pixels(mask):
    """
    Extract boundary pixels from a binary 2D component mask.
    Boundary = mask - eroded(mask)
    """
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    boundary = mask & (~eroded)
    return boundary

# ==========================================
# 3. CROP MIDDLE THIRD
# ==========================================
def crop_middle_third_x(img):
    """
    Keep only the middle third of the image along x.
    For a 2D image shaped (y, x), remove the left third and right third.
    """
    if img.ndim != 2:
        raise ValueError(f"crop_middle_third_x expects a 2D image, got shape {img.shape}")

    h, w = img.shape
    x1 = w // 3
    x2 = (2 * w) // 3
    return img[:, x1:x2]


# ==========================================
# 3. BINARY STITCHED IMAGE
# ==========================================
def make_binary_for_display(img, threshold, use_abs=True, min_size=0, fill_holes=False):
    """
    Convenience wrapper for making a binary display image.
    """
    img_bin, _ = apply_binary_cutoff(
        img,
        threshold=threshold,
        use_abs=use_abs,
        min_size=min_size,
        fill_holes=fill_holes
    )
    return img_bin
# ==========================================
# 3. FOURIER BOUNDARY DESCRIPTOR
# ==========================================

def radial_fourier_descriptor_from_boundary_2d(component, n_bins=32, fourier_terms=8, eps=1e-10):
    """
    Build a Fourier descriptor from the 2D boundary radial-distance signal.

    Steps:
    - get boundary pixels
    - compute centroid-to-boundary distances
    - normalize distances by max distance
    - histogram those normalized distances
    - FFT the histogram
    - keep the first `fourier_terms` magnitudes
    - L2 normalize descriptor
    """
    mask_local = component["mask_local"]
    boundary = extract_boundary_pixels(mask_local)

    boundary_coords_local = np.argwhere(boundary)
    if boundary_coords_local.shape[0] == 0:
        return None, None

    slc = component["slice"]
    offset = np.array([slc[0].start, slc[1].start])
    boundary_coords_global = boundary_coords_local + offset

    centroid = component["centroid"]

    dists = np.linalg.norm(boundary_coords_global - centroid[None, :], axis=1)

    dmax = np.max(dists)
    if dmax < eps:
        return None, None

    dists_norm = dists / dmax

    radial_hist, _ = np.histogram(dists_norm, bins=n_bins, range=(0.0, 1.0))
    radial_hist = radial_hist.astype(np.float32)

    if np.sum(radial_hist) == 0:
        return None, None

    F = np.fft.fft(radial_hist)
    mag = np.abs(F).astype(np.float32)

    k = min(fourier_terms, mag.shape[0])
    desc = mag[:k].copy()

    if desc.shape[0] > 0:
        desc[0] *= 0.25

    norm = np.linalg.norm(desc)
    if norm < eps:
        return None, None

    desc /= norm

    meta = {
        "centroid": centroid,
        "size": component["size"],
        "bbox": component["slice"],
        "boundary_count": int(boundary_coords_global.shape[0]),
    }

    return desc, meta


def cosine_similarity(a, b, eps=1e-10):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


# ==========================================
# 4. DESCRIBE ALL COMPONENTS
# ==========================================

def describe_components_fourier_2d(img_bin, min_size=20, n_bins=32, fourier_terms=8):
    """
    Find all connected components and compute one Fourier descriptor per component.
    """
    components = extract_components_2d(img_bin, min_size=min_size)

    described = []
    for comp in components:
        desc, meta = radial_fourier_descriptor_from_boundary_2d(
            comp,
            n_bins=n_bins,
            fourier_terms=fourier_terms
        )
        if desc is None:
            continue

        described.append({
            "component": comp,
            "descriptor": desc,
            "meta": meta,
        })

    return described


# ==========================================
# 5. MATCH COMPONENTS WITH Y CONSTRAINT
# ==========================================

def match_components_x_only_2d(descs1, descs2, y_tolerance=10.0, min_similarity=0.7):
    """
    Match components between two 2D images assuming motion is only in x.

    Conditions:
    - y centroids must be close
    - descriptor similarity must be high enough
    """
    matches = []

    for i, a in enumerate(descs1):
        cy1, cx1 = a["meta"]["centroid"]

        best = None
        best_score = -np.inf

        for j, b in enumerate(descs2):
            cy2, cx2 = b["meta"]["centroid"]

            y_dist = abs(cy1 - cy2)
            if y_dist > y_tolerance:
                continue

            sim = cosine_similarity(a["descriptor"], b["descriptor"])
            if sim < min_similarity:
                continue

            s1 = a["meta"]["size"]
            s2 = b["meta"]["size"]
            size_ratio = min(s1, s2) / max(s1, s2)

            score = sim * size_ratio

            if score > best_score:
                best_score = score
                best = {
                    "idx1": i,
                    "idx2": j,
                    "sim": sim,
                    "score": score,
                    "y_dist": y_dist,
                    "shift_x": float(cx1 - cx2),
                    "centroid1": a["meta"]["centroid"],
                    "centroid2": b["meta"]["centroid"],
                    "size1": s1,
                    "size2": s2,
                }

        if best is not None:
            matches.append(best)

    return matches


# ==========================================
# 6. ESTIMATE FINAL X SHIFT
# ==========================================

def estimate_shift_from_matches(matches, rounding="nearest"):
    """
    Estimate one final x-shift from all component matches.
    """
    if not matches:
        raise ValueError("No valid component matches found.")

    shifts = np.array([m["shift_x"] for m in matches], dtype=np.float32)
    weights = np.array([m["score"] for m in matches], dtype=np.float32)

    if rounding == "nearest":
        shifts_rounded = np.round(shifts).astype(int)
    else:
        shifts_rounded = np.floor(shifts + 0.5).astype(int)

    unique_shifts = np.unique(shifts_rounded)
    vote_scores = {}

    for s in unique_shifts:
        vote_scores[int(s)] = float(np.sum(weights[shifts_rounded == s]))

    final_shift = max(vote_scores, key=vote_scores.get)

    return final_shift, vote_scores


# ==========================================
# 7. MAIN PIPELINE
# ==========================================

def run_component_boundary_stitcher_2d(
    img1,
    img2,
    binary_threshold=0.57,
    use_abs=True,
    min_size=20,
    fill_holes=False,
    descriptor_bins=32,
    fourier_terms=8,
    y_tolerance=10.0,
    min_similarity=0.7,
    verbose=True
):
    """
    Full 2D pipeline:
    1. threshold to binary
    2. find connected regions
    3. make one Fourier boundary descriptor per region
    4. match regions using y proximity + descriptor similarity
    5. estimate x shift
    """
    i1_bin, s1 = apply_binary_cutoff(
        img1,
        threshold=binary_threshold,
        use_abs=use_abs,
        min_size=min_size,
        fill_holes=fill_holes
    )
    i2_bin, s2 = apply_binary_cutoff(
        img2,
        threshold=binary_threshold,
        use_abs=use_abs,
        min_size=min_size,
        fill_holes=fill_holes
    )

    if verbose:
        print(f"[Binary cutoff] threshold={binary_threshold}")
        print(f" -> Img1 Sparsity: {s1:.1f}%")
        print(f" -> Img2 Sparsity: {s2:.1f}%")

    descs1 = describe_components_fourier_2d(
        i1_bin,
        min_size=min_size,
        n_bins=descriptor_bins,
        fourier_terms=fourier_terms
    )
    descs2 = describe_components_fourier_2d(
        i2_bin,
        min_size=min_size,
        n_bins=descriptor_bins,
        fourier_terms=fourier_terms
    )

    if verbose:
        print(f"[Components] Img1: {len(descs1)} descriptors")
        print(f"[Components] Img2: {len(descs2)} descriptors")
        print(f"[Descriptor settings] bins={descriptor_bins}, fourier_terms={fourier_terms}")

    matches = match_components_x_only_2d(
        descs1,
        descs2,
        y_tolerance=y_tolerance,
        min_similarity=min_similarity
    )

    if verbose:
        print(f"[Matches] {len(matches)} candidate matches")

    final_shift, vote_scores = estimate_shift_from_matches(matches)

    if verbose:
        print(f"\n[Result] Estimated x shift: {final_shift} pixels")

    return final_shift, i1_bin, i2_bin, descs1, descs2, matches, vote_scores


# ==========================================
# 8. OPTIONAL PLOT
# ==========================================

def plot_vote_scores(vote_scores):
    shifts = np.array(sorted(vote_scores.keys()))
    vals = np.array([vote_scores[s] for s in shifts])

    plt.figure(figsize=(10, 4))
    plt.plot(shifts, vals, marker="o")
    plt.xlabel("Estimated x shift (pixels)")
    plt.ylabel("Weighted vote score")
    plt.title("Component Match Voting for X Shift (2D)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 9. STITCH TWO 2D IMAGES
# ==========================================

def stitch_2d_images(img1, img2, shift_x):
    """
    Stitch img2 onto img1 using a shift along x.
    """
    shift = int(round(shift_x))
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    i1_start, i2_start = (abs(shift), 0) if shift < 0 else (0, shift)
    total_w = max(i1_start + w1, i2_start + w2)

    stitched = np.zeros((max(h1, h2), total_w), dtype=np.float32)
    stitched[:h1, i1_start:i1_start + w1] = img1

    inter_s = max(i1_start, i2_start)
    inter_e = min(i1_start + w1, i2_start + w2)

    if inter_e > inter_s:
        overlap = inter_e - inter_s
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)[None, :]
        if shift < 0:
            ramp = 1.0 - ramp

        stitched[:h1, inter_s:inter_e] = (
            img1[:, inter_s - i1_start:inter_e - i1_start] * (1.0 - ramp) +
            img2[:, inter_s - i2_start:inter_e - i2_start] * ramp
        )

    if i2_start < inter_s:
        stitched[:h2, i2_start:inter_s] = img2[:, :inter_s - i2_start]

    if i2_start + w2 > inter_e:
        stitched[:h2, inter_e:i2_start + w2] = img2[:, inter_e - i2_start:w2]

    return stitched


# ==========================================
# 10. EXECUTION
# ==========================================

if __name__ == "__main__":
    print(Path.cwd())
    IN_DIR = Path.cwd() / "DATA" / "1D TFM Data" / "Cu Pure 10MHz 16022026 Filtered"

    # Settings
    binary_threshold = 0.4
    use_abs = True
    min_size = 20
    fill_holes = False
    descriptor_bins = 32
    fourier_terms = 8
    y_tolerance = 10.0
    min_similarity = 0.7

    try:
        img1_raw = load_image_any(IN_DIR / "Cu_70_1_1_filtered_TFM.png")
        img2_raw = load_image_any(IN_DIR / "Cu_70_2_1_filtered_TFM.png")

        img1_raw = normalize_image_01(img1_raw)
        img2_raw = normalize_image_01(img2_raw)

        print("Img1 shape:", img1_raw.shape, "dtype:", img1_raw.dtype)
        print("Img2 shape:", img2_raw.shape, "dtype:", img2_raw.dtype)

    except FileNotFoundError:
        print("Data not found.")
        raise SystemExit

    # ------------------------------------------
    # CROP TO MIDDLE THIRD IN X
    # ------------------------------------------
    img1_crop = crop_middle_third_x(img1_raw)
    img2_crop = crop_middle_third_x(img2_raw)

    # Binary versions for display
    i1_crop_bin = make_binary_for_display(
        img1_crop,
        threshold=binary_threshold,
        use_abs=use_abs,
        min_size=min_size,
        fill_holes=fill_holes
    )
    i2_crop_bin = make_binary_for_display(
        img2_crop,
        threshold=binary_threshold,
        use_abs=use_abs,
        min_size=min_size,
        fill_holes=fill_holes
    )

    print("Img1 cropped binary foreground:", np.count_nonzero(i1_crop_bin), "out of", i1_crop_bin.size)
    print("Img2 cropped binary foreground:", np.count_nonzero(i2_crop_bin), "out of", i2_crop_bin.size)
    print("Img1 cropped binary unique values:", np.unique(i1_crop_bin))
    print("Img2 cropped binary unique values:", np.unique(i2_crop_bin))

    # ------------------------------------------
    # STITCH USING CROPPED IMAGES
    # ------------------------------------------
    stitch_shift_cropped, _, _, descs1_crop, descs2_crop, matches_crop, vote_scores_crop = run_component_boundary_stitcher_2d(
        img1_crop,
        img2_crop,
        binary_threshold=binary_threshold,
        use_abs=use_abs,
        min_size=min_size,
        fill_holes=fill_holes,
        descriptor_bins=descriptor_bins,
        fourier_terms=fourier_terms,
        y_tolerance=y_tolerance,
        min_similarity=min_similarity,
        verbose=True
    )

    stitched_cropped = stitch_2d_images(img1_crop, img2_crop, stitch_shift_cropped)

    plot_vote_scores(vote_scores_crop)

    # ------------------------------------------
    # SHOW ONLY CROPPED INPUTS + CROPPED STITCH
    # ------------------------------------------
    # ------------------------------------------
# SHOW CROPPED INPUTS + CROPPED STITCH
# ------------------------------------------

viewer = napari.Viewer(title="2D Fourier Descriptor Stitcher (Cropped)")

# Img1 raw
viewer.add_image(
    img1_crop,
    name="Img 1 Cropped (Raw)",
    colormap="gray"
)

# Img1 binary
viewer.add_image(
    i1_crop_bin.astype(np.float32),
    name="Img 1 Cropped (Binary)",
    colormap="yellow",
    contrast_limits=[0, 1],
    opacity=1.0
)

# Img2 raw shifted
viewer.add_image(
    img2_crop,
    name=f"Img 2 Cropped (Raw, shifted {stitch_shift_cropped}px)",
    colormap="magenta",
    blending="additive",
    translate=(0, stitch_shift_cropped)
)

# Img2 binary shifted
viewer.add_image(
    i2_crop_bin.astype(np.float32),
    name=f"Img 2 Cropped (Binary, shifted {stitch_shift_cropped}px)",
    colormap="green",
    contrast_limits=[0, 1],
    opacity=1.0,
    translate=(0, stitch_shift_cropped)
)

# Stitched result
viewer.add_image(
    stitched_cropped,
    name=f"Stitched Cropped ({stitch_shift_cropped}px)",
    colormap="viridis"
)

napari.run()