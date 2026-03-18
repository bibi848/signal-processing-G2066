import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import napari
from scipy import ndimage as ndi


# ==========================================
# 1. BASIC PREP
# ==========================================

def normalize_volume_01(vol):
    """
    Normalize a volume to the range [0,1].
    """
    v = vol.astype(np.float32)
    v_min = np.min(v)
    v_max = np.max(v)

    if v_max == v_min:
        return np.zeros_like(v)

    return (v - v_min) / (v_max - v_min)


def apply_binary_cutoff(vol, threshold, use_abs=True, min_size=0, fill_holes=False):
    """
    Convert grayscale volume to binary 0/1 volume.
    """
    v = np.abs(vol) if use_abs else vol
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

def extract_components(vol_bin, min_size=20):
    """
    Find connected regions of 1s in a binary 3D volume.
    Returns a list of dicts, one per component.
    """
    structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity
    labels, num = ndi.label(vol_bin, structure=structure)

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
        z0, y0, x0 = slc[0].start, slc[1].start, slc[2].start
        coords_global = coords_local + np.array([z0, y0, x0])

        centroid = coords_global.mean(axis=0)  # [z, y, x]

        components.append({
            "label": label_id,
            "slice": slc,
            "mask_local": comp_mask,
            "coords_global": coords_global,
            "centroid": centroid,
            "size": size,
        })

    return components


def extract_boundary_voxels(mask):
    """
    Extract surface voxels from a binary 3D component mask.
    Boundary = mask - eroded(mask)
    """
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool))
    boundary = mask & (~eroded)
    return boundary


# ==========================================
# 3. FOURIER BOUNDARY DESCRIPTOR
# ==========================================

def radial_fourier_descriptor_from_boundary(component, n_bins=32, fourier_terms=8, eps=1e-10):
    """
    Build a Fourier descriptor from the boundary radial-distance signal.

    Steps:
    - get boundary voxels
    - compute centroid-to-boundary distances
    - normalize distances by max distance
    - histogram those normalized distances
    - FFT the histogram
    - keep only the first `fourier_terms` magnitudes
    - L2 normalize descriptor
    """
    mask_local = component["mask_local"]
    boundary = extract_boundary_voxels(mask_local)

    boundary_coords_local = np.argwhere(boundary)
    if boundary_coords_local.shape[0] == 0:
        return None, None

    slc = component["slice"]
    offset = np.array([slc[0].start, slc[1].start, slc[2].start])
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

def describe_components_fourier(vol_bin, min_size=20, n_bins=32, fourier_terms=8):
    """
    Find all connected components and compute one Fourier descriptor per component.
    """
    components = extract_components(vol_bin, min_size=min_size)

    described = []
    for comp in components:
        desc, meta = radial_fourier_descriptor_from_boundary(
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
# 5. MATCH COMPONENTS WITH Y-Z CONSTRAINT
# ==========================================

def match_components_x_only(descs1, descs2, yz_tolerance=10.0, min_similarity=0.7):
    """
    Match components between two volumes assuming motion is only in x.

    Conditions:
    - y-z centroids must be close
    - descriptor similarity must be high enough
    """
    matches = []

    for i, a in enumerate(descs1):
        cz1, cy1, cx1 = a["meta"]["centroid"]

        best = None
        best_score = -np.inf

        for j, b in enumerate(descs2):
            cz2, cy2, cx2 = b["meta"]["centroid"]

            yz_dist = np.sqrt((cz1 - cz2) ** 2 + (cy1 - cy2) ** 2)
            if yz_dist > yz_tolerance:
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
                    "yz_dist": yz_dist,
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

def run_component_boundary_stitcher(
    vol1,
    vol2,
    binary_threshold=0.57,
    use_abs=True,
    min_size=20,
    fill_holes=False,
    descriptor_bins=32,
    fourier_terms=8,
    yz_tolerance=10.0,
    min_similarity=0.7,
    verbose=True
):
    """
    Full pipeline:
    1. threshold to binary
    2. find connected regions
    3. make one Fourier boundary descriptor per region
    4. match regions using y-z proximity + descriptor similarity
    5. estimate x shift
    """
    v1_bin, s1 = apply_binary_cutoff(
        vol1,
        threshold=binary_threshold,
        use_abs=use_abs,
        min_size=min_size,
        fill_holes=fill_holes
    )
    v2_bin, s2 = apply_binary_cutoff(
        vol2,
        threshold=binary_threshold,
        use_abs=use_abs,
        min_size=min_size,
        fill_holes=fill_holes
    )

    if verbose:
        print(f"[Binary cutoff] threshold={binary_threshold}")
        print(f" -> Vol1 Sparsity: {s1:.1f}%")
        print(f" -> Vol2 Sparsity: {s2:.1f}%")

    descs1 = describe_components_fourier(
        v1_bin,
        min_size=min_size,
        n_bins=descriptor_bins,
        fourier_terms=fourier_terms
    )
    descs2 = describe_components_fourier(
        v2_bin,
        min_size=min_size,
        n_bins=descriptor_bins,
        fourier_terms=fourier_terms
    )

    if verbose:
        print(f"[Components] Vol1: {len(descs1)} descriptors")
        print(f"[Components] Vol2: {len(descs2)} descriptors")
        print(f"[Descriptor settings] bins={descriptor_bins}, fourier_terms={fourier_terms}")

    matches = match_components_x_only(
        descs1,
        descs2,
        yz_tolerance=yz_tolerance,
        min_similarity=min_similarity
    )

    if verbose:
        print(f"[Matches] {len(matches)} candidate matches")

    final_shift, vote_scores = estimate_shift_from_matches(matches)

    if verbose:
        print(f"\n[Result] Estimated x shift: {final_shift} voxels")

    return final_shift, v1_bin, v2_bin, descs1, descs2, matches, vote_scores


# ==========================================
# 8. OPTIONAL PLOT
# ==========================================

def plot_vote_scores(vote_scores):
    shifts = np.array(sorted(vote_scores.keys()))
    vals = np.array([vote_scores[s] for s in shifts])

    plt.figure(figsize=(10, 4))
    plt.plot(shifts, vals, marker="o")
    plt.xlabel("Estimated x shift (voxels)")
    plt.ylabel("Weighted vote score")
    plt.title("Component Match Voting for X Shift")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 9. EXECUTION
# ==========================================

if __name__ == "__main__":
    IN_DIR = Path.cwd() / "DATA" / "2D TFM Data" / "Cu Pure 7.5MHz Ex 11032026 Filtered"

    try:
        vol1_raw = np.load(IN_DIR / "A2_filtered_3D_TFM.npy")
        vol2_raw = np.load(IN_DIR / "A3_filtered_3D_TFM.npy")

        vol1_raw = normalize_volume_01(vol1_raw)
        vol2_raw = normalize_volume_01(vol2_raw)

    except FileNotFoundError:
        print("Data not found.")
        raise SystemExit

    stitch_shift, v1_bin, v2_bin, descs1, descs2, matches, vote_scores = run_component_boundary_stitcher(
        vol1_raw,
        vol2_raw,
        binary_threshold=0.57,
        use_abs=True,
        min_size=20,
        fill_holes=False,
        descriptor_bins=32,
        fourier_terms=8,
        yz_tolerance=10.0,
        min_similarity=0.7,
        verbose=True
    )

    plot_vote_scores(vote_scores)

    clim_raw = sorted([
        float(np.percentile(vol1_raw, 0.1)),
        float(np.percentile(vol1_raw, 99.9))
    ])
    if clim_raw[0] == clim_raw[1]:
        clim_raw = [clim_raw[0], clim_raw[0] + 1]

    viewer = napari.Viewer(title="3D Component Fourier Descriptor Stitcher")

    viewer.add_image(
        vol1_raw,
        name="Vol 1 (Raw)",
        colormap="cyan",
        contrast_limits=clim_raw,
        opacity=0.35
    )

    viewer.add_image(
        v1_bin,
        name="Vol 1 (Binary)",
        colormap="yellow",
        contrast_limits=[0, 1],
        opacity=0.6
    )

    trans = [0, 0, 0]
    trans[2] = stitch_shift

    viewer.add_image(
        v2_bin,
        name=f"Vol 2 binary (Shifted {stitch_shift}px)",
        colormap="green",
        blending="additive",
        translate=trans,
        contrast_limits=clim_raw
    )

    viewer.add_image(
        vol2_raw,
        name=f"Vol 2 raw (Shifted {stitch_shift}px)",
        colormap="magenta",
        blending="additive",
        translate=trans,
        contrast_limits=clim_raw
    )

    napari.run()