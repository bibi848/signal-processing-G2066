import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import napari
from scipy import ndimage as ndi


# ==========================================
# 1. BASIC PREPROCESSING
# ==========================================

def normalize_volume_01(vol):
    """
    Normalize volume to [0, 1].
    """
    v = vol.astype(np.float32)
    v_min = np.min(v)
    v_max = np.max(v)

    if v_max == v_min:
        return np.zeros_like(v, dtype=np.float32)

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
# 2. COMPONENT + BOUNDARY EXTRACTION
# ==========================================

def extract_components(vol_bin, min_size=20):
    """
    Find connected regions of 1s in a binary 3D volume.
    Returns a list of component dictionaries.
    """
    structure = np.ones((3, 3, 3), dtype=np.uint8)
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
        offset = np.array([slc[0].start, slc[1].start, slc[2].start])
        coords_global = coords_local + offset
        centroid = coords_global.mean(axis=0)  # z, y, x

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
    Boundary = mask - eroded(mask)
    """
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool))
    boundary = mask & (~eroded)
    return boundary


# ==========================================
# 3. FOURIER DESCRIPTOR FOR ONE COMPONENT
# ==========================================

def radial_fourier_descriptor_from_boundary(component, n_bins=32, fourier_terms=8, eps=1e-10):
    """
    Build a Fourier descriptor from the boundary radial-distance signal.

    Steps:
    - extract boundary voxels
    - compute distance of each boundary voxel from centroid
    - normalize distances
    - histogram those distances into n_bins
    - take FFT magnitude of that histogram
    - keep only the first `fourier_terms`
    - normalize the descriptor
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
        "boundary_count": int(boundary_coords_global.shape[0]),
        "bbox": component["slice"],
    }

    return desc, meta


def describe_components_fourier(vol_bin, min_size=20, n_bins=32, fourier_terms=8):
    """
    Find connected components and compute one Fourier descriptor per component.
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


def cosine_similarity(a, b, eps=1e-10):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


# ==========================================
# 4. MATCH COMPONENTS (X-ONLY MOTION)
# ==========================================

def match_components_x_only(descs1, descs2, yz_tolerance=10.0, min_similarity=0.7):
    """
    Match components assuming movement only in x.
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


def estimate_shift_from_matches(matches):
    """
    Estimate one final x-shift from all matches by weighted voting.
    """
    if not matches:
        return 0, {}, 0.0

    shifts = np.array([m["shift_x"] for m in matches], dtype=np.float32)
    weights = np.array([m["score"] for m in matches], dtype=np.float32)

    shifts_rounded = np.round(shifts).astype(int)
    unique_shifts = np.unique(shifts_rounded)

    vote_scores = {}
    for s in unique_shifts:
        vote_scores[int(s)] = float(np.sum(weights[shifts_rounded == s]))

    final_shift = max(vote_scores, key=vote_scores.get)
    confidence = vote_scores[final_shift]

    return final_shift, vote_scores, confidence


# ==========================================
# 5. FOURIER-DESCRIPTOR STITCHER CLASS
# ==========================================

class TFMFourierDescriptorStitcher:
    def __init__(self, vol1, vol2, axis=2):
        self.vol1 = vol1.astype(np.float32)
        self.vol2 = vol2.astype(np.float32)
        self.axis = axis
        self.shift_index = 0
        self.best_score = -1.0
        self.matches = []
        self.vote_scores = {}

    def find_optimal_shift(
        self,
        binary_threshold=0.57,
        use_abs=True,
        min_size=20,
        fill_holes=False,
        descriptor_bins=32,
        fourier_terms=8,
        yz_tolerance=10.0,
        min_similarity=0.7,
        ignore_top=30,
        expected=0,
        tolerance=200,
        positive_only=True,
        verbose=True
    ):
        if self.axis != 2:
            raise NotImplementedError("This stitcher currently supports x-only motion (axis=2).")

        v1_use = self.vol1[ignore_top:, :, :] if ignore_top > 0 else self.vol1
        v2_use = self.vol2[ignore_top:, :, :] if ignore_top > 0 else self.vol2

        v1_bin, s1 = apply_binary_cutoff(
            v1_use,
            threshold=binary_threshold,
            use_abs=use_abs,
            min_size=min_size,
            fill_holes=fill_holes
        )
        v2_bin, s2 = apply_binary_cutoff(
            v2_use,
            threshold=binary_threshold,
            use_abs=use_abs,
            min_size=min_size,
            fill_holes=fill_holes
        )

        if verbose:
            print(f"[Binary cutoff] threshold={binary_threshold}")
            print(f" -> Vol1 Sparsity: {s1:.1f}% | Vol2 Sparsity: {s2:.1f}%")

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
            print(f"[Components] Vol1: {len(descs1)} descriptors | Vol2: {len(descs2)} descriptors")
            print(f"[Descriptor settings] bins={descriptor_bins}, fourier_terms={fourier_terms}")

        matches = match_components_x_only(
            descs1,
            descs2,
            yz_tolerance=yz_tolerance,
            min_similarity=min_similarity
        )

        filtered = []
        for m in matches:
            sx = m["shift_x"]
            if sx < expected - tolerance or sx > expected + tolerance:
                continue
            if positive_only and sx < 0:
                continue
            filtered.append(m)

        self.matches = filtered

        if verbose:
            print(f"[Matches] {len(filtered)} candidate matches after constraints")

        if not filtered:
            print("  !! Warning: No valid Fourier-descriptor matches found for this pair.")
            self.shift_index = 0
            self.best_score = 0.0
            self.vote_scores = {}
            return 0

        final_shift, vote_scores, confidence = estimate_shift_from_matches(filtered)

        self.shift_index = int(final_shift)
        self.best_score = float(confidence)
        self.vote_scores = vote_scores

        if verbose:
            print(f" -> Descriptor Shift: {self.shift_index} px | Conf: {self.best_score:.3f}")

        return self.shift_index

    def stitch(self, blend_mode='linear'):
        """
        Stitch vol2 onto vol1 using self.shift_index along self.axis.
        """
        shift = int(round(self.shift_index))
        s1, s2 = self.vol1.shape, self.vol2.shape
        L1, L2 = s1[self.axis], s2[self.axis]

        v1_start, v2_start = (abs(shift), 0) if shift < 0 else (0, shift)
        total_len = max(v1_start + L1, v2_start + L2)

        stitched = np.zeros((*s1[:self.axis], total_len, *s1[self.axis+1:]), dtype=np.float32)

        sl1 = [slice(None)] * 3
        out1 = [slice(None)] * 3
        sl1[self.axis] = slice(0, L1)
        out1[self.axis] = slice(v1_start, v1_start + L1)
        stitched[tuple(out1)] = self.vol1[tuple(sl1)]

        inter_s = max(v1_start, v2_start)
        inter_e = min(v1_start + L1, v2_start + L2)

        if inter_e > inter_s:
            overlap = inter_e - inter_s
            ramp_shape = [1, 1, 1]
            ramp_shape[self.axis] = overlap
            ramp = np.linspace(0, 1, overlap, dtype=np.float32).reshape(ramp_shape)
            if shift < 0:
                ramp = 1.0 - ramp

            idx_out = [slice(None)] * 3
            idx1 = [slice(None)] * 3
            idx2 = [slice(None)] * 3

            idx_out[self.axis] = slice(inter_s, inter_e)
            idx1[self.axis] = slice(inter_s - v1_start, inter_e - v1_start)
            idx2[self.axis] = slice(inter_s - v2_start, inter_e - v2_start)

            stitched[tuple(idx_out)] = (
                self.vol1[tuple(idx1)] * (1.0 - ramp) +
                self.vol2[tuple(idx2)] * ramp
            )

        if v2_start < inter_s:
            idx_out = [slice(None)] * 3
            idx2 = [slice(None)] * 3
            idx_out[self.axis] = slice(v2_start, inter_s)
            idx2[self.axis] = slice(0, inter_s - v2_start)
            stitched[tuple(idx_out)] = self.vol2[tuple(idx2)]

        if v2_start + L2 > inter_e:
            idx_out = [slice(None)] * 3
            idx2 = [slice(None)] * 3
            idx_out[self.axis] = slice(inter_e, v2_start + L2)
            idx2[self.axis] = slice(inter_e - v2_start, L2)
            stitched[tuple(idx_out)] = self.vol2[tuple(idx2)]

        return stitched


# ==========================================
# 6. OPTIONAL PLOT
# ==========================================

def plot_vote_scores(vote_scores, title="Fourier Descriptor Shift Votes"):
    if not vote_scores:
        print("No vote scores to plot.")
        return

    shifts = np.array(sorted(vote_scores.keys()))
    vals = np.array([vote_scores[s] for s in shifts])

    plt.figure(figsize=(10, 4))
    plt.plot(shifts, vals, marker="o")
    plt.xlabel("Estimated x shift (voxels)")
    plt.ylabel("Weighted vote score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 7. MAIN SEQUENTIAL EXECUTION
# ==========================================

if __name__ == "__main__":
    IN_DIR = Path.cwd() / "DATA" / "2D TFM Data" / "FeC Smile 3MHz 04022026 Filtered"

    selected = [
        IN_DIR / "FeC_40_4_filtered_3D_TFM.npy",
        IN_DIR / "FeC_40_3_filtered_3D_TFM.npy",
        IN_DIR / "FeC_40_2_filtered_3D_TFM.npy"
    ]

    print(f"Found {len(selected)} volumes. Starting Sequential Fourier-Descriptor Stitch...")

    global_vol = np.load(selected[0]).astype(np.float32)
    global_vol = normalize_volume_01(global_vol)

    for i in range(1, len(selected)):
        next_vol = np.load(selected[i]).astype(np.float32)
        next_vol = normalize_volume_01(next_vol)

        print(f"\nMerging {i+1}/{len(selected)}: {selected[i].name}")

        window_size = int(next_vol.shape[2] * 1.2)
        offset_removed = max(0, global_vol.shape[2] - window_size)
        active_tail = global_vol[:, :, offset_removed:]

        stitcher = TFMFourierDescriptorStitcher(active_tail, next_vol, axis=2)
        relative_shift = stitcher.find_optimal_shift(
            binary_threshold=0.57,
            use_abs=True,
            min_size=20,
            fill_holes=False,
            descriptor_bins=32,
            fourier_terms=8,
            yz_tolerance=10.0,
            min_similarity=0.7,
            ignore_top=30,
            expected=0,
            tolerance=200,
            positive_only=True,
            verbose=True
        )

        global_shift = offset_removed + relative_shift

        print(
            f" -> Global Shift: {global_shift} px "
            f"(Local: {relative_shift}) | Conf: {stitcher.best_score:.3f}"
        )

        full_stitcher = TFMFourierDescriptorStitcher(global_vol, next_vol, axis=2)
        full_stitcher.shift_index = global_shift
        global_vol = full_stitcher.stitch()

        plot_vote_scores(
            stitcher.vote_scores,
            title=f"Shift votes for merge {i}/{len(selected)-1}"
        )

    v_min = float(np.percentile(global_vol, 0.1))
    v_max = float(np.percentile(global_vol, 99.9))
    clim = sorted([v_min, v_max])
    if clim[0] == clim[1]:
        clim = [clim[0], clim[0] + 1]

    viewer = napari.Viewer(title="Sequential Fourier Descriptor Assembly")
    viewer.add_image(
        global_vol,
        name="Final Assembly",
        contrast_limits=clim,
        colormap="viridis"
    )

    napari.run()