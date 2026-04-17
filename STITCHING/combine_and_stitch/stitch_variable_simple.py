# %%
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, label, find_objects
import napari


@dataclass
class VariableStitcher:
    """
    Tile-based 3D stitching for volumes with shape (z, x, y).

    Main workflow
    -------------
    stitcher = VariableStitcher(...)
    result = stitcher.stitch(vol1, vol2)

    Returned result contains:
    - best_shift
    - shifts
    - corr_values
    - diagnostics
    - canvas1
    - canvas2
    """

    axis: str = "x"
    max_shift: int = 100

    grid: Tuple[int, int] = (4, 4)
    adaptive_grid: bool = False

    grid_binary_threshold: float = 0.35
    corr_binary_threshold: Optional[float] = None

    ignore_top: int = 0
    min_voxels: int = 10
    tile_multiple: Tuple[float, float] = (2.0, 2.0)
    min_grid: Tuple[int, int] = (4, 4)
    max_grid: Tuple[int, int] = (12, 12)
    opening_structure: Optional[np.ndarray] = None
    size_statistic: str = "median"

    # Stores latest stitch result for convenience
    last_result: Optional[Dict[str, Any]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")

        if self.max_shift < 0:
            raise ValueError("max_shift must be >= 0")

        self._validate_threshold(self.grid_binary_threshold, "grid_binary_threshold")

        if self.corr_binary_threshold is not None:
            self._validate_threshold(self.corr_binary_threshold, "corr_binary_threshold")

        if self.size_statistic not in ("mean", "median"):
            raise ValueError("size_statistic must be 'mean' or 'median'")

    # ================================
    # Public API
    # ================================
    def stitch(self, vol1: np.ndarray, vol2: np.ndarray) -> Dict[str, Any]:
        """
        Compute tiled correlation stitch and return full result.
        """
        vol1 = self._as_float32_3d(vol1, name="vol1")
        vol2 = self._as_float32_3d(vol2, name="vol2")

        best_shift, shifts, corr_values, diagnostics = self._normalised_correlation_3D_tiled(
            vol1=vol1,
            vol2=vol2,
        )

        canvas1, canvas2 = self._stitch_volumes(
            vol1=vol1,
            vol2=vol2,
            shift=best_shift,
        )

        result = {
            "best_shift": best_shift,
            "shifts": shifts,
            "corr_values": corr_values,
            "diagnostics": diagnostics,
            "canvas1": canvas1,
            "canvas2": canvas2,
        }

        self.last_result = result
        return result

    def stitch_from_files(self, vol1_path: str | Path, vol2_path: str | Path) -> Dict[str, Any]:
        """
        Load two .npy files and stitch them.
        """
        vol1_path = Path(vol1_path)
        vol2_path = Path(vol2_path)

        if not vol1_path.exists():
            raise FileNotFoundError(f"Could not find: {vol1_path}")
        if not vol2_path.exists():
            raise FileNotFoundError(f"Could not find: {vol2_path}")

        vol1 = np.load(vol1_path).astype(np.float32)
        vol2 = np.load(vol2_path).astype(np.float32)

        result = self.stitch(vol1, vol2)
        result["vol1_path"] = vol1_path
        result["vol2_path"] = vol2_path
        return result

    def print_summary(self, result: Optional[Dict[str, Any]] = None) -> None:
        """
        Print a compact summary of a stitch result.
        """
        result = self._get_result(result)
        diagnostics = result["diagnostics"]

        print("\nTiled stitcher complete.")
        print(f"Axis: {self.axis}")
        print(f"Grid: {diagnostics['grid']}")
        print(f"Valid tiles: {diagnostics['valid_tile_count']}")
        print(f"Best shift: {result['best_shift']} voxels")
        print(f"Best summed correlation: {np.max(result['corr_values']):.6f}")
        print(f"Grid binary threshold: {diagnostics['grid_binary_threshold']}")
        print(f"Correlation binary threshold: {diagnostics['corr_binary_threshold']}")

        if diagnostics["grid_info"] is not None:
            gi = diagnostics["grid_info"]
            print("\nAdaptive grid info:")
            print(f"  Raw grid: {gi['raw_grid']}")
            print(f"  Final grid: {gi['grid']}")
            print(f"  Components: {gi['num_components_total']}")
            print(
                "  Representative sizes: "
                f"({gi['representative_size_axis0']:.2f}, {gi['representative_size_axis1']:.2f})"
            )
            print(f"  Tile sizes: ({gi['tile_size_axis0']}, {gi['tile_size_axis1']})")

    def plot_correlation(self, result: Optional[Dict[str, Any]] = None) -> None:
        result = self._get_result(result)

        shifts = result["shifts"]
        corr_values = result["corr_values"]
        best_shift = result["best_shift"]

        plt.figure(figsize=(10, 4))
        plt.plot(shifts, corr_values, linewidth=2)
        plt.axvline(best_shift, linestyle="--", label=f"Best shift = {best_shift}")
        plt.xlabel("Shift (voxels)")
        plt.ylabel("Summed normalised correlation")
        plt.title("Tiled 3D Normalised Cross-Correlation")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_vote_map(self, result: Optional[Dict[str, Any]] = None) -> None:
        result = self._get_result(result)
        diagnostics = result["diagnostics"]

        vote_map = diagnostics["tile_vote_map"]

        plt.figure(figsize=(6, 5))
        heat = np.ma.masked_invalid(vote_map)
        im = plt.imshow(heat, aspect="auto", interpolation="nearest")
        plt.colorbar(im, label="Tile best shift")
        plt.title("Tile Vote Map")
        plt.xlabel("Tile column")
        plt.ylabel("Tile row")
        plt.tight_layout()
        plt.show()

    def plot_binary_projections(self, result: Optional[Dict[str, Any]] = None) -> None:
        result = self._get_result(result)
        diagnostics = result["diagnostics"]
        grid_info = diagnostics["grid_info"]

        if grid_info is None:
            print("No adaptive grid info available.")
            return

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(grid_info["mask1_projection"], aspect="auto", interpolation="nearest")
        plt.title("Vol 1 Binary Projection")
        plt.xlabel("Column axis")
        plt.ylabel("Row axis")

        plt.subplot(1, 2, 2)
        plt.imshow(grid_info["mask2_projection"], aspect="auto", interpolation="nearest")
        plt.title("Vol 2 Binary Projection")
        plt.xlabel("Column axis")
        plt.ylabel("Row axis")

        plt.tight_layout()
        plt.show()

    def view_in_napari(
        self,
        result: Optional[Dict[str, Any]] = None,
        title: str = "Tiled 3D Stitcher",
        show_binary_masks: bool = True,
) -> None:

        result = self._get_result(result)

        canvas1 = self.normalise_for_display(result["canvas1"])
        canvas2 = self.normalise_for_display(result["canvas2"])
        best_shift = result["best_shift"]

        viewer = napari.Viewer(title=title)

        viewer.add_image(
            canvas1,
            name="Volume 1",
            colormap="magenta",
            blending="additive",
        )
        viewer.add_image(
            canvas2,
            name=f"Volume 2 shifted by {best_shift}",
            colormap="cyan",
            blending="additive",
        )

        if show_binary_masks:
            diagnostics = result["diagnostics"]
            grid_info = diagnostics.get("grid_info", None)

            if grid_info is not None:
                mask1 = grid_info.get("mask1", None)
                mask2 = grid_info.get("mask2", None)

                if mask1 is not None and mask2 is not None:

                    if self.axis == "x":
                        left_offset = max(0, -best_shift)

                        mask_canvas1 = np.zeros_like(result["canvas1"], dtype=np.float32)
                        mask_canvas2 = np.zeros_like(result["canvas2"], dtype=np.float32)

                        mask_canvas1[:, left_offset:left_offset + mask1.shape[1], :] = mask1.astype(np.float32)

                        x2_start = left_offset + best_shift
                        mask_canvas2[:, x2_start:x2_start + mask2.shape[1], :] = mask2.astype(np.float32)

                    else:
                        left_offset = max(0, -best_shift)

                        mask_canvas1 = np.zeros_like(result["canvas1"], dtype=np.float32)
                        mask_canvas2 = np.zeros_like(result["canvas2"], dtype=np.float32)

                        mask_canvas1[:, :, left_offset:left_offset + mask1.shape[2]] = mask1.astype(np.float32)

                        y2_start = left_offset + best_shift
                        mask_canvas2[:, :, y2_start:y2_start + mask2.shape[2]] = mask2.astype(np.float32)

                    viewer.add_image(
                        mask_canvas1,
                        name="Binary Mask 1",
                        colormap="yellow",
                        blending="additive",
                        opacity=0.35,
                    )

                    viewer.add_image(
                        mask_canvas2,
                        name="Binary Mask 2",
                        colormap="green",
                        blending="additive",
                        opacity=0.35,
                    )

        napari.run()

    @staticmethod
    def normalise_for_display(vol: np.ndarray) -> np.ndarray:
        vmin = np.min(vol)
        vmax = np.max(vol)

        if vmax > vmin:
            return (vol - vmin) / (vmax - vmin)
        return np.zeros_like(vol, dtype=np.float32)

    # ================================
    # Internal utilities
    # ================================
    @staticmethod
    def _validate_threshold(value: float, name: str) -> None:
        if not (0 <= value <= 1):
            raise ValueError(f"{name} must be between 0 and 1")

    @staticmethod
    def _as_float32_3d(vol: np.ndarray, name: str = "volume") -> np.ndarray:
        vol = np.asarray(vol)
        if vol.ndim != 3:
            raise ValueError(f"{name} must be a 3D array, got shape {vol.shape}")
        return vol.astype(np.float32, copy=False)

    def _get_result(self, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if result is not None:
            return result
        if self.last_result is None:
            raise ValueError("No result provided and no previous result stored.")
        return self.last_result

    # ================================
    # Binary mask and adaptive grid
    # ================================
    def make_binary_mask(self, vol: np.ndarray, binary_threshold: float, opening_structure=None) -> np.ndarray:
        """
        Create a binary mask from normalized absolute intensity.
        """
        self._validate_threshold(binary_threshold, "binary_threshold")

        v_abs = np.abs(vol)
        v_max = np.max(v_abs)

        if v_max == 0:
            return np.zeros_like(vol, dtype=bool)

        v_norm = v_abs / v_max
        mask = v_norm >= binary_threshold

        if opening_structure is not None:
            mask = binary_opening(mask, structure=opening_structure)

        return mask

    def measure_component_sizes_3D(
        self,
        binary_vol: np.ndarray,
        ignore_top: int = 0,
        min_voxels: int = 10,
    ) -> list[dict]:
        """
        Measure connected-component extents in 3D binary volume.
        """
        z_dim = binary_vol.shape[0]
        z_start = min(ignore_top, z_dim)
        binary_crop = binary_vol[z_start:, :, :]

        labeled, _ = label(binary_crop)
        slices = find_objects(labeled)

        component_sizes = []

        for comp_idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue

            z_slc, x_slc, y_slc = slc
            component_mask = labeled[slc] == comp_idx
            voxels = int(np.count_nonzero(component_mask))

            if voxels < min_voxels:
                continue

            z_size = int(z_slc.stop - z_slc.start)
            x_size = int(x_slc.stop - x_slc.start)
            y_size = int(y_slc.stop - y_slc.start)

            component_sizes.append(
                {
                    "label": comp_idx,
                    "z_size": z_size,
                    "x_size": x_size,
                    "y_size": y_size,
                    "voxels": voxels,
                }
            )

        return component_sizes

    def estimate_adaptive_grid_from_binary_mask(
        self,
        vol1: np.ndarray,
        vol2: np.ndarray,
    ) -> tuple[tuple[int, int], dict]:
        """
        Estimate adaptive grid using binary masking only.

        For axis='x', the grid is defined in the (z, y) plane.
        For axis='y', the grid is defined in the (z, x) plane.
        """
        mask1 = self.make_binary_mask(
            vol1,
            binary_threshold=self.grid_binary_threshold,
            opening_structure=self.opening_structure,
        )
        mask2 = self.make_binary_mask(
            vol2,
            binary_threshold=self.grid_binary_threshold,
            opening_structure=self.opening_structure,
        )

        sizes1 = self.measure_component_sizes_3D(
            mask1,
            ignore_top=self.ignore_top,
            min_voxels=self.min_voxels,
        )
        sizes2 = self.measure_component_sizes_3D(
            mask2,
            ignore_top=self.ignore_top,
            min_voxels=self.min_voxels,
        )
        all_sizes = sizes1 + sizes2

        if not all_sizes:
            raise ValueError(
                "No binary components found for adaptive grid sizing. "
                "Try lowering grid_binary_threshold or min_voxels."
            )

        if self.axis == "x":
            size0_vals = np.array([s["z_size"] for s in all_sizes], dtype=float)
            size1_vals = np.array([s["y_size"] for s in all_sizes], dtype=float)
            plane_dim_0 = vol1.shape[0] - min(self.ignore_top, vol1.shape[0])
            plane_dim_1 = vol1.shape[2]
        else:
            size0_vals = np.array([s["z_size"] for s in all_sizes], dtype=float)
            size1_vals = np.array([s["x_size"] for s in all_sizes], dtype=float)
            plane_dim_0 = vol1.shape[0] - min(self.ignore_top, vol1.shape[0])
            plane_dim_1 = vol1.shape[1]

        if self.size_statistic == "mean":
            rep0 = float(np.mean(size0_vals))
            rep1 = float(np.mean(size1_vals))
        else:
            rep0 = float(np.median(size0_vals))
            rep1 = float(np.median(size1_vals))

        tile0 = max(1, int(round(self.tile_multiple[0] * rep0)))
        tile1 = max(1, int(round(self.tile_multiple[1] * rep1)))

        raw_rows = max(1, plane_dim_0 // tile0)
        raw_cols = max(1, plane_dim_1 // tile1)

        rows = int(np.clip(raw_rows, self.min_grid[0], self.max_grid[0]))
        cols = int(np.clip(raw_cols, self.min_grid[1], self.max_grid[1]))

        if self.axis == "x":
            proj1 = np.any(mask1[min(self.ignore_top, mask1.shape[0]):, :, :], axis=1)
            proj2 = np.any(mask2[min(self.ignore_top, mask2.shape[0]):, :, :], axis=1)
        else:
            proj1 = np.any(mask1[min(self.ignore_top, mask1.shape[0]):, :, :], axis=2)
            proj2 = np.any(mask2[min(self.ignore_top, mask2.shape[0]):, :, :], axis=2)

        info = {
            "grid": (rows, cols),
            "raw_grid": (raw_rows, raw_cols),
            "representative_size_axis0": rep0,
            "representative_size_axis1": rep1,
            "tile_size_axis0": tile0,
            "tile_size_axis1": tile1,
            "num_components_vol1": len(sizes1),
            "num_components_vol2": len(sizes2),
            "num_components_total": len(all_sizes),
            "mask1_projection": proj1,
            "mask2_projection": proj2,
            "mask1": mask1,
            "mask2": mask2,
            "binary_threshold": self.grid_binary_threshold,
            "ignore_top": self.ignore_top,
            "min_voxels": self.min_voxels,
            "tile_multiple": self.tile_multiple,
            "size_statistic": self.size_statistic,
        }

        return (rows, cols), info

    # ================================
    # Correlation
    # ================================
    def _normalised_correlation_3D(
        self,
        vol1: np.ndarray,
        vol2: np.ndarray,
        binary_threshold: Optional[float] = None,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Compute normalized cross-correlation between two volumes.

        If binary_threshold is provided, correlation is computed only on voxels
        where both overlapping regions pass the binary mask.
        """
        z1, x1, y1 = vol1.shape
        z2, x2, y2 = vol2.shape

        if binary_threshold is not None:
            self._validate_threshold(binary_threshold, "binary_threshold")

        v1_abs = np.abs(vol1)
        v2_abs = np.abs(vol2)

        v1_max = np.max(v1_abs)
        v2_max = np.max(v2_abs)

        if binary_threshold is not None:
            mask1_full = np.zeros_like(vol1, dtype=bool) if v1_max == 0 else (v1_abs / v1_max) >= binary_threshold
            mask2_full = np.zeros_like(vol2, dtype=bool) if v2_max == 0 else (v2_abs / v2_max) >= binary_threshold
        else:
            mask1_full = None
            mask2_full = None

        shifts = np.arange(-self.max_shift, self.max_shift + 1)
        corr_values = []

        for d in shifts:
            if self.axis == "x":
                a1_start = max(0, d)
                a1_end = min(x1, x2 + d)

                a2_start = max(0, -d)
                a2_end = min(x2, x1 - d)

                if (a1_end - a1_start) <= 0:
                    corr_values.append(0.0)
                    continue

                region1 = vol1[:, a1_start:a1_end, :]
                region2 = vol2[:, a2_start:a2_end, :]

                if binary_threshold is not None:
                    mask1 = mask1_full[:, a1_start:a1_end, :]
                    mask2 = mask2_full[:, a2_start:a2_end, :]

            else:
                a1_start = max(0, d)
                a1_end = min(y1, y2 + d)

                a2_start = max(0, -d)
                a2_end = min(y2, y1 - d)

                if (a1_end - a1_start) <= 0:
                    corr_values.append(0.0)
                    continue

                region1 = vol1[:, :, a1_start:a1_end]
                region2 = vol2[:, :, a2_start:a2_end]

                if binary_threshold is not None:
                    mask1 = mask1_full[:, :, a1_start:a1_end]
                    mask2 = mask2_full[:, :, a2_start:a2_end]

            if binary_threshold is not None:
                joint_mask = mask1 & mask2

                if not np.any(joint_mask):
                    corr_values.append(0.0)
                    continue

                r1 = region1[joint_mask]
                r2 = region2[joint_mask]
            else:
                r1 = region1.ravel()
                r2 = region2.ravel()

            numerator = np.sum(r1 * r2)
            denom = np.sqrt(np.sum(r1 ** 2) * np.sum(r2 ** 2))
            corr_values.append(numerator / denom if denom > 0 else 0.0)

        corr_values = np.array(corr_values, dtype=float)
        best_index = int(np.argmax(corr_values))
        best_shift = int(shifts[best_index])

        return best_shift, shifts, corr_values

    def _normalised_correlation_3D_tiled(
        self,
        vol1: np.ndarray,
        vol2: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray, dict]:
        """
        Compute normalized cross-correlation tile-by-tile, then combine scores.
        """
        z1, x1, y1 = vol1.shape
        z2, x2, y2 = vol2.shape

        if self.adaptive_grid:
            grid, grid_info = self.estimate_adaptive_grid_from_binary_mask(vol1, vol2)
        else:
            grid = self.grid
            grid_info = None

        rows, cols = grid
        shifts = np.arange(-self.max_shift, self.max_shift + 1)
        corr_values = np.zeros_like(shifts, dtype=float)

        tile_vote_map = np.full((rows, cols), np.nan, dtype=float)
        tile_peak_map = np.full((rows, cols), np.nan, dtype=float)
        valid_tile_count = 0

        if self.axis == "x":
            if z1 != z2 or y1 != y2:
                raise ValueError("For x stitching, z and y dimensions must match")

            z_start = min(self.ignore_top, z1)
            z_usable = z1 - z_start
            tile_z = z_usable // rows
            tile_y = y1 // cols

            if tile_z <= 0 or tile_y <= 0:
                raise ValueError(f"Grid {grid} is too fine for volume shape {vol1.shape}")

            for r in range(rows):
                for c in range(cols):
                    zs = z_start + r * tile_z
                    ze = z_start + (r + 1) * tile_z if r < rows - 1 else z1

                    ys = c * tile_y
                    ye = (c + 1) * tile_y if c < cols - 1 else y1

                    tile1 = vol1[zs:ze, :, ys:ye]
                    tile2 = vol2[zs:ze, :, ys:ye]

                    if tile1.size == 0 or tile2.size == 0:
                        continue

                    _, _, tile_corr = self._normalised_correlation_3D(
                        tile1,
                        tile2,
                        binary_threshold=self.corr_binary_threshold,
                    )

                    corr_values += tile_corr
                    peak_idx = int(np.argmax(tile_corr))
                    tile_vote_map[r, c] = shifts[peak_idx]
                    tile_peak_map[r, c] = tile_corr[peak_idx]
                    valid_tile_count += 1

        else:
            if z1 != z2 or x1 != x2:
                raise ValueError("For y stitching, z and x dimensions must match")

            z_start = min(self.ignore_top, z1)
            z_usable = z1 - z_start
            tile_z = z_usable // rows
            tile_x = x1 // cols

            if tile_z <= 0 or tile_x <= 0:
                raise ValueError(f"Grid {grid} is too fine for volume shape {vol1.shape}")

            for r in range(rows):
                for c in range(cols):
                    zs = z_start + r * tile_z
                    ze = z_start + (r + 1) * tile_z if r < rows - 1 else z1

                    xs = c * tile_x
                    xe = (c + 1) * tile_x if c < cols - 1 else x1

                    tile1 = vol1[zs:ze, xs:xe, :]
                    tile2 = vol2[zs:ze, xs:xe, :]

                    if tile1.size == 0 or tile2.size == 0:
                        continue

                    _, _, tile_corr = self._normalised_correlation_3D(
                        tile1,
                        tile2,
                        binary_threshold=self.corr_binary_threshold,
                    )

                    corr_values += tile_corr
                    peak_idx = int(np.argmax(tile_corr))
                    tile_vote_map[r, c] = shifts[peak_idx]
                    tile_peak_map[r, c] = tile_corr[peak_idx]
                    valid_tile_count += 1

        if valid_tile_count == 0:
            raise ValueError("No valid tiles were found")

        best_index = int(np.argmax(corr_values))
        best_shift = int(shifts[best_index])

        diagnostics = {
            "grid": grid,
            "tile_vote_map": tile_vote_map,
            "tile_peak_map": tile_peak_map,
            "valid_tile_count": valid_tile_count,
            "best_shift": best_shift,
            "adaptive_grid": self.adaptive_grid,
            "grid_info": grid_info,
            "corr_binary_threshold": self.corr_binary_threshold,
            "grid_binary_threshold": self.grid_binary_threshold,
        }

        return best_shift, shifts, corr_values, diagnostics

    # ================================
    # Canvas placement
    # ================================
    def _stitch_volumes(
        self,
        vol1: np.ndarray,
        vol2: np.ndarray,
        shift: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Place two volumes onto canvases according to the computed shift.
        """
        z1, x1, y1 = vol1.shape
        z2, x2, y2 = vol2.shape

        if z1 != z2:
            raise ValueError(f"z dimensions must match, got {z1} and {z2}")

        if self.axis == "x":
            if y1 != y2:
                raise ValueError(f"y dimensions must match for x stitching, got {y1} and {y2}")

            left_offset = max(0, -shift)
            right_extent = max(x1, x2 + shift)
            total_x = left_offset + right_extent

            canvas1 = np.zeros((z1, total_x, y1), dtype=vol1.dtype)
            canvas2 = np.zeros((z2, total_x, y2), dtype=vol2.dtype)

            canvas1[:, left_offset:left_offset + x1, :] = vol1
            x2_start = left_offset + shift
            canvas2[:, x2_start:x2_start + x2, :] = vol2

        else:
            if x1 != x2:
                raise ValueError(f"x dimensions must match for y stitching, got {x1} and {x2}")

            left_offset = max(0, -shift)
            right_extent = max(y1, y2 + shift)
            total_y = left_offset + right_extent

            canvas1 = np.zeros((z1, x1, total_y), dtype=vol1.dtype)
            canvas2 = np.zeros((z2, x2, total_y), dtype=vol2.dtype)

            canvas1[:, :, left_offset:left_offset + y1] = vol1
            y2_start = left_offset + shift
            canvas2[:, :, y2_start:y2_start + y2] = vol2

        return canvas1, canvas2


# %%
if __name__ == "__main__":
    IN_DIR = Path.cwd().parent.parent / "PROCESSING" / "Rotation NPYs"

    file_1 = "position_4_fused_mean.npy"
    file_2 = "position_3_fused_mean.npy"

    stitcher = VariableStitcher(
        axis="x",
        max_shift=200,
        grid=(4, 4),
        adaptive_grid=True,
        grid_binary_threshold=0.9,
        corr_binary_threshold=0.9,
        ignore_top=0,
        min_voxels=50,
        tile_multiple=(1.5, 1.5),
        min_grid=(4, 4),
        max_grid=(45, 45),
        opening_structure=np.ones((3, 3, 3), dtype=bool),
        size_statistic="median",
    )

    result = stitcher.stitch_from_files(
        IN_DIR / file_1,
        IN_DIR / file_2,
    )

    print(f"Volume 1: {file_1}")
    print(f"Volume 2: {file_2}")
    stitcher.print_summary(result)

    stitcher.plot_correlation(result)
    stitcher.plot_vote_map(result)

    if result["diagnostics"]["grid_info"] is not None:
        stitcher.plot_binary_projections(result)

    stitcher.view_in_napari(result)
# %%
