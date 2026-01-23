import numpy as np
import h5py
from pathlib import Path


def mat_to_csv(mat_path: str, dataset_name: str, csv_path: str, transpose_if_needed: bool = True):
    with h5py.File(mat_path, "r") as f:
        obj = f[dataset_name]
        if isinstance(obj, h5py.Group):
            raise ValueError(f"{dataset_name} is a group. Available datasets: {list(obj.keys())}")
        arr = obj[()]
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if transpose_if_needed and arr.ndim == 2:
        arr = arr.T

    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    np.savetxt(csv_path, arr, delimiter=",")
    print(f"Saved {dataset_name} with shape {arr.shape} -> {csv_path}")

MAT_ROOT = Path(__file__).resolve().parents[1]
mat_file = MAT_ROOT / "1D Raw Data" / "Initial Aluminium Data" / "exp_data_Al_gain35.mat"
csv_file = MAT_ROOT / "DATA PROCESSING" / "exp_data_Al_gain35.csv"

with h5py.File(mat_file, "r") as f:
    print("Top-level keys:", list(f.keys()))
    exp_data = f['exp_data']
    print(f"exp_data type: {type(exp_data)}")
    
    if isinstance(exp_data, h5py.Group):
        print(f"exp_data is a group with keys: {list(exp_data.keys())}")
        # Explore nested structure
        exp_data.visititems(lambda name, obj: print(f"  {name}: {type(obj)} {getattr(obj, 'shape', '')}"))
    elif isinstance(exp_data, h5py.Dataset):
        print(f"exp_data shape: {exp_data.shape}, dtype: {exp_data.dtype}")


mat_to_csv(str(mat_file), "exp_data/time_data", str(csv_file))