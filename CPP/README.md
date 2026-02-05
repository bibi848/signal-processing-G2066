# CPP

This folder contains the C++ source and build configuration for the Total Focusing Method (TFM) image-reconstruction kernels. Two backends are provided: a CPU version parallelised with OpenMP, and a GPU version that targets AMD hardware via ROCm/HIP. Both expose their functionality to Python through pybind11.

---

## Folder Structure

```
CPP/
├── CMakeLists.txt          # Top-level: discovers and adds the two sub-projects
├── TFM/                    # CPU/OpenMP backend
│   ├── CMakeLists.txt      # Builds the tfm_cpp pybind11 module
│   ├── tfm.h               # Public declarations (tfm1D, tfm2D)
│   ├── tfm.cpp             # OpenMP-parallelised kernel implementations
│   └── bindings.cpp        # pybind11 glue: numpy to C++ array conversion
└── TFM_GPU/                # AMD GPU backend (ROCm / HIP)
    ├── CMakeLists.txt      # Builds the tfm_gpu pybind11 module via hipcc
    ├── tfm_gpu.h           # Public declarations (tfm1D_GPU, tfm2D_GPU)
    ├── tfm_gpu.cpp         # HIP kernel + host-side memory management
    └── bindings_gpu.cpp    # pybind11 glue: numpy to C++ array conversion
```

---

## How the CMake Build Works

### Top-level `CPP/CMakeLists.txt`

This file simply pulls in both sub-projects:

```cmake
add_subdirectory(TFM)
add_subdirectory(TFM_GPU)
```

### `TFM/CMakeLists.txt`: CPU Module

Straightforward pybind11 module build. `tfm.cpp` and `bindings.cpp` are compiled together with the standard compiler, and `OpenMP::OpenMP_CXX` is linked so that the `#pragma omp parallel for` directives in the kernels are honoured at runtime.

### `TFM_GPU/CMakeLists.txt`: GPU Module

Because `tfm_gpu.cpp` contains HIP kernel code (the `__global__` functions), it **cannot** be compiled by a regular C++ compiler. The CMakeLists here works around this in three steps:

1. **Locate ROCm.** It checks, in order: the `ROCM_PATH` environment variable, the first entry of `CMAKE_PREFIX_PATH`, and finally the default `/opt/rocm`. A status message confirms which path was chosen.

2. **Compile the kernel with `hipcc`.** An `add_custom_command` invokes `hipcc -fPIC -c` on `tfm_gpu.cpp`, producing a standalone object file (`tfm_gpu_kernel.o`). This is the only file that needs the HIP toolchain.

3. **Link everything together.** `pybind11_add_module` compiles `bindings_gpu.cpp` with the normal compiler, then links it against the pre-built kernel object and `libamdhip64.so` (the AMD HIP runtime).

---

## ROCm / HIP

HIP is AMD's CUDA-like programming model. The key pieces in this project are:

* **`__global__` kernels** (`tfm1D_kernel`, `tfm2D_kernel`) — these run on the GPU. Each thread handles one pixel of the output image, iterating over every transmit/receive firing and accumulating the delay-and-sum with linear interpolation.
* **Host wrapper functions** (`tfm1D_GPU`, `tfm2D_GPU`) — these run on the CPU. They allocate device memory with `hipMalloc`, copy the input arrays to the GPU with `hipMemcpy`, launch the kernel with `hipLaunchKernelGGL`, copy the result back, and free every allocation.

The `threads` parameter (passed through from Python) controls the HIP block size. The number of blocks is calculated automatically as `ceil(Np / threads)`, where `Np` is the total number of image pixels.

---

## CPU vs GPU — Letting the Caller Choose

The two backends are built as **separate Python modules** (`tfm_cpp` and `tfm_gpu`). Neither one knows about the other at compile time or at runtime. The decision of which module to call is made in the Python layer, as shown in `Imaging.py` and `Imaging3D.py`:

```python
if engine == 'cpp':
    import platform
    if platform.system() == 'Windows':
        build_dir = os.path.join(os.path.dirname(__file__), "build", "CPP", "TFM", "Debug")
    else:
        build_dir = os.path.join(os.path.dirname(__file__), "build", "CPP", "TFM")
    sys.path.insert(0, build_dir)
    import tfm_cpp
    print('CPP Setup Successful')
    print()


elif engine == 'gpu':
    build_dir = os.path.join(os.path.dirname(__file__), "build", "CPP", "TFM_GPU")
    sys.path.insert(0, build_dir)
    import tfm_gpu
    print('GPU Setup Successful')
    print()

# TFM Computation
if engine == 'cpp':
  img = tfm_cpp.tfm2D(time_data, time_sec, tx0, rx0, xc, yc, zc, X, Y, Z, c)

elif engine == 'gpu':
  img = tfm_gpu.tfm2D_GPU(time_data, time_sec, tx0, rx0, xc, yc, zc, X, Y, Z, c, threads)

```

This means:

* A machine zith a compatible AMD GPU and ROCm installed will have both modules available; the Python code can prefer the GPU path for speed.
* A machine zithout ROCm (e.g. Windows, Mac) will simply fail to import `tfm_gpu`, and the CPU/OpenMP fallback should be used.
* No code changes are needed when moving between machines — only the availability of the compiled module matters.

---

## Function Signatures (Python-Facing)

### CPU (`tfm_cpp`)

```python
tfm_cpp.tfm1D(time_data, time, tx, rx, xc, zc, X, Z, c)            -> np.ndarray  # shape (Nz, Nx)
tfm_cpp.tfm2D(time_data, time, tx, rx, xc, yc, zc, X, Y, Z, c)     -> np.ndarray  # shape (Nz, Ny, Nx)
```

### GPU (`tfm_gpu`)

```python
tfm_gpu.tfm1D_GPU(time_data, time, tx, rx, xc, zc, X, Z, c, threads)           -> np.ndarray  # shape (Nz, Nx)
tfm_gpu.tfm2D_GPU(time_data, time, tx, rx, xc, yc, zc, X, Y, Z, c, threads)    -> np.ndarray  # shape (Nz, Ny, Nx)
```

The GPU variants take an extra `threads` argument that sets the HIP block size. All other arguments and return shapes are identical between the two backends.

### Argument Reference

| Argument | Type | Description |
|---|---|---|
| `time_data` | `(Nf, Nt)` float64 | Raw A-scan traces, one row per firing |
| `time` | `(Nt,)` float64 | Uniformly-spaced time vector for the traces |
| `tx` | `(Nf,)` int32 | Transmitter element index for each firing |
| `rx` | `(Nf,)` int32 | Receiver element index for each firing |
| `xc` | `(Nelem,)` float64 | Element x-coordinates |
| `yc` | `(Nelem,)` float64 | Element y-coordinates *(2D only)* |
| `zc` | `(Nelem,)` float64 | Element z-coordinates |
| `X, Y, Z` | image-grid arrays | Flattened meshgrid coordinates of every pixel |
| `c` | float | Wave speed in the medium |
| `threads` | int | HIP block size *(GPU only)* |

---

## Prerequisites

| Backend | Requirement |
|---|---|
| CPU | A C++17-capable compiler, OpenMP support, pybind11 |
| GPU | All of the above, plus an AMD GPU with ROCm installed (`hipcc` on `PATH` or `ROCM_PATH` set) |

Both modules are built from the same top-level `cmake` invocation. If ROCm is not present the GPU sub-project will fail at configure time; the CPU module will still build successfully.
