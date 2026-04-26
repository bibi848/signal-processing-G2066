# CPP

This folder contains the C++ source and build configuration for the TFM image reconstruction kernels. Three backends are provided: 
* **CPU with OpenMP**: CPU multithreaded option which works on Mac, Windows and Linux.
* **GPU with ROCm/HIP**: Single image GPU acceleration targeting AMD hardware via ROCm/HIP.
* **Batched GPU with ROCm/HIP**: Batched GPU implementation which aims to reduce kernel launch and memory transfer overhead. 
All expose their functionality to Python through pybind11.

---

## Folder Structure

```
CPP/
├── CMakeLists.txt          # Discovers the sub-projects
├── TFM/                    # CPU/OpenMP backend
│   ├── CMakeLists.txt      
│   ├── tfm.h               
│   ├── tfm.cpp             
│   └── bindings.cpp        
├── TFM_GPU/                # Standard GPU backend (sequential)
|   ├── CMakeLists.txt      
|   ├── tfm_gpu.h           
|   ├── tfm_gpu.cpp         
|   └── bindings_gpu.cpp    
└── TFM_U/                  # Batched GPU backend
    ├── CMakeLists.txt      
    ├── tfm_ultra.h           
    ├── tfm_ultra.cpp         
    └── bindings_ultra.cpp
```

---

## How the CMake Build Works

### Top-level `CPP/CMakeLists.txt`

This file pulls in all sub-projects:

```cmake
add_subdirectory(TFM)
add_subdirectory(TFM_GPU)
add_subdirectory(TFM_U)
```

## ROCm / HIP

HIP is AMD's CUDA-like programming model. The key pieces in this project are:

* **`__global__` kernels** (`tfm1D_kernel`, `tfm2D_kernel`) which run on the GPU. Each thread handles one pixel of the output image, iterating over every transmit/receive firing and accumulating the delay-and-sum with linear interpolation.
* **Host wrapper functions** (`tfm1D_GPU`, `tfm2D_GPU`) which run on the CPU. They allocate device memory with `hipMalloc`, copy the input arrays to the GPU with `hipMemcpy`, launch the kernel with `hipLaunchKernelGGL`, copy the result back, and free every allocation.

The `threads` parameter controls the HIP block size. The number of blocks is calculated automatically as `ceil(Np / threads)`, where `Np` is the total number of image pixels.

---

## CPU vs GPU

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

elif engine == 'gpu':
    build_dir = os.path.join(os.path.dirname(__file__), "build", "CPP", "TFM_GPU")
    sys.path.insert(0, build_dir)
    import tfm_gpu

elif engine == 'gpu_u':
    build_dir = os.path.join(os.path.dirname(__file__), "build", "CPP", "TFM_U")
    sys.path.insert(0, build_dir)
    import tfm_ultra


# TFM Computation
if engine == 'cpp':
    img = tfm_cpp.tfm2D(time_data, time_sec, tx0, rx0, xc, yc, zc, X, Y, Z, c)

elif engine == 'gpu':
    img = tfm_gpu.tfm2D_GPU(time_data, time_sec, tx0, rx0, xc, yc, zc, X, Y, Z, c, threads)

elif engine == 'gpu_u':
    # Shared geometry is prepared once and cached on the GPU.
    tfm_ultra.prepare_tfm1D_GPU(time_sec, tx0, rx0, xc, zc, X, Z, c, batch_size, threads)

    imgs = tfm_ultra.tfm1D_batch_GPU(time_data_batch)

    # Clear cached GPU memory once processing is finished
    tfm_ultra.clear_gpu_cache()
```
This means:

* A machine with a compatible AMD GPU and ROCm installed will have all modules available; the Python code can prefer the GPU path for speed.
* A machine without ROCm (e.g. Windows or Mac) will simply fail to import `tfm_gpu`, and the CPU/OpenMP fallback should be used.
* No code changes are needed when moving between machines as only the availability of the compiled module matters.
* The GPU variants take an extra `threads` argument that sets the HIP block size. All other arguments and return shapes are identical between the backends.
* The `batch_size` variable corresponds to the number of images that are being processed in parallel.

### Argument Reference

| Argument | Type | Description |
|---|---|---|
| `time_data` | `(Nf, Nt)` float64 | Raw A-scan traces, one row per firing |
| `time` | `(Nt,)` float64 | Uniformly spaced time vector for the traces |
| `tx` | `(Nf,)` int32 | Transmitter element index for each firing |
| `rx` | `(Nf,)` int32 | Receiver element index for each firing |
| `xc` | `(Nelem,)` float64 | Element x-coordinates |
| `yc` | `(Nelem,)` float64 | Element y-coordinates |
| `zc` | `(Nelem,)` float64 | Element z-coordinates |
| `X, Y, Z` | image-grid arrays | Flattened meshgrid coordinates of every pixel |
| `c` | float | Speed of Sound |
| `threads` | int | HIP block size |
| `batch_size` | int | Number of images processed per batch |

---
