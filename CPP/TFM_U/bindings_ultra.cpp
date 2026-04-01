#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "tfm_ultra.h"

namespace py = pybind11;

/*
imgs = tfm_ultra.tfm1D_batch_GPU(
    time_data_list,   # list of 2-D numpy arrays (Nf x Nt), float64
    time,             # 1-D numpy array (Nt,),              float64
    tx,               # 1-D numpy array (Nf,),              int32
    rx,               # 1-D numpy array (Nf,),              int32
    xc,               # 1-D numpy array (Nelem,),           float64
    zc,               # 1-D numpy array (Nelem,),           float64
    X,                # 2-D numpy array (Nz x Nx),          float64
    Z,                # 2-D numpy array (Nz x Nx),          float64
    c,                # float  — speed of sound m/s
    threads,          # int    — GPU threads per block
) return list[np.ndarray]  # N_images arrays each shaped (Nz, Nx)
 */
py::list tfm1D_batch_GPU_bind(
    py::list                                                                 time_data_list,
    py::array_t<double, py::array::c_style | py::array::forcecast>           time,
    py::array_t<int,    py::array::c_style | py::array::forcecast>           tx,
    py::array_t<int,    py::array::c_style | py::array::forcecast>           rx,
    py::array_t<double, py::array::c_style | py::array::forcecast>           xc,
    py::array_t<double, py::array::c_style | py::array::forcecast>           zc,
    py::array_t<double, py::array::c_style | py::array::forcecast>           X,
    py::array_t<double, py::array::c_style | py::array::forcecast>           Z,
    double c,
    int    threads
) {
    auto tb  = time.request();
    auto txb = tx.request();
    auto rxb = rx.request();
    auto xcb = xc.request();
    auto zcb = zc.request();
    auto Xb  = X.request();
    auto Zb  = Z.request();

    const int Nt    = static_cast<int>(tb.shape[0]);
    const int Nf    = static_cast<int>(txb.shape[0]);
    const int Nelem = static_cast<int>(xcb.shape[0]);
    const int Nz    = static_cast<int>(Xb.shape[0]);
    const int Nx    = static_cast<int>(Xb.shape[1]);

    const int N_images = static_cast<int>(py::len(time_data_list));

    std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> td_arrays;
    td_arrays.reserve(N_images);

    std::vector<const double*> td_ptrs(N_images);

    for (int i = 0; i < N_images; ++i) {
        td_arrays.push_back(
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                time_data_list[i]
            )
        );
        td_ptrs[i] = static_cast<const double*>(td_arrays.back().request().ptr);
    }

    py::list result;
    std::vector<py::array_t<double>> out_arrays;
    out_arrays.reserve(N_images);
    std::vector<double*> out_ptrs(N_images);

    for (int i = 0; i < N_images; ++i) {
        py::array_t<double> arr({Nz, Nx});
        out_ptrs[i] = static_cast<double*>(arr.request().ptr);
        out_arrays.push_back(std::move(arr));
    }

    tfm1D_batch_GPU(
        td_ptrs.data(),
        static_cast<const double*>(tb.ptr),
        static_cast<const int*>(txb.ptr),
        static_cast<const int*>(rxb.ptr),
        static_cast<const double*>(xcb.ptr),
        static_cast<const double*>(zcb.ptr),
        static_cast<const double*>(Xb.ptr),
        static_cast<const double*>(Zb.ptr),
        Nf, Nt, Nx, Nz, Nelem,
        c, threads, N_images,
        out_ptrs.data()
    );

    for (auto& arr : out_arrays) {
        result.append(arr);
    }
    return result;
}

PYBIND11_MODULE(tfm_ultra, m) {
    m.doc() = "Batched GPU TFM";

    m.def(
        "tfm1D_batch_GPU",
        &tfm1D_batch_GPU_bind,
        py::arg("time_data_list"),
        py::arg("time"),
        py::arg("tx"),
        py::arg("rx"),
        py::arg("xc"),
        py::arg("zc"),
        py::arg("X"),
        py::arg("Z"),
        py::arg("c"),
        py::arg("threads") = 256,
        R"doc(
Batch 1-D TFM on AMD GPU (ROCm/HIP) with a 2-stream double-buffer pipeline.

Geometry (xc, zc, tx, rx, X, Z) is uploaded to VRAM once and shared across
all images. Only time_data transfers per image, maximising GPU utilisation.

Parameters
----------
time_data_list : list of np.ndarray, each shape (Nf, Nt), dtype float64
time           : np.ndarray (Nt,), dtype float64
tx, rx         : np.ndarray (Nf,), dtype int32
xc, zc         : np.ndarray (Nelem,), dtype float64
X, Z           : np.ndarray (Nz, Nx), dtype float64  — from np.meshgrid
c              : float  — speed of sound in m/s
threads        : int    — GPU threads per block (default 256)

Return
-------
list of np.ndarray, each shape (Nz, Nx), dtype float64 (raw TFM values)
        )doc"
    );
}
