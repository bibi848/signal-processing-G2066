#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tfm_ultra.h"

namespace py = pybind11;

void tfm1D_prepare_GPU_bind(
    py::array_t<double, py::array::c_style | py::array::forcecast> time,
    py::array_t<int,    py::array::c_style | py::array::forcecast> tx,
    py::array_t<int,    py::array::c_style | py::array::forcecast> rx,
    py::array_t<double, py::array::c_style | py::array::forcecast> xc,
    py::array_t<double, py::array::c_style | py::array::forcecast> zc,
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> Z,
    double c,
    int    max_batch,
    int    threads
) {
    auto tb  = time.request();
    auto txb = tx.request();
    auto rxb = rx.request();
    auto xcb = xc.request();
    auto zcb = zc.request();
    auto Xb  = X.request();
    auto Zb  = Z.request();

    if (Xb.ndim != 2 || Zb.ndim != 2) {
        throw std::runtime_error("X and Z must be 2-D arrays shaped (Nz, Nx)");
    }
    if (tb.ndim != 1 || txb.ndim != 1 || rxb.ndim != 1 || xcb.ndim != 1 || zcb.ndim != 1) {
        throw std::runtime_error("time, tx, rx, xc, zc must all be 1-D arrays");
    }

    const int Nt    = static_cast<int>(tb.shape[0]);
    const int Nf    = static_cast<int>(txb.shape[0]);
    const int Nelem = static_cast<int>(xcb.shape[0]);
    const int Nz    = static_cast<int>(Xb.shape[0]);
    const int Nx    = static_cast<int>(Xb.shape[1]);

    tfm1D_init_geometry_GPU(
        static_cast<const double*>(tb.ptr),
        static_cast<const int*>(txb.ptr),
        static_cast<const int*>(rxb.ptr),
        static_cast<const double*>(xcb.ptr),
        static_cast<const double*>(zcb.ptr),
        static_cast<const double*>(Xb.ptr),
        static_cast<const double*>(Zb.ptr),
        Nf, Nt, Nx, Nz, Nelem,
        max_batch, threads, c
    );
}

py::array_t<double> tfm1D_batch_GPU_bind(
    py::array_t<double, py::array::c_style | py::array::forcecast> time_data_batch
) {
    auto tdb = time_data_batch.request();
    if (tdb.ndim != 3) {
        throw std::runtime_error("time_data_batch must be a contiguous 3-D array shaped (N_images, Nf, Nt)");
    }

    const int N_images = static_cast<int>(tdb.shape[0]);
    const int Nz = tfm1D_current_Nz();
    const int Nx = tfm1D_current_Nx();
    if (Nz <= 0 || Nx <= 0) {
        throw std::runtime_error("GPU geometry not prepared. Call prepare_tfm1D_GPU first.");
    }

    py::array_t<double> out({N_images, Nz, Nx});
    auto outb = out.request();

    tfm1D_batch_GPU_stacked(
        static_cast<const double*>(tdb.ptr),
        N_images,
        static_cast<double*>(outb.ptr)
    );

    return out;
}

PYBIND11_MODULE(tfm_ultra, m) {
    m.doc() = "Persistent batched GPU TFM";

    m.def(
        "prepare_tfm1D_GPU",
        &tfm1D_prepare_GPU_bind,
        py::arg("time"),
        py::arg("tx"),
        py::arg("rx"),
        py::arg("xc"),
        py::arg("zc"),
        py::arg("X"),
        py::arg("Z"),
        py::arg("c"),
        py::arg("max_batch"),
        py::arg("threads") = 256,
        R"doc(
Initialise and cache shared TFM geometry on the GPU.
Call this once per geometry/configuration change.
        )doc"
    );

    m.def(
        "tfm1D_batch_GPU",
        &tfm1D_batch_GPU_bind,
        py::arg("time_data_batch"),
        R"doc(
Run a stacked batch of time-domain data on the cached GPU geometry.

time_data_batch must be a contiguous array shaped (N_images, Nf, Nt).
Returns an array shaped (N_images, Nz, Nx).
        )doc"
    );

    m.def(
        "clear_gpu_cache",
        &clear_tfm1D_GPU_cache,
        R"doc(
Free persistent device buffers and streams used by tfm_ultra.
        )doc"
    );
}
