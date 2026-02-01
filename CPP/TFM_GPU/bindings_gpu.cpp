#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "tfm_gpu.h"

namespace py = pybind11;

py::array_t<double> tfm1D_GPU_bind(
    py::array_t<double, py::array::c_style | py::array::forcecast> time_data,
    py::array_t<double, py::array::c_style | py::array::forcecast> time,
    py::array_t<int,    py::array::c_style | py::array::forcecast> tx,
    py::array_t<int,    py::array::c_style | py::array::forcecast> rx,
    py::array_t<double, py::array::c_style | py::array::forcecast> xc,
    py::array_t<double, py::array::c_style | py::array::forcecast> zc,
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> Z,
    double c,
    int threads
)
{
    auto td  = time_data.request();
    auto t   = time.request();
    auto txb = tx.request();
    auto rxb = rx.request();
    auto xcb = xc.request();
    auto zcb = zc.request();
    auto Xb  = X.request();
    auto Zb  = Z.request();

    int Nf    = td.shape[0];
    int Nt    = td.shape[1];
    int Nz    = Xb.shape[0];
    int Nx    = Xb.shape[1];
    int Nelem = xcb.shape[0];

    py::array_t<double> img({Nz, Nx});
    auto imgb = img.request();

    tfm1D_GPU(
        static_cast<double*>(td.ptr),
        static_cast<double*>(t.ptr),
        static_cast<int*>(txb.ptr),
        static_cast<int*>(rxb.ptr),
        static_cast<double*>(xcb.ptr),
        static_cast<double*>(zcb.ptr),
        static_cast<double*>(Xb.ptr),
        static_cast<double*>(Zb.ptr),
        Nf, Nt, Nx, Nz, Nelem, c, threads,
        static_cast<double*>(imgb.ptr)
    );

    return img;
}

py::array_t<double> tfm2D_GPU_bind(
    py::array_t<double, py::array::c_style | py::array::forcecast> time_data,
    py::array_t<double, py::array::c_style | py::array::forcecast> time,
    py::array_t<int,    py::array::c_style | py::array::forcecast> tx,
    py::array_t<int,    py::array::c_style | py::array::forcecast> rx,
    py::array_t<double, py::array::c_style | py::array::forcecast> xc,
    py::array_t<double, py::array::c_style | py::array::forcecast> yc,
    py::array_t<double, py::array::c_style | py::array::forcecast> zc,
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> Y,
    py::array_t<double, py::array::c_style | py::array::forcecast> Z,
    double c,
    int threads
)
{
    auto td  = time_data.request();
    auto t   = time.request();
    auto txb = tx.request();
    auto rxb = rx.request();
    auto xcb = xc.request();
    auto ycb = yc.request();
    auto zcb = zc.request();
    auto Xb  = X.request();
    auto Yb  = Y.request();
    auto Zb  = Z.request();

    int Nf    = td.shape[0];
    int Nt    = td.shape[1];
    int Nz    = Xb.shape[0];
    int Ny    = Xb.shape[1];
    int Nx    = Xb.shape[2];
    int Nelem = xcb.shape[0];

    py::array_t<double> img({Nz, Ny, Nx});
    auto imgb = img.request();

    tfm2D_GPU(
        static_cast<double*>(td.ptr),
        static_cast<double*>(t.ptr),
        static_cast<int*>(txb.ptr),
        static_cast<int*>(rxb.ptr),
        static_cast<double*>(xcb.ptr),
        static_cast<double*>(ycb.ptr),
        static_cast<double*>(zcb.ptr),
        static_cast<double*>(Xb.ptr),
        static_cast<double*>(Yb.ptr),
        static_cast<double*>(Zb.ptr),
        Nf, Nt, Nx, Ny, Nz, Nelem, c, threads,
        static_cast<double*>(imgb.ptr)
    );

    return img;
}

PYBIND11_MODULE(tfm_gpu, m)
{
    m.def(
        "tfm1D_GPU",
        &tfm1D_GPU_bind,
        "1D TFM GPU"
    );
    m.def(
        "tfm2D_GPU",
        &tfm2D_GPU_bind,
        "2D TFM GPU"
    );
}