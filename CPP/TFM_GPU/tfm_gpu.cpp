#include "tfm_gpu.h"
#include <hip/hip_runtime.h>
#include <cstdio>

__global__
void tfm1D_kernel(
    const double* time_data,
    const double* time,
    const int* tx,
    const int* rx,
    const double* xc,
    const double* zc,
    const double* X,
    const double* Z,
    int Nf,
    int Nt,
    int Np,
    double inv_c,
    double* img
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= Np) return;

    const double dt = time[1] - time[0];
    const double t0 = time[0];

    double acc = 0.0;

    const double xp = X[p];
    const double zp = Z[p];

    for (int f = 0; f < Nf; ++f) {

        const int txi = tx[f];
        const int rxi = rx[f];

        const double dx_t = xp - xc[txi];
        const double dz_t = zp - zc[txi];
        const double dx_r = xp - xc[rxi];
        const double dz_r = zp - zc[rxi];

        const double dtot = (sqrt(dx_t * dx_t + dz_t * dz_t) + sqrt(dx_r * dx_r + dz_r * dz_r)) * inv_c;

        const double idx = (dtot - t0) / dt;
        const int i0 = (int)floor(idx);

        if (i0 < 0 || i0 >= Nt - 1) continue;

        const double w = idx - i0;
        const double* trace = &time_data[f * Nt];

        acc += (1.0 - w) * trace[i0] + w * trace[i0 + 1];
    }

    img[p] = acc;
}

void tfm1D_GPU(
    const double* time_data,
    const double* time,
    const int* tx,
    const int* rx,
    const double* xc,
    const double* zc,
    const double* X,
    const double* Z,
    int Nf,
    int Nt,
    int Nx,
    int Nz,
    int Nelem,
    double c,
    const int threads,
    double* img
) {
    const int Np = Nx * Nz;
    double *d_time_data, *d_time, *d_xc, *d_zc, *d_X, *d_Z, *d_img;
    int *d_tx, *d_rx;
    double inv_c = 1.0 / c;

    hipMalloc(&d_time_data, sizeof(double) * Nf * Nt);
    hipMalloc(&d_time,      sizeof(double) * Nt);
    hipMalloc(&d_tx,        sizeof(int)    * Nf);
    hipMalloc(&d_rx,        sizeof(int)    * Nf);
    hipMalloc(&d_xc,        sizeof(double) * Nelem);
    hipMalloc(&d_zc,        sizeof(double) * Nelem);
    hipMalloc(&d_X,         sizeof(double) * Np);
    hipMalloc(&d_Z,         sizeof(double) * Np);
    hipMalloc(&d_img,       sizeof(double) * Np);

    hipMemcpy(d_time_data, time_data, sizeof(double) * Nf * Nt, hipMemcpyHostToDevice);
    hipMemcpy(d_time,      time,      sizeof(double) * Nt,      hipMemcpyHostToDevice);
    hipMemcpy(d_tx,        tx,        sizeof(int)    * Nf,      hipMemcpyHostToDevice);
    hipMemcpy(d_rx,        rx,        sizeof(int)    * Nf,      hipMemcpyHostToDevice);
    hipMemcpy(d_xc,        xc,        sizeof(double) * Nelem,   hipMemcpyHostToDevice);
    hipMemcpy(d_zc,        zc,        sizeof(double) * Nelem,   hipMemcpyHostToDevice);
    hipMemcpy(d_X,         X,         sizeof(double) * Np,      hipMemcpyHostToDevice);
    hipMemcpy(d_Z,         Z,         sizeof(double) * Np,      hipMemcpyHostToDevice);

    const int blocks = (Np + threads - 1) / threads;

    hipLaunchKernelGGL(
        tfm1D_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        0,
        d_time_data,
        d_time,
        d_tx,
        d_rx,
        d_xc,
        d_zc,
        d_X,
        d_Z,
        Nf,
        Nt,
        Np,
        inv_c,
        d_img
    );

    hipMemcpy(img, d_img, sizeof(double) * Np, hipMemcpyDeviceToHost);

    hipFree(d_time_data);
    hipFree(d_time);
    hipFree(d_tx);
    hipFree(d_rx);
    hipFree(d_xc);
    hipFree(d_zc);
    hipFree(d_X);
    hipFree(d_Z);
    hipFree(d_img);
}

__global__
void tfm2D_kernel(
    const double* time_data,
    const double* time,
    const int* tx,
    const int* rx,
    const double* xc,
    const double* yc,
    const double* zc,
    const double* X,
    const double* Y,
    const double* Z,
    int Nf,
    int Nt,
    int Np,
    double inv_c,
    double* img
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= Np) return;

    const double dt = time[1] - time[0];
    const double t0 = time[0];

    double acc = 0.0;

    const double xp = X[p];
    const double yp = Y[p];
    const double zp = Z[p];

    for (int f = 0; f < Nf; ++f) {

        const int txi = tx[f];
        const int rxi = rx[f];

        const double dx_t = xp - xc[txi];
        const double dy_t = yp - yc[txi];
        const double dz_t = zp - zc[txi];

        const double dx_r = xp - xc[rxi];
        const double dy_r = yp - yc[rxi];
        const double dz_r = zp - zc[rxi];

        const double dtot = (sqrt(dx_t * dx_t + dy_t * dy_t + dz_t * dz_t) + sqrt(dx_r * dx_r + dy_r * dy_r + dz_r * dz_r)) * inv_c;

        const double idx = (dtot - t0) / dt;
        const int i0 = (int)floor(idx);

        if (i0 < 0 || i0 >= Nt - 1) continue;

        const double w = idx - i0;
        const double* trace = &time_data[f * Nt];

        acc += (1.0 - w) * trace[i0] + w * trace[i0 + 1];
    }

    img[p] = acc;
}

void tfm2D_GPU(
    const double* time_data,
    const double* time,
    const int* tx,
    const int* rx,
    const double* xc,
    const double* yc,
    const double* zc,
    const double* X,
    const double* Y,
    const double* Z,
    int Nf,
    int Nt,
    int Nx,
    int Ny,
    int Nz,
    int Nelem,
    double c,
    const int threads,
    double* img
) {
    const int Np = Nx * Ny * Nz;
    double *d_time_data, *d_time, *d_xc, *d_yc, *d_zc, *d_X, *d_Y, *d_Z, *d_img;
    int *d_tx, *d_rx;
    double inv_c = 1.0 / c;

    hipMalloc(&d_time_data, sizeof(double) * Nf * Nt);
    hipMalloc(&d_time,      sizeof(double) * Nt);
    hipMalloc(&d_tx,        sizeof(int)    * Nf);
    hipMalloc(&d_rx,        sizeof(int)    * Nf);
    hipMalloc(&d_xc,        sizeof(double) * Nelem);
    hipMalloc(&d_yc,        sizeof(double) * Nelem);
    hipMalloc(&d_zc,        sizeof(double) * Nelem);
    hipMalloc(&d_X,         sizeof(double) * Np);
    hipMalloc(&d_Y,         sizeof(double) * Np);
    hipMalloc(&d_Z,         sizeof(double) * Np);
    hipMalloc(&d_img,       sizeof(double) * Np);

    hipMemcpy(d_time_data, time_data, sizeof(double) * Nf * Nt, hipMemcpyHostToDevice);
    hipMemcpy(d_time,      time,      sizeof(double) * Nt,      hipMemcpyHostToDevice);
    hipMemcpy(d_tx,        tx,        sizeof(int)    * Nf,      hipMemcpyHostToDevice);
    hipMemcpy(d_rx,        rx,        sizeof(int)    * Nf,      hipMemcpyHostToDevice);
    hipMemcpy(d_xc,        xc,        sizeof(double) * Nelem,   hipMemcpyHostToDevice);
    hipMemcpy(d_yc,        yc,        sizeof(double) * Nelem,   hipMemcpyHostToDevice);
    hipMemcpy(d_zc,        zc,        sizeof(double) * Nelem,   hipMemcpyHostToDevice);
    hipMemcpy(d_X,         X,         sizeof(double) * Np,      hipMemcpyHostToDevice);
    hipMemcpy(d_Y,         Y,         sizeof(double) * Np,      hipMemcpyHostToDevice);
    hipMemcpy(d_Z,         Z,         sizeof(double) * Np,      hipMemcpyHostToDevice);

    const int blocks = (Np + threads - 1) / threads;

    hipLaunchKernelGGL(
        tfm2D_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        0,
        d_time_data,
        d_time,
        d_tx,
        d_rx,
        d_xc,
        d_yc,
        d_zc,
        d_X,
        d_Y,
        d_Z,
        Nf,
        Nt,
        Np,
        inv_c,
        d_img
    );

    hipMemcpy(img, d_img, sizeof(double) * Np, hipMemcpyDeviceToHost);

    hipFree(d_time_data);
    hipFree(d_time);
    hipFree(d_tx);
    hipFree(d_rx);
    hipFree(d_xc);
    hipFree(d_yc);
    hipFree(d_zc);
    hipFree(d_X);
    hipFree(d_Y);
    hipFree(d_Z);
    hipFree(d_img);
}