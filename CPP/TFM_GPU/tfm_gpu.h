#pragma once

void tfm1D_GPU(
    const double* time_data, const double* time, const int* tx, const int* rx,
    const double* xc, const double* zc, const double* X, const double* Z,
    int Nf, int Nt, int Nx, int Nz, int Nelem, double c, int threads, double* img
);

void tfm2D_GPU(
    const double* time_data, const double* time, const int* tx, const int* rx,
    const double* xc, const double* yc, const double* zc,
    const double* X, const double* Y, const double* Z,
    int Nf, int Nt, int Nx, int Ny, int Nz, int Nelem, double c, int threads, double* img
);