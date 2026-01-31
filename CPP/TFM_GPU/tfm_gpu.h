#pragma once

void tfm1D_GPU(
    const double* time_data, const double* time, const int* tx, const int* rx,
    const double* xc, const double* zc, const double* X, const double* Z,
    int Nf, int Nt, int Nx, int Nz, double c, double* img
);