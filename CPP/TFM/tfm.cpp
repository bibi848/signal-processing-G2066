#include "tfm.h"
#include <cmath>
#include <omp.h>

void tfm1D(
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
    double c,
    double* img
) {
    const double dt = time[1] - time[0];
    const double t0 = time[0];
    const int Np = Nx * Nz;

    #pragma omp parallel for schedule(static)
    for (int p = 0; p < Np; ++p) {

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

            const double dtot = (std::sqrt(dx_t * dx_t + dz_t * dz_t) + std::sqrt(dx_r * dx_r + dz_r * dz_r)) / c;

            const double idx = (dtot - t0) / dt;
            const int i0 = static_cast<int>(idx);

            if (i0 < 0 || i0 >= Nt - 1) continue;

            const double w = idx - i0;
            const double* trace = &time_data[f * Nt];

            acc += (1.0 - w) * trace[i0] + w * trace[i0 + 1];
        }

        img[p] = acc;
    }
}

void tfm2D(
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
    double c,
    double* img
) {
    const double dt = time[1] - time[0];
    const double t0 = time[0];
    const int Np = Nx * Ny * Nz;

    #pragma omp parallel for schedule(static)
    for (int p = 0; p < Np; ++p) {

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

            const double dtot = (std::sqrt(dx_t * dx_t + dy_t * dy_t + dz_t * dz_t) + std::sqrt(dx_r * dx_r + dy_r * dy_r + dz_r * dz_r)) / c;

            const double idx = (dtot - t0) / dt;
            const int i0 = static_cast<int>(idx);

            if (i0 < 0 || i0 >= Nt - 1) continue;

            const double w = idx - i0;
            const double* trace = &time_data[f * Nt];

            acc += (1.0 - w) * trace[i0] + w * trace[i0 + 1];
        }

        img[p] = acc;
    }
}
