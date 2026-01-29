#include "tfm.h"
#include <cmath>

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
    double dt = time[1] - time[0];
    double t0 = time[0];

    for (int i = 0; i < Nx * Nz; ++i){
        img[i] = 0.0;
    }
    
    for (int f = 0; f < Nf; ++f) {
        int txi = tx[f];
        int rxi = rx[f];

        double xt = xc[txi];
        double zt = zc[txi];
        double xr = xc[rxi];
        double zr = zc[rxi];

        for (int iz = 0; iz < Nz; ++iz){
            for (int ix = 0; ix < Nx; ++ix){
                int p = iz * Nx + ix;

                double dx_t = X[p] - xt;
                double dz_t = Z[p] - zt;
                double dx_r = X[p] - xr;
                double dz_r = Z[p] - zr;

                double ttot = (std::sqrt(dx_t * dx_t + dz_t * dz_t) + (std::sqrt(dx_r * dx_r + dz_r * dz_r))) / c;
                double idx = (ttot - t0) / dt;
                int i0 = static_cast<int>(std::floor(idx));

                if (i0 < 0 || i0 >= Nt - 1){
                    continue;
                }

                double w = idx - i0;
                double s0 = time_data[f * Nt + i0];
                double s1 = time_data[f * Nt + i0 + 1];

                img[p] += (1.0 - w) * s0 + w * s1;
            }
        }
    }
}