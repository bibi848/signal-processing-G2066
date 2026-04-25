// Author: OD

#pragma once
#include <cstddef>

// One-time geometry initialisation.
void tfm1D_init_geometry_GPU(
    const double* time,
    const int*    tx,
    const int*    rx,
    const double* xc,
    const double* zc,
    const double* X,
    const double* Z,
    int           Nf,
    int           Nt,
    int           Nx,
    int           Nz,
    int           Nelem,
    int           max_batch,
    int           threads,
    double        c
);

void tfm1D_batch_GPU_stacked(
    const double* time_data_stacked,
    int           N_images,
    double*       img_stacked
);

// Frees all persistent host buffers
void clear_tfm1D_GPU_cache();

int tfm1D_current_Nx();
int tfm1D_current_Nz();
