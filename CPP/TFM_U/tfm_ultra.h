#pragma once

#include <cstddef>

/*
Processes a batch of N_images TFM datasets on the GPU using a 2-stream
double-buffer pipeline. Geometry arrays (xc, zc, tx, rx, X, Z) are
uploaded once and reused across every image in the batch. Only time_data
and the output img buffers move per image.

Parameters

time_data_batch  : array of N_images pointers, each pointing to a contiguous (Nf x Nt) array
time             : shared time axis, length Nt
tx, rx           : transmitter / receiver element indices, length Nf
xc, zc           : element centre coordinates, length Nelem
X, Z             : flattened image-point coordinate grids, length Np = Nx*Nz
Nf               : number of firings (tx/rx pairs)
Nt               : number of time samples per trace
Nx, Nz           : image grid dimensions
Nelem            : number of transducer elements
c                : longitudinal wave speed (m/s)
threads          : threads per block
N_images         : number of datasets in this batch
img_batch        : output array of N_images pointers, each pointing to a
                   pre-allocated (Np) double buffer for the result
 */
void tfm1D_batch_GPU(
    const double* const* time_data_batch,
    const double*        time,
    const int*           tx,
    const int*           rx,
    const double*        xc,
    const double*        zc,
    const double*        X,
    const double*        Z,
    int                  Nf,
    int                  Nt,
    int                  Nx,
    int                  Nz,
    int                  Nelem,
    double               c,
    int                  threads,
    int                  N_images,
    double**             img_batch
);
