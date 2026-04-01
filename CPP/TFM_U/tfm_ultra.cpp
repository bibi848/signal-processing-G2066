#include "tfm_ultra.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Helpers

#define HIP_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d  '%s'\n",                     \
                    __FILE__, __LINE__, hipGetErrorString(e));              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)


// Kernel
__global__
void tfm1D_ultra_kernel(
    const double* __restrict__ time_data,   // Nf x Nt
    const double* __restrict__ time,        // Nt
    const int*   __restrict__ tx,           // Nf
    const int*   __restrict__ rx,           // Nf
    const double* __restrict__ xc,          // Nelem
    const double* __restrict__ zc,          // Nelem
    const double* __restrict__ X,           // Np
    const double* __restrict__ Z,           // Np
    int    Nf,
    int    Nt,
    int    Np,
    double inv_c,
    double* __restrict__ img               // Np  (output)
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= Np) return;

    const double dt = time[1] - time[0];
    const double t0 = time[0];

    const double xp = X[p];
    const double zp = Z[p];

    double acc = 0.0;

    for (int f = 0; f < Nf; ++f) {
        const int txi = tx[f];
        const int rxi = rx[f];

        const double dx_t = xp - xc[txi];
        const double dz_t = zp - zc[txi];
        const double dx_r = xp - xc[rxi];
        const double dz_r = zp - zc[rxi];

        const double dtot = (sqrt(dx_t*dx_t + dz_t*dz_t)
                           + sqrt(dx_r*dx_r + dz_r*dz_r)) * inv_c;

        const double idx_d = (dtot - t0) / dt;
        const int    i0    = (int)floor(idx_d);

        if (i0 < 0 || i0 >= Nt - 1) continue;

        const double w = idx_d - i0;
        const double* trace = time_data + (size_t)f * Nt;

        acc += (1.0 - w) * trace[i0] + w * trace[i0 + 1];
    }

    img[p] = acc;
}

// Host batch function

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
) {
    if (N_images <= 0) return;

    const int    Np    = Nx * Nz;
    const double inv_c = 1.0 / c;
    const int    blocks = (Np + threads - 1) / threads;

    // Upload geometry
    double *d_time, *d_xc, *d_zc, *d_X, *d_Z;
    int    *d_tx,   *d_rx;

    HIP_CHECK(hipMalloc(&d_time, sizeof(double) * Nt));
    HIP_CHECK(hipMalloc(&d_tx,   sizeof(int)    * Nf));
    HIP_CHECK(hipMalloc(&d_rx,   sizeof(int)    * Nf));
    HIP_CHECK(hipMalloc(&d_xc,   sizeof(double) * Nelem));
    HIP_CHECK(hipMalloc(&d_zc,   sizeof(double) * Nelem));
    HIP_CHECK(hipMalloc(&d_X,    sizeof(double) * Np));
    HIP_CHECK(hipMalloc(&d_Z,    sizeof(double) * Np));

    HIP_CHECK(hipMemcpy(d_time, time, sizeof(double) * Nt,    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tx,   tx,   sizeof(int)    * Nf,    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rx,   rx,   sizeof(int)    * Nf,    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_xc,   xc,   sizeof(double) * Nelem, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_zc,   zc,   sizeof(double) * Nelem, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_X,    X,    sizeof(double) * Np,    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Z,    Z,    sizeof(double) * Np,    hipMemcpyHostToDevice));

    // Allocate 2 ping-pong device buffers for time_data + img 
    // Buffer [s] is owned by stream [s].
    const int    N_STREAMS      = 2;
    const size_t time_data_bytes = (size_t)Nf * Nt * sizeof(double);
    const size_t img_bytes       = (size_t)Np      * sizeof(double);

    double* d_time_data[N_STREAMS];
    double* d_img[N_STREAMS];
    hipStream_t streams[N_STREAMS];

    for (int s = 0; s < N_STREAMS; ++s) {
        HIP_CHECK(hipMalloc(&d_time_data[s], time_data_bytes));
        HIP_CHECK(hipMalloc(&d_img[s],       img_bytes));
        HIP_CHECK(hipStreamCreate(&streams[s]));
    }

    // Pinned (page-locked) staging buffers on the host
    double* h_pinned_in[N_STREAMS];
    double* h_pinned_out[N_STREAMS];

    for (int s = 0; s < N_STREAMS; ++s) {
        HIP_CHECK(hipHostMalloc(&h_pinned_in[s],  time_data_bytes, hipHostMallocDefault));
        HIP_CHECK(hipHostMalloc(&h_pinned_out[s], img_bytes,       hipHostMallocDefault));
    }

    // Double-buffer pipeline
    // For image i assigned to slot s = i:
    // a) Synchronise stream s (ensures previous work on s is done)
    // b) Copy user time_data[i] > pinned staging buffer
    // c) Async copy pinned > device on stream s
    // d) Launch kernel on stream s
    // e) Async copy device img pinned on stream s
    // After enqueuing all images, drain all streams, then copy pinned results back to the user output buffers.
    
    int slot_image_idx[N_STREAMS] = {-1, -1};

    for (int i = 0; i < N_images; ++i) {
        const int s = i % N_STREAMS;

        // (a) Synchronise this slot
        HIP_CHECK(hipStreamSynchronize(streams[s]));

        // If there was a previous image in this slot collect its result
        if (slot_image_idx[s] >= 0) {
            const int prev = slot_image_idx[s];
            std::memcpy(img_batch[prev], h_pinned_out[s], img_bytes);
        }

        // (b) Stage input in pinned memory
        std::memcpy(h_pinned_in[s], time_data_batch[i], time_data_bytes);

        // (c) Async H2D transfer
        HIP_CHECK(hipMemcpyAsync(
            d_time_data[s], h_pinned_in[s],
            time_data_bytes, hipMemcpyHostToDevice, streams[s]
        ));

        // (d) Launch kernel — all geometry pointers are already on device.
        hipLaunchKernelGGL(
            tfm1D_ultra_kernel,
            dim3(blocks), dim3(threads),
            0, streams[s],
            d_time_data[s], d_time,
            d_tx, d_rx,
            d_xc, d_zc,
            d_X, d_Z,
            Nf, Nt, Np, inv_c,
            d_img[s]
        );

        // (e) Async D2H transfer
        HIP_CHECK(hipMemcpyAsync(
            h_pinned_out[s], d_img[s],
            img_bytes, hipMemcpyDeviceToHost, streams[s]
        ));

        slot_image_idx[s] = i;
    }

    // Drain remaining work
    for (int s = 0; s < N_STREAMS; ++s) {
        HIP_CHECK(hipStreamSynchronize(streams[s]));
        if (slot_image_idx[s] >= 0) {
            std::memcpy(img_batch[slot_image_idx[s]], h_pinned_out[s], img_bytes);
        }
    }

    // Cleanup
    for (int s = 0; s < N_STREAMS; ++s) {
        hipFree(d_time_data[s]);
        hipFree(d_img[s]);
        hipHostFree(h_pinned_in[s]);
        hipHostFree(h_pinned_out[s]);
        hipStreamDestroy(streams[s]);
    }

    hipFree(d_time);
    hipFree(d_tx);
    hipFree(d_rx);
    hipFree(d_xc);
    hipFree(d_zc);
    hipFree(d_X);
    hipFree(d_Z);
}
