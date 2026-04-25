// Author: OD

#include "tfm_ultra.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(cmd)                                                      
    do {                                                      
        hipError_t e = (cmd);                                 
        if (e != hipSuccess) {                                
            fprintf(stderr, "HIP error %s:%d '%s'\n",         
                    __FILE__, __LINE__, hipGetErrorString(e));
            std::exit(EXIT_FAILURE);                          
        }                                                     
    } while (0)

namespace {

struct TFMUltraContext {
    int Nf = 0;
    int Nt = 0;
    int Nx = 0;
    int Nz = 0;
    int Np = 0;
    int Nelem = 0;
    int max_batch = 0;
    int threads = 256;
    double c = 0.0;
    double inv_c = 0.0;
    double dt = 0.0;
    double t0 = 0.0;

    double *d_time = nullptr, *d_xc = nullptr, *d_zc = nullptr, *d_X = nullptr, *d_Z = nullptr;
    int *d_tx = nullptr, *d_rx = nullptr;

    double *d_td_flat[2]  = {nullptr, nullptr};
    double *d_img_flat[2] = {nullptr, nullptr};
    double *h_td_flat[2]  = {nullptr, nullptr};
    double *h_img_flat[2] = {nullptr, nullptr};

    hipStream_t streams[2] = {nullptr, nullptr};
    hipEvent_t  done[2]    = {nullptr, nullptr};

    bool initialised = false;
};

TFMUltraContext g_ctx;

void free_buffers_only() {
    for (int i = 0; i < 2; ++i) {
        if (g_ctx.done[i])    { hipEventDestroy(g_ctx.done[i]); g_ctx.done[i] = nullptr; }
        if (g_ctx.streams[i]) { hipStreamDestroy(g_ctx.streams[i]); g_ctx.streams[i] = nullptr; }
        if (g_ctx.h_td_flat[i])  { hipHostFree(g_ctx.h_td_flat[i]);  g_ctx.h_td_flat[i] = nullptr; }
        if (g_ctx.h_img_flat[i]) { hipHostFree(g_ctx.h_img_flat[i]); g_ctx.h_img_flat[i] = nullptr; }
        if (g_ctx.d_td_flat[i])  { hipFree(g_ctx.d_td_flat[i]);      g_ctx.d_td_flat[i] = nullptr; }
        if (g_ctx.d_img_flat[i]) { hipFree(g_ctx.d_img_flat[i]);     g_ctx.d_img_flat[i] = nullptr; }
    }
}

void free_all_context() {
    free_buffers_only();

    if (g_ctx.d_time) { hipFree(g_ctx.d_time); g_ctx.d_time = nullptr; }
    if (g_ctx.d_tx)   { hipFree(g_ctx.d_tx);   g_ctx.d_tx   = nullptr; }
    if (g_ctx.d_rx)   { hipFree(g_ctx.d_rx);   g_ctx.d_rx   = nullptr; }
    if (g_ctx.d_xc)   { hipFree(g_ctx.d_xc);   g_ctx.d_xc   = nullptr; }
    if (g_ctx.d_zc)   { hipFree(g_ctx.d_zc);   g_ctx.d_zc   = nullptr; }
    if (g_ctx.d_X)    { hipFree(g_ctx.d_X);    g_ctx.d_X    = nullptr; }
    if (g_ctx.d_Z)    { hipFree(g_ctx.d_Z);    g_ctx.d_Z    = nullptr; }

    g_ctx = TFMUltraContext{};
}

void allocate_reusable_batch_buffers() {
    const size_t td_all  = static_cast<size_t>(g_ctx.max_batch) * g_ctx.Nf * g_ctx.Nt * sizeof(double);
    const size_t img_all = static_cast<size_t>(g_ctx.max_batch) * g_ctx.Np * sizeof(double);

    for (int i = 0; i < 2; ++i) {
        HIP_CHECK(hipHostMalloc(&g_ctx.h_td_flat[i], td_all, hipHostMallocDefault));
        HIP_CHECK(hipHostMalloc(&g_ctx.h_img_flat[i], img_all, hipHostMallocDefault));
        HIP_CHECK(hipMalloc(&g_ctx.d_td_flat[i], td_all));
        HIP_CHECK(hipMalloc(&g_ctx.d_img_flat[i], img_all));
        HIP_CHECK(hipStreamCreate(&g_ctx.streams[i]));
        HIP_CHECK(hipEventCreateWithFlags(&g_ctx.done[i], hipEventDisableTiming));
    }
}

void ensure_context_matches(
    const int Nf,
    const int Nt,
    const int Nx,
    const int Nz,
    const int Nelem,
    const int max_batch,
    const int threads,
    const double c
) {
    const bool same_shape = g_ctx.initialised &&
        g_ctx.Nf == Nf && g_ctx.Nt == Nt &&
        g_ctx.Nx == Nx && g_ctx.Nz == Nz &&
        g_ctx.Nelem == Nelem &&
        g_ctx.max_batch == max_batch &&
        g_ctx.threads == threads &&
        g_ctx.c == c;

    if (same_shape) return;

    free_all_context();

    g_ctx.Nf = Nf;
    g_ctx.Nt = Nt;
    g_ctx.Nx = Nx;
    g_ctx.Nz = Nz;
    g_ctx.Np = Nx * Nz;
    g_ctx.Nelem = Nelem;
    g_ctx.max_batch = max_batch;
    g_ctx.threads = threads;
    g_ctx.c = c;
    g_ctx.inv_c = 1.0 / c;

    HIP_CHECK(hipMalloc(&g_ctx.d_time, sizeof(double) * Nt));
    HIP_CHECK(hipMalloc(&g_ctx.d_tx,   sizeof(int)    * Nf));
    HIP_CHECK(hipMalloc(&g_ctx.d_rx,   sizeof(int)    * Nf));
    HIP_CHECK(hipMalloc(&g_ctx.d_xc,   sizeof(double) * Nelem));
    HIP_CHECK(hipMalloc(&g_ctx.d_zc,   sizeof(double) * Nelem));
    HIP_CHECK(hipMalloc(&g_ctx.d_X,    sizeof(double) * g_ctx.Np));
    HIP_CHECK(hipMalloc(&g_ctx.d_Z,    sizeof(double) * g_ctx.Np));

    allocate_reusable_batch_buffers();
    g_ctx.initialised = true;
}

__global__
void tfm1D_ultra_kernel(
    const double* __restrict__ time_data_flat, // [N_images, Nf, Nt]
    const int*    __restrict__ tx,             // [Nf]
    const int*    __restrict__ rx,             // [Nf]
    const double* __restrict__ xc,             // [Nelem]
    const double* __restrict__ zc,             // [Nelem]
    const double* __restrict__ X,              // [Np]
    const double* __restrict__ Z,              // [Np]
    int    Nf,
    int    Nt,
    int    Np,
    int    N_images,
    double inv_c,
    double dt,
    double t0,
    double* __restrict__ img_flat
) {
    const int p       = blockIdx.x * blockDim.x + threadIdx.x;
    const int img_idx = blockIdx.y;

    if (p >= Np || img_idx >= N_images) return;

    const double xp = X[p];
    const double zp = Z[p];

    const double* my_time_data = time_data_flat + static_cast<size_t>(img_idx) * Nf * Nt;
    double acc = 0.0;

    for (int f = 0; f < Nf; ++f) {
        const int txi = tx[f];
        const int rxi = rx[f];

        const double dx_t = xp - xc[txi];
        const double dz_t = zp - zc[txi];
        const double dx_r = xp - xc[rxi];
        const double dz_r = zp - zc[rxi];

        const double dtot = (sqrt(dx_t * dx_t + dz_t * dz_t)
                           + sqrt(dx_r * dx_r + dz_r * dz_r)) * inv_c;

        const double idx_d = (dtot - t0) / dt;
        const int i0 = static_cast<int>(floor(idx_d));

        if (i0 < 0 || i0 >= Nt - 1) continue;

        const double w = idx_d - i0;
        const double* trace = my_time_data + static_cast<size_t>(f) * Nt;
        acc += (1.0 - w) * trace[i0] + w * trace[i0 + 1];
    }

    img_flat[static_cast<size_t>(img_idx) * Np + p] = acc;
}

void launch_one_chunk_async(const double* stacked_src, int batch_offset, int n_chunk, int slot) {
    const size_t td_one   = static_cast<size_t>(g_ctx.Nf) * g_ctx.Nt;
    const size_t img_one  = static_cast<size_t>(g_ctx.Np);
    const size_t td_chunk = static_cast<size_t>(n_chunk) * td_one;
    const size_t img_chunk = static_cast<size_t>(n_chunk) * img_one;

    std::memcpy(
        g_ctx.h_td_flat[slot],
        stacked_src + static_cast<size_t>(batch_offset) * td_one,
        td_chunk * sizeof(double)
    );

    HIP_CHECK(hipMemcpyAsync(
        g_ctx.d_td_flat[slot],
        g_ctx.h_td_flat[slot],
        td_chunk * sizeof(double),
        hipMemcpyHostToDevice,
        g_ctx.streams[slot]
    ));

    const int pixel_blocks = (g_ctx.Np + g_ctx.threads - 1) / g_ctx.threads;
    const dim3 grid(pixel_blocks, n_chunk);
    const dim3 block(g_ctx.threads);

    hipLaunchKernelGGL(
        tfm1D_ultra_kernel,
        grid, block, 0, g_ctx.streams[slot],
        g_ctx.d_td_flat[slot],
        g_ctx.d_tx, g_ctx.d_rx,
        g_ctx.d_xc, g_ctx.d_zc,
        g_ctx.d_X, g_ctx.d_Z,
        g_ctx.Nf, g_ctx.Nt, g_ctx.Np, n_chunk,
        g_ctx.inv_c, g_ctx.dt, g_ctx.t0,
        g_ctx.d_img_flat[slot]
    );
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpyAsync(
        g_ctx.h_img_flat[slot],
        g_ctx.d_img_flat[slot],
        img_chunk * sizeof(double),
        hipMemcpyDeviceToHost,
        g_ctx.streams[slot]
    ));

    HIP_CHECK(hipEventRecord(g_ctx.done[slot], g_ctx.streams[slot]));
}

void collect_one_chunk(double* img_stacked, int batch_offset, int n_chunk, int slot) {
    const size_t img_one = static_cast<size_t>(g_ctx.Np);
    const size_t img_chunk = static_cast<size_t>(n_chunk) * img_one;

    HIP_CHECK(hipEventSynchronize(g_ctx.done[slot]));

    std::memcpy(
        img_stacked + static_cast<size_t>(batch_offset) * img_one,
        g_ctx.h_img_flat[slot],
        img_chunk * sizeof(double)
    );
}

}

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
) {
    ensure_context_matches(Nf, Nt, Nx, Nz, Nelem, max_batch, threads, c);

    g_ctx.dt = time[1] - time[0];
    g_ctx.t0 = time[0];

    HIP_CHECK(hipMemcpy(g_ctx.d_time, time, sizeof(double) * Nt, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_ctx.d_tx,   tx,   sizeof(int)    * Nf,    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_ctx.d_rx,   rx,   sizeof(int)    * Nf,    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_ctx.d_xc,   xc,   sizeof(double) * Nelem, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_ctx.d_zc,   zc,   sizeof(double) * Nelem, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_ctx.d_X,    X,    sizeof(double) * g_ctx.Np, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_ctx.d_Z,    Z,    sizeof(double) * g_ctx.Np, hipMemcpyHostToDevice));
}

void tfm1D_batch_GPU_stacked(
    const double* time_data_stacked,
    int           N_images,
    double*       img_stacked
) {
    if (N_images <= 0) return;

    int previous_slot = -1;
    int previous_offset = 0;
    int previous_count = 0;

    for (int batch_offset = 0, chunk_idx = 0; batch_offset < N_images; batch_offset += g_ctx.max_batch, ++chunk_idx) {
        const int slot = chunk_idx % 2;
        const int n_chunk = std::min(g_ctx.max_batch, N_images - batch_offset);

        if (previous_slot == slot) {
            collect_one_chunk(img_stacked, previous_offset, previous_count, previous_slot);
            previous_slot = -1;
        }

        launch_one_chunk_async(time_data_stacked, batch_offset, n_chunk, slot);

        if (previous_slot != -1) {
            collect_one_chunk(img_stacked, previous_offset, previous_count, previous_slot);
        }

        previous_slot = slot;
        previous_offset = batch_offset;
        previous_count = n_chunk;
    }

    if (previous_slot != -1) {
        collect_one_chunk(img_stacked, previous_offset, previous_count, previous_slot);
    }
}

void clear_tfm1D_GPU_cache() {
    free_all_context();
}

int tfm1D_current_Nx() { return g_ctx.Nx; }
int tfm1D_current_Nz() { return g_ctx.Nz; }
