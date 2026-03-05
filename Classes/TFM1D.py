import numpy as np
from scipy.signal import hilbert

def TFM1D(time_data, time, tx, rx, xc, zc, c, x_img, z_img):
    """
    time_data : (N_fmc, N_t)
    time      : (N_t,)
    tx, rx    : (N_fmc,) 1-based indices
    xc, zc    : (N_el,)
    c         : wave speed (m/s)
    x_img     : (Nx,)
    z_img     : (Nz,)
    """

    tx0 = tx - 1
    rx0 = rx - 1
    Nf, Nt = time_data.shape
    dt = time[1] - time[0]
    t0 = time[0]

    X, Z = np.meshgrid(x_img, z_img)
    img  = np.zeros_like(X)

    for i in range(Nf):
        # Distances for TX/RX pair 
        d_tx = np.sqrt((X - xc[tx0[i]])**2 + (Z - zc[tx0[i]])**2)
        d_rx = np.sqrt((X - xc[rx0[i]])**2 + (Z - zc[rx0[i]])**2)

        idx_f      = ((d_tx + d_rx) / c - t0) / dt
        i0         = np.floor(idx_f).astype(int)
        w          = idx_f - i0
        valid      = (i0 >= 0) & (i0 < Nt - 1)
        i0_clipped = np.clip(i0, 0, Nt - 2)

        s0 = time_data[i, i0_clipped]
        s1 = time_data[i, i0_clipped + 1]

        img += valid * ((1.0 - w) * s0 + w * s1)

    return img

def CTFM1D(time_data, time, tx, rx, xc, zc, c, x_img, z_img,
           output_db=True):

    tx0 = tx - 1
    rx0 = rx - 1
    Nf, Nt = time_data.shape
    dt = time[1] - time[0]
    t0 = time[0]
    eps = 1e-10

    X, Z = np.meshgrid(x_img, z_img)
    img  = np.zeros_like(X)         

    for i in range(Nf):
        # Distances for TX/RX pair
        d_tx = np.sqrt((X - xc[tx0[i]])**2 + (Z - zc[tx0[i]])**2)
        d_rx = np.sqrt((X - xc[rx0[i]])**2 + (Z - zc[rx0[i]])**2)

        # Sample index
        idx_f  = ((d_tx + d_rx) / c - t0) / dt
        i0     = np.floor(idx_f).astype(int)
        w      = idx_f - i0

        valid      = (i0 >= 0) & (i0 < Nt - 1)
        i0_clipped = np.clip(i0, 0, Nt - 2)

        s0 = time_data[i, i0_clipped]
        s1 = time_data[i, i0_clipped + 1]

        img += valid * ((1.0 - w) * s0 + w * s1)

    # Envelope
    img_analytic = hilbert(img, axis=0)
    img_envelope = np.abs(img_analytic)

    if output_db:
        img_db = 20 * np.log10(img_envelope / (img_envelope.max() + eps) + eps)
        return img_db
    else:
        return img_envelope


def TFM_angular1D(time_data, time, tx, rx, xc, zc, c,
                 x_img, z_img, half_angle_deg, min_els,
                 output_db=True):
    tx0 = tx - 1
    rx0 = rx - 1
    Nf, Nt = time_data.shape
    dt = time[1] - time[0]
    t0 = time[0]
    eps = 1e-10

    X, Z = np.meshgrid(x_img, z_img)
    Nz, Nx = X.shape

    # Angular mask
    theta_rad = np.deg2rad(half_angle_deg)
    x_spread  = Z * np.tan(theta_rad)
    min_x, max_x = xc.min(), xc.max()

    illumination    = np.abs(xc[np.newaxis, np.newaxis, :] - X[:, :, None]) <= x_spread[:, :, None]
    num_illuminated = illumination.sum(axis=2)
    cone_fits       = (X - x_spread >= min_x) & (X + x_spread <= max_x)
    mask            = (num_illuminated >= min_els)

    valid_idx = np.flatnonzero(mask)
    x_v = X.ravel()[valid_idx]
    z_v = Z.ravel()[valid_idx]

    # Accumulate over FMC
    pixel_sums = np.zeros(len(valid_idx))

    for i in range(Nf):
        # Distances for TX/RX pair
        d_tx = np.sqrt((x_v - xc[tx0[i]])**2 + (z_v - zc[tx0[i]])**2)
        d_rx = np.sqrt((x_v - xc[rx0[i]])**2 + (z_v - zc[rx0[i]])**2)

        idx_f      = ((d_tx + d_rx) / c - t0) / dt
        i0         = np.floor(idx_f).astype(int)
        w          = idx_f - i0
        in_bounds  = (i0 >= 0) & (i0 < Nt - 1)
        i0_clipped = np.clip(i0, 0, Nt - 2)

        s0 = time_data[i, i0_clipped]
        s1 = time_data[i, i0_clipped + 1]

        pixel_sums += in_bounds * ((1.0 - w) * s0 + w * s1)

    # Scatter back to true image
    img_flat          = np.zeros(Nz * Nx)
    img_flat[valid_idx] = pixel_sums
    img               = img_flat.reshape(Nz, Nx)

    img_analytic = hilbert(img, axis=0)
    img_envelope = np.abs(img_analytic)

    if output_db:
        img_db = 20 * np.log10(img_envelope / (img_envelope.max() + eps) + eps)
        return img_db
    else:
        return img_envelope