import numpy as np
from scipy.signal import hilbert

def TFM1D(time_data, time, tx, rx, xc, zc, c, x_img, z_img):
    """
    Vectorised 2D TFM (x,z)

    time_data : (N_fmc, N_t)
    time      : (N_t,)
    tx, rx    : (N_fmc,) 1-based indices
    xc, zc    : (N_el,)
    c         : wave speed (m/s)
    x_img     : (Nx,)
    z_img     : (Nz,)
    """

    # Convert to 0-based indices
    tx0 = tx - 1
    rx0 = rx - 1

    Nf, Nt = time_data.shape

    dt = time[1] - time[0]
    t0 = time[0]

    # Image Grid
    X, Z = np.meshgrid(x_img, z_img)

    # Pre-compute Distances
    d_tx = np.sqrt(
        (X[None, :, :] - xc[tx0][:, None, None])**2 +
        (Z[None, :, :] - zc[tx0][:, None, None])**2
    )

    d_rx = np.sqrt(
        (X[None, :, :] - xc[rx0][:, None, None])**2 +
        (Z[None, :, :] - zc[rx0][:, None, None])**2
    )

    # Travel Time
    t_tot = (d_tx + d_rx) / c
    idx = (t_tot - t0) / dt

    i0 = np.floor(idx).astype(int)
    w = idx - i0

    # Valid Indices
    valid = (i0 >= 0) & (i0 < Nt - 1)

    # Clamp Indices
    i0_clipped = np.clip(i0, 0, Nt - 2)

    # Safe samples
    s0 = np.take_along_axis(
        time_data[:, :, None, None],
        i0_clipped[:, None, :, :],
        axis=1
    ).squeeze(axis=1)

    s1 = np.take_along_axis(
        time_data[:, :, None, None],
        (i0_clipped + 1)[:, None, :, :],
        axis=1
    ).squeeze(axis=1)

    # Validity Mask
    s0 *= valid
    s1 *= valid

    # Linear interpolation
    img = np.sum((1.0 - w) * s0 + w * s1, axis=0)

    return img

def CTFM1D(time_data, time, tx, rx, xc, zc, c, x_img, z_img, 
            output_db=True):
    
    # Convert to 0-based indices
    tx0 = tx - 1
    rx0 = rx - 1

    Nf, Nt = time_data.shape

    dt = time[1] - time[0]
    t0 = time[0]
    eps = 1e-10

    # Image Grid
    X, Z = np.meshgrid(x_img, z_img)

    # Pre-compute Distances
    d_tx = np.sqrt(
        (X[None, :, :] - xc[tx0][:, None, None])**2 +
        (Z[None, :, :] - zc[tx0][:, None, None])**2
    )

    d_rx = np.sqrt(
        (X[None, :, :] - xc[rx0][:, None, None])**2 +
        (Z[None, :, :] - zc[rx0][:, None, None])**2
    )

    # Diffusion attenuation compensation factor
    C_ij = 1.0 / np.sqrt(d_tx * d_rx + eps)

    # Travel Time
    t_tot = (d_tx + d_rx) / c
    idx = (t_tot - t0) / dt

    i0 = np.floor(idx).astype(int)
    w = idx - i0

    # Valid Indices
    valid = (i0 >= 0) & (i0 < Nt - 1)

    # Clamp Indices
    i0_clipped = np.clip(i0, 0, Nt - 2)

    # Safe samples
    s0 = np.take_along_axis(
        time_data[:, :, None, None],
        i0_clipped[:, None, :, :],
        axis=1
    ).squeeze(axis=1)

    s1 = np.take_along_axis(
        time_data[:, :, None, None],
        (i0_clipped + 1)[:, None, :, :],
        axis=1
    ).squeeze(axis=1)

    # Validity Mask
    s0 *= valid
    s1 *= valid

    # Linear interp
    weighted_samples = C_ij * ((1.0 - w) * s0 + w * s1)
    img = np.sum(weighted_samples, axis=0)

    # Hilbert transform
    img_analytic = hilbert(img, axis=0)
    img_envelope = np.abs(img_analytic)
    
    if output_db:
        img_max = np.max(img_envelope)
        img_db = 20 * np.log10(img_envelope / img_max + eps)
        return img_db
    else:
        return img_envelope