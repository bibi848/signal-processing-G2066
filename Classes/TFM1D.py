import numpy as np

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