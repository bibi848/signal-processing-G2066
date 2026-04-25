'''
Author: OD
'''

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

def CTFM1D(time_data, time, tx, rx, xc, zc, c, x_img, z_img, output='db'):
    '''
    output : 'db':       dB-normalised envelope
             'envelope': linear envelope
             'complex':  complex analytic signal
             'real':     real TFM sum before envelope
    '''
    tx0 = tx - 1
    rx0 = rx - 1
    Nf, Nt = time_data.shape
    dt = time[1] - time[0]
    t0 = time[0]
    eps = 1e-10

    X, Z = np.meshgrid(x_img, z_img)
    img  = np.zeros_like(X)

    for i in range(Nf):
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

    if output == 'real':
        return img

    img_analytic = hilbert(img, axis=0)

    if output == 'complex':
        return img_analytic

    img_envelope = np.abs(img_analytic)

    if output == 'envelope':
        return img_envelope

    return 20 * np.log10(img_envelope / (img_envelope.max() + eps) + eps)