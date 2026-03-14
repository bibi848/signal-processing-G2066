import numpy as np

def normalised_correlation_3D(vol1, vol2, axis='x', max_shift=100):
    '''
    Compute normalized cross-correlation between two volumes

    Parameters
    vol1, vol2: Volumes with shape (z, x, y)
    axis:       Axis to shift along: 'x' or 'y'
    max_shift:  Maximum shift to test in both directions

    Returns
    best_shift:   Shift that gives maximum correlation
    shifts:       All tested shifts
    corr_values:  Correlation values for each shift
    '''

    z1, x1, y1 = vol1.shape
    z2, x2, y2 = vol2.shape

    shifts = range(-max_shift, max_shift + 1)
    corr_values = []

    for d in shifts:

        if axis == 'x':
            a1_start = max(0, d)
            a1_end   = min(x1, x2 + d)

            a2_start = max(0, -d)
            a2_end   = min(x2, x1 - d)

            if (a1_end - a1_start) <= 0:
                corr_values.append(0)
                continue

            region1 = vol1[:, a1_start:a1_end, :]
            region2 = vol2[:, a2_start:a2_end, :]

        elif axis == 'y':
            a1_start = max(0, d)
            a1_end   = min(y1, y2 + d)

            a2_start = max(0, -d)
            a2_end   = min(y2, y1 - d)

            if (a1_end - a1_start) <= 0:
                corr_values.append(0)
                continue

            region1 = vol1[:, :, a1_start:a1_end]
            region2 = vol2[:, :, a2_start:a2_end]

        numerator = np.sum(region1 * region2)
        denom = np.sqrt(np.sum(region1**2) * np.sum(region2**2))

        corr_values.append(numerator / denom if denom > 0 else 0)

    corr_values = np.array(corr_values)

    best_index = np.argmax(corr_values)
    best_shift = shifts[best_index]

    return best_shift, shifts, corr_values

def stitch_volumes(vol1, vol2, shift, axis='x'):

    z1, x1, y1 = vol1.shape
    z2, x2, y2 = vol2.shape

    if axis == 'x':

        left_offset = max(0, -shift)
        right_extent = max(x1, x2 + shift)
        total_x = left_offset + right_extent

        canvas1 = np.zeros((z1, total_x, y1))
        canvas2 = np.zeros((z1, total_x, y1))

        canvas1[:, left_offset:left_offset + x1, :] = vol1

        x2_start = left_offset + shift
        canvas2[:, x2_start:x2_start + x2, :] = vol2

    elif axis == 'y':

        left_offset = max(0, -shift)
        right_extent = max(y1, y2 + shift)
        total_y = left_offset + right_extent

        canvas1 = np.zeros((z1, x1, total_y))
        canvas2 = np.zeros((z1, x1, total_y))

        canvas1[:, :, left_offset:left_offset + y1] = vol1

        y2_start = left_offset + shift
        canvas2[:, :, y2_start:y2_start + y2] = vol2

    return canvas1, canvas2