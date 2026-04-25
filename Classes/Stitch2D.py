'''
Author: OD
'''

import numpy as np

def normalised_correlation_2D(img1, img2):

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    max_shift = w1

    corr_values = []

    shifts = range(-max_shift, max_shift + 1)

    for dx in shifts:
        x1_start = max(0, dx)
        x1_end   = min(w1, w2 + dx)

        x2_start = max(0, -dx)
        x2_end   = min(w2, w1 - dx)

        if (x1_end - x1_start) <= 0:
            corr_values.append(0)
            continue

        region1 = img1[:, x1_start:x1_end]
        region2 = img2[:, x2_start:x2_end]

        # Normalised Correlation
        numerator = np.sum(region1 * region2)
        denom = np.sqrt(np.sum(region1**2) * np.sum(region2**2))

        if denom > 0:
            corr_values.append(numerator / denom)
        else:
            corr_values.append(0)

    corr_values = np.array(corr_values)

    best_index = np.argmax(corr_values)
    best_dx = shifts[best_index]

    return best_dx, shifts, corr_values

def stitch_images(img1, img2, dx, colour_bool=True):

    h1, w1 = img1.shape
    h2, w2 = img2.shape

    left_offset = max(0, -dx)
    right_extent = max(w1, w2 + dx)

    total_width = left_offset + right_extent
    height = h1

    canvas1 = np.zeros((height, total_width))
    canvas2 = np.zeros((height, total_width))

    canvas1[:, left_offset:left_offset + w1] = img1
    x2 = left_offset + dx
    canvas2[:, x2:x2 + w2] = img2

    if colour_bool:
        stitched = np.zeros((height, total_width, 3))

        stitched[:, :, 0] += canvas1 * 1.0
        stitched[:, :, 2] += canvas1 * 0.8

        stitched[:, :, 0] += canvas2 * 0.3
        stitched[:, :, 1] += canvas2 * 0.85
        stitched[:, :, 2] += canvas2 * 1.0

        overlap = (canvas1 > 0) & (canvas2 > 0)
        stitched[overlap] = [1, 1, 1]

        stitched = np.clip(stitched, 0, 1)

    else:
        stitched = np.maximum(canvas1, canvas2)

    return stitched, left_offset, w1, x2, w2