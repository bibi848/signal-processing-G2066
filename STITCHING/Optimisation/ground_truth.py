def voxel_size(meta):
    return meta["scan_grid"]["cube_side_m"] / meta["tfm"]["n_pixels"]

def true_shift_vox(meta):
    vx = voxel_size(meta)
    return meta["scan_grid"]["step_x_m"] / vx

def half_wavelength_vox(meta, wavespeed):
    f = meta["array"]["frequency_Hz"]
    vx = voxel_size(meta)

    wavelength = wavespeed / f
    return 0.5 * wavelength / vx