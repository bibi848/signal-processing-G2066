%% Load data (HDF5 format — real and imag stored separately)
mat_path = 'output/recon_3d_clean/result_tfm_for_matlab_v73.mat';
result_tfm = h5read(mat_path, '/result_tfm_real') + ...
             1i * h5read(mat_path, '/result_tfm_imag');
theta = double(h5read(mat_path, '/theta')');   % row vector, degrees
x     = double(h5read(mat_path, '/x')');       % lateral coords (m)
z     = double(h5read(mat_path, '/z')');       % depth coords (m)

% HDF5 stores in column-major (same as MATLAB), but h5read transposes
% relative to Python's row-major. Permute back to (Nz, Nx, N_theta).
result_tfm = permute(result_tfm, [3, 2, 1]);

Nz      = size(result_tfm, 1);
Nx      = size(result_tfm, 2);
N_theta = size(result_tfm, 3);
fprintf('Loaded: Nz=%d, Nx=%d, N_theta=%d\n', Nz, Nx, N_theta);
fprintf('theta: [%.1f, %.1f] deg (%d angles)\n', theta(1), theta(end), N_theta);
fprintf('x: [%.2f, %.2f] mm\n', x(1)*1e3, x(end)*1e3);
fprintf('z: [%.2f, %.2f] mm\n', z(1)*1e3, z(end)*1e3);

assert(N_theta == length(theta), ...
    'Mismatch: result_tfm has %d angles but theta has %d', N_theta, length(theta));

%% Reconstruction parameters
interp_iradon = 'pchip';
filter_iradon = 'Shepp-Logan';

%% Inverse Radon per depth slice
for iz = 1:Nz
    tmp = squeeze(result_tfm(iz, :, :));  % (Nx, N_theta)
    tmp_xy = iradon(real(tmp), theta, interp_iradon, filter_iradon) + ...
             1i * iradon(imag(tmp), theta, interp_iradon, filter_iradon);

    if iz == 1
        Ny = size(tmp_xy, 1);
        half_width = (x(end) - x(1)) / 2;
        xp = linspace(-half_width, half_width, Ny);
        yp = xp;
        result_tfm_3D = complex(zeros(Ny, Ny, Nz));
    end
    result_tfm_3D(:, :, iz) = tmp_xy;
end

%% Coordinate vectors and visualisation
x3d = xp; y3d = yp; z3d = z;
V = abs(result_tfm_3D);
fprintf('Reconstructed volume: %d x %d x %d\n', size(V));
volumeViewer(V);
