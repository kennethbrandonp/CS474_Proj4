from pgm import PGM
import numpy as np
import fft
import math
import matplotlib.pyplot as plt

# Helper function for complex multiplication
def complex_multiply(H_real, H_imag, F_real, F_imag):
    result_real = H_real * F_real - H_imag * F_imag
    result_imag = H_real * F_imag + H_imag * F_real
    return result_real, result_imag

# Create filter and plot, save as jpg
def plot_filter_shape(tempImg, fig_name, D0, gammaL, gammaH, c):
    # Compute D_uv, the distance from the center
    delta_u = 1 / tempImg.x
    delta_v = 1 / tempImg.y
    u_min, u_max, v_min, v_max, _, _ = tempImg.compute_frequency_ranges(delta_u, delta_v)
    u_range = np.linspace(u_min, u_max, num=tempImg.x)
    v_range = np.linspace(v_min, v_max, num=tempImg.y)
    u, v = np.meshgrid(u_range, v_range)
    D_uv = np.sqrt(u**2 + v**2)
    # Compute H(u,v) based on the provided formula
    H_uv = ((gammaH - gammaL) * (1 - np.exp(-c * (D_uv**2 / D0**2)))) + gammaL

    # Plot the filter shape
    plt.figure(figsize=(8, 6))
    plt.plot(D_uv.flatten(), H_uv.flatten(), 'b-', label='H(u,v)')
    plt.title(f'{fig_name}')
    plt.xlabel('D(u,v)')
    plt.ylabel('H(u,v)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{fig_name}.png") 
    plt.close()

    return H_uv

def plot_spectra(freal=0, fimag=0, plot_name="default_name", H_uv=None):
    if H_uv is None:
        magnitude = magnitude = np.sqrt(freal**2 + fimag**2)
        
    else:
        magnitude = (np.abs(H_uv))**2

    # Stretch values for visualize
    magnitude = np.log(1 + magnitude)
    plt.figure(figsize=(8, 6))
    plt.imshow(magnitude, cmap='viridis')
    plt.colorbar(label='Magnitude Spectrum')
    plt.title('Magnitude Spectrum of the Filter')
    plt.savefig(f'{plot_name}.png')
    plt.close()

def homomorphic_filtering(D0, gammaL, gammaH, c, filter_spectra_fname, filter_spatial_fname, img_spectra_fname, output_img_name):
    # Create PGM object for input image
    img = PGM('girl')

    # Log transform img pixels (c = 1)
    img.log_transform(0.75)

    # Perform forward 2D FFT
    real = np.array(img.pixels)
    imag = np.zeros_like(real)

    freal, fimag = fft.fft2D(img.x, img.y, real, imag, -1)

    data = freal + 1j * fimag

    # Generate Filter
    H_uv = plot_filter_shape(img, filter_spatial_fname, D0, gammaL, gammaH, c)
    H_uv_imag = np.zeros_like(H_uv)
    
    plot_spectra(plot_name=filter_spectra_fname, H_uv=H_uv)

    # F(u,v) * H(u,v)
    filt_real = freal * H_uv
    filt_imag = fimag * H_uv

    plot_spectra(plot_name=img_spectra_fname, freal=filt_real, fimag=filt_imag)

    # INV 2d FFT
    ireal, iimag = fft.fft2D(img.x, img.y, filt_real, filt_imag, 1)

    # EXP transform (c = 1)
    ireal = ireal[::-1, ::1]
    img.pixels = ireal
    img.exp_transform(0.75)

    # Export filtered image
    img.save(f'{output_img_name}')
    
    
def exp3():
    # EXP3_1
    # INITIAL FILTER PARAMS BY DR. BEBIS
    homomorphic_filtering(1.8, 0.5, 1.5, 1.0, "f_spectra_initial", "f_spatial_initial", "img_spectra_initial", "exp3_initial_filtered_img")
    
    # EXP3_2
    # Lowering gammaL (0.0) and gammaH(1.0)
    homomorphic_filtering(1.8, 0.0, 1.0, 1.0, "f_spectra_lowered", "f_spatial_lowered", "img_spectra_lowered", "exp3_lowered_output_img")

    # EXP3_3
    # Original Filter Params with modified slope value c
    homomorphic_filtering(1.8, 0.7, 1.3, 1.0, "f_spectra_small_slope", "f_spatial_small_slope ", "img_spectra_small_slope", "exp3_small_slope_output_img")

    # EXP3_4
    homomorphic_filtering(1.8, 0.5, 1.5, 1.0, "f_spectra_lowgL", "f_spatial_low_gL", "img_spectra_low_gL", "low_gl_output_img")

    



exp3()