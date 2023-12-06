from pgm import PGM
import numpy as np
import fft
import math
import matplotlib.pyplot as plt


# Create filter and plot
def plot_filter_shape(tempImg, fig_name, D0, gammaL, gammaH, c):
    # Compute D_uv, the distance from the center
    width = tempImg.x
    height = tempImg.y
    u, v = np.meshgrid(np.arange(-width // 2, width // 2), np.arange(-height // 2, height // 2))
    # u = u - tempImg.x // 2
    # v = v - tempImg.y // 2
    D_uv = np.sqrt(u**2 + v**2)
    # Compute H(u,v) based on the provided formula
    H_uv = (gammaH - gammaL) * (1- np.exp(-c * ((D_uv**2) / (2 * (D0**2))))) + gammaL

    # Plot the filter shape
    plt.figure(figsize=(8, 6))
    plt.plot(D_uv.flatten(), H_uv.flatten(), 'b-', label='H(u,v)')
    plt.title('High-Pass Filter Shape')
    plt.xlabel('D(u,v)')
    plt.ylabel('H(u,v)')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_name, format='pdf')

    return H_uv

# Create Filter
# Can modify to pass in img.x, img.y as params
def homomorphic_high_pass_filter(gammaL, gammaH, D0, c):
    # Step 1: Generate frequency domain coordinates
    rows, cols = 512, 512 
    u, v = np.meshgrid(np.fft.fftfreq(rows), np.fft.fftfreq(cols))

    # Step 2: Compute distance from the center
    D_uv = np.sqrt(u**2 + v**2)
    
    # Step 3: Define homomorphic filter function
    H_uv = (gammaH - gammaL) * (1- np.exp(-c * ((D_uv**2) / ((D0**2))))) + gammaL
    return H_uv

# Example usage
img = PGM('girl')

# Bebis starting params for filter
D0 = 1.8
gammaL = 0.5
gammaH = 1.5
c = 1.0

# Create Homomorphic Filter
H_uv = plot_filter_shape(img, 'filter_fig', gammaL, gammaH, D0, c)

# Transform image pixels to ln space
img.log_transform()

# FWD 2DFFT
real = np.array(img.pixels)
imag = np.zeros_like(real)
freal, fimag = fft.fft2D(img.x, img.y, real, imag, -1)  

# H(u,v) * F(u,v)
filt_freal = freal * H_uv
filt_fimag = fimag * H_uv

# INV 2D FFT
ireal, iimag = fft.fft2D(img.x, img.y, filt_freal, filt_fimag, 1)

# CLEAN DATA
ireal = np.clip(np.ceil(ireal).astype(int), 0, 255) #Round everything
ireal = ireal[::-1, ::1]
img.pixels = ireal

# Bring back from ln space
img.exp_transform()

# Store filtered image
img.save('homo_filt_stock_gH_1.5_gL0.5')


