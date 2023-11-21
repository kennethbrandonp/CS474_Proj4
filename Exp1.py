from pgm import PGM
import numpy as np
import fft

def partA(pgm):
    data = np.array(pgm.pixels) #Grab pixels from the image

    #Apply forward FFT:
    real, imag = fft.fft2D(pgm.x, pgm.y, data, np.zeros_like(data), -1)

    #Magnitude spectrum:
    magnitude = np.abs(real + 1j * imag)
    magnitude = np.clip(np.ceil(magnitude).astype(int), 0, 255)

    pgm.pixels = fft.magShift(magnitude)
    #Save original magnitude spectrum:
    pgm.save("_original_spectrum")  #Produces result but it's extremely dark, likely need to call upon partB in Exp3.py

    #Band-Reject Filter:
    ##Band Reject filter and what parameters we're using:
    ###Which frequency range we're rejecting
    ###We'll have to tune the size of the band
    ###Filter frequency domain
    ###Get filtered 2D DFT from image
    ###Save filtered image

    #Spatial Domain Gaussian Filtering:
    ##Create Gaussian filter
    ##Experiment with different filter sizes (7x7, 15x15)
    ##Apply filter to original image in spatial domain
    ##Save filtered image

    #Compare the two filtered spectras.(Report)

#Make our image, pass it into partA:
image = PGM("boy_noisy")
partA(image)