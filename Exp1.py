from pgm import PGM
import numpy as np
import fft

def bandRejectFilter(pgm, centerx, centery, radius):
    data = np.array(pgm.pixels)

    #Apply forward:
    real, imag = fft.fft2D(pgm.x, pgm.y, data, np.zeros_like(data), -1)

    #Create a filter mask:
    mask = np.ones_like(real)
    y, x = np.ogrid[:pgm.y, :pgm.x] #Makes a circle around the origin
    mask[((x - centerx)**2 + (y - centery)**2) < radius**2] = 0  #Set any values within band to zero

    #Apply this filter to the DFT of the image (Complex multiplication)
    filteredReal = real * mask
    filteredImag = imag * mask

    #Apply inverse to get original image filtered back:
    inverseReal, inverseImag = fft.fft2D(pgm.x, pgm.y, filteredReal, filteredImag, 1)

    #Round to the nearest positive integer:
    inverseReal = np.clip(np.ceil(inverseReal).astype(int), 0, 255)

    #Flip it back to the original orientation
    inverseReal = inverseReal[::-1, ::1]

    return inverseReal

def partA(pgm):
    #Get the original spectrum so we can apply filters:
    tempPGM = pgm
    # originalSpectrum = fft.visDFT(tempPGM, 1) #To visualize the spectrum
    # originalSpectrum = np.clip(np.ceil(originalSpectrum).astype(int), 0, 255)
    # tempPGM.pixels = originalSpectrum
    
    # tempPGM.save("originalSpectrum")#For visualizing the spectrum

    #Band-Reject Filter:
    tempPGM.pixels = bandRejectFilter(pgm, 32, 16, 1)  #Adjusted band(Still needs tuning)
    bandReject = fft.visDFT(tempPGM, 1) #Visualize band reject filter
    bandReject = np.clip(np.ceil(bandReject).astype(int), 0, 255)
    tempPGM.pixels = bandReject

    tempPGM.save("bandRejectFilter")
    
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