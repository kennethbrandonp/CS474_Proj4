from pgm import PGM
import numpy as np
import fft
import math

def sampleBandReject(u, v, w, radius, centerX, centerY, n):
    dd = (u - centerX) * (u - centerX) + (v - centerY) * (v - centerY)
    if dd == radius * radius:
        return 0.0  #Avoids division by zero in the case it happens
    denom = 1 + (((math.sqrt(dd) * w) / (dd - (radius * radius))) ** (2 * n))

    return 1 / denom

def filterDataBand(pgm):    #For visualizing the band filter
    for i in range(0, pgm.x):
        for k in range(0, pgm.y):
            # Use the band reject filter function:
            filter = sampleBandReject(k - pgm.x // 2, pgm.y // 2 - i, 4, 36, 0, 0, 1) #Parameters are adjusted according to where the noise is
            # Update the output image pixels:
            pgm.pixels[i][k] = filter

    return pgm.pixels

def applyBandFilter(pgm):   #Complex mutliplication is not working correctly, images are jumbled up
    #Establish real and imaginary components:
    real = np.array(pgm.pixels)
    imag = np.zeros_like(real)

    #Forward FFT:
    freal, fimag = fft.fft2D(pgm.x, pgm.y, real, imag, -1)

    for i in range(0, pgm.x):
        for k in range(0, pgm.y):
            s = np.zeros(2, dtype=complex)
            # s[0] = samp_bw_notch_reject(x, y, 3, 32, 16, 1)
            # s[0] *= samp_bw_notch_reject(x, y, 3, -32, 16, 1)
            s[0] = sampleBandReject(k - pgm.x // 2, pgm.y // 2 - i, 4, 36, 0, 0, 1)
            s[1] = 0    #Complex portion

            #Perform complex multiplication
            filter = np.multiply([freal[i][k], fimag[i][k]], s)

            #Update the real and imaginary parts of pgm.pixels
            freal[i][k] = filter[0]
            fimag[i][k] = filter[1]

    #Inverse FFT to get our image back:
    ireal, iimag = fft.fft2D(pgm.x, pgm.y, freal, fimag, 1)

    return ireal

def partA(pgm):
    #Get the original spectrum so we can apply filters: *Done
    # tempPGM = pgm
    # originalSpectrum = fft.visualizeDFT(tempPGM, 1) #To visualize the spectrum
    # originalSpectrum = np.clip(np.ceil(originalSpectrum).astype(int), 0, 255)
    # tempPGM.pixels = originalSpectrum
    
    #tempPGM.save("originalSpectrum")#For visualizing the spectrum

    #Band-Reject Filter:    ***In progress
    tempPGM = pgm
    bandFilteredImage = applyBandFilter(tempPGM)
    bandFilteredImage = np.clip(np.ceil(bandFilteredImage).astype(int), 0, 255)
    bandFilteredImage = bandFilteredImage[::-1, ::1]
    tempPGM.pixels = bandFilteredImage
    
    tempPGM.save("bandRejectFilter")

    #Visualizing:   *Done
    # tempPGM = pgm
    # bandFilter = filterDataBand(tempPGM)
    # bandFilter = fft.visualizeFilter(bandFilter)
    # bandFilter = np.clip(np.ceil(bandFilter).astype(int), 0, 255)
    # tempPGM.pixels = bandFilter
    # tempPGM.save("filter") #Visualize our filter
    
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