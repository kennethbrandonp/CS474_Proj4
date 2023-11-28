import math
import matplotlib.pyplot as plt
import numpy as np

def fft(data, nn, isign): #Cooley-Tukey modified
    n = nn
    data = np.array(data, dtype = complex) #For imaginary numbers
    
    if isign == -1:  #Forward FFT
        if n > 1:
            evenPoints = fft(data[0::2], n // 2, isign) #Recursive split of data points for even
            oddPoints = fft(data[1::2], n // 2, isign)  #and odd data points

            w = np.exp(-2j * np.pi * np.arange(n) / n) # e^(-2pij * k/N)
            half = n // 2   #Truncate decimals

            return np.concatenate([evenPoints + w[:half] * oddPoints, evenPoints + w[half:] * oddPoints]) 
        else:
            return data
    
    elif isign == 1:  #Inverse FFT
        data = np.conj(data)        #Perform Complex conjugate 
        result = fft(data, nn, -1)  #Recall FFT

        result.imag = np.round(result.imag, 10)
        return result

def fft2D(n, m, real, imag, isign):
    #Combine to put into our 1D DFT:
    data = real + 1j * imag

    #FFT on rows:
    for i in range(n):
        data[i, :] = fft(data[i, :], m, isign)

    #FFT on columns:
    for j in range(m):
        data[:, j] = fft(data[:, j], n, isign)
    
    if isign == -1:
        data /= n * m   #Normlization

    real = data.real
    imag = data.imag

    return real, imag    

###For Inverse 2D FFT, use following to round to ints and flip to original orientation:###
    # #Round our real numbers to the nearest positive integer:
    # inverseReal = np.clip(np.ceil(inverseReal).astype(int), 0, 255)
    # #Flip it back to original orientation:
    # inverseReal = inverseReal[::-1, ::1]

def plotDFT(data, title, xlabel, ylabel, filename):
    graph = plt.figure()    #Different graph every time
    plt.title(title)
    plt.plot(np.arange(data.size), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename + ".png")

def magShift(data):
    n = len(data)
    half = n // 2
    return np.concatenate((data[half:], data[:half]))

def visDFT(pgm, centered):
    #Center spectrum:
    if centered == 1:
        for i in range(0, pgm.y):
            for k in range(0, pgm.x):
                pgm.pixels[i][k] = pgm.pixels[i][k] * math.pow(-1, i + k)

    #Establish real and imaginary components:
    real = np.array(pgm.pixels)
    imag = np.zeros_like(real)

    #Forward FFT:
    freal, fimag = fft2D(pgm.x, pgm.y, real, imag, -1)

    #Calculate magnitude and get everything in grayscale:
    for i in range(0, pgm.y):
        for k in range(0, pgm.x):
            index = i * pgm.x + k
            m = freal[i][k] * freal[i][k] + fimag[i][k] * fimag[i][k]
            pgm.pixels[i][k] = clamp(0, 255, 120 * math.log2(1 + math.sqrt(m)))

    return pgm.pixels   #We can specify what the name is saved outside the function
    #Save centered image:
    # if centered == 1:
    #     pgm.save("_centered_dft")
    # else:
    #     pgm.save("_dft")

def clamp(a, b, x):
    if x > b:
        return b
    if x < a:
        return a
    return x