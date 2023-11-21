import numpy as np
import pandas as pd
import random
from statistics import mean
from matplotlib import pyplot as plt

# PGM object constructor
class PGM():
    def __init__(self, filename):
        self.name = filename
        self.magic_number = ""
        self.resolution = ""
        self.x = 0
        self.y = 0
        self.pixels = []
        self.quantization = 0
        self.histogram = []

        # Open file
        file = open((self.name + ".pgm"), "r")

        # Read magic number from line 1 and store as string
        self.magic_number = file.readline()

        # Skip comment on line 2
        file.readline()

        # Read columns x rows from line 3 and store as a string as well as integers in the "x" and "y" variables
        self.resolution = file.readline()
        split_resolution = self.resolution.split()
        self.x = int(split_resolution[0])
        self.y = int(split_resolution[1])

        # Read quantization level from line 4 and store as integer
        self.quantization = int(file.readline())

        # Read body and store pixels as integers in a 2D array
        for x in range(self.x):
            self.pixels.append([])
            for y in range(self.y):
                self.pixels[x].append(int(file.readline().strip('\n')))

        # Create histogram
        # self.pixels_flat = np.array(self.pixels).flatten().astype(np.int32)
        # self.histogram = np.histogram(self.pixels_flat.flatten(),256,[0,255])
        
        # Close file
        file.close()

    # Print header data
    def debug_header(self):
        print("Magic number: " + self.magic_number.strip('\n'))
        print("Resolution: " + str(self.resolution[0]) + " x " + str(self.resolution[1]))
        print("Quantization: " + str(self.quantization))

    # Print row data
    def debug_row(self, row):
        print("\n" + "---Line " + str(row) + "---")
        print(self.pixels[row])

    # Print body data
    def debug_body(self):
        for x in range(self.x):
            print("\n" + "---Line " + str(x) + "---")
            print(self.pixels[x])

    # Save pgm image with a suffix indicating how it was modified
    # Note that this will overwrite an exiting file with that suffix
    def save(self, suffix):
        # Create or overwrite file
        file = open(self.name + suffix + ".pgm", "w")

        # Write magic number line 1
        file.write(self.magic_number)

        # Write comment on line 2
        file.write("# This is a modified file\n")

        # Write resolution on line 3
        file.write(self.resolution)

        # Write quantization on line 4
        file.write(str(self.quantization) + "\n")

        # Write body on the rest of the file
        for x in range(self.x):
            line = " ".join(str(pixel) for pixel in self.pixels[x])
            line = line.join(" \n")
            file.write(line)
        
        file.close()