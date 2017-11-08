import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as img
import numpy as np
import os
import math
from heapq import heappush, heappop, heapify
from collections import defaultdict


img_src = "lenaTest3.jpg"
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, img_src)


# Quantizes a vector using R bits
def quantize(vector, R, minv=0, maxv=256):
    L = pow(2, R)
    bucket = abs(maxv - minv)/L

    bins = np.linspace(minv, maxv, L+1)
    indexes = np.digitize(vector, bins)

    return indexes

# Dequantizes a vector given a bucket size
def dequantize(indexes, bucket, minv):
    result = []
    for i in indexes:
        result.append(i*bucket - bucket/2.0 + minv)

    return result

# Quantize the image passed as parameter (it has to be a matrix)
def quantize_image(matrix, R):
    #print("MIN:" + str(matrix.min()) + " MAX:" + str(matrix.max()))
    return np.apply_along_axis(quantize, 1, matrix, R, minv=matrix.min(), maxv=matrix.max())


# Encodes (Haar wavelet transform) the line passed as parameter.
def encode_line(vec):
    averages = []
    diffs = []

    for i in range(0,len(vec),2):
        averages.append(((vec[i]) + (vec[i+1]))/2.0)
        diffs.append(((vec[i]) - (vec[i+1]))/2.0)
    
    return averages + diffs

# Perfroms the Haar wavelet synthesis of the provided line
def decode_line(vec):
    averages = vec[:len(vec)/2]
    diffs = vec[len(vec)/2:]

    line = []
    
    for a,d in zip(averages, diffs):
        line.append(a + d)
        line.append(a - d)

    return line

# Receives a matrix and encodes it, row-wise and column-wise.
# Returns the encoded matrix
def encode_matrix(matrix, func):
    matrix = np.apply_along_axis(func, 1, matrix)
    matrix = np.apply_along_axis(func, 0, matrix)

    return matrix

# Encodes a given image
def image_analysis(matrix, N=None):
    newmatrix = np.empty_like(matrix)
    newmatrix[:] = matrix

    if N is None:
        N = int(np.floor(np.log(len(matrix)))) - 1 # Number of iterations 

    # For each iteration, find the subset of the image which has to be encoded,
    # pass it to the encode_matrix function, and substitute it with the returned value
    for i in range(0, N):
        # The index of the first column which doesnt have to be encoded in this iteration
        # (Cols further to the right correspond to the high pass component of previous iterations)
        col_limit = len(matrix)/pow(2,i) 
        row_limit = len(matrix[0])/pow(2,i) # Same thing with rows (this way we can work with non square imgs)

        newmatrix[:row_limit,:col_limit] = encode_matrix(newmatrix[:row_limit, :col_limit], encode_line) 

    return newmatrix


# Decodes a given image
def image_synthesis(matrix, N=None):
    newmatrix = np.empty_like(matrix)
    newmatrix[:] = matrix
    
    # If no N is provided, find the optimal number of decomposition levels
    if N is None:
        N = int(np.floor(np.log(len(matrix)))) - 1 # Number of iterations (levels)

    # For each iteration, find the subset of the image which has to be encoded,
    # pass it to the encode_matrix function, and substitute it with the returned value
    for i in reversed(range(0, N)):
        # The index of the first column which doesnt have to be encoded in this iteration
        # (Cols further to the right correspond to the high pass component of previous iterations)
        col_limit = len(matrix)/pow(2,i) 
        row_limit = len(matrix[0])/pow(2,i) # Same thing with rows (this way we can work with non square imgs)

        newmatrix[:row_limit,:col_limit] = encode_matrix(newmatrix[:row_limit, :col_limit], decode_line) 

    return newmatrix

# Creates a dictionary where each symbol has it's frequency associated to it
def get_symbol2freq(vals):
    hist = {}

    # Get the histogram
    for v in vals:
        if v in hist:
            hist[v] = hist[v] + 1
        else:
            hist[v] = 1

    return hist


# Calculate the entropy of the image passed as parameter (matrix)
def entropy(image):
    # Flatten to 1D array
    vals = image.flatten()
    hist = get_symbol2freq(vals)

    # Normalize the freqs
    total = float(sum(hist.values()))

    entropy = 0
    for count in hist.values():
        if count != 0:
            norm = count/total
            entropy += norm * np.math.log(norm, 2)

    return (entropy*(-1))

# Returns an array containing the different subbands present in a N-level Haar wavelet decomposition
# N: number of decomposition levels
def get_subbands(matrix, N):
    subbands = []

    # Find the subbands
    for i in range(1, N+1):
        # The index of the first column which doesnt have to be encoded in this iteration
        # (Cols further to the right correspond to the high pass component of previous iterations)
        col_limit = len(matrix)/pow(2,i)
        row_limit = len(matrix[0])/pow(2,i) # Same thing with rows (this way we can work with non square imgs)

        subbands.append(matrix[row_limit:row_limit*2, 0:col_limit])
        subbands.append(matrix[row_limit:row_limit*2, col_limit:col_limit*2])
        subbands.append(matrix[0:row_limit, col_limit:col_limit*2])

    # Find the dimensions of the lowpass component
    size_lp_horiz = len(matrix)/pow(2, N)
    size_lp_vert = len(matrix[0])/pow(2, N)

    lowpass = np.zeros(shape = (size_lp_horiz, size_lp_vert))
    lowpass[:] = matrix[0:size_lp_horiz, 0:size_lp_vert]
    return (subbands, lowpass)



# Given the array of subbands, reconstructs the original matrix
# The first subbands in the array should be those corresponding to the first decomposition level
# (the outer ones)
# Assuming square input!
def reconstruct_subbands(subbands, lowpass):
    size = len(subbands[0])*2 # First subband is half the size of the original image
    matrix = np.zeros(shape=(size,size))

    N = len(subbands)/3 # There are three subbands per decomposition level


    for i in range(0,N):
        middle = size/pow(2,i+1)

        matrix[middle:middle*2, 0:middle] = subbands[i*3]
        matrix[middle:middle*2, middle:middle*2] = subbands[i*3 + 1]
        matrix[0:middle, middle:middle*2] = subbands[i*3 + 2]

    matrix[0:size/pow(2, N), 0:size/pow(2, N)] = lowpass

    return matrix


def img_codec(image):
    """ Applies the still image codec to a given image
    Params:
      image: matrix representing the grayscale img"""

    # 1. HAAR Transformation
    transformed_image = image_analysis(image[:], N=2)

    subbands,lowpass= get_subbands(transformed_image, 2)

    arr_min = []
    arr_bucket = []

    # 2. Quantization
    # R = 5 for outer subbands
    # R = 6 for inner subbands
    # LL subband is not quantized
    quantized_subbands = []
    total_entropy = entropy(lowpass)
    total_entropy_quant = entropy(lowpass)
    for idx, band in enumerate(subbands):

        # Different bitrate for subbands
        # Weight is used to calculate the contribution of each subband to the total entropy
        if idx < 3:
            R = 5
            weight = 1/4.0
        else:
            R = 6
            weight = 1/16.0

        #print("Bitrate R: " + str(R))

        # quantize subband with fixed bit rate R
        quant_band = quantize_image(band, R)

        total_entropy += entropy(band)*weight
        total_entropy_quant += entropy(quant_band)*weight

        # calculate bucket for current subband
        L = pow(2, R)
        bucket = abs(band.max() - band.min()) / L

        # save band min and bucket size
        arr_min.append(band.min())
        arr_bucket.append(bucket)

        # save quantized subband
        quantized_subbands.append(quant_band)

    # 5. Synthesis
    # 5.2 Synthesis of quantized
    dequantized_subbands = []
    for idx, band in enumerate(quantized_subbands):
        dequantized_subband = dequantize(band, arr_bucket[idx], arr_min[idx])
        dequantized_subbands.append(dequantized_subband)

    transformed_quantized = reconstruct_subbands(dequantized_subbands, lowpass)
    reconstructed_quantized = image_synthesis(transformed_quantized, N = 2)

    return reconstructed_quantized