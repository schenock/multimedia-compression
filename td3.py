import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as img
import numpy as np
import os
import math
from heapq import heappush, heappop, heapify
from collections import defaultdict

img_src = "lenaTest3.jpg"
dir = os.path.dirname(__file__)
filename = os.path.join(dir, img_src)

vec = [88, 88, 89, 90, 92, 94, 96, 97]


# Encodes the line passed as parameter.
def encode_line(vec):
    averages = []
    diffs = []

    for i in range(0,len(vec),2):
        averages.append(((vec[i]) + (vec[i+1]))/2.0)
        diffs.append(((vec[i]) - (vec[i+1]))/2.0)
    
    return averages + diffs


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
    # rowmatrix = [] # Matrix to store the results of encoding by row
    # newmatrix = [] # Matrix to store the final results

    # Encode all rows in our original image
    # for row in matrix:
    #     newmatrix.append(encode_line(row))
    
    # Transpose the matrix so that we operate with columns this time
    # transposed = np.transpose(rowmatrix)
    # for col in transposed:
    #     newmatrix.append(encode_line(col))
    
    # # We have to transpose it again before returning it
    # return np.array((np.transpose(newmatrix)))

    matrix = np.apply_along_axis(func, 1, matrix)
    matrix = np.apply_along_axis(func, 0, matrix)

    return matrix


#print(encode_line(encode_line(vec)))


# Encodes a given image
def image_analysis(image, N=None):
    matrix = img.imread(image)
    matrix = matrix.astype(float)
    
    if N is None:
        N = int(np.floor(np.log(len(matrix)))) - 1 # Number of iterations 

    # For each iteration, find the subset of the image which has to be encoded,
    # pass it to the encode_matrix function, and substitute it with the returned value
    for i in range(0, N):
        # The index of the first column which doesnt have to be encoded in this iteration
        # (Cols further to the right correspond to the high pass component of previous iterations)
        col_limit = len(matrix)/pow(2,i) 
        row_limit = len(matrix[0])/pow(2,i) # Same thing with rows (this way we can work with non square imgs)

        matrix[:row_limit,:col_limit] = encode_matrix(matrix[:row_limit, :col_limit], encode_line) 

    # Show the image
    plt.imshow(matrix)
    plt.gray()
    plt.show()

    return matrix


# Decodes a given image
def image_synthesis(image, N=None):
    # TODO: Figure out what to do with image/matrix as input (which one to choose)
    # matrix = img.imread(image)
    # matrix = matrix.astype(float)
    matrix = image
    
    if N is None:
        N = int(np.floor(np.log(len(matrix)))) - 1 # Number of iterations 

    # For each iteration, find the subset of the image which has to be encoded,
    # pass it to the encode_matrix function, and substitute it with the returned value
    for i in reversed(range(0, N)):
        # The index of the first column which doesnt have to be encoded in this iteration
        # (Cols further to the right correspond to the high pass component of previous iterations)
        col_limit = len(matrix)/pow(2,i) 
        row_limit = len(matrix[0])/pow(2,i) # Same thing with rows (this way we can work with non square imgs)

        matrix[:row_limit,:col_limit] = encode_matrix(matrix[:row_limit, :col_limit], decode_line) 

        # Show the image
        plt.imshow(matrix)
        plt.gray()
        plt.show()

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



    # #print("L:" + str(L)+" R:" + str(R)+" min:"+str(minv)+ " max:" + str(maxv) + " bucket:" +str(bucket))
    # quantized = []

    # for val in vector:
    #     interval = (val)/bucket
    #     representative = (interval*bucket + bucket/2)
    #     quantized.append(representative)
        
    # return quantized


# # Quantizes a vector using R bits
# def quantize(vector, step, minv):
#     quantized = []

#     for val in vector:
#         interval = (val-minv)/step
#         representative = interval*step + step/2
#         quantized.append(representative)
#         #print representative
#     return quantized

# Quantize the image passed as parameter (it has to be a matrix)
def quantize_image(matrix, R):
    #print("MIN:" + str(matrix.min()) + " MAX:" + str(matrix.max()))
    return np.apply_along_axis(quantize, 1, matrix, R, minv=matrix.min(), maxv=matrix.max())


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
    # calculate mean value from RGB channels and flatten to 1D array
    vals = image.flatten()
    
    hist = get_symbol2freq(vals)
    
    entropy = 0
    
    for count in hist.values():
        norm = (count/float(sum(hist.values())))
        if norm != 0:
            entropy += norm * np.math.log(norm, 2)

    return -entropy

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

    return subbands

def quantize_subbands(img_src):
    # 1. HAAR Transformation
    transformed_image = image_analysis(img_src, N=2)

    subbands = get_subbands(transformed_image, 2)

    arr_min = []
    arr_bucket = []

    # 2. Quantization
    for R in range(1,8):
        quantized_subbands = []
        print("Bitrate R: " + str(R))
        for band in subbands:

            # quantize subband with fixed bit rate R
            quant_band = quantize_image(band, R)

            # calculate bucket for current subband
            L = pow(2, R)
            bucket = abs(band.max() - band.min()) / L

            # save band min and bucket size
            arr_min.append(band.min())
            arr_bucket.append(bucket)

            # save quantized subband
            quantized_subbands.append(quant_band)


            print("Entropy before quant = ", entropy(band))
            #print(band)
            print("Entropy after quant = ", entropy(quant_band))
            #print(quant_band)
            print("Diff = ", entropy(band)-entropy(quant_band))

    # 3. Huffman Coding


def calc_MSE(original, quantized):
    MSE = 0
    for (val, quant) in zip(original, quantized):
        MSE += pow(abs(val-quant), 2)
        print MSE
    return MSE/len(original)

def huff_encode(symb2freq):
    """Huffman encode the given dict mapping symbols to weights"""
    # Source: https://rosettacode.org/wiki/Huffman_coding#Python

    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

  
    
quantize_subbands(img_src)

# for subband in quantize_decomposed(image_analysis(img_src, N=2), 2):
#     # Show the image
#     plt.imshow(subband)
#     plt.gray()
#     plt.show()


# image_analysis(img_src, N=2)

