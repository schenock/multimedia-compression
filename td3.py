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
def image_analysis(matrix, N=None):
    
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
    #plt.imshow(matrix)
    #plt.gray()
    #plt.show()

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

        print "i = " + str(i)
        # Show the image
        #plt.imshow(matrix)
        #plt.gray()
        #plt.show()

    return matrix

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

    # Normalize the freqs
    total = float(sum(hist.values()))
    for v in hist.values():
        v = v / total

    return hist


# Calculate the entropy of the image passed as parameter (matrix)
def entropy(image):
    # calculate mean value from RGB channels and flatten to 1D array
    vals = image.flatten()
    hist = get_symbol2freq(vals)

    entropy = 0
    for count in hist.values():
        if count != 0:
            entropy += count * np.math.log(count, 2)

    return -entropy

# N: number of decomposition levels
def get_subbands(matrix, N):
    subbands = []

    plt.imshow(matrix[0:len(matrix)/pow(2,N), 0:len(matrix[0])/pow(2,N)])
    plt.gray()
    plt.show()
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


def main(img_src):

    # Setup
    image = img.imread(img_src)
    image = image.astype(float)

    # 1. HAAR Transformation
    transformed_image = image_analysis(image, N=2)

    subbands, lowpass= get_subbands(transformed_image, 2)

    arr_min = []
    arr_bucket = []

    # 2. Quantization
    # R = 5 for outer subbands
    # R = 6 for inner subbands
    # LL subband is not quantized
    quantized_subbands = []
    for idx, band in enumerate(subbands):

        # different bitrate for subbands
        if idx < 3:
            R = 5
        else:
            R = 6

        print("Bitrate R: " + str(R))

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

        #print("Entropy before quant = ", entropy(band))
        #print(band)
        #print("Entropy after quant = ", entropy(quant_band))
        #print(quant_band)
        print("Entropy Ratio = ", entropy(band)/entropy(quant_band))

    # 5. Synthesis

    # 5.1 Synthesis of non quantized
    reconstructed_non_quantized = image_synthesis(transformed_image, N = 2)

    # 5.2 Synthesis of quantized
    dequantized_subbands = []
    for idx, band in enumerate(quantized_subbands):
        dequantized_subband = dequantize(band, arr_bucket[idx], arr_min[idx])
        dequantized_subbands.append(dequantized_subband)

    reconstructed_quantized = image_synthesis(reconstruct_subbands(dequantized_subbands, lowpass), N = 2)

    # Non quantized synthesis
    reconstructed  = image_synthesis(transformed_image, N = 2)


    # 6. Peak Signal to Noise Ratio
    r = calc_PSNR(image, reconstructed)
    rq = calc_PSNR(image, reconstructed_quantized)
    print r
    print rq
    D = r/rq
    print "Distortion:  " + str(D)



def calc_MSE(original, quantized):
    return ((original - quantized) ** 2).mean(axis=None)



# 4. Calculate Peak Signal to Noise Ratio
def calc_PSNR(original, quantized):
    mse = calc_MSE(original, quantized)
    return 10*np.log10(pow(255, 2))/mse


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



main(img_src)




# for subband in quantize_decomposed(image_analysis(img_src, N=2), 2):
#     # Show the image
#     plt.imshow(subband)
#     plt.gray()
#     plt.show()


# image_analysis(img_src, N=2)

