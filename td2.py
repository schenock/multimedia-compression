import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as img
import numpy as np

vec = [88, 88, 89, 90, 92, 94, 96, 97]
img_src = "/home/jose/Universidad/DataScience/Compression/lenaTest3.jpg"


# Encodes the line passed as parameter.
def encode_line(vec):
    averages = []
    diffs = []
    vec = map(int, vec)
    for i in range(0,len(vec),2):
        averages.append(((vec[i]) + (vec[i+1]))/2.0)
        diffs.append(((vec[i]) - (vec[i+1]))/2.0)
    
    return averages + diffs

# Receives a matrix and encodes it, row-wise and column-wise.
# Returns the encoded matrix
def encode_matrix(matrix):
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

    matrix = np.apply_along_axis(encode_line, 0, matrix)
    matrix = np.apply_along_axis(encode_line, 1, matrix)

    return matrix


#print(encode_line(encode_line(vec)))


# Encodes a given image
def image_analysis(image):
    matrix = img.imread(image)
    matrix.astype(float)
    N = int(np.floor(np.log(len(matrix)))) # Number of iterations 

    N = 3 #TEST ONLY

    # For each iteration, find the subset of the image which has to be encoded,
    # pass it to the encode_matrix function, and substitute it with the returned value
    for i in range(0, N):
        # The index of the first column which doesnt have to be encoded in this iteration
        # (Cols further to the right correspond to the high pass component of previous iterations)
        col_limit = len(matrix)/pow(2,i) 
        row_limit = len(matrix[0])/pow(2,i) # Same thing with rows (this way we can work with non square imgs)

        matrix[:row_limit,:col_limit] = encode_matrix(matrix[:row_limit, :col_limit]) 

        # Show the image
        plt.imshow(matrix)
        plt.gray()
        plt.show()

    # #show image
    # plt.imshow(matrix, interpolation='nearest')
    # plt.gray()
    # plt.show()

image_analysis(img_src)