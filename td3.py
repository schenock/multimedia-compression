def quantize(vector, R):
    minv = 0
    maxv = 256

    L = pow(2, R)
    bucket = (maxv - minv)/L

    #print "Bucket: " + str(bucket)

    quantized = []

    for val in vector:
        interval = val/bucket
        representative = interval*bucket + bucket/2
        quantized.append(representative)
        #print representative
    return quantized

# Quantize the image passed as parameter (it has to be a matrix)
def quantize_image(matrix, R):
   
    return np.apply_along_axis(quantize, 1, matrix, R)



