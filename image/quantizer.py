import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as img
import numpy as np

vec = [0, 255, 20, 15, 1]

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


def plot_characteristic(R):
    rangeV = range(0,255)
    values = quantize(rangeV,R)
    # plot range, values
    plt.plot(rangeV, values)
    plt.show()
    return values


def calc_MSE(original, quantized):
    MSE = 0
    for (val, quant) in zip(original, quantized):
        MSE += pow(abs(val-quant), 2)
        print MSE
    return MSE/len(original)


def quantize_image(R):
    matrix = img.imread('/home/schenock/Desktop/lenaTest3.jpg')
    print matrix

    quantized_lena = []
    for line in matrix:
        quantized_line = quantize(line, R)
        quantized_lena.append(quantized_line)
    #print quantized_lena
    arr = np.array(quantized_lena)
    return arr


def quantize_lena():

    matrix = img.imread('/home/schenock/Desktop/lenaTest3.jpg')

    entropy_arr = []
    mse_arr = []

    for R in range(1, 8):
        arr = quantize_image(R)

        e = entropy2(arr)
        entropy_arr.append(e)

        mse = calc_MSE2(matrix, arr)
        mse_arr.append(mse)
        #print arr
        #plt.imshow(arr, interpolation='nearest')
        #plt.gray()
        #plt.title("Levels: " + str(pow(2,R)))
        #plt.show()

    print "MSE Array: " + str(mse_arr)
    print "Entropy Array: " + str(entropy_arr)
    plt.plot(entropy_arr, mse_arr)
    plt.xlabel("Entropy")
    plt.ylabel("Distortion")
    plt.show()

def plot_distortion():
    errors = []
    R = range(1,7)
    for r in R:
        MSE = calc_MSE(vec, quantize(vec, r))
        errors.append(MSE)
    plt.plot(R, errors)
    plt.xlabel("R")
    plt.ylabel("Distortion")
    plt.show()


def entropy():
    im = quantize_image(2)

    # calculate mean value from RGB channels and flatten to 1D array
    vals = im.flatten()
    # plot histogram with 255 bins
    b, bins, patches = plt.hist(vals, 255)
    plt.xlim([0, 255])
    plt.show()
    entropy = 0
    for count in b:
        norm = (count/sum(b))
        entropy += norm * np.log(norm, 2)
        print norm

    return -entropy



# MSE
def calc_MSE2(original, quantized):
    return (np.square(original - quantized)).mean(axis=None)




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
def entropy2(image):
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





print quantize(vec, 2)
print plot_characteristic(3)
plot_distortion()
print calc_MSE(vec, quantize(vec, 2))
quantize_lena()
entropy()