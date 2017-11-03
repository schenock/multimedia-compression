
import numpy as np
import imageio
import os
import pylab
import matplotlib.pyplot as plt
from numpy import ma
from skimage import color
import td5

def main():
    # Load the video
    vid_src = "sample_video.mp4"
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, vid_src)
    vid = imageio.get_reader(filename, 'mp4')

    # Extract two frames from the video and convert them to grayscale
    # Reference Frame
    fr = color.rgb2gray(vid.get_data(50))
    # Current Frame
    fc = color.rgb2gray(vid.get_data(55))
    
    # Perform exhaustive search to find the motion vectors for 
    # each macroblock in fr
    block_size = 20
    p = 4
    
    # U, V represent the length of both components of the arrows
    #U, V = td5.exhaustive_search(fr, fr, block_size, p)
    
    height, width = fr.shape
    plt.imshow(fr, cmap='gray', extent=[0, width, 0, height])
    
    # X, Y are the coords of the arrows tails
    X, Y = np.meshgrid(np.arange(0+block_size/2, width, block_size),
                       np.arange(0+block_size/2, height, block_size))

    U = np.ones_like(X) * 10
    V = np.ones_like(Y) * 10

    # Can be useful if we want the arrows to be colored
    # depending on their length
    M = np.hypot(U, V)
    
    plt.quiver(X, Y, U, V, M, scale=1, units='xy', color="w")
    plt.show()

    # Create a new frame, fcc, placing each of the macroblocks in fr
    # in the position their motion vectors indicate
    fcc = np.zeros_like(fc)
    fcc[:] = fc


if __name__ == "__main__":
    main()

    
# https://matplotlib.org/examples/pylab_examples/quiver_demo.html
# https://stackoverflow.com/questions/34458251/plot-over-an-image-background-in-python