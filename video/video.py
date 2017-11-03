
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from skimage import color
import td5
import math


def main():
    # Load the video
    vid_src = "sample_video.mp4"
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, vid_src)
    vid = imageio.get_reader(filename, 'mp4')

    # Extract two frames from the video and convert them to grayscale
    # Reference Frame
    fr = color.rgb2gray(vid.get_data(170))
    # Current Frame
    fc = color.rgb2gray(vid.get_data(173))

    # Perform exhaustive search to find the motion vectors for 
    # each macroblock in fr
    block_size = 10
    p = 8
    
    # U, V represent the two components of the movement vectors
    U, V = td5.exhaustive_search(fc, fr, block_size, p) # TODO: Which one is fr and which one is fc?

    fcc = motion_copy(fc, U, V, block_size)
    plt.imshow(fcc)
    plt.show()

    height, width = fr.shape # 320 x 240
    plt.imshow(fr, cmap='gray', extent=[0, width, 0, height])
    
    # X, Y are the coords of the arrows tails
    X, Y = np.meshgrid(np.arange(0+block_size/2, width, block_size),
                       np.arange(0+block_size/2, height, block_size))

    # U = np.ones_like(X) * 10
    # V = np.ones_like(Y) * 10

    # Can be useful if we want the arrows to be colored
    # depending on their length
    M = np.hypot(U, V)
    
    plt.quiver(X, Y, np.asarray(U), np.asarray(V), np.asarray(M), scale=1, units='xy', color="w")
    plt.show()

    # Create a new frame, fcc, placing each of the macroblocks in fr
    # in the position their motion vectors indicate
    fcc = np.zeros_like(fc)
    fcc[:] = fc


def motion_copy(ref, xmov, ymov, block_size):
    # Create new frame and fill it with ref
    new_frame = np.zeros_like(ref)
    #new_frame[:] = ref # TODO: Should it be like this? Or what to place in the empty spaces? 
    
    # Block by block, find where they should be
    for i in range(0, math.ceil(len(ref)/block_size)): 
        for j in range(0, math.ceil(len(ref[0])/block_size)):
            # Actual x and y coordinates on the ref matrix
            xref = i*block_size
            yref = j*block_size
            # New position will be
            x = xref + xmov[i,j]
            y = yref + ymov[i,j]
            # Update the corresponding bits in the new frame
            new_frame[x:x+block_size, y:y+block_size] = ref[xref:xref+block_size, yref:yref+block_size]

    return new_frame


if __name__ == "__main__":
    main()

    
# https://matplotlib.org/examples/pylab_examples/quiver_demo.html
# https://stackoverflow.com/questions/34458251/plot-over-an-image-background-in-python