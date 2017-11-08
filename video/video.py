
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
    fr = color.rgb2gray(vid.get_data(102))#170
    # Current Frame
    fc = color.rgb2gray(vid.get_data(103))#173

    # Perform exhaustive search to find the motion vectors for 
    # each macroblock in fr
    block_size = 8
    p = 8
    
    # U, V represent the two components of the movement vectors
    U, V = td5.exhaustive_search(fr, fc, block_size, p) # TODO: Which one is fr and which one is fc?

    height, width = fr.shape # 320 x 240
    plt.imshow(fc, cmap='gray', extent=[0, width, 0, height])
    
    # X, Y are the coords of the arrows tails
    X, Y = np.meshgrid(np.arange(0+block_size/2, width, block_size),
                       np.arange(0+block_size/2, height, block_size))

    # U = np.ones_like(X) * 10
    # V = np.ones_like(Y) * 10

    # Can be useful if we want the arrows to be colored
    # depending on their length
    M = np.hypot(U, np.flip(V, axis=0))
    
    plt.quiver(X, Y, np.asarray(U), np.asarray(np.flip(V, axis=0)), np.asarray(M), scale=1, units='xy', color="w")
    plt.show()

    # Create a new frame, fcc, placing each of the macroblocks in fr
    # in the position their motion vectors indicate
    fcc = motion_copy(fr, U, V, block_size)

    # Difference between fc and fr
    plt.imshow(fc -fr)
    plt.gray()
    plt.show()

    eres =  fc - fcc
    plt.imshow(eres)
    plt.gray()
    plt.show()

    # Avg motion compensated error
    mae = np.absolute(eres).mean(axis = None)

    # Calculate mae for the first 20 frames
    mae_20 = []
    psnr_20 = []
    for i in range(170,190):
        # Extract the current frame and the following one
        fr = color.rgb2gray(vid.get_data(i))
        fc = color.rgb2gray(vid.get_data(i+1))

        # U, V represent the two components of the movement vectors
        U, V = td5.exhaustive_search(fr, fc, block_size, p)

        # Create a new frame, fcc, placing each of the macroblocks in fr
        # in the position their motion vectors indicate
        fcc = motion_copy(fr, U, V, block_size)

        eres =  fc - fcc

        # Calculate mae and psnr and append to the lists
        mae = np.absolute(eres).mean(axis = None)
        mae_20.append(mae)
        psnr_20.append(10*np.log10(pow(255,2)/mae))

    # Plot the results
    plt.plot(range(len(mae_20)), mae_20)
    plt.ylabel("Mae")
    plt.xlabel("Frame")
    plt.xticks(range(0, len(mae_20), 2))
    plt.show()

    plt.plot(range(len(psnr_20)), psnr_20)
    plt.ylabel("PSNR[Mae] (dB)")
    plt.xlabel("Frame")
    plt.xticks(range(0, len(psnr_20), 2))
    plt.show()




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