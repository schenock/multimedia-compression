import numpy as np
import imageio
import os
import pylab
import matplotlib.pyplot as plt
from numpy import ma


def main():
    # Load the video
    vid_src = "sample_video.mp4"
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, vid_src)
    vid = imageio.get_reader(filename, 'mp4')

    # Extract two frames from the video
    fr = vid.get_data(50)  # Reference Frame
    fc = vid.get_data(55)  # Current Frame
    # TODO: Convert images to grayscale

    # Perform exhaustive search

    height, width, d = fr.shape
    fig, ax = plt.subplots()
    ax.imshow(fr, extent=[0, width, 0, height])
    plt.show()


if __name__ == "__main__":
    main()

    # # X, Y are the coords of the arrows tails
    # X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
    # # U, V represent the length of both components of the arrows
    # U = np.cos(X)
    # V = np.sin(Y)

    # plt.figure()
    # plt.title('Arrows scale with plot width, not view')
    # Q = plt.quiver(X, Y, U, V, scale=1, units='xy')
    # qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
    #                    coordinates='figure')

    # plt.show()
    # https://matplotlib.org/examples/pylab_examples/quiver_demo.html
    # https://stackoverflow.com/questions/34458251/plot-over-an-image-background-in-python
