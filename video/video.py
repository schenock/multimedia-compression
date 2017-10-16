import numpy as np
import imageio
import os
import pylab

def main:
	# Load the video
	vid_src = "sample_video.mp4"
	dirname = os.path.dirname(__file__)
	filename = os.path.join(dirname, vid_src) 
	vid = imageio.get_reader(filename,  'mp4')

	# Extract two frames from the video
	fr = vid.get_data(50) # Reference Frame
	fc = vid.get_data(55) # Current Frame

	# Perform exhaustive search


# MSE
def mse(original, quantized):
    return (np.square(original - quantized)).mean(axis=None)


    