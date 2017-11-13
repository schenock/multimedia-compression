import math
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from skimage import color
import cv2

import td5
import imgcodec

import math

def main():

    # Load the video
    vid_src = "sample_video2.mp4"
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
    block_size = 16
    p = 8

    # U, V represent the two components of the movement vectors
    U, V = td5.get_motion_vectors(fr, fc, block_size, p)

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
    plt.title("Motion vectors")
    plt.show()

    # Create a new frame, fcc, placing each of the macroblocks in fr
    # in the position their motion vectors indicate
    fcc = motion_copy(fr, U, V, block_size)

    # Difference between fc and fr
    plt.imshow(fc -fr)
    plt.title("Fc - Fr")
    plt.gray()
    plt.show()

    eres =  fc - fcc
    plt.imshow(eres)
    plt.title("Eres")
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
        U, V = td5.get_motion_vectors(fr, fc, block_size, p)

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

    # TD7
    # Load the video
    shortvid_src = "trimmed_sample_video2.mp4"
    shortvid = imageio.get_reader(os.path.join(dirname, shortvid_src), 'mp4')
    fr = color.rgb2gray(shortvid.get_data(0))

    # Compress the whole video to extract the motion vectors and the error
    mean_errors = []
    mean_psnr = []
    bitrates = []

    # Lets compress the video using different bitrates 
    for R in range(1, 8):
        first, motion, eres = compress_video(shortvid, block_size, p, R)

        # Reconstruct the video from what we have stored 
        reconstructed = decompress_video(first, motion, eres, block_size)

        # Find the distortion
        errors = []
        for idx,frame in enumerate(reconstructed):
            original = color.rgb2gray(shortvid.get_data(idx))
            error = original - frame
            mae = np.absolute(error).mean(axis = None)
            errors.append(mae)
        # Avg error for the whole video
        mae = sum(errors)/len(errors)
        mean_errors.append(mae)
        mean_psnr.append(10*np.log10(pow(255,2)/mae))
        
        bitrates.append(get_bitrate(shortvid, motion, R, block_size, p))

    # Plot the results
    plt.plot(bitrates/1000000, mean_errors)
    plt.ylabel("Total average distortion")
    plt.xlabel("Bitrate (Mbps)")
    plt.show()

    plt.plot(bitrates/1000000, mean_psnr)
    plt.ylabel("PSNR")
    plt.xlabel("Bitrate (Mbps)")
    plt.show()    


    # Get frames of video
    frames = []
    for index in range(shortvid.get_length()):
        frames.append(shortvid.get_data(index))

    # Save as MPEG - test.
    print("Saving mp4 video..")
    save_MPEG(frames=frames, filename='saved-video', quality=7)


def get_bitrate(vid, motion, R, block_size, p):
    # Get video metadata
    fps = vid.get_meta_data()["fps"]
    dimensions = vid.get_meta_data()["size"]
    nframes = vid.get_meta_data()["nframes"]

    duration = nframes/float(fps)

    # Pixels per frame 
    pixels = dimensions[0] * dimensions[1]

    # Bits per pixel
    bpp = 1/16.0*8 + 3/16.0*(R+1) + 3/4.0*R

    # Bits used to encode motion vectors
    # (max possible motion value is p)
    bitrate_motion = np.math.log(np.absolute(2*p), 2)

    total_bits_motion = pixels/pow(block_size, 2) * 2 * nframes * bitrate_motion

    return (pixels*bpp*nframes + total_bits_motion)/duration


def compress_video(vid, block_size, p, R):
    """Compresses the video passed as parameter, using a given block size and search
    parameter for the block matching algorithm, and bitrate R for image compression.i
    """
    print("Compressing video...")
    all_motion = []
    all_eres = []

    first = imgcodec.compress(color.rgb2gray(vid.get_data(0)), R)

    for i in range(1, vid.get_length()):
        print("Frame {} of {}:".format(i, vid.get_length()))
        fr = color.rgb2gray(vid.get_data(i - 1))
        fc = color.rgb2gray(vid.get_data(i))

        print("Getting motion vectors...")
        # Motion compensated frame
        u, v = td5.get_motion_vectors(fr, fc, block_size, p)
        mot_comp = motion_copy(fr, u, v, block_size)

        print("Calculating and compressing Eres...")
        # Calculate the error and compress it
        eres = fc - mot_comp
        comp_eres = imgcodec.compress(eres, R)
        
        # Store eres and motion vectors
        all_eres.append(comp_eres)
        all_motion.append([u,v])

    return first, all_motion, all_eres


def decompress_video(first, motion, eres, block_size):
    """Given the first frame of a video (compressed), the motion vectors, the
    Eres (also compressed), and the block size used for the block matching algorithm,
    reconstructs the original video."""

    print("Decompressing video...")
    # Decompress the video
    frames = []

    # Decompress the first frame
    fr = imgcodec.decompress(first)
    frames.append(fr)
    for i in range(len(motion)):
        # Get the predicted frame, compensating with the error
        fc = motion_copy(fr, motion[i][0], motion[i][1], block_size)
        fc = fc + imgcodec.decompress(eres[i])
        frames.append(fc)
        # The current frame becomes the reference for next iteration 
        fr = fc
    print("Done.")
    return frames


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


def save_MPEG(frames, filename, quality):
    """Saves array of frames as .mp4 video"""

    filename = filename + '.mp4'

    with imageio.get_writer(filename, quality=quality) as writer:
        for frame in frames:
            writer.append_data(frame)

    writer.close()



if __name__ == "__main__":
    main()

    
# https://matplotlib.org/examples/pylab_examples/quiver_demo.html
# https://stackoverflow.com/questions/34458251/plot-over-an-image-background-in-python
