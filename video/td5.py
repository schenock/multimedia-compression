import matplotlib
import numpy as np

def exhaustive_search(reference_frame, current_frame, block_size, padding):
    """
    Performs exhaustive serach, given 2 frames: the current and the reference frame
    :param int block_size: Block-size - in pixels
    :param int padding: Search area around the current block - in pixels

    """
    i = 0
    j = 0
    m = 0
    n = 0

    search_size = block_size + 2 * padding
    print "padding: " + str(padding)
    print "block size: " + str(block_size)
    print "search length: " + str(search_size)

    for i in range(0, len(current_frame) - block_size + 1, block_size):
        for j in range(0, len(current_frame[0]) - block_size + 1, block_size):
            bx = i + block_size
            by = j + block_size
            print "bx = " + str(bx)
            print "Current Frame: "
            print current_frame[i:i + block_size, j:j + block_size]
            print " ----------------------------- "
            current_block = current_frame[i:bx,j:by]
            # Perform exhaustive search in search-area defined by P for each block - current_block current_frame[i:i + N, j:j + N]
            x = i - padding if i - padding >= 0 else 0
            y = i + block_size + padding if i + block_size + padding >= 0 else 0
            z = j - padding if j - padding >= 0 else 0
            w = j + block_size + padding if j + block_size + padding >= 0 else  0

            search_area = reference_frame[x:y, z:w]

            print "Searching for search area:\n"
            print search_area

            search_for_block(current_block, search_area, block_size)


def search_for_block(current_block, search_area, block_size):

    print "=================================================================="
    for i in range(0, len(search_area) - block_size+1, 1):
        for j in range(0, len(search_area[0]) - block_size+1, 1):
            search_block = search_area[i:(i + block_size), j:(j + block_size)]
            print "Current block\n" + str(current_block)
            print "Search block: \n" + str(search_block)

            print np.allclose(search_block, current_block)

    print "=================================================================="



a = np.arange(100).reshape(10,10)
b = np.arange(100).reshape(10,10)

exhaustive_search(a, b, 3, 2)
