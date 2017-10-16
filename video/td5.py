import matplotlib
import numpy as np

def exhaustive_search(reference_frame, current_frame, N, P):
    """
    Performs exhaustive serach, given 2 frames: the current and the reference frame
    :param int N: Block-size - in pixels
    :param int P: Search area around the current block - in pixels

    """
    i = 0
    j = 0
    m = 0
    n = 0

    print current_frame
    print '--------------'
    #print len(current_frame)
    #print len(current_frame[0])

    for i in range(0, len(current_frame), N):
        for j in range(0, len(current_frame[0]), N):
            bx = i + N
            by = j + N
            print "bx = " + str(bx)
            print "Current Frame: "
            print current_frame[i:i + N, j:j + N]
            print " ----------------------------- "
            current_block = current_frame[i:bx,j:by]
            print current_block
            # Perform exhaustive search in search-area defined by P for each block - current_block current_frame[i:i + N, j:j + N]
            print "i-p = " +  str(i-P)
            print "i+N+P = " + str(i+N+P)
            print "j-p = " + str(j - P)
            print "i+N+P = " + str(i+N+P)
            x = i - P if i - P >= 0 else 0
            y = i + N + P if i + N + P >= 0 else 0
            z = j - P if j - P >= 0 else 0
            w = j + N + P if j + N + P >= 0 else  0

            print x, y , z, w

            search_area = reference_frame[x:y, z:w]
            print search_area
            search_for_block(current_block, search_area, N)


def search_for_block(current_block, search_area, N):
    for i in range(0, len(search_area), N):
        for j in range(0, len(search_area[0]), N):
            search_block = search_area[i:(i + N), j:(j + N)]
            print "Current block" + current_block
            print "Search block: " + search_block
            print "............................."



a = np.arange(36).reshape(6,6)
b = np.arange(36).reshape(6,6)



exhaustive_search(a, b, 2, 1)
