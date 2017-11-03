import matplotlib
import numpy as np


# TODO: Remove comments 1
def exhaustive_search(reference_frame, current_frame, block_size, padding):
    """
    Performs exhaustive serach, given 2 frames: the current and the reference frame
    :param int block_size: Block-size - in pixels
    :param int padding: Search area around the current block - in pixels

    """
    row = 0
    col = 0
    m = 0
    n = 0

    search_size = block_size + padding * 2

    xAxis = []
    yAxis = []

    for row in range(0, len(current_frame) - block_size + 1, block_size):
        xAxisList = []
        yAxisList = []
        for col in range(0, len(current_frame[0]) - block_size + 1, block_size):
            bx = row + block_size
            by = col + block_size
            current_block = current_frame[row:bx,col:by]
            # Call search function here:
            # current_block, reference_frame, block_size, padding
            # Perform exhaustive search in search-area defined by P for each block - current_block current_frame[row:row + N, col:col + N]
            xAxisCoord, yAxisCoord = search_for_block(current_block, reference_frame, block_size, padding, row, col)
            # append coords to list
            xAxisList.append(xAxisCoord)
            yAxisList.append(yAxisCoord)
        # append list to matrix
        xAxis.append(xAxisList)
        yAxis.append(yAxisList)

    return xAxis, yAxis

#TODO: Remove prints 2
def search_for_block(current_block, reference_frame, block_size, padding, row, col):

    # Calculate search area
    x = row - padding if row - padding >= 0 else 0
    y = row + block_size + padding if row + block_size + padding >= 0 else 0
    z = col - padding if col - padding >= 0 else 0
    w = col + block_size + padding if col + block_size + padding >= 0 else  0

    search_area = reference_frame[x:y, z:w]
    min_mse = 9999999
    xAxisCoordinates = 99999999
    yAxisCoordinates = 99999999

    for search_row in range(0, len(search_area) - block_size + 1, 1):
        for search_col in range(0, len(search_area[0]) - block_size + 1, 1):
            search_block = search_area[search_row:(search_row + block_size), search_col:(search_col + block_size)]
            mse = ((current_block - search_block) ** 2).mean(axis = None)
            if(mse < min_mse):
                min_mse = mse
                # Save coordinates of current block
                xAxisCoordinates = col - (search_col + x)
                yAxisCoordinates = (row - (search_row + z))*-1 # Multiply by -1 because the origin is top left here and bottom left in the plot

    return xAxisCoordinates, yAxisCoordinates

    #if min_mse == 0:
    #   print "MIN MSE: " + str(min_mse)
    #   print "RES X: " + str(xAxisCoordinates)
    #   print "RES Y: " + str(yAxisCoordinates)



def main():
    reference = np.arange(100).reshape(10,10)
    b = np.arange(100).reshape(10,10)

    current_frame = np.zeros(100).reshape(10, 10)
    current_frame[6, 2] = 5
    current_frame[6, 3] = 5
    current_frame[7, 2] = 5
    current_frame[7, 3] = 5

    #reference[2, 1] = 5
    #reference[2, 2] = 5
    #reference[3, 1] = 5
    #reference[3, 2] = 5

    reference[1, 2] = 5
    reference[1, 3] = 5
    reference[2, 2] = 5
    reference[2, 3] = 5

    #print a
    exhaustive_search(reference, current_frame, 2, 2)

if __name__ == "__main__":
    main()
