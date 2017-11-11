import matplotlib
import sys
import numpy as np


# TODO: Remove comments 1
def get_motion_vectors(reference_frame, current_frame, block_size, padding):
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

    yMovement = []
    xMovement = []

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
        yMovement.append(xAxisList)
        xMovement.append(yAxisList)

    return np.matrix(xMovement), np.matrix(yMovement)*-1


def search_for_block(current_block, reference_frame, block_size, padding, row, col):
    """
    Given a block and a reference frame, searches for the most similar block in search area defined with: block_size + padding*2
    :param int block_size : Block size (in pixels)
    :param int padding : Padding (in pixels)
    :param int row : Row number (y-coordinate) of the current_block in the original frame
    :param int col : Col number (x-coordinate) of the current_block in the original frame
    """

    # Calculate search area
    x = row - padding if row - padding >= 0 else 0
    y = row + block_size + padding if row + block_size + padding >= 0 else 0
    z = col - padding if col - padding >= 0 else 0
    w = col + block_size + padding if col + block_size + padding >= 0 else  0

    search_area = reference_frame[x:y, z:w]

    min_mse = sys.maxsize
    xMovement = sys.maxsize
    yMovement = sys.maxsize

    for search_row in range(0, len(search_area) - block_size + 1, 1):
        for search_col in range(0, len(search_area[0]) - block_size + 1, 1):
            search_block = search_area[search_row:(search_row + block_size), search_col:(search_col + block_size)]
            mse = ((current_block - search_block) ** 2).mean(axis = None)

            # movement of the current block
            current_x_movement = col - (search_col + z)
            current_y_movement = (row - (search_row + x))*-1 # Multiply by -1 because the origin is top left here and bottom left in the plot

            # in case of equal mse for two blocks, take the closest one
            if mse == min_mse:
                dist = np.hypot(xMovement, yMovement)
                dist_current = np.hypot(current_x_movement, current_y_movement)
                if dist_current < dist:
                    xMovement = current_x_movement
                    yMovement = current_y_movement

            # update coordinates if more similar block found
            if mse < min_mse:
                min_mse = mse
                # Save coordinates of current block
                xMovement = current_x_movement
                yMovement = current_y_movement

    return xMovement, yMovement
    