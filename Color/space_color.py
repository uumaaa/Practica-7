import numpy as np
import numpy as np

def convert_gray_scale(I):
    # Get the size of the image
    n, m = I.shape[0:2]
    img = np.zeros((n, m), dtype=np.uint8)

    # Convert to grayscale
    for x in range(n):
        for y in range(m):
            img[x][y] = 0.299 * I[x][y][0] + 0.587 * I[x][y][1] + 0.114 * I[x][y][2]

    return img

def invert_binary(I:np.ndarray):
    n,m = I.shape[:2]
    img = np.zeros((n,m),dtype=np.uint8)
    for x in range(n):
        for y in range(m):
            img[x][y] = 255 if I[x][y] == 0 else 0
    return img