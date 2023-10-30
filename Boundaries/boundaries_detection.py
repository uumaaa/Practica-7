import numpy as np
def moore_boundary_detection(image: np.ndarray) -> [[int, int]]:
    m_ngbh = {0: [0, -1], 1: [-1, -1], 2: [-1, 0], 3: [-1, 1], 4: [0, 1], 5: [1, 1], 6: [1, 0], 7: [1, -1]}
    def find_moore_neighborhood(pixel: [int, int],c:int) -> (dict, int):
        neighbors = {}
        first = None
        for n in range(8):
                neighbor_y = pixel[0] + m_ngbh[(n+c)%8][0]
                neighbor_x = pixel[1] + m_ngbh[(n+c)%8][1]
                if image_pad[neighbor_y][neighbor_x] == 255:
                    if first is None:
                        first = (n+c)%8
                    neighbors[(n+c)%8] = [neighbor_y, neighbor_x]
        return neighbors, first
    
    image_pad = np.pad(image, pad_width=1, mode="constant", constant_values=0)
    rows, columns = image_pad.shape
    boundariesImage = np.zeros(image.shape)
    boundary = []
    b0 = None
    bk = None
    c = 0
    for y in range(rows):
        for x in range(columns):
            if image_pad[y][x] == 255:
                b0 = [y, x]
                boundary.append([y, x])
            if(b0 is not None):
                break
        if(b0 is not None):
                break
    b = b0
    while b0 != bk:
        neighbors, first = find_moore_neighborhood(b,c)
        if first is not None:
            bk = neighbors.get(first, None)
            if bk is not None:
                c = (first + 6)%8
                boundary.append(bk)
                b = bk
            else:
                break
        else:
            break
    boundary = boundary[:-1]
    for i in range(len(boundary)):
        boundary[i][0] = boundary[i][0] -1
        boundary[i][1] = boundary[i][1] -1
        boundariesImage[boundary[i][0]][boundary[i][1]] = 255
    return boundariesImage,boundary

