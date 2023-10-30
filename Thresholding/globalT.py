import numpy as np

def global_thresholding(img, T):
    DeltaT = 1 
    image_thresholded = np.copy(img).astype(np.uint8)
    while True:
        g1 = img[img > T]
        g2 = img[img <= T]

        m1 = np.mean(g1)
        m2 = np.mean(g2)

        new_T = 0.5 * (m1 + m2)


        if np.isnan(new_T):
            break
        
        if abs(T - new_T) < DeltaT: 
            break
        T = round(new_T)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] > T:
                image_thresholded[i][j] = 255
            else:
                image_thresholded[i][j] = 0
    print(F'Umbral global optimo: {T}')
    return image_thresholded

