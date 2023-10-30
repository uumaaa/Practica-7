import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def BersenThreshold(image: np.ndarray, radius: int, cmin: float, K: int = 128, bg:int =1) -> np.ndarray:
    M, N = image.shape
    q = K
    image_thresholded = np.copy(image).astype(np.uint8)
    for m in range(M):
        for n in range(N):
            neighborhood = image[max(0, m - radius):min(M, m + radius + 1), 
                                max(0, n - radius):min(N, n + radius + 1)]
            diff_value = np.max(neighborhood) - np.min(neighborhood)
            threshold = (np.max(neighborhood) + np.min(neighborhood)) / 2 if diff_value >= cmin else q
            if(bg == 1):
                image_thresholded[m, n] = 255 if image[m, n] > threshold else 0
            else:
                image_thresholded[m, n] = 0 if image[m, n] > threshold else 255
    return image_thresholded

