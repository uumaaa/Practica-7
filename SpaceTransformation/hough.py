import numpy as np
from Thresholding.otsu import otsu
from Border_detection import border_detection
import cv2
import math
def houghTransform_lines(image, error):

    # Define the values of theta (angle) and rho (distance from origin)
    theta_res = 1  # Resolution of theta in degrees
    rho_res = 1    # Resolution of rho in pixels

    # Define theta and rho ranges
    theta = np.deg2rad(np.arange(-90, 90, theta_res))
    height, width, _ = image.shape
    max_rho = int(np.sqrt(height**2 + width**2))
    rho = np.arange(-max_rho, max_rho, rho_res)

    # Create Hough space
    hough_space = np.zeros((2 * max_rho, len(theta)), dtype=np.uint64)

    # Find edges
    edges = border_detection.canny_bordering(image)

    # Get edge coordinates
    y_idx, x_idx = np.where(edges > 0)

    # Calculaten the Hough space
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        for t_idx in range(len(theta)):
            r = int(x * np.cos(theta[t_idx]) + y * np.sin(theta[t_idx]))
            hough_space[r + max_rho, t_idx] += 1
    
    # Get the max and min values of the Hough space
    max_value = int(np.max(hough_space))
    min_value = int(np.min(hough_space))

    # Calculate the threshold with Otsu's method getting the maximun intra-class variance
    threshold = otsu.otsu(hough_space) - (max_value) * error # 0 <= error <= 1
    # Get the coordinates of the peaks
    y_peaks, x_peaks = np.where(hough_space > threshold)

    return hough_space,y_peaks,x_peaks

def houghTrasnform_circles(image:np.ndarray,error:float,radius:float):
    
    # Define the values of theta (angle) and rho (distance from origin)
    theta_res = 1  # Resolution of theta in degrees

    # Define theta and rho ranges
    theta = np.deg2rad(np.arange(0, 360, theta_res))
    height, width, _ = image.shape

    # Create Hough space
    hough_space = np.zeros((2 * radius + height, 2 * radius + width), dtype=np.uint64)

    # Find edges
    edges = border_detection.canny_bordering(image)

    # Get edge coordinates
    y_idx, x_idx = np.where(edges > 0)

    # Calculaten the Hough space
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        for t_idx in range(len(theta)):
            a = round(x - radius * math.cos(t_idx))
            b = round(y - radius * math.sin(t_idx))
            hough_space[b,a] += 1
    
    # Get the max and min values of the Hough space
    max_value = np.max(hough_space)
    # Calculate the threshold with Otsu's method getting the maximun intra-class variance
    threshold = 140 # 0 <= error <= 1
    print(max_value)
    # Get the coordinates of the peaks
    y_peaks, x_peaks = np.where(hough_space > threshold)
    
    hough_space[hough_space >= threshold] == 255
    hough_space[hough_space < threshold] == 0

    return hough_space