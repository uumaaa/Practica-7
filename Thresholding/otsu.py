import numpy as np
import matplotlib.pyplot as plt

def otsu(img):

    # Calculate the histogram
    hist = np.histogram(img, bins=256, range=(0, 255))[0]

    # Calculate total number of pixels
    total = np.sum(hist)

    ThresholdList = np.arange(0, 256)
    var_intra = np.zeros(256)

    for Threshold in ThresholdList:
        # Calculate background weight
        w_bg = np.sum(hist[0:Threshold]) / total if total > 0 else 0
        # Calculate foreground weight
        w_fg = np.sum(hist[Threshold:256]) / total if total > 0 else 0
        # Calculate background mean
        m_bg = (np.sum(hist[0:Threshold] * np.arange(0, Threshold)) / np.sum(hist[0:Threshold])) if np.sum(hist[0:Threshold]) > 0 else 0
        # Calculate foreground mean
        m_fg = (np.sum(hist[Threshold:256] * np.arange(Threshold, 256)) / np.sum(hist[Threshold:256])) if np.sum(hist[Threshold:256]) > 0 else 0
        # Calculate variance of background
        var_bg = (np.sum(hist[0:Threshold] * (np.arange(0, Threshold) - m_bg)**2) / np.sum(hist[0:Threshold])) if np.sum(hist[0:Threshold]) > 0 else 0
        # Calculate variance of foreground
        var_fg = (np.sum(hist[Threshold:256] * (np.arange(Threshold, 256) - m_fg)**2) / np.sum(hist[Threshold:256])) if np.sum(hist[Threshold:256]) > 0 else 0

        # Calculate intra-class variance
        var_intra[Threshold] = w_bg * var_bg + w_fg * var_fg    

    # Get the threshold that minimizes intra-class variance
    Threshold = np.argmin(var_intra)

    return Threshold


def otsu_image(img):

    # Calculate the histogram
    hist = np.histogram(img, bins=256, range=(0, 255))[0]

    # Calculate total number of pixels
    total = np.sum(hist)

    ThresholdList = np.arange(0, 256)
    var_intra = np.zeros(256)

    for Threshold in ThresholdList:
        # Calculate background weight
        w_bg = np.sum(hist[0:Threshold]) / total if total > 0 else 0
        # Calculate foreground weight
        w_fg = np.sum(hist[Threshold:256]) / total if total > 0 else 0
        # Calculate background mean
        m_bg = (np.sum(hist[0:Threshold] * np.arange(0, Threshold)) / np.sum(hist[0:Threshold])) if np.sum(hist[0:Threshold]) > 0 else 0
        # Calculate foreground mean
        m_fg = (np.sum(hist[Threshold:256] * np.arange(Threshold, 256)) / np.sum(hist[Threshold:256])) if np.sum(hist[Threshold:256]) > 0 else 0
        # Calculate variance of background
        var_bg = (np.sum(hist[0:Threshold] * (np.arange(0, Threshold) - m_bg)**2) / np.sum(hist[0:Threshold])) if np.sum(hist[0:Threshold]) > 0 else 0
        # Calculate variance of foreground
        var_fg = (np.sum(hist[Threshold:256] * (np.arange(Threshold, 256) - m_fg)**2) / np.sum(hist[Threshold:256])) if np.sum(hist[Threshold:256]) > 0 else 0

        # Calculate intra-class variance
        var_intra[Threshold] = w_bg * var_bg + w_fg * var_fg    

    # Get the threshold that minimizes intra-class variance
    Threshold = np.argmin(var_intra)
    img[img < Threshold] = 0
    img[img >= Threshold] = 255
    return img