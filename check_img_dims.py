import numpy as np


def check_img_dims(imgstack, lblstack, minsize):
    """
 Check Canvas Size of training images and training labels
 to match same size and to fullfill min canvas size

----------------------------------------------------------------------------------------
 CDeep3M -- NCMIR/NBCR, UCSD -- Author: M Haberl -- Date: 03/2018
----------------------------------------------------------------------------------------"""

    print('Checking image dimensions')
    if imgstack.shape[1:] != lblstack.shape[1:]:
        raise Exception(
            'Image dimension mismatch in x/y between images and labels')
        return
    elif imgstack.shape[0] != lblstack.shape[0]:
        raise Exception(
            'Image dimension mismatch in z between images and labels')
        return

    x1 = imgstack.shape[1]
    y1 = imgstack.shape[2]
    z1 = imgstack.shape[0]

    if x1 < minsize:
        z = np.zeros((z1, minsize - x1, y1, 1), dtype=imgstack.dtype)
        imgstack = np.concatenate((imgstack, z), axis=1)
        lblstack = np.concatenate((lblstack, z), axis=1)

    x1 = imgstack.shape[1]
    if y1 < minsize:
        z = np.zeros((z1, x1, minsize - y1, 1), dtype=imgstack.dtype)
        imgstack = np.concatenate((imgstack, z), axis=2)
        lblstack = np.concatenate((lblstack, z), axis=2)

    return imgstack, lblstack
