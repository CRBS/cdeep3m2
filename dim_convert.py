"""
Convert stack to h5 format for CDeep3M

NCMIR/NBCR, UCSD -- CDeep3M -- Update: 08/2019 @mhaberl

"""

import numpy as np


def dim_convert(img, lb):
    shape = img.shape
    print(shape)
    temp_img = np.zeros([shape[1], shape[2], shape[0]], dtype=img.dtype)
    temp_lb = np.zeros([shape[1], shape[2], shape[0]], dtype=lb.dtype)
    # temp_img = np.zeros([1024, 1024, shape[0]], dtype=img.dtype)
    # temp_lb = np.zeros([1024,1024, shape[0]], dtype=lb.dtype)
    for z in range(0, shape[0]):
        temp_img[0:shape[1], 0:shape[2], z] = np.squeeze(img[z, :, :])
        temp_lb[0:shape[1], 0:shape[2], z] = np.squeeze(lb[z, :, :])
    # print(temp_img.shape)
    # print(temp_lb.shape)
    img = np.squeeze(temp_img)
    lb = np.squeeze(temp_lb)
    # print(img.shape)
    # print(lb.shape)
    return img, lb
