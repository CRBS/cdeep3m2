"""
Adds 2 planes for z padding in the begin and end of the image stack

NCMIR, UCSD -- CDeep3M -- Last update: 11/2019
@MHaberl
"""

import numpy as np


def add_z_padding(im_stack):
    print('Adding Z padding')
    print(np.shape(im_stack))
    im_stack = np.concatenate((im_stack[:1],
                               im_stack[:1],
                               im_stack,
                               im_stack[-1:],
                               im_stack[-1:]),
                              axis=0)
    print(np.shape(im_stack))
    return im_stack
