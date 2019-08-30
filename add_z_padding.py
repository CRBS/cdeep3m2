"""
Adds 2 planes in the begin and end of the image stack

NCMIR/NBCR, UCSD -- CDeep3M -- Last update: 08/2019

"""

import numpy as np
def add_z_padding(im_stack):
    print('Adding Z padding')
    print(np.shape(im_stack))
    im_stack =  np.concatenate((im_stack[2:3], im_stack[1:2], im_stack, im_stack[-2:-1], im_stack[-3:-2]), axis=0)
    print(np.shape(im_stack))
    return im_stack
