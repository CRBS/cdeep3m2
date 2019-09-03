
# ------------------------------------------------
# augment_package for CDeep3M -- NCMIR/NBCR, UCSD
# ------------------------------------------------
import os
import numpy as np
import h5py
from augment_data import augment_img


def augment_package(original, outsubdir, fmnumber, speed):
    allowed_speed = [1, 2, 4, 10]
    if speed not in allowed_speed:
        speed = allowed_speed[abs(np.array(allowed_speed) - speed).argmin()]

    if speed == 10:
        augment_choices = {
            1: [1],
            3: [1],
            5: [1]
        }
    elif speed == 1:
        augment_choices = {
            1: [1, 2, 3, 4, 5, 6, 7, 8],
            3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        }
    elif speed == 2:
        augment_choices = {
            1: [1, 6, 11, 15],
            3: [1, 2, 3, 4, 13, 14, 15, 16],
            5: [5, 6, 7, 8, 9, 10, 11, 12]
        }
    elif speed == 4:
        augment_choices = {
            1: [2, 3, 6, 7],
            3: [7, 8, 10, 12],
            5: [1, 6, 11, 15]
        }

    do_var = augment_choices.get(fmnumber)

    original_flip = np.flip(original, 0)
    d_details = '/data'

    for i in do_var:
        if i < 9:
            stack_out = augment_img(original, i - 1)
        else:
            stack_out = augment_img(original_flip, i - 9)

        print('Create Hd5 file Variation ', str(i))

        stack_out = stack_out.transpose([1, 2, 0])

        filename = os.path.join(outsubdir, 'image_stacks_v%d.h5' % (i))
        print('Saving: ', filename)
        hdf5_file = h5py.File(filename, mode='w')
        hdf5_file.create_dataset(d_details, data=stack_out)
        hdf5_file.close()
