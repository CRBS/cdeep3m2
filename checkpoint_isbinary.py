import numpy as np


def checkpoint_isbinary(imagestack):
    if len(np.unique(imagestack[:])) > 2:
        print('Your labels do not seem to be binary files')
        reply = input(
            'Type S to stop label augmentation here?  Otherwise label augmentation will proceed now')
        if reply == 's' or reply == 'S':
            print('Augmentation cancelled')
