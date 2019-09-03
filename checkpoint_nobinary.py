import numpy as np


def checkpoint_nobinary(imagestack):
    if len(np.unique(imagestack[:])) < 3:
        print('Images are not 8 or 16bit')
        print('Please be sure you did not use binary labels by mistake here')
        reply = input(
            'Type S to stop image augmentation?  Otherwise images will be augmented now')
        if reply == 's' or reply == 'S':
            print('Augmentation cancelled')
