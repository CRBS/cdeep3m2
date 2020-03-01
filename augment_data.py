"""
augment_data


-----------------------------------------------------------------------------
 NCMIR, UCSD -- CDeep3M -- Last Update: 08/2019
-----------------------------------------------------------------------------


"""

import numpy as np
from factor import factor
from SecondaryOperations import (
    HighContrast,
    LowContrast,
    Blur,
    Sharpen,
    UniformNoise,
    TV_Chambolle,
    HistogramEqualization,
    Skew,
    ElasticDistortion,
    Resize)


def augment_img(img_in, i=0):
    print('\nCreate variation {0} and {1}'.format(str(i), str(i + 8)))
    augment_choices = {
        0: (img_in),
        1: (np.flip(img_in, 1)),
        2: (np.flip(img_in, 2)),
        3: (np.rot90(img_in, 1, (1, 2))),
        4: (np.rot90(img_in, -1, (1, 2))),
        5: (np.rot90(np.flip(img_in, 1), 1, (1, 2))),
        6: (np.rot90(np.flip(img_in, 2), 1, (1, 2))),
        7: (np.rot90(img_in, 2, (1, 2)))
    }
    img_out = augment_choices.get(i, (img_in))

    return img_out


def augment_data(img_in, lbl_in, i=0):
    augment_choices = {
        0: (img_in, lbl_in),
        1: (np.flip(img_in, 1), np.flip(lbl_in, 1)),
        2: (np.flip(img_in, 2), np.flip(lbl_in, 2)),
        3: (np.rot90(img_in, 1, (1, 2)), np.rot90(lbl_in, 1, (1, 2))),
        4: (np.rot90(img_in, -1, (1, 2)), np.rot90(lbl_in, -1, (1, 2))),
        5: (np.rot90(np.flip(img_in, 1), 1, (1, 2)), np.rot90(np.flip(lbl_in, 1), 1, (1, 2))),
        6: (np.rot90(np.flip(img_in, 2), 1, (1, 2)), np.rot90(np.flip(lbl_in, 2), 1, (1, 2))),
        7: (np.rot90(img_in, 2, (1, 2)), np.rot90(lbl_in, 2, (1, 2)))
    }

    (img_out, lb_out) = augment_choices.get(i, (img_in, lbl_in))
    return img_out, lb_out


# build pipelines for secondary augmentations
def addtl_augs(strength, img_in, lbl_in, i=0):
    print('\nCreate additional variation {0}'.format(str(i + 1)))
    addtl_choices = {
        0: img_in,
        1: HighContrast,
        2: LowContrast,
        3: Sharpen,
        4: Blur,
        5: UniformNoise,
        6: TV_Chambolle,
        7: HistogramEqualization,
        8: [img_in, Skew],
        9: [HighContrast, Skew],
        10: [LowContrast, Skew],
        11: [Sharpen, Skew],
        12: [Blur, ElasticDistortion],
        13: [UniformNoise, ElasticDistortion],
        14: [TV_Chambolle, ElasticDistortion],
        15: [HistogramEqualization, ElasticDistortion]
    }

    if factor(strength, i) == 0:
        img_out, lbl_out = img_in, lbl_in
    else:
        if i == 0:
            img_out, lbl_out = img_in, lbl_in
        elif i < 8:
            img_out, lbl_out = addtl_choices[i](
                img_in, factor(strength, i)), lbl_in
        else:
            if i == 8:
                img1 = img_in
            else:
                img1 = addtl_choices[i][0](img_in, factor(strength, i - 8))
            if i < 12:
                img_out, lbl_out = addtl_choices[i][1](
                    img1, lbl_in, factor(strength, 8))
            else:
                img_out, lbl_out = addtl_choices[i][1](
                    img1, lbl_in, factor(strength, 9))

    return img_out, lbl_out, addtl_choices


# build pipelines for resizing
def third_augs(strength, img_in, lbl_in, i):
    print('\nApplying tertiary augmentation to stack {0}'.format(str(i + 1)))
    if strength == '0':
        img_out, lbl_out = img_in, lbl_in
    else:
        if i % 2 == 0:
            img_out, lbl_out = img_in, lbl_in
        # downscale stacks 1, 3, 5, 7
        elif i < 8:
            img_out, lbl_out = Resize(img_in, lbl_in, strength, 0)
        # upscale stacks 9, 11, 13, 15
        else:
            img_out, lbl_out = Resize(img_in, lbl_in, strength, 1)

    return img_out, lbl_out
