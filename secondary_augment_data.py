#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:51:48 2019

@author: jihyeonje
"""

# Secondary augmentations [1: light distortion, 10: heavy distortion]
# img_in = input pathway

from SecondaryOperations import (
    HighContrast,
    LowContrast,
    Blur,
    Sharpen,
    TV_Chambolle,
    TV_Bregman,
    UniformNoise,
    HistogramEqualization)
import strength
from read_config import probabilities, power, original
import new_Pipeline

"""
#operation order
1: High Contrast
2: Low Contrast
3: Blur
4: Sharpen
5: Uniform Noise
6: Total Variation - Chambolle
7: Total Variation - Bregman
8: Histogram Equalization

"""


def secondary_augment_data(i):

    secondary_augment_choices = {
        # high contrast
        1: HighContrast(probabilities[0], strength(power[0])[0]),
        # low contrast
        2: LowContrast(probabilities[1], strength(power[1])[1]),
        # blur
        3: Blur(probabilities[2], strength(power[2])[2]),
        # sharpen
        4: Sharpen(probabilities[3], strength(power[3])[3]),
        # uniform noise
        5: UniformNoise(probabilities[4], strength(power[4])[4]),
        # total variation_chambolle
        6: TV_Chambolle(probabilities[5], strength(power[5])[5]),
        # total variation_bregman
        7: TV_Bregman(probabilities[6], strength(power[6])[6]),
        # histogram equalization
        8: HistogramEqualization(probabilities[7], strength(power[7])[7])
    }

    secondary_aug_op = secondary_augment_choices.get(i)
    return secondary_aug_op


def run_secondary_augmentations(num_pipelines, operations):

    # execute 0 to 3 of the following augmenters per image
    # don't execute all of them, as that would often be way too strong
    piplist = []
    for i in range(0, num_pipelines):
        pipeline = new_Pipeline.Pipeline(emimg)
        piplist.append(pipeline)
        for j in range(len(operations[i])):
            pipeline.add_operation(secondary_augment_data(operations[i][j]))

    return piplist
