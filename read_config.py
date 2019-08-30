#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:21:46 2019

@author: jihyeonje
"""

from configparser import ConfigParser


def read_cfg(cfg_file):
    config = ConfigParser()
    config.read(cfg_file)

    power = []
    scale = config['Additional Augmentations']
    power.append(int(scale['HighContrast']))
    power.append(int(scale['LowContrast']))
    power.append(int(scale['Blur']))
    power.append(int(scale['Sharpen']))
    power.append(int(scale['UniformNoise']))
    power.append(int(scale['TV_Chambolle']))
    power.append(int(scale['HistogramEqualization']))
    power.append(int(scale['Skew']))
    power.append(int(scale['ElasticDistortion']))

    return power
