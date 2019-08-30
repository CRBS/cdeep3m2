#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:07:02 2019

@author: jihyeonje
"""

import configparser

config = configparser.ConfigParser()

config['Additional Augmentations'] = {}
power = config['Additional Augmentations']
power['HighContrast'] = '3'
power['LowContrast'] = '3'
power['Blur'] = '3'
power['Sharpen'] = '4'
power['UniformNoise'] = '5'
power['TV_Chambolle'] = '3'
power['HistogramEqualization'] = '2'
power['Skew'] = '5'
power['ElasticDistortion'] = '5'

with open('augcfg.ini', 'w') as configfile:
    config.write(configfile)
