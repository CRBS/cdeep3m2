#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:56:12 2019

@author: jihyeonje
"""
from read_config import read_cfg


def factor(strength, i=0):

    if strength.isdigit():
        power = [int(strength)] * 10
    else:
        power = read_cfg(strength)

    factor_choices = {
        0: 0,
        1: 1.25 + 0.25 * power[0],  # HighContrast
        2: 0.95 - 0.05 * power[1],  # LosContrast
        3: power[2],  # Blur
        4: 0.125 + 0.125 * power[3],  # Sharpen
        5: 0.05 + 0.015 * power[4],  # UniformNoise
        6: 0.1 * power[5],  # TV_Chambolle
        7: 261 - 5 * power[6],  # HistogramEqualization
        8: power[7],  # Skew
        9: 27 - power[8]  # ElasticDistortion

    }
    if strength == '0':
        return factor_choices.get(0)
    else:
        return factor_choices.get(i)
