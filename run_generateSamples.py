#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:08:37 2019

@author: jihyeonje
"""

from secondary_augmentations import run_secondary_augmentations
from read_config import num_of_pipelines, combinations


for i in range(0, num_of_pipelines):
    run_secondary_augmentations(num_of_pipelines, combinations)[i].sample(5)
