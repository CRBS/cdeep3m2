#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script will write configuration file to set limits to the CPU utilization if needed.

CDeep3M normally performs cpu counts and intends to increase speed by assigning CPUs for the different tasks.
Values set to 0 here mean that no manual limits are set and that the number of CPUs used are assigned automatically, all other values stand for the number of CPUs (or threads if applicable).

#-----------------------------------------------------------------------------
## CDeep3M -- Author: M Haberl -- Date: 06/2020
#-----------------------------------------------------------------------------
"""

#import os.path
import json

#os.path.isfile('./configs/cpu_limits.txt')

cpu_limits = {
        "enhance_stack": 1,
        "enhance_stack_threadlimit": 2,
        "crop_png": 0,
        "Merge_LargeData": 1,
        "EnsemblePredictions": 1,
        "ensemble": 1,
        "Create_overlays": 1
}
with open('./configs/cpu_resources.txt', 'w') as outfile:
    json.dump(cpu_limits, outfile)
