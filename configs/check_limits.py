#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads the json file with respective limitations
Currently used for CPUs only

#-----------------------------------------------------------------------------
## CDeep3M -- Author: M Haberl -- Date: 06/2020
#-----------------------------------------------------------------------------
"""

import os.path
import json

#json.load('./configs/cpu_resources.txt')
def cpus():
    if os.path.isfile('./configs/cpu_resources.txt'):
        with open('./configs/cpu_resources.txt') as json_file:
            cpu_limits = json.load(json_file)
    if os.path.isfile('/home/cdeep3m/configs/cpu_resources.txt'):
        with open('/home/cdeep3m/configs/cpu_resources.txt') as json_file:
            cpu_limits = json.load(json_file)
    return cpu_limits
