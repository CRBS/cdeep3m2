#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

# conv2uint8 rescales image contrast
# performs image measurements of noise levels and automated enhancements
#
# Syntax: conv2uint8.py inputfolder outputfolder
# optional: conv2uint8.py inputfolder outputfolder 3
#
# Input: images inside inputfolder, signed or unsigned
# optional input: single number, to define lower and upper percentile of greyvalues to clip
#
# Processing: rescales to positive values, crops lower and higher percentile (standard: 5% and 95%)
# Output: images, inside outputfolder, filenames match images in inputfolder
#
#----------------------------------------------------------------------------
## CDeep3M -- NCMIR, UCSD -- Author: M Haberl -- Date: 10/2019
#-----------------------------------------------------------------------------

"""

import sys
import os
import numpy as np
import skimage
from joblib import Parallel, delayed
from read_files_in_folder import read_files_in_folder


print(sys.argv)
inputfolder = sys.argv[1]
outputfolder = sys.argv[2]
cutperc = 5
if len(sys.argv) > 3:
    cutperc = int(sys.argv[3])
sys.stdout.write('Removing ' + str(cutperc) + ' percentile of grey values \n')
os.mkdir(outputfolder)
file_list = read_files_in_folder(inputfolder)[0]
sys.stdout.write('Processing ' + str(len(file_list)) + ' images \n')

def processInput(x):
    file_in = os.path.join(inputfolder, file_list[x])
    sys.stdout.write('Loading: ' + str(file_in) + '\n')
    img = skimage.io.imread(file_in)
    img_float64 = skimage.img_as_float64(img)
    image = skimage.exposure.rescale_intensity(img_float64, in_range=(np.percentile(img_float64, cutperc), np.percentile(img_float64, 100-cutperc)), out_range=(0, 1))
    img_uint8 = skimage.img_as_ubyte(image)
    file_out = os.path.join(outputfolder, file_list[x])
    sys.stdout.write('Saving: ' + str(file_out) + '\n')
    skimage.io.imsave(file_out, img_uint8)

p_tasks = 5
sys.stdout.write('Running ' + str(p_tasks) + ' parallel tasks\n')
results = Parallel(n_jobs=p_tasks)(delayed(processInput)(i) for i in range(0, len(file_list)))
