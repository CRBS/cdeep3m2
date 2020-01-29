#!/usr/bin/python
# image_enhance performs image measurements of noise levels and automated enhancements
#
# syntax: python3 image_enhance.py inputfolder outputfolder 3
# Inputarguments: inputfolder outputfolder
# Optional argument: integer value defining percentile to cut for stretching the histogram to remove outlier pixels
# Output: 8 bit images saved in outputfolder
#
#----------------------------------------------------------------------------
## CDeep3M -- NCMIR/NBCR, UCSD -- Author: M Haberl -- Date: 10/2019
#-----------------------------------------------------------------------------

import sys
import os
#import h5py
#from PIL import Image
import numpy as np
import skimage
from joblib import Parallel, delayed
from read_files_in_folder import read_files_in_folder
import cv2


print(sys.argv)
inputfolder = sys.argv[1]
outputfolder = sys.argv[2]
cutperc = 2
if len(sys.argv) > 3:
    cutperc = int(sys.argv[3])
sys.stdout.write('Removing ' + str(cutperc) + ' percentile of grey values \n')
os.mkdir(outputfolder)
file_list = read_files_in_folder(inputfolder)[0]
sys.stdout.write('Processing ' + str(len(file_list)) + ' images \n')

def processInput(x):
    file_in = os.path.join(inputfolder, file_list[x])
    sys.stdout.write('Loading: ' + str(file_in) + ' -> ')
    img = cv2.imread(file_in, cv2.IMREAD_UNCHANGED)
    sys.stdout.write('Type: ' + str(img.dtype) + '\n')
    print(str(img.shape) + '\n')
    # Check 3rd dimension here, if loaded as RGB, remove 3rd dimension here
    if len(img.shape) > 2:
        print('Converting RGB  to grey level image')
        img = img[:,:,0]
    img = skimage.img_as_float64(img)
    # remove extreme outlier pixels before denoising
    #if img.dtype~=uint8
    img = skimage.exposure.rescale_intensity(img, in_range=(np.percentile(img, 1), np.percentile(img, 99)), out_range=(0, 1))
    sigma_est = skimage.restoration.estimate_sigma(skimage.img_as_float(img))
    print(file_in + ": Estimated Gaussian noise standard deviation before denoising = {}".format(sigma_est))
    #img = skimage.filters.gaussian(img, sigma=1, output=None, mode='nearest', cval=0, multichannel=None, preserve_range=False, truncate=4.0)
    img = skimage.restoration.denoise_tv_chambolle(img, weight=sigma_est/2, multichannel=False)
    #img = skimage.restoration.denoise_tv_bregman(img, weight=0.2, max_iter=100, eps=0.001, isotropic=True);
    img = skimage.exposure.rescale_intensity(img, in_range=(np.percentile(img, cutperc), np.percentile(img, 100-cutperc)), out_range=(0, 1))
    file_out = os.path.join(outputfolder, file_list[x])
    sigma_est = skimage.restoration.estimate_sigma(skimage.img_as_float(img))
    print(file_out + ": Estimated Gaussian noise standard deviation after denoising = {}".format(sigma_est))
    sys.stdout.write('Saving: ' + str(file_out) + '\n')
    img = skimage.img_as_ubyte(img)
    skimage.io.imsave(file_out, img)
p_tasks = 5
sys.stdout.write('Running ' + str(p_tasks) + ' parallel tasks\n')
results = Parallel(n_jobs=p_tasks)(delayed(processInput)(i) for i in range(0, len(file_list)))
