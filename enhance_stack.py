#!/usr/bin/python3
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
import numpy as np
import skimage
import skimage.util
import skimage.restoration
import skimage.exposure
import skimage.io
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import cv2
import prox_tv as ptv
from read_files_in_folder import read_files_in_folder
import configs.check_limits

sys.stdout.write('Runnning image enhancements\n')
#print(sys.argv)
inputfolder = sys.argv[1]
outputfolder = sys.argv[2]
cutperc = 2
if len(sys.argv) > 3:
    cutperc = int(sys.argv[3])
os.mkdir(outputfolder)
file_list = read_files_in_folder(inputfolder)[0]
sys.stdout.write('Processing ' + str(len(file_list)) + ' images \n')
sys.stdout.write('Removing ' + str(cutperc) + ' percentile of grey values \n')
cpu_limits = configs.check_limits.cpus()
if cpu_limits['enhance_stack'] > 0:
    p_tasks = min(len(file_list), cpu_limits['enhance_stack'])
else:
    p_tasks = max(1, min(len(file_list), int(cpu_count()/2)))
if cpu_limits['enhance_stack_threadlimit'] > 0:
    num_threads = cpu_limits['enhance_stack_threadlimit']
else:
    num_threads = int(round((cpu_count() / p_tasks), ndigits = None))
def processInput(x):
    file_in = os.path.join(inputfolder, file_list[x])
    sys.stdout.write('Loading: ' + str(file_in) + ' -> ')
    img = cv2.imread(file_in, cv2.IMREAD_UNCHANGED)
    sys.stdout.write('Type: ' + str(img.dtype) + '\n')
    # Check 3rd dimension here, if loaded as RGB, remove 3rd dimension here
    if len(img.shape) > 2:
        #print('Converting RGB  to grey level image')
        img = img[:, :, 0]
    try:
        img = skimage.util.img_as_float(img)
    except:
        img = skimage.img_as_float64(img)
    # remove extreme outlier pixels before denoising
    img = skimage.exposure.rescale_intensity(img, in_range=(np.percentile(img, 1), np.percentile(img, 99)), out_range=(0, 1))
    sigma_est1 = skimage.restoration.estimate_sigma(skimage.img_as_float(img))
    img = ptv.tv1_2d(img, sigma_est1/2, n_threads=num_threads)
    #img = skimage.restoration.denoise_tv_chambolle(img, weight=sigma_est1/2, multichannel=False)
    #img = skimage.restoration.denoise_tv_bregman(img, weight=sigma_est1/2, max_iter=100, eps=0.001, isotropic=True);
    img = skimage.exposure.rescale_intensity(img, in_range=(np.percentile(img, cutperc), np.percentile(img, 100-cutperc)), out_range=(0, 1))
    file_out = os.path.join(outputfolder, file_list[x])
    sigma_est2 = skimage.restoration.estimate_sigma(skimage.img_as_float(img))
    #sys.stdout.write(file_out + ": Estimated Gaussian noise stdev before " + str(sigma_est1) + " vs after denoising = " + str(sigma_est2))
    sys.stdout.write('Saving: ' + str(file_out) + '\n')
    img = 255 * img
    img = img.astype(np.uint8)
    cv2.imwrite(file_out, img)

sys.stdout.write('Running ' + str(p_tasks * num_threads) + ' parallel threads\n')
results = Parallel(n_jobs=p_tasks)(delayed(processInput)(i) for i in range(0, len(file_list)))
sys.stdout.write('Image enhancements completed\n')
sys.stdout.write('Enhanced images are stored in' + str(outputfolder) + '\n')
