#!/usr/bin/python3
# generates overlays with the original image
#
# syntax: python3 create_overlays.py inputfolder outputfolder
# Inputarguments: inputfolder_segmentation inputfolder_rawimages outputfolder
# Optional argument: none
# Output: RGB images saved in outputfolder
#
#----------------------------------------------------------------------------
## CDeep3M -- NCMIR/NBCR, UCSD -- Author: M Haberl -- Date: 10/2019
#-----------------------------------------------------------------------------

import sys
import os
import numpy as np
import skimage.io
import skimage.exposure
import skimage
from joblib import Parallel, delayed
import cv2
from read_files_in_folder import read_files_in_folder


print(sys.argv)
inputfolder_seg = sys.argv[1]
inputfolder_img = sys.argv[2]
outputfolder = sys.argv[3]

os.mkdir(outputfolder)
file_list_raw = read_files_in_folder(inputfolder)[0]
sys.stdout.write('Processing ' + str(len(file_list_raw)) + ' images \n')
file_list_seg = read_files_in_folder(inputfolder_seg)[0]

def processInput(x):
    file_in = os.path.join(inputfolder_seg, file_list_raw[x])
    sys.stdout.write('Loading: ' + str(file_in) + ' -> ')
    raw_image = cv2.imread(file_in, -1)
    
    file_in = os.path.join(inputfolder_img, file_list_seg[x])
    sys.stdout.write('Loading: ' + str(file_in) + ' -> ')
    seg = cv2.imread(file_in, -1)    
    seg = np.uint8(seg)
    seg = np.clip(seg, 40, 255)
    seg = skimage.exposure.rescale_intensity(seg, in_range=(40, 215), out_range=(0, 255))
    rgb_seg = np.dstack((np.zeros_like(seg), seg, np.zeros_like(seg))) #convert to RGB for overlay
    raw_image = np.uint8(raw_image)
    raw_image = skimage.exposure.rescale_intensity(raw_image, in_range=(0, 235), out_range=(0, 255))
    rgb_img = np.dstack((raw_image, raw_image, raw_image)) #convert to RGB image for overlay
    overlayed = rgb_img # initialize
    overlayed = np.uint8(np.uint8(0.60* rgb_img) + np.uint8(0.40* rgb_seg))
    overlayed = skimage.exposure.rescale_intensity(overlayed, in_range=(0, 225), out_range=(0, 255))
    
    file_out = os.path.join(outputfolder, file_list_seg[x])
    sys.stdout.write('Saving: ' + str(file_out) + '\n')
    skimage.io.imsave(file_out, overlayed)


p_tasks = 5
sys.stdout.write('Running ' + str(p_tasks) + ' parallel tasks\n')
results = Parallel(n_jobs=p_tasks)(delayed(processInput)(i) for i in range(0, len(file_list)))