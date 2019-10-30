#!/usr/bin/python
#image_enhance performs image measurements of noise levels and automated enhancements
#
#
#
#----------------------------------------------------------------------------
## CDeep3M -- NCMIR/NBCR, UCSD -- Author: M Haberl -- Date: 05/2019
#-----------------------------------------------------------------------------

import sys
#import getopt
import os
#import h5py
#from PIL import Image
#import numpy as np
import skimage
from joblib import Parallel, delayed
from read_files_in_folder import read_files_in_folder
#import cv2


"""

def main(argv):
   inputfolder = ''
   outputfolder = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('image_enhance.py -i <inputfolder> -e <enhancement> -o <outputfolder>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('image_enhance.py -i <inputfolder> -e <enhancement> -o <outputfolder>')
         sys.exit()
      elif opt in ("-i", "--ifolder"):
         inputfolder = arg
      elif opt in ("-e", "--enhance"):
         enhance_op = arg
      elif opt in ("-o", "--ofolder"):
         outputfolder = arg
#   print 'Input file is "', inputfile
#   print 'Output file is "', outputfile
#   print 'Performing:"',enhance_op


if __name__ == "__main__":
   main(sys.argv[1:])

"""

print(sys.argv)
inputfolder = sys.argv[1]
outputfolder = sys.argv[2]
os.mkdir(outputfolder)
file_list = read_files_in_folder(inputfolder)[0]
sys.stdout.write('Processing ' + str(len(file_list)) + ' images \n')

def processInput(x):
    file_in = os.path.join(inputfolder, file_list[x])
    sys.stdout.write('Loading: ' + str(file_in) + '\n')
    img = skimage.io.imread(file_in)
    sigma_est = skimage.restoration.estimate_sigma(skimage.img_as_float(img))
    print(file_in + ": Estimated Gaussian noise standard deviation = {}".format(sigma_est))
    #img = skimage.filters.gaussian(img, sigma=1, output=None, mode='nearest', cval=0, multichannel=None, preserve_range=False, truncate=4.0)
    img = skimage.restoration.denoise_tv_chambolle(img, weight=sigma_est, multichannel=False)
    #img = skimage.restoration.denoise_tv_bregman(img, weight=0.2, max_iter=100, eps=0.001, isotropic=True);
    img = skimage.exposure.rescale_intensity(img)
    file_out = os.path.join(outputfolder, file_list[x])
    sigma_est = skimage.restoration.estimate_sigma(skimage.img_as_float(img))
    print(file_out + ": Estimated Gaussian noise standard deviation = {}".format(sigma_est))
    sys.stdout.write('Saving: ' + str(file_out) + '\n')
    skimage.io.imsave(file_out, img)

p_tasks = 5
sys.stdout.write('Running ' + str(p_tasks) + ' parallel tasks\n')
results = Parallel(n_jobs=p_tasks)(delayed(processInput)(i) for i in range(0, len(file_list)))
