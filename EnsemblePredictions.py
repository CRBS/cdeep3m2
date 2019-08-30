## EnsemblePredictions
# different predictions coming from files e.g. from 1fm 3fm and 5fm will be averaged here
# flexible number of inputs
# last argument has to be the outputdirectory where the average files are stored
#
# -----------------------------------------------------------------------------
## NCMIR, UCSD -- Author: M Haberl -- Data: 10/2017 -- Update: 10/2018
# -----------------------------------------------------------------------------
#

## Initialize

import os
import sys
import time
import numpy as np 
import skimage
from read_files_in_folder import read_files_in_folder

tic = time.time()

if len(sys.argv) < 3: 
    print('Please specify more than one input directory to average: EnsemblePredictions ./inputdir1 ./inputdir2 ./inputdir3 ./outputdir\n')
    exit() 

png_list = [None]*(len(sys.argv)-2)
for i in range(1, len(sys.argv)-1):
    if not os.path.isdir(sys.argv[i]):
        print('%s not a directory\nPlease check if predictions ran successfully or ensure to use: EnsemblePredictions ./inputdir1 ./inputdir2 ./inputdir3 ./outputdir\n'%(sys.argv[i]))
        exit()
    png_list[i-1] = [f for f in read_files_in_folder(sys.argv[i])[0] if f.endswith('.png')]

outputdir = sys.argv[len(sys.argv)-1]
os.mkdir(outputdir)

## =============== Generate ensemble predictions =================================

total_zplanes = len(png_list[0])

for z in range(0, total_zplanes):
    image_list = [skimage.io.imread(os.path.join(sys.argv[proc+1], png_list[proc][z])) 
                  for proc in range(0, len(png_list))]  #Cumulate all average predictions of this plane
    
    prob_map = np.uint8(np.round(np.mean(image_list, 0)))

    save_file_save = os.path.join(outputdir, png_list[0][z])
    print('Saving Image # %d of %d: %s\n'%(z+1, total_zplanes, save_file_save))
    try:
        skimage.io.imsave(save_file_save, prob_map, as_grey=True)
    except:
        skimage.io.imsave(save_file_save, prob_map)

print('Elapsed time for merging predictions is %06d seconds.\n'% (np.round(time.time()-tic)))
