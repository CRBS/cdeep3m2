# EnsemblePredictions
# different predictions coming from files e.g. from 1fm 3fm and 5fm will be averaged here
# flexible number of inputs
# last argument has to be the outputdirectory where the average files are stored
#
# -----------------------------------------------------------------------------
# NCMIR, UCSD -- Author: M Haberl -- Data: 10/2017 -- Update: 10/2018
# -----------------------------------------------------------------------------
#

# Initialize

import os
import sys
import time
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import numpy as np
#import skimage
import cv2
from read_files_in_folder import read_files_in_folder
import configs.check_limits

tic = time.time()

if len(sys.argv) < 3:
    print('Please specify more than one input directory to average: ')
    print('EnsemblePredictions ./inputdir1 ./inputdir2 ./inputdir3 ./outputdir\n')
    sys.exit()

png_list = [None] * (len(sys.argv) - 2)
for i in range(1, len(sys.argv) - 1):
    if not os.path.isdir(sys.argv[i]):
        print(
            '%s not a directory\nPlease check if predictions ran successfully or ensure to use: EnsemblePredictions ./inputdir1 ./inputdir2 ./inputdir3 ./outputdir\n' %
            (sys.argv[i]))
        sys.exit()
    png_list[i -
             1] = [f for f in read_files_in_folder(sys.argv[i])[0] if f.endswith('.png')]

outputdir = sys.argv[len(sys.argv) - 1]
os.mkdir(outputdir)

# =============== Generate ensemble predictions ==========================

total_zplanes = len(png_list[0])

def ensembleImgs(z):
    prob_maps = [cv2.imread(os.path.join(sys.argv[proc + 1], png_list[proc][z]), -1)
                  for proc in range(0, len(png_list))]  # Cumulate all average predictions of this plane
    ens_map = np.uint8(np.round(np.mean(prob_maps, 0)))

    save_filename = os.path.join(outputdir, png_list[0][z])
    print('Saving Image # %d of %d: %s\n' %
          (z + 1, total_zplanes, save_filename))
    cv2.imwrite(save_filename, ens_map)
    #try:
    #    skimage.io.imsave(save_filename, prob_map, as_grey=True)
    #except BaseException:
    #    skimage.io.imsave(save_filename, prob_map)

cpu_limits = configs.check_limits.cpus()
if cpu_limits['EnsemblePredictions'] > 0:
    p_tasks = cpu_limits['EnsemblePredictions']
else:
    p_tasks = max(1, int(cpu_count() - 2))
Parallel(n_jobs=p_tasks)(delayed(ensembleImgs)(z)
                             for z in range(0, total_zplanes))

print('Elapsed time for merging predictions is %06d seconds.\n' %
      (np.round(time.time() - tic)))
