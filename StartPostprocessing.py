# New Postprocessing
# Syntax: StartPostprocessing /example/seg1/predict/ /example/seg2/predict/
#
# Runtime estimate 2min for 1024x1024x100 dataset
#
#------------------------------------------------------------------
## New -- NCMIR/NBCR, UCSD -- Author: M Haberl -- Date: 10/2017
#------------------------------------------------------------------

import os
import sys
import time
import numpy as np
from merge_16_probs_v3 import merge_16_probs_v3

if len(sys.argv) < 2: 
    print('Please specify at least 1 input directory')
    print('Use -> StartPostprocessing /example/seg1/predict/ /example/seg2/predict/')
    exit() 

tic = time.time()

## Enable batch processing all predictions
print('Starting to merge de-augment data')
print('Starting to process %d datasets \n' % (len(sys.argv)-1))
for i in range(1, len(sys.argv)):
    inputdir = sys.argv[i]

    if not os.path.isdir(inputdir):
        raise Exception('%s not a input directory' %(inputdir))
        exit()

    print('Generating Average Prediction of %s\n' %(inputdir))
    average_prob_folder = merge_16_probs_v3(inputdir)

print('Elapsed runtime for data-deaugmentation: %04d seconds.\n' % (np.round(time.time()-tic)))

## Run Merge Predictions now
#MergePredictions 

## Run 3D Watershed if required
#if regexpi(arg_list{end},'water','once'):
#    readvars =1;   
#    Run_3DWatershed_onPredictions
#end

