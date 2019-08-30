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
from ensemble import ensemble
import numpy as np
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
#raw_image_full_path = arg_list{end};

## =============== Generate ensemble predictions =================================

tempmat_infile = os.path.join(outputdir,'infolders.txt')

if os.path.isfile(tempmat_infile):
        os.remove(tempmat_infile)

with open(tempmat_infile, 'a') as fid:
	for fl in range(1, len(sys.argv)-1):
		fid.write(os.path.join(sys.argv[fl], '')+'\n')

ensemble([tempmat_infile, outputdir])

print('Elapsed time for merging predictions is %06d seconds.\n'% (np.round(time.time()-tic)))

