#!/usr/bin/env python3
# PreprocessValidation.py
# Makes hdf5 validation file from raw images and corresponding labels
#
# Syntax : PreprocessValidation ~/validation/images/ ~/validation/labels/ ~/validation/combined
#
#
# ----------------------------------------------------------------------------------------
# PreprocessTraining for Deep3M -- NCMIR/NBCR, UCSD -- Author: M Haberl -- Date: 04/2018
# ----------------------------------------------------------------------------------------
#
# Runtime: <1 min
#
# ----------------------------------------------------------------------------------------
# Initialize
# ----------------------------------------------------------------------------------------
import os
import sys
import time
import h5py
import numpy as np
from imageimporter import imageimporter
from checkpoint_nobinary import checkpoint_nobinary
from checkpoint_isbinary import checkpoint_isbinary

print('Starting Validation data Preprocessing')

arg_list = sys.argv[1:]

second_aug = '-1'
if len(arg_list) < 3:
    print('Use -> PreprocessValidation.py /validation/images/ /validation/labels/ /validation/combined')
    print('Or without denoising: PreprocessValidation.py /validation/images/ /validation/labels/ 0 /validation/combined')
    sys.exit()
if len(arg_list) > 3:
    second_aug = arg_list[2]

tic = time.time()

valid_img_path = arg_list[0]
print('Validation Image Path:', valid_img_path)
label_img_path = arg_list[1]
print('Validation Label Path:', label_img_path)
outdir = arg_list[-1]
print('Output Path:', outdir)
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# ----------------------------------------------------------------------------------------
# apply denoising
# ----------------------------------------------------------------------------------------

if second_aug == '-1':
    print('Running image enhancement')
    enhanced_path = os.path.join(outdir, 'enhanced')
    run_enhancement = 'enhance_stack.py ' + valid_img_path + ' ' + enhanced_path + ' 2'
    os.system(run_enhancement)
    valid_img_path = enhanced_path

# ----------------------------------------------------------------------------------------
# Load images
# ----------------------------------------------------------------------------------------

print('Loading:')
print(valid_img_path)
imgstack = imageimporter(valid_img_path)
print('Verifying images')

# ----------------------------------------------------------------------------------------
# Load labels
# ----------------------------------------------------------------------------------------

print('Loading:')
print(label_img_path)
lblstack = imageimporter(label_img_path)
print('Verifying labels')
checkpoint_isbinary(lblstack)
lblstack = lblstack > 0

# ----------------------------------------------------------------------------------------
# Convert and save
# ----------------------------------------------------------------------------------------

img = imgstack.astype('float32')
lb = lblstack.astype('float32')

d_details = '/data'
l_details = '/label'

ext = ".h5"

filename = os.path.abspath(outdir) + '/' + \
    'validation_stack_v{0}{1}'.format(str(1), ext)
print('Saving: ', filename)
hdf5_file = h5py.File(filename, mode='w')
hdf5_file.create_dataset(d_details, data=img)
hdf5_file.create_dataset(l_details, data=lb)
hdf5_file.close()

# ----------------------------------------------------------------------------------------
# Completed
# ----------------------------------------------------------------------------------------

print('Validation data stored in %06d seconds.\n' %
      (np.round(time.time() - tic)))
