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
import shutil
import numpy as np
import skimage
import skimage.util
import skimage.restoration
import skimage.exposure
import skimage.io
from multiprocessing import cpu_count
from joblib import dump, load
from joblib import Parallel, delayed
import cv2
import prox_tv as ptv
from read_files_in_folder import read_files_in_folder
import configs.check_limits


def processInput(x, **kwargs):
    """
    Processes images based on the given input 1) List of filenames 2) z-index of stack mem-map

    if x is [str] the input to the script is assumed to be a folder of multiple .png/.tif image files

    if x is [int] the input to the script is assumed to be single tif stack
    """
    if isinstance(x, str):
        file_in = os.path.join(inputfolder, x)
        sys.stdout.write('Loading: ' + str(file_in) + ' -> ')
        name, extension = os.path.splitext(x)
        img = cv2.imread(file_in, cv2.IMREAD_UNCHANGED)
        file_out = os.path.join(outputfolder, str(name+'.png'))
    elif isinstance(x, int):
        mem_map_path = kwargs['mmap']
        stack_arr = load(mem_map_path, mmap_mode='r')
        sys.stdout.write('Loading: image layer no. ' + str(x+1) + ' -> ')
        img = stack_arr[x, :, :]
        name_with_path, extension = os.path.splitext(inputfolder)
        out_name = str(os.path.basename(name_with_path)) + "_" + str(x+1) + ".png"
        file_out = os.path.join(outputfolder, out_name)
    else:
        raise ValueError("Input supplied is incompatible")

    sys.stdout.write('Type: ' + str(img.dtype) + '\n')
    # Check 3rd dimension here, if loaded as RGB, remove 3rd dimension here
    if len(img.shape) > 2:
        # print('Converting RGB  to grey level image')
        img = img[:, :, 0]
    try:
        img = skimage.util.img_as_float(img)
    except:
        img = skimage.img_as_float64(img)
    # remove extreme outlier pixels before denoising
    img = skimage.exposure.rescale_intensity(img, in_range=(np.percentile(img, 1),
                                                            np.percentile(img, 99)), out_range=(0, 1))
    sigma_est1 = skimage.restoration.estimate_sigma(skimage.img_as_float(img))
    img = ptv.tv1_2d(img, sigma_est1/2, n_threads=num_threads)
    # img = skimage.restoration.denoise_tv_chambolle(img, weight=sigma_est1/2, multichannel=False)
    # img = skimage.restoration.denoise_tv_bregman(img, weight=sigma_est1/2, max_iter=100, eps=0.001, isotropic=True);
    img = skimage.exposure.rescale_intensity(img, in_range=(np.percentile(img, cutperc),
                                                            np.percentile(img, 100-cutperc)), out_range=(0, 1))
    sigma_est2 = skimage.restoration.estimate_sigma(skimage.img_as_float(img))
    # sys.stdout.write(file_out + ": Estimated Gaussian noise stdev before " + str(sigma_est1) + " vs after denoising = " + str(sigma_est2))
    sys.stdout.write('Saving: ' + str(file_out) + '\n')
    img = 255 * img
    img = img.astype(np.uint8)
    cv2.imwrite(file_out, img)


def configure_limits(num_images):
    cpu_limits = configs.check_limits.cpus()
    if cpu_limits['enhance_stack'] > 0:
        ptasks = min(num_images, cpu_limits['enhance_stack'])
    else:
        ptasks = max(1, min(num_images, int(cpu_count()/2)))
    if cpu_limits['enhance_stack_threadlimit'] > 0:
        n_threads = cpu_limits['enhance_stack_threadlimit']
    else:
        n_threads = int(round((cpu_count() / ptasks), ndigits=None))

    return ptasks, n_threads


#print(sys.argv)
inputfolder = sys.argv[1]
outputfolder = sys.argv[2]
cutperc = 2
if len(sys.argv) > 3:
    cutperc = int(sys.argv[3])

sys.stdout.write('Running image enhancements\n')

try:
    os.mkdir(outputfolder)
except FileExistsError:
    shutil.rmtree(outputfolder)
    os.mkdir(outputfolder)

dirpath, ext = os.path.splitext(inputfolder)
mem_map_folder = os.path.join(os.path.dirname(outputfolder), 'joblib_mmaps')

if ext:
    if ext.lower() in ['.tif', '.tiff']:
        retflag, stack_im_arr = cv2.imreadmulti(inputfolder, flags=cv2.IMREAD_UNCHANGED)
        if retflag is True:
            stack_im_arr = np.array(stack_im_arr)
            stack_shape = stack_im_arr.shape
            sys.stdout.write('Processing image stack with ' + str(stack_im_arr.shape[0]) + ' images \n')
            sys.stdout.write('Removing ' + str(cutperc) + ' percentile of grey values \n')
            try:
                os.mkdir(mem_map_folder)
            except FileExistsError:
                pass
            stack_mem_map = os.path.join(mem_map_folder, 'stack_mem_map')
            dump(stack_im_arr, stack_mem_map)  # dump a numpy mem_map to disk
            del stack_im_arr
            p_tasks, num_threads = configure_limits(stack_shape[0])
            data = load(stack_mem_map, mmap_mode='r')
            sys.stdout.write('Running ' + str(p_tasks * num_threads) + ' parallel threads\n')
            results = Parallel(n_jobs=p_tasks)(delayed(processInput)(i, mmap=stack_mem_map)
                                               for i in range(0, stack_shape[0]))
        else:
            raise Exception("Could not load the single .tif stack/file")
    else:
        raise Exception("Only multi-page tif/tiff stacks are supported for enhancement for now.")
elif os.path.isdir(inputfolder):
    file_list = read_files_in_folder(inputfolder)[0]
    sys.stdout.write('Processing ' + str(len(file_list)) + ' images \n')
    p_tasks, num_threads = configure_limits(len(file_list))
    sys.stdout.write('Running ' + str(p_tasks * num_threads) + ' parallel threads\n')
    results = Parallel(n_jobs=p_tasks)(delayed(processInput)(i) for i in file_list)
else:
    raise Exception("could not read the contents of foldr/file please check inputdir.")

sys.stdout.write('Image enhancements completed\n')
sys.stdout.write('Enhanced images are stored in' + str(outputfolder) + '\n')
# cleanup
if os.path.isdir(mem_map_folder):
    try:
        shutil.rmtree(mem_map_folder)
    except:
        print('Could not clean-up automatically.')
