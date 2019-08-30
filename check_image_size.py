
#check_image_size: to see how to break large image data
#
# 
# 
#----------------------------------------------------------------------------
## CDeep3M -- NCMIR/NBCR, UCSD -- Author: M Haberl -- Date: 02/2019
#-----------------------------------------------------------------------------

import os
import h5py
from PIL import Image
import numpy as np
import skimage
from read_files_in_folder import read_files_in_folder

Image.MAX_IMAGE_PIXELS=10000000000000

def check_image_size(img_path):
    print ('Check image size of: ', img_path)
    #check if a folder of png/tif files or a single stack to load
    if os.path.isfile(img_path) == 1:
        filename, file_extension = os.path.splitext(img_path)
        if file_extension == '.h5':
            print ('Reading H5 image file')
            data = h5py.File(img_path, 'r')
            keys = list(data.keys())
            imagesize =  data[keys[0]].shape
            return imagesize

        elif file_extension == '.tif':
            im = Image.open(img_path)
            imarray = np.array(im)
            imagesize = imarray.shape
            im.close()
            return imagesize
        '''
        elif file_extension == '.png':
            im = Image.open(img_path)
            imarray = np.array(im)
            imagesize = imarray.shape
            return imagesize
        '''
    elif os.path.isdir(img_path) == 1:
        file_list = read_files_in_folder(img_path)[0]
        png_list = [f for f in file_list if f.endswith('.png')]
        tif_list = [f for f in file_list if f.endswith('.tif')]
        if len(tif_list) + len(png_list) == 0:
            print ('No Tifs or PNGs found in the directory')
            return
        else:
            #only read tif or pngs if ambiguous
            if len(png_list) > len(tif_list):
                file_list = png_list
            else:
                file_list = tif_list

            filename = os.path.join(img_path, file_list[0])
            print ('Reading file: ', filename)
            imarray = skimage.io.imread(filename)
            imagesize = (imarray.shape[0], imarray.shape[1], len(file_list))
            return imagesize

    else:
        raise Exception ('No images found')





