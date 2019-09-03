import os
import sys
import h5py
import numpy as np
from PIL import Image as pilimage
import skimage


def imageimporter(img_path):
    """
imageimporter: loads image data from folder or from an individual image stack
-----------------------------------------------------------------------------
 CDeep3M -- NCMIR/NBCR, UCSD -- Date: 01/2019
-----------------------------------------------------------------------------"""

    print('Image importer loading ... ')
    print(img_path)
    # check if a folder of png/tif files or a single stack to load
    # [Dir,name,ext] = fileparts(img_path);
    name_with_path, ext = os.path.splitext(img_path)
    imgstack = []
    if ext:
        if '.h5' in ext:
            print('Reading H5 image file', img_path, '\n')
            h5file = h5py.File(img_path, 'r')
            imgstack = [np.stack((h5file[key].value,), axis=-1)
                        for key in h5file.keys()]
        elif '.tif' in ext:
            dataset = pilimage.open(img_path)
            print(
                'Reading image stack with', str(
                    dataset.n_frames), 'images\n')
            for i in range(dataset.n_frames):
                dataset.seek(i)
                imgstack.append(np.stack((np.array(dataset),), axis=-1))
    elif os.path.isdir(img_path):
        png_list = [f for f in os.listdir(img_path) if f.endswith('.png')]
        tif_list = [f for f in os.listdir(img_path) if f.endswith('.tif')]
        tif_list_len = len(tif_list)
        png_list_len = len(png_list)

        if tif_list_len + png_list_len == 0:
            print('No Tifs or PNGs found in training directory')
            return
        else:
            if tif_list_len > png_list_len:
                file_list = sorted(tif_list)
            else:
                file_list = sorted(png_list)

            for filename in file_list:
                print('Reading file:', img_path + filename)
            imgstack = [
                np.stack(
                    (np.array(
                        skimage.io.imread(
                            os.path.join(
                                img_path,
                                filename))),
                     ),
                    axis=-
                    1) for filename in file_list]
            # imgstack = np.array([pilimage.open(os.path.join(img_path,file_list[i])) for i in range(0, len(file_list))])
            shape = np.shape(imgstack)
            print(shape)
    else:
        raise Exception('No images found')
        return
    print(np.array(imgstack).shape)
    return np.array(imgstack)
