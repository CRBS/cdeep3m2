# imageimporter_large: loads subarea of large image data
# from folder or from an individual image stack
#
#
#
# -----------------------------------------------------------------------------
# CDeep3M -- NCMIR/NBCR, UCSD -- Author: M Haberl -- Date: 03/2019
# -----------------------------------------------------------------------------
import os
import h5py
import cv2
import numpy as np
from PIL import Image as pilimage
from read_files_in_folder import read_files_in_folder
from crop_png import crop_png


def imageimporter_large(img_path, area, z_stack, outfolder):
    print('Image importer loading ... ')
    print(img_path)

    # check if a folder of png/tif files or a single stack to load
    name_with_path, ext = os.path.splitext(img_path)

    if ext:
        if '.h5' in ext:
            h5file = h5py.File(img_path, 'r')
            imgstack = [h5file[key].value for key in h5file.keys()]
            imgstack = imgstack[area[0]:area[1],
                                area[2]:area[3], z_stack[0]:z_stack[1]]
            print('Processed size:', imgstack.shape)
        elif ext.lower() in ['.tif', '.tiff']:
            retflag, im = cv2.imreadmulti(img_path, flags=cv2.IMREAD_UNCHANGED)
            if retflag is True:
                imarray = np.array(im)
                # imarray = np.expand_dims(imarray, axis=len(imarray.shape))
                imgstack = imarray[z_stack[0]:z_stack[1], area[0]:area[1],
                                   area[2]:area[3]]
                imagesize = imgstack.shape
                print('Successfully read image stack with', str(
                    imagesize[0]), 'images\n')
            else:
                raise Exception("Something went wrong while loading the multi-page TIF file")

    elif os.path.isdir(img_path):
        file_list = read_files_in_folder(img_path)[0]
        png_list = [f for f in file_list if f.lower().endswith('.png')]
        tif_list = [f for f in file_list if f.lower().endswith(('.tif', '.tiff'))]
        tif_list_len = len(tif_list)
        png_list_len = len(png_list)

        if tif_list_len + png_list_len == 0:
            print('No Tifs or PNGs found in training directory')
            return
        else:
            if tif_list_len > png_list_len:
                file_list = tif_list
            else:
                file_list = png_list

            tempdir = os.path.join(outfolder, 'temp')
            if not os.path.isdir(tempdir):
                os.mkdir(tempdir)

            tempmat_infile = os.path.join(tempdir, 'infiles.txt')
            with open(tempmat_infile, 'w') as f:
                for fl in range(z_stack[0], z_stack[1]):
                    f.write(os.path.join(img_path, file_list[fl]) + '\n')

            tempmat_outfile = os.path.join(tempdir, 'outfiles.txt')

            with open(tempmat_outfile, 'w') as f:
                for fl in range(z_stack[0], z_stack[1]):
                    f.write(os.path.join(
                        tempdir, file_list[fl][:-3] + 'tif') + '\n')

            crop_png(
                tempmat_infile,
                tempmat_outfile,
                area[0],
                area[1],
                area[2],
                area[3])

            print('Reading images')
            imgstack = np.array([cv2.imread(os.path.join(
                tempdir, file_list[i][:-3] + 'tif'), -1) for i in range(z_stack[0], z_stack[1])])
            # shape = np.shape(imgstack)
            # print (shape)

    else:
        raise Exception('No images found')
        return

    # Add padding
    # Left and upper side
    if area[0] == 0 and imgstack.shape[1] <= 1012:  # first in y
        imgstack = np.concatenate(
            (np.flipud(imgstack[:, 1:13, :]), imgstack), axis=1)
    if area[2] == 0 and imgstack.shape[2] <= 1012:  # then in x
        imgstack = np.concatenate(
            (np.fliplr(imgstack[:, :, 1:13]), imgstack), axis=2)

    x_size = imgstack.shape[1]
    y_size = imgstack.shape[2]

    # Right and lower end

    if x_size < 1024:
        max_padsize = 1024 - x_size
        max_padsize = min(max_padsize, 12)
        imgstack = np.concatenate((imgstack, np.flipud(
            imgstack[:, x_size - max_padsize - 1:x_size - 1, :])), axis=1)

    if y_size < 1024:
        max_padsize = 1024 - y_size
        max_padsize = min(max_padsize, 12)
        imgstack = np.concatenate((imgstack, np.fliplr(
            imgstack[:, :, y_size - max_padsize - 1:y_size - 1])), axis=2)

    # Add zeros to fill 1024*1024 Image size
    x_size = imgstack.shape[1]
    y_size = imgstack.shape[2]

    if x_size < 1024 or y_size < 1024:
        temp_img = np.zeros(
            (imgstack.shape[0], 1024, 1024), dtype=imgstack.dtype)
        temp_img[:, 0:imgstack.shape[1], 0:imgstack.shape[2]] = imgstack
        imgstack = temp_img

    return imgstack
