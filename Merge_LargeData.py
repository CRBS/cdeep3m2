# Merge LargeData
#
# After segmentation of smaller image packages this
# script will stitch the initial dataset back together
# Assumes Packages are in the subdirectories of 1fm / 3fm / 5fm
# an expects a de_augmentation_info.mat in the parent directory thereof.
#
# Runs after StartPostProcessing which merges the 16variations
# and already removed z-padding.
#
#
# Use: Merge_LargeData ~/prediction/1fm
# expects de_augmentation_info.mat in the parent directory
#
# ------------------------------------------------------------------
# NCMIR/NBCR, UCSD -- Author: M Haberl -- Date: 10/2017
# ------------------------------------------------------------------

import os
import sys
import time
import json
import numpy as np
import skimage
import skimage.io
from read_files_in_folder import read_files_in_folder
from multiprocessing import cpu_count
from joblib import Parallel, delayed


def merge_images(z_plane):
    print('Merging image no. %s\n' % (str(z_plane)))

    # Initialize empty image in x/y 2 in  z
    merger_image = np.array(np.zeros(imagesize[0:2]))

    for x_y_num in range(0, len(packages)):
        packagedir = os.path.join(fm_dir, 'Pkg_%03d' % (x_y_num + 1))
        filename = os.path.join(packagedir, filelist[z_plane])
        small_patch = skimage.io.imread(filename)

        # bitdepth = single(2.^([1:16]));
        # [~,idx] = min(abs(bitdepth - max(small_patch(:))));
        # fprintf('Scaling %s bit image\n', num2str(idx));
        # save_plane = uint8((255 /bitdepth(idx))*combined_plane);
        # small_patch = single((255 /bitdepth(idx))*small_patch);
        # small_patch = single((255 /max(small_patch(:)))*small_patch);
        area = packages[x_y_num]
        if len(packages) > 1:

            corners = [area[0] + 12, area[1] - 12, area[2] + 12, area[3] - 12]
            if area[0] == 0:
                corners[0] = 0
            if area[1] == np.shape(merger_image)[0]:
                corners[1] = np.shape(merger_image)[0]

            if area[2] == 0:
                corners[2] = 0
            if area[3] == np.shape(merger_image)[1]:
                corners[3] = np.shape(merger_image)[1]

            if corners[1] > np.shape(merger_image)[0]:
                corners[1] = np.shape(merger_image)[0]

            if corners[3] > np.shape(merger_image)[1]:
                corners[3] = np.shape(merger_image)[1]

            insertsize = [corners[1] - corners[0], corners[3] - corners[2]]

            merger_image[corners[0]:corners[1], corners[2]:corners[3]
                         ] = small_patch[12:insertsize[0] + 12, 12:insertsize[1] + 12]

        else:  # if there is only one package
            start = [0, 0]
            if imagesize[0] <= 1012:  # define where the image has been padded
                start[0] = 12
            else:
                start[0] = 0

            if imagesize[1] <= 1012:  # define where the image has been padded
                start[1] = 12
            else:
                start[1] = 0

            # clear merger_image;
            merger_image = small_patch[start[0]:(
                imagesize[0] + start[0]), start[1]:(imagesize[1] + start[1])]

    bitdepth = [2**i for i in range(1, 17)]
    # print('Scaling %s bit image\n' %(num2str(idx)))
    idx = abs(np.array(bitdepth) - max(merger_image.flatten())).argmin()
    save_plane = np.uint8(np.round((255.0 / bitdepth[idx]) * merger_image))
    outfile = os.path.join(fm_dir, 'Segmented_%04d.png' % (z_plane + 1))
    # print('Saving image %s\n' %(outfile))
    try:
        skimage.io.imsave(outfile, save_plane, as_grey=True)
    except BaseException:
        skimage.io.imsave(outfile, save_plane)
        

print('Starting to merge large image dataset')

if len(sys.argv) == 1:
    print('Use -> Merge_LargeData ~/prediction/1fm')
    exit()
else:
    fm_dir = sys.argv[1]


tic = time.time()

path_separator = os.path.join(fm_dir, '')[-1]

if fm_dir[-1] == path_separator:  # fixing special case which can cause error
    fm_dir = fm_dir[:-1]

parent_dir = path_separator.join(fm_dir.split(path_separator)[:-1])
de_aug_file = os.path.join(parent_dir, 'de_augmentation_info.json')
print('Processing:', de_aug_file)

with open(de_aug_file, 'r') as json_file:
    json_file_contents = json.load(json_file)

packages = json_file_contents['packages']
num_of_pkg = json_file_contents['num_of_pkg']
imagesize = json_file_contents['imagesize']
# zplanes = json_file_contents['zplanes']
z_blocks = json_file_contents['z_blocks']

# Merge Z-sections
# first combine images from the same x/y areas through all z-planes
print('Combining image stacks')
for x_y_num in range(1, len(packages) + 1):
    imcounter = 0  # Reset imagecounter to combine next Package
    combined_folder = os.path.join(fm_dir, "Pkg_%03d" % (x_y_num))
    os.mkdir(combined_folder)
    for z_plane in range(1, len(z_blocks)):
        in_folder = os.path.join(fm_dir, 'Pkg%03d_Z%02d' % (x_y_num, z_plane))
        print('Reading:', in_folder)
        imlist = read_files_in_folder(in_folder)[0]
        imlist = [
            file_name for file_name in imlist if file_name.endswith('.png')]
        for filenum in range(0, len(imlist)):
            imcounter = imcounter + 1
            in_filename = os.path.join(in_folder, imlist[filenum])
            out_filename = os.path.join(
                combined_folder,
                'segmentation_%04d.png' %
                (imcounter))
            os.rename(in_filename, out_filename)

z_found = len([file_name for file_name in read_files_in_folder(
    os.path.join(fm_dir, 'Pkg_001'))[0] if file_name.endswith('.png')])
print('Expected number of planes: %s ... Found: %s planes\n' %
      (str(z_blocks[-1]), str(z_found)))
# Now stitch individual sections
combined_folder = os.path.join(
    fm_dir, 'Pkg_%03d' %
    (1))  # read in the filenames of the first Pkg
filelist = read_files_in_folder(combined_folder)[0]
p_tasks = max(1, min(z_found - 1, int(cpu_count()/2)))
print('Running ' + str(p_tasks) + ' parallel tasks\n')
Parallel(n_jobs=p_tasks)(delayed(merge_images)(z_plane)
                        for z_plane in range(0, z_found))  # one z-plane at a time


print('Merging large image dataset completed')
print("Total time = ", time.time() - tic)
print('Your results are in: %s\n' % (fm_dir))

with open(os.path.join(fm_dir, "DONE"), "w") as done_file:
    done_file.write("0\n")
