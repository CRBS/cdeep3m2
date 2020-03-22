# preprocess_package
# receives package index numbers to process
# requires data_packagedef to have run before
# -> Makes augmented hdf5 datafiles from raw images based on defining parameters
#
# Syntax : preprocess_package indir outdir xy_package z_stack augmentation speed
# Example: preprocess_package ~/EMdata1/ ~/AugmentedEMData/ 15 2 1fm 10
#
# Speed: supported values 1,2,4 or 10
# speeds up processing potentially with a negative effect on accuracy (speed of 1 equals highest accuracy)
#
#
# ----------------------------------------------------------------------------------------
# preprocess_package for Deep3M -- NCMIR/NBCR, UCSD -- Author: M Haberl -- Date: 02/2019
# ----------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------
# Initialize
# ----------------------------------------------------------------------------------------

import sys
import os
import json
from imageimporter_large import imageimporter_large
from checkpoint_nobinary import checkpoint_nobinary
from augment_package import augment_package
from add_z_padding import add_z_padding


def main():
    print('Starting Image Augmentation')

    in_img_path = sys.argv[1]
    outdir = sys.argv[2]
    ii = int(sys.argv[3])
    zz = int(sys.argv[4])
    fmtype = sys.argv[5]
    fmnumber = int(fmtype[0])
    speed = int(sys.argv[6])
    fmdir = os.path.join(outdir, str(fmnumber) + 'fm')
    if os.path.isdir(fmdir) == 0:
        os.mkdir(fmdir)

    with open(os.path.join(outdir, 'de_augmentation_info.json'), 'r') as json_file:
        json_file_contents = json.load(json_file)

    packages = json_file_contents['packages']
    num_of_pkg = json_file_contents['num_of_pkg']
    imagesize = json_file_contents['imagesize']
    z_blocks = json_file_contents['z_blocks']

    # print("ii "+str(ii))
    # print("zz "+str(zz))
    # print("z_blocks", z_blocks)
    # print("packages", packages)
    z_stack = [z_blocks[zz - 1], z_blocks[zz]]

    area = packages[ii - 1]

    stack = imageimporter_large(
        in_img_path,
        area,
        z_stack,
        outdir)  # load only subarea here
    print('Padding images\n')
    stack = add_z_padding(stack)  # adds 2 planes in the beginning and end

    # augment_and_saveSave image data
    pkg_dir = 'Pkg' + "{:03}".format(ii) + '_Z' + "{:02}".format(zz)
    outsubdir = os.path.join(fmdir, pkg_dir)

    if os.path.isdir(outsubdir) == 0:
        os.mkdir(outsubdir)

    augment_package(stack, outsubdir, fmnumber, speed)

    with open(os.path.join(outsubdir, "DONE"), 'w') as done_file:
        done_file.write("0\n")


if __name__ == "__main__":
    main()
