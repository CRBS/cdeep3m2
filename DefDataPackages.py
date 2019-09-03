# -> Defines size of datapackages used for augmentation of image data
# Input: Image folder and output directory to store de_augmentation file
# Output: de_augmentation_info.mat
#
# Syntax : def_datapackages /ImageData/EMdata1/ /ImageData/AugmentedEMData/
#
#
# ----------------------------------------------------------------------------------------
# PreProcessImageData for CDeep3M -- NCMIR/NBCR, UCSD -- Author:
# ---------------------------------------------------------------------------------------
#

import sys
import os
import json
from check_image_size import check_image_size
from break_large_img import break_large_img


def main():
    arg_list = []
    for arg in sys.argv:
        arg_list.append(arg)

    if len(arg_list) < 2:
        print(
            'Use -> PreProcessImageData /ImageData/EMdata1/ /ImageData/AugmentedEMData/')
        return

    in_img_path = arg_list[1]
    outdir = arg_list[2]

    if os.path.isdir(outdir) == 0:
        os.mkdir(outdir)

    imagesize = check_image_size(in_img_path)
    packages, z_blocks = break_large_img(imagesize)
    num_of_pkg = len(packages)

    json_dump = {}
    json_dump['packages'] = packages
    json_dump['num_of_pkg'] = num_of_pkg
    json_dump['imagesize'] = imagesize
    json_dump['z_blocks'] = z_blocks

    with open(os.path.join(outdir, 'de_augmentation_info.json'), 'w') as json_file:
        json.dump(json_dump, json_file)

    with open(os.path.join(outdir, 'package_processing_info.txt'), 'w') as document:
        document.write('\nNumber of XY Packages\n')
        document.write(str(num_of_pkg))
        document.write('\nNumber of z-blocks\n')
        document.write(str(len(z_blocks) - 1))


if __name__ == "__main__":
    print('Starting Image Augmentation:')
    main()
