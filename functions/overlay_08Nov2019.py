#!/usr/bin/env python

"""
Overlay image frame for CDeep3M
NCMIR/NBCR, UCSD -- Authors: M Haberl -- Date: 06/2019


Example:
python overlay.py ~/rawimage/image_001.png ~/pediction/segmented_001.png ~/overlay_001.png


"""

import sys
import argparse
import numpy as np
from skimage import io, exposure
import skimage

def _parse_arguments(desc, theargs):
    """Parses command line arguments using argparse
    """
    help_formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=help_formatter)
    parser.add_argument('inputrawimage_name',
                        help='File containing list of input raw image name')
    parser.add_argument('inputsegmentedimage_name',
                        help='File containing list of input segmented image name')
    parser.add_argument('outputoverlayimage_name',
                        help='File containing list of output overlay image name')
    return parser.parse_args(theargs)

desc = """
"""

# Parse arguments
theargs = _parse_arguments(desc, sys.argv[1:])

raw_image = io.imread(theargs.inputrawimage_name)
seg = io.imread(theargs.inputsegmentedimage_name)
seg = np.uint8(seg)
seg = np.clip(seg, 40, 255)
seg = skimage.exposure.rescale_intensity(seg, in_range=(40, 215), out_range=(0, 255))

rgb_seg = np.dstack((np.zeros_like(seg), seg, np.zeros_like(seg)))

raw_image = np.uint8(raw_image)
raw_image = skimage.exposure.rescale_intensity(raw_image, in_range=(0, 235), out_range=(0, 255))

rgb_img = np.dstack((raw_image, raw_image, raw_image))
overlayed = rgb_img
overlayed = np.uint8(np.uint8(0.60* rgb_img) + np.uint8(0.40* rgb_seg))
overlayed = exposure.rescale_intensity(overlayed, in_range=(0, 225), out_range=(0, 255))

sys.stdout.write('Saving: ' + str(theargs.outputoverlayimage_name) + '\n')
io.imsave(theargs.outputoverlayimage_name, overlayed)
