"""
EnsemblePredictions for CDeep3M
different predictions coming from files e.g. from 1fm 3fm and 5fm will be averaged here
flexible number of inputs
last argument has to be the outputdirectory where the average files will be stored

-----------------------------------------------------------------------------
 NCMIR, UCSD -- Author: M Haberl -- Data: 10/2018
 ----------------------------------------------------------------------------

"""
import sys
import os
import argparse
from multiprocessing import cpu_count
from time import time
import cv2
from joblib import Parallel, delayed
# from multiprocessing import Pool, TimeoutError
import numpy as np
from PIL import Image
import skimage

Image.MAX_IMAGE_PIXELS = 10000000000000


def ensemble(sys_argv):

    def _parse_arguments(desc, theargs):
        """Parses command line arguments using argparse
        """
        help_formatter = argparse.RawDescriptionHelpFormatter
        parser = argparse.ArgumentParser(description=desc,
                                         formatter_class=help_formatter)
        parser.add_argument('inputlistfile',
                            help='File containing list of paths')
        parser.add_argument('outputfolder',
                            help='Path to write output in')

        return parser.parse_args(theargs)

    desc = """
    Given a file with a list of folder (inputlistfile),
    """

    # Parse arguments
    theargs = _parse_arguments(desc, sys_argv)
    outfolder = theargs.outputfolder

    file = open(theargs.inputlistfile, "r")
    infolders = [line.rstrip('\n') for line in file]
    file.close()

    folder1 = infolders[0]
    sys.stdout.write('Reading ' + str(folder1) + ' \n')
    filelist1 = [fileb for fileb in os.listdir(
        folder1) if fileb.endswith('.png')]
    print(infolders)
    print(filelist1)
    sys.stdout.write('Merging ' + str(len(filelist1)) + ' files \n')

    def average_img(x):
        sys.stdout.write(
            'Loading: ' + str(os.path.join(infolders[0], filelist1[x])) + '\n')
        t0 = time()
        temp = cv2.imread(
            os.path.join(
                infolders[0],
                filelist1[x]),
            cv2.IMREAD_UNCHANGED)
        # img[:,:,0]
        for n in range(1, len(infolders)):
            temp = np.dstack(
                (temp,
                 cv2.imread(
                     os.path.join(
                         infolders[n],
                         filelist1[x]),
                     cv2.IMREAD_UNCHANGED)))
            print(time() - t0)
            print(temp.shape)
        arr = np.array(np.mean(temp, axis=(2)), dtype=np.uint8)
        # aver = Image.fromarray(arr)
        # cv2.imwrite(os.path.join(outfolder,filelist1[x]), arr)
        print("ensemble size", arr.shape)
        # saveimage.imsave(os.path.join(outfolder,filelist1[x]), arr, cmap='gray')
        try:
            skimage.io.imsave(
                os.path.join(
                    outfolder,
                    filelist1[x]),
                arr,
                as_grey=True)
        except BaseException:
            skimage.io.imsave(os.path.join(outfolder, filelist1[x]), arr)
        return

    # p_tasks = 2
    p_tasks = max(1, min(6, int(cpu_count() / 2.5)))
    sys.stdout.write('Running ' + str(p_tasks) + ' parallel tasks\n')
    Parallel(n_jobs=p_tasks)(delayed(average_img)(i)
                             for i in range(0, len(filelist1)))
