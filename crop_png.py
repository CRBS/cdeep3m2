"""
Crop image frame for CDeep3M


NCMIR/NBCR, UCSD -- CDeep3M -- Update: 08/2019 @mhaberl

"""
import sys
import os
#from multiprocessing import cpu_count
from joblib import Parallel, delayed
import cv2
import numpy as np
#import skimage


def crop_png(
        inputlistfile,
        outputlistfile,
        leftxcoord,
        rightxcoord,
        topycoord,
        bottomycoord):

    in1 = leftxcoord
    in2 = rightxcoord
    in3 = topycoord
    in4 = bottomycoord

    sys.stdout.write(str(in1) + '\n')
    sys.stdout.write(str(in2) + '\n')
    sys.stdout.write(str(in3) + '\n')
    sys.stdout.write(str(in4) + '\n')

    file = open(inputlistfile, "r")
    lines = [line.rstrip('\n') for line in file]
    file.close()

    file = open(outputlistfile, "r")
    outfiles = [line.rstrip('\n') for line in file]
    file.close()

    def processInput(x):
        img = cv2.imread(lines[x], 0)
        cv2.imwrite(outfiles[x], img[in1:in2, in3:in4])
        #try:
        #    skimage.io.imsave(outfiles[x], cropped, as_grey=True)
        #except BaseException:
        #    skimage.io.imsave(outfiles[x], cropped)
        #return
    niceness = os.nice(0)
    os.nice(5 - niceness)
    p_tasks = max(1, min(10, int(os.cpu_count() - 1)))
    sys.stdout.write('Running ' + str(p_tasks) + ' parallel tasks\n')
    Parallel(n_jobs=p_tasks)(delayed(processInput)(i)
                             for i in range(0, len(lines)))
