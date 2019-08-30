"""
Crop image frame for CDeep3M


NCMIR/NBCR, UCSD -- CDeep3M -- Update: 08/2019 @mhaberl 

"""
import sys
import os
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import cv2
import numpy as np
import skimage

def crop_png(inputlistfile, outputlistfile, leftxcoord, rightxcoord, topycoord, bottomycoord):

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
        img = cv2.imread(lines[x], cv2.IMREAD_UNCHANGED)
        cropped = img[in1:in2, in3:in4]
        sys.stdout.write('Saving: ' + str(outfiles[x]) + '\n')
        info = np.iinfo(cropped.dtype) # Get the information of the incoming image type
        cropped = cropped.astype(np.float64) / info.max # normalize the data to 0 - 1
        cropped = 255 * cropped # Now scale by 255
        cropped = cropped.astype(np.uint8)
        #cv2.imwrite(outfiles[x], cropped)
        try:
            skimage.io.imsave(outfiles[x], cropped, as_grey=True)
        except:
            skimage.io.imsave(outfiles[x], cropped)
        return
    niceness=os.nice(0)
    os.nice(10-niceness)
    p_tasks = max(1, min(6, int(cpu_count()/2.5)))
    #p_tasks = 2
    sys.stdout.write('Running ' + str(p_tasks) + ' parallel tasks\n')
    Parallel(n_jobs=p_tasks)(delayed(processInput)(i) for i in range(0, len(lines)))

