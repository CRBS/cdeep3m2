import os
from PIL import Image
import cv2
#import skimage
import numpy as np

image_dir = '/scratch/test_tutorial/test_testsample/testset_bigimage/'
outdir = os.path.join(image_dir, 'trimmed_bigimage_1024')
if not os.path.isdir(outdir):
    os.mkdir(outdir)
for filename in os.listdir(image_dir):
    print(filename)
    if filename.endswith(".png"):
        print(filename)
        im = Image.open(os.path.join(image_dir,filename))
        im = np.array(im)
        fname = filename+'_1024.png'
        cv2.imwrite(os.path.join(outdir, fname), im[0:1024, 0:1024])
        #skimage.io.imsave(os.path.join(outdir, fname), skimage.img_as_ubyte(im), as_grey=True)
