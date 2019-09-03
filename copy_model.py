# Usage copy_model( base_dir, the_model, dest_dir )
##
# Starting from base_dir directory this function
# copies *txt files from model/inception_residual_train_prediction_<the_model>
# to directory specified by dest_dir argument. If copy fails
# error() is invoked describing the issue
##
import glob
import shutil
import os


def copy_model(base_dir, the_model, dest_dir):
    src_files = os.path.join(
        base_dir,
        'model',
        'inception_residual_train_prediction_' +
        the_model,
        '*txt')
    # print (src_files)
    for filename in glob.glob(src_files):
        try:
            shutil.copy(filename, dest_dir)
        except Exception as e:
            print('Error copying model ', the_model, '\n', str(e))

    return
