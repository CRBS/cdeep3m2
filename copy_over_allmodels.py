#  Usage [onefm_dest, threefm_dest, fivefm_dest] = copy_over_allmodels( base_dir, outdir )
##
# Create outdir directory and copy over model files for
# 1fm, 3fm, and 5fm models. It is assumed that
# base_dir directory contains the deep3m source tree
# and there exists model/inception_residual_train_prediction_<model>
# directories
##
# Upon success three directory paths are returned,
# one for each model
import os
import copy_model


def create_dir(dir):
    if os.path.isdir(dir) == 0:
        os.mkdir(dir)


def copy_over_allmodels(base_dir, outdir):
    # ----------------------------------------------------------------------------
    # Create output directory and copy over model files and
    # adjust configuration files
    # ----------------------------------------------------------------------------
    # copy over 1fm, 3fm, and 5fm model data to separate directories
    onefm_dest = os.path.join(outdir, '1fm')
    create_dir(onefm_dest)
    copy_model.copy_model(base_dir, '1fm', onefm_dest)

    threefm_dest = os.path.join(outdir, '3fm')
    create_dir(threefm_dest)
    copy_model.copy_model(base_dir, '3fm', threefm_dest)

    fivefm_dest = os.path.join(outdir, '5fm')
    create_dir(fivefm_dest)
    copy_model.copy_model(base_dir, '5fm', fivefm_dest)

    return (onefm_dest, threefm_dest, fivefm_dest)
