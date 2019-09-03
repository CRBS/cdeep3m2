# Usage run_train(arg_list)
##
# Sets up directory and scripts to run training on CDeep3M model by caffe.
# arg_list should contain two a element cell array with first value
# set to path to augmented training data and the second argument the
# destination output directory
##
# Example: arg_list =
# {
# [1,1] = /foo/traindata
# [2,1] = /foo/output
# [3,1] = /foo/validationdata
# }
import os
import sys
import verify_and_create_train_file
import copy_over_allmodels
import update_solverproto_txt_file
import update_train_val_prototxt
import copy_version
import write_train_readme


def main():
    arg_list = []
    for arg in sys.argv:
        arg_list.append(arg)
    run_train(arg_list)


def run_train(arg_list):
    # Runs CDeep3M train using caffe.
    # Usage runtrain(cell array of strings)
    # by first verifying first argument is path to training data and
    # then copying over models under model/ directory to output directory
    # suffix for hdf5 files

    H_FIVE_SUFFIX = '.h5'
    base_dir = os.path.dirname(arg_list[0])
    print("base_dir is: ", base_dir)

    if len(arg_list) < 2:
        print('run_train expects at least two command line arguments\n\n')
        msg = 'Usage: run_train <Input train data directory> <output directory> <validatoin data directory>(validation data is optional)\n'
        raise Exception(msg)

    in_img_path = arg_list[1]
    if os.path.isdir(in_img_path) == 0:
        raise Exception(
            'First argument is not a directory and its supposed to be')

    outdir = arg_list[2]

    validation_img_path = arg_list[3]

    if os.path.isdir(validation_img_path) == 0:
        raise Exception(
            'Third argument is not a directory and its supposed to be')

    # ---------------------------------------------------------------------------
    # Examine input training data and generate list of h5 files
    # ---------------------------------------------------------------------------

    print('Verifying input training data is valid ... ')
    (status,
     errmsg,
     train_file,
     valid_file) = verify_and_create_train_file.verify_and_create_train_file(in_img_path,
                                                                             outdir,
                                                                             validation_img_path)

    if status != 0:
        raise Exception(errmsg)

    print('success')

    # ----------------------------------------------------------------------------
    # Create output directory and copy over model files and
    # adjust configuration files
    # ----------------------------------------------------------------------------
    print('Copying over model files and creating run scripts ... ')

    (onefm_dest, threefm_dest,
     fivefm_dest) = copy_over_allmodels.copy_over_allmodels(base_dir, outdir)
    max_iterations = 10000

    update_solverproto_txt_file.update_solverproto_txt_file(outdir, '1fm')
    update_solverproto_txt_file.update_solverproto_txt_file(outdir, '3fm')
    update_solverproto_txt_file.update_solverproto_txt_file(outdir, '5fm')

    update_train_val_prototxt.update_train_val_prototxt(
        outdir, '1fm', train_file, valid_file)
    update_train_val_prototxt.update_train_val_prototxt(
        outdir, '3fm', train_file, valid_file)
    update_train_val_prototxt.update_train_val_prototxt(
        outdir, '5fm', train_file, valid_file)

    copy_version.copy_version(base_dir, outdir)
    write_train_readme.write_train_readme(outdir)
    print('success\n\n')

    print('A new directory has been created: ' + outdir)
    print('In this directory are 3 directories 1fm,3fm,5fm which')
    print('correspond to 3 caffe models that need to be trained')


if __name__ == "__main__":
    main()
