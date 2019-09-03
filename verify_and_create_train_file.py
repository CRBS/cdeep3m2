# Usage [status, errmsg, train_file, valid_file] = verify_and_create_train_file ( train_input, outdir, valid_input="" )
##
# 1st Looks for files ending with .h5 in train_input directory
# and verifies there are 16 of them. 2nd code creates a
# train_file.txt in the outdir directory that has a list
# of full paths to these h5 files.
##
# Upon success train_file will have the path to the train_file.txt created
# by this function.
##
# If there is an error status will be set to a non zero numeric
# value and errmsg will explain the issue.
import os
import glob


def verify_and_create_train_file(train_input, outdir, valid_input=""):
    errmsg = ''
    train_file = ''
    valid_file = ''
    status = 0
    H_FIVE_SUFFIX = '.h5'

    if os.path.isdir(train_input) == 0:
        errmsg = train_input + 'is not a directory'
        status = 1
        return (status, errmsg, train_file, valid_file)

    train_files = (glob.glob(train_input + '/*' + H_FIVE_SUFFIX))

    if len(train_files) % 16 != 0:
        errmsg = 'Expecting 16 .h5 files, but got: ' + str(len(train_files))
        status = 3
        return (status, errmsg, train_file, valid_file)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    train_file = os.path.join(outdir, 'train_file.txt')
    train_out = open(train_file, "w")
    for filename in train_files:
        train_out.write(filename + '\n')
    train_out.close()

    # If user specified validation file
    if valid_input != '':
        if os.path.isdir(valid_input) == 0:
            errmsg = valid_input + 'is not a directory'
            status = 1
            return (status, errmsg, train_file, valid_file)

        valid_files = glob.glob(valid_input + '/*' + H_FIVE_SUFFIX)

        valid_file = os.path.join(outdir, 'valid_file.txt')
        valid_out = open(valid_file, "w")
        for filename in valid_files:
            valid_out.write(filename + '\n')
        valid_out.close()
        return (status, errmsg, train_file, valid_file)

    else:
        valid_file = train_file
        return (status, errmsg, train_file, valid_file)
