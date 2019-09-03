# Usage [errmsg] = copy_version( base_dir, dest_dir )
##
# Starting from base_dir directory this function
# copies the VERSION from to directory specified
# by dest_dir argument. If copy fails errmsg is set
# to string describing error otherwise its empty string
##
import shutil
import os


def copy_version(base_dir, dest_dir):
    errmsg = ''
    src_file = os.path.join(base_dir, 'VERSION')
    # print(src_file)
    try:
        shutil.copy(src_file, dest_dir)
    except Exception as e:
        errmsg = 'Error copying VERSION ' + src_file + '\n' + str(e)
        print(errmsg)

    # return errmsg
