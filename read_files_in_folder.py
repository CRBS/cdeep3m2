import os
import sys
import stat


def read_files_in_folder(input_directory):
    """
Read_files_in_folder
   Get the complete list of good excluding hidden Files, excluding any subfolders in the input folder


   INPUT FORMAT
   --------------------------
   (InputDirectory)

   OUTPUT FORMAT
   --------------------------
   [fileList, file_list_length]


   --------------------------
   -- National Center for Microscopy and Imaging Research, NCMIR
   -- Matthias Haberl -- San Diego, 02/2016"""

    fileList = [file_name for file_name in os.listdir(input_directory)
                if not os.path.isdir(file_name) and not file_name.startswith('.')]

    if sys.platform.startswith("win"):
        fileList = [
            file_name for file_name in fileList if not bool(
                os.stat(
                    os.path.join(
                        input_directory,
                        file_name)).st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)]

    return [sorted(fileList), len(fileList)]
