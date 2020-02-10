import os
import shutil
import h5py
import numpy as np
import skimage
import skimage.io
from read_files_in_folder import read_files_in_folder


def merge_16_probs(folder):

    # get variation directories aka directories that start with v
    vfolderlist = [
        dir_name for dir_name in os.listdir(
            os.path.join(
                folder,
                '')) if os.path.isdir(
                    os.path.join(
                        folder,
                        dir_name)) and dir_name.startswith('v')]

    folder_name = os.path.join(folder, vfolderlist[0])
    # print(folder_name)
    all_files, all_files_length = read_files_in_folder(folder_name)
    # print(all_files)
    # print(all_files_length)
    first_file_name, ext = os.path.splitext(all_files[0])
    filebasename = first_file_name[:-1]  # drop the last is the digit

    for fff in range(
            2,
            all_files_length -
            2):  # predictions start with 0; Ignore 0&1 and last two, since they are z-padding

        loadfile = filebasename + str(fff) + '.h5'
        print('Merging 16 variations of file ', filebasename,
              ' ... number ', str(fff - 1), ' of ', str(all_files_length - 3))
        image = []
        for i in range(1, 9):  # File 1:8 are 1:100
            folder_name = os.path.join(folder, 'v' + str(i))

            if os.path.isdir(folder_name):
                filename = os.path.join(folder_name, loadfile)
                # fileinfo = h5info(filename);
                # load_im = h5read(filename, '/data');
                load_im = h5py.File(filename, mode='r')
                load_im = list(load_im['/data'])
                print('H5 Dimensions: ', np.shape(load_im))
                # scale = max(max(load_im(:,:,2)));
                # inputim = np.transpose(load_im[0][1, :, :])
                inputim = load_im[0][1, :, :]
                inputim = {
                    0: inputim,
                    1: np.flip(inputim, 0),
                    2: np.flip(inputim, 1),
                    3: np.rot90(inputim, -1),
                    4: np.rot90(inputim, 1),
                    5: np.flip(np.rot90(inputim, -1), 0),
                    6: np.flip(np.rot90(inputim, -1), 1),
                    7: np.rot90(inputim, 2)
                }.get(i - 1, inputim)

                image.append(inputim)
            # prob=combinePredicctionSlice_v2(folder_name);
            # data{i}=prob;

    # Variations 9-16 are inverse organized
        loadfile_revert = filebasename + \
            str(all_files_length - (fff + 1)) + '.h5'
        for i in range(1, 9):  # File 9:16 are 100:1
            folder_name = os.path.join(folder, 'v' + str(i + 8))

            if os.path.isdir(folder_name):
                filename = os.path.join(folder_name, loadfile_revert)
                # load_im = h5read(filename, '/data');
                load_im = h5py.File(filename, mode='r')
                load_im = list(load_im['/data'])
                # scale = max(max(load_im(:,:,2)));
                # inputim = np.transpose(load_im[0][1, :, :])
                inputim = load_im[0][1, :, :]
                inputim = {
                    0: inputim,
                    1: np.flip(inputim, 0),
                    2: np.flip(inputim, 1),
                    3: np.rot90(inputim, -1),
                    4: np.rot90(inputim, 1),
                    5: np.flip(np.rot90(inputim, -1), 0),
                    6: np.flip(np.rot90(inputim, -1), 1),
                    7: np.rot90(inputim, 2)
                }.get(i - 1, inputim)
                image.append(inputim)
        # {
        # To check if 16 variations are good uncomment here
        # output_filename = os.path.join(folder, "%s_%04d.tiff" % (filebasename, (fff+1)))
        # for z in range(1, 17):
        #    imwrite(sixteen_vars(:,:,z),output_filename,'WriteMode','append');
        #    print("Saving: %s ... Image #%s   \n" %(output_filename, str(z)))
        # }
        # print(np.shape(image))
        # print('Dim2:', np.shape(image))
        # if np.shape(image)[0]>1:
        image = np.mean(image, 0)
        # image2 = mode(sixteen_vars,3) #mode weighting vs mean 
        # image_stack=de_augment_data(b);
        output_filename = os.path.join(
            folder, '%s_%04d.png' %
            (filebasename, (fff - 2)))

        print('write: ', output_filename)
        try:
            skimage.io.imsave(
                output_filename,
                skimage.img_as_ubyte(image),
                as_grey=True)
        except BaseException:
            skimage.io.imsave(output_filename, skimage.img_as_ubyte(image))

    print('Deleting intermediate .h5 files')
    for folder_name in vfolderlist:
        removefolders = os.path.join(folder, folder_name)
        print('Deleting %s\n' % (removefolders))
        shutil.rmtree(removefolders)

    return folder
