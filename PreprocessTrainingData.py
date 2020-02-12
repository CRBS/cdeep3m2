#!/usr/bin/env python3
# PreprocessTraining
# Generates augmented hdf5 datafiles from raw and label images
#
# Syntax : PreprocessTraining.py /ImageData/training/images/ /ImageData/training/labels/ /Augmentation level/ /ImageData/augmentedtraining/
#
# Example usage: ./PreprocessTrainingData.py ./mito_testsample/training/images/ ./mito_testsample/training/labels/ -1 3 mito_testsample/training/augmented_data
# 
#
# Primary 16 augmentations (rotations, flips, z-invert) are always performed
# Secondary augmentations (injecting and removing noise, contrast adjustments) can be regulated in intensity (0-10), default in Version 2.0.0 is set to perform denoising and contrast optimization (-1)
# Tertiary augmentaiotns (resizing) can be regulated in intensity (0-10), default is no 
#
# ----------------------------------------------------------------------------------------
# PreprocessTraining for CDeep3M (Version 2.0.0) -- NCMIR/NBCR, UCSD -- Date: 08/2019
# ----------------------------------------------------------------------------------------
#
import os
import sys
import h5py
import numpy as np
from imageimporter import imageimporter
from checkpoint_nobinary import checkpoint_nobinary
from check_img_dims import check_img_dims
from config_export import writecfg
from augment_data import augment_data, addtl_augs, third_augs
from checkpoint_isbinary import checkpoint_isbinary
from dim_convert import dim_convert

def main():
    print('Starting Training Data Preprocessing')
    arg_list = []
    for arg in sys.argv[1:]:
        arg_list.append(arg)

    if len(arg_list) < 3:
        print('Use -> python3 PreprocessTrainingData.py /ImageData/training/images/ /ImageData/training/labels/ ')
        print('Secondary augmentation strength(int value or path to config file, no input for default 0)')
        print('Tertiary augmentation strength /ImageData/augmentedtraining/')
        return

# counting number of training sets provided
    ends = []
    augmentation_level = []
    third_aug_lvl = []

    i = 2
    while i <= len(arg_list):
        if arg_list[i].isdigit() or arg_list[i] == '-1' or  arg_list[i].endswith('.ini'):
            if arg_list[i + 1].isdigit():
                ends.append(i - 1)
                augmentation_level.append(arg_list[i])
                third_aug_lvl.append(arg_list[i + 1])
                i += 4
            else:
                ends.append(i - 1)
                augmentation_level.append(arg_list[i])
                third_aug_lvl.append('0')
                i += 3
        else:
            ends.append(i - 1)
            augmentation_level.append('-1')
            third_aug_lvl.append('0')
            i += 2

    num_training_sets = len(augmentation_level)
    print('num_training_sets:', num_training_sets)
    print('augmentation_level:', augmentation_level)
    print('thrid_augmentation_level:', third_aug_lvl)
    # print('ends:', ends)

    for j in range(num_training_sets):

        training_img_path = arg_list[ends[j] - 1]
        print('Training Image Path:', training_img_path)
        label_img_path = arg_list[ends[j]]
        print('Training Label Path:', label_img_path)

        strength = augmentation_level[j]
        print('Secondary Augmentation level:', strength)
        third_str = third_aug_lvl[j]
        print('Tertiary Augmentation level:', third_str)

        outdir = arg_list[len(arg_list) - 1]
        print('Output Path:', outdir)

        # ----------------------------------------------------------------------------------------
        # Load train labels
        # ----------------------------------------------------------------------------------------

        print('Loading:')
        print(label_img_path)
        lblstack = imageimporter(label_img_path)
        print('Verifying labels')
        checkpoint_isbinary(lblstack)
        if np.max(lblstack[:]) != 1:
            lblstack = np.divide(lblstack, np.max(lblstack[:]))

        # ----------------------------------------------------------------------------------------
        # apply denoising
        # ----------------------------------------------------------------------------------------

        if strength == -1:
            print('Running image enhancement')
            enhanced_path = os.path.join(training_img_path, 'enhanced')
            run_enhancement = 'python3 enhance_stack.py ' + training_img_path + ' ' + enhanced_path + ' ' + 2
            os.system(run_enhancement)
            training_img_path = enhanced_path

        # ----------------------------------------------------------------------------------------
        # Load training images
        # ----------------------------------------------------------------------------------------

        print('Loading:')
        print(training_img_path)
        imgstack = imageimporter(training_img_path)
        print('Verifying images')
        checkpoint_nobinary(imgstack)

        # ----------------------------------------------------------------------------------------
        # Check size of images and labels
        # ----------------------------------------------------------------------------------------

        [imgstack, lblstack] = check_img_dims(imgstack, lblstack, 325)

        # ----------------------------------------------------------------------------------------
        # Augment the data, generating 16 versions and save
        # ----------------------------------------------------------------------------------------

        img_v1 = imgstack.astype('float32')
        lb_v1 = lblstack.astype('float32')
        del imgstack
        del lblstack

        d_details = '/data'
        l_details = '/label'

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        ext = ".h5"

        print('Augmenting training data 1-8 and 9-16')
        for i in range(8):
            # v1-8

            img, lb = augment_data(img_v1, lb_v1, i)
            img_n, lb_n = img.astype(
                np.uint8), lb.astype(
                    np.uint8)  # augmentations 1-8
            inv_img_n = np.flip(img, 0).astype(np.uint8)  # augmentations 9-16
            inv_lb_n = np.flip(lb, 0).astype(np.uint8)  # augmentations 9-16
            del img, lb

            if strength != '-1':
                img_result, lb_result, addtl_choices = addtl_augs(
                    strength, img_n, lb_n, i)  # apply secondary augmentations
            else:
                img_result, lb_result = img_n, lb_n
            del img_n, lb_n
            img_result_r, lb_result_r = third_augs(
                third_str, img_result, lb_result, i)  # apply tertiary augmentations
            img_result_f = img_result_r.astype('float32')
            lb_result_f = lb_result_r.astype('float32')

            filename = os.path.abspath(
                outdir) + '/' + 'training_full_stacks_v{0}_{1}.h5'.format(str(j + 1), str(i + 1), ext)
            print('Saving: ', filename)
            hdf5_file = h5py.File(filename, mode='w')
            img_result_f, lb_result_f = dim_convert(img_result_f, lb_result_f)
            hdf5_file.create_dataset(d_details, data=img_result_f)
            hdf5_file.create_dataset(l_details, data=lb_result_f)
            hdf5_file.close()
            del img_result, img_result_r, img_result_f
            del lb_result, lb_result_r, lb_result_f

            # v9-16
            if strength != '-1':
                inv_img_result, inv_lb_result, inv_addtl_choices = addtl_augs(
                    strength, inv_img_n, inv_lb_n, i + 8) # apply secondary augmentations
            else:
                inv_img_result, inv_lb_result = inv_img_n, inv_lb_n
            del inv_img_n, inv_lb_n
            inv_img_result_r, inv_lb_result_r = third_augs(
                third_str, inv_img_result, inv_lb_result, i + 8)
            inv_img_result_f = inv_img_result_r.astype('float32')
            inv_lb_result_f = inv_lb_result_r.astype('float32')

            filename = os.path.abspath(
                outdir) + '/' + 'training_full_stacks_v{0}_{1}.h5'.format(str(j + 1), str(i + 1 + 8), ext)
            print('Saving: ', filename)
            hdf5_file = h5py.File(filename, mode='w')
            inv_img_result_f, inv_lb_result_f = dim_convert(
                inv_img_result_f, inv_lb_result_f)
            hdf5_file.create_dataset(d_details, data=inv_img_result_f)
            hdf5_file.create_dataset(l_details, data=inv_lb_result_f)
            hdf5_file.close()
            del inv_img_result, inv_img_result_r, inv_img_result_f
            del inv_lb_result, inv_lb_result_r, inv_lb_result_f

        #writecfg(outdir, j+1, addtl_choices, strength, third_str) # turned off temporarily, usefull for tracking secondary augm.
        #print('Saving: ', filename)

    print('\n-> Training data augmentation completed')
    print('Training data stored in ', outdir)
    print('For training your model please run runtraining.sh ',
          outdir, '<desired output directory>\n')


if __name__ == "__main__":
    main()
