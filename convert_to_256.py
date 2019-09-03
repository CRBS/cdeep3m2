import numpy as np
import os
import h5py

# h5_folder = '/scratch/converted_all_python/test_testsample_processed'
h5_folder = '/scratch/test_training/prashant_augmentedtraining'
if not os.path.isdir(os.path.join(h5_folder, 'converted')):
    os.mkdir(os.path.join(h5_folder, 'converted'))
    outdir = os.path.join(h5_folder, 'converted')
else:
    outdir = os.path.join(h5_folder, 'converted')

for file_name in os.listdir(h5_folder):
    if file_name.endswith('h5'):

        f = h5py.File(os.path.join(h5_folder, file_name), 'r')
        X_data = f['data'][:, :, :]
        Y_label = f['label'][:, :, :]
        X_data = np.transpose(np.array([X_data]), [3, 1, 2, 0])
        Y_label = np.transpose(np.array([Y_label]), [3, 1, 2, 0])

        w = h5py.File(os.path.join(outdir, ('256_' + file_name)), 'w')

        flag = 0
        for i in range(4):
            for j in range(4):
                if (flag):
                    a = np.concatenate(
                        [a, X_data[:, i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :]], axis=0)
                    b = np.concatenate(
                        [b, Y_label[:, i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :]], axis=0)
                else:
                    a = X_data[:, i * 256:(i + 1) * 256,
                               j * 256:(j + 1) * 256, :]
                    b = Y_label[:, i * 256:(i + 1) * 256,
                                j * 256:(j + 1) * 256, :]
                    flag = 1

        w['data'] = a
        w['label'] = b
