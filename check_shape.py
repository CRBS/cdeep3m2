import os
import h5py


for filename in os.listdir(
        '/scratch/converted_all_python/test_testsample_processed/converted/'):
    if filename.endswith('h5'):
        f = h5py.File(
            '/scratch/converted_all_python/test_testsample_processed/converted/' +
            filename,
            'r')
        print(f['data'].shape)
