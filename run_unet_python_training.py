from keras.preprocessing.image import ImageDataGenerator
import h5py
from keras.utils.io_utils import HDF5Matrix
import os
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from Unet import unet
from keras.utils import multi_gpu_model
#from keras import backend as K
#print (K.backend())
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print ("GPU's: ",get_available_gpus())

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)

image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()

#h5_folder = '/scratch/converted_all_python/test_testsample_processed'
h5_folder = '/scratch/converted_all_python/test_testsample_processed/converted/'
for file_name in os.listdir(h5_folder):
    if file_name.endswith('h5'):

        #X_train = HDF5Matrix(os.path.join(h5_folder, file_name), 'data')
        #y_train = HDF5Matrix(os.path.join(h5_folder, file_name), 'label')
    
        f = h5py.File(os.path.join(h5_folder, file_name),'r')
        X_train = f['data']
        y_train = f['label']

        image_generator = image_datagen.flow(X_train, None)
        mask_generator = mask_datagen.flow(y_train, None)

        # combine generators into one which yields image and masks
        train_generator = zip(image_generator, mask_generator)
    
        model.fit_generator(train_generator, steps_per_epoch=1, epochs=1, callbacks=[model_checkpoint])

