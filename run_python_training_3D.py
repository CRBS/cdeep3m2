from keras.preprocessing.image import ImageDataGenerator
import h5py
from keras.utils.io_utils import HDF5Matrix
import os
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from MultiResUnet3D import MultiResUnet3D
from keras.utils import multi_gpu_model
from keras.models import Sequential, load_model
#from keras import backend as K
#print (K.backend())
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print ("GPU's: ",get_available_gpus())

#model = MultiResUnet(1024, 1024,1)
model = MultiResUnet3D(256, 256,3,1)
#model = load_model("multiresunet_membrane.h5")
#model = load_model("multiresunet_membrane_1024.h5")
#model = multi_gpu_model(model, gpus=1)
lr = 0.01
opt = Adam(lr=0.001, decay=1e-6)
batch_size = 8
numberOfIterations = 100
#multiresunet_model.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['accuracy'])

model_checkpoint = ModelCheckpoint('multiresunet_membrane_1024.h5', monitor='loss', verbose=1, save_best_only=True)

image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()

#h5_folder = '/scratch/converted_all_python/test_testsample_processed'
#h5_folder = '/scratch/converted_all_python/test_testsample_processed/converted/'
h5_folder = '/scratch/test_training/prashant_augmentedtraining/converted/'
for iteration in range(numberOfIterations):
    print ("Iteration = ",iteration)    
    for file_name in os.listdir(h5_folder):
        print ("Training on ",file_name)
        if file_name.endswith('h5'):

            #X_train = HDF5Matrix(os.path.join(h5_folder, file_name), 'data')
            #y_train = HDF5Matrix(os.path.join(h5_folder, file_name), 'label')
    
            f = h5py.File(os.path.join(h5_folder, file_name),'r')
            num_images = f['data'].shape[0]
            for i in range(0, num_images, batch_size):
                if (i+batch_size) < num_images:
                    X_train = f['data'][i:i+batch_size, :, :, :]
                    y_train = f['label'][i:i+batch_size, :, :, :]
                else:
                    X_train = f['data'][i:, :, :, :]
                    y_train = f['label'][i:, :, :, :]
            
                #X_train = f['data'][0:8, :, :, :]
                #y_train = f['label'][0:8, :, :, :]

                image_generator = image_datagen.flow(X_train, None)
                mask_generator = mask_datagen.flow(y_train, None)
        
                # combine generators into one which yields image and masks
                train_generator = zip(image_generator, mask_generator)
    
                model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=1, callbacks=[model_checkpoint])
            
            print ("Training completed on ",file_name)

print ("Hurry!! Training Completed")



