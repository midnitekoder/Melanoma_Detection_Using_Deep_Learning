from __future__ import print_function

import os
#from skimage.transform import resize
import PIL
from PIL import Image
#from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D
#from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from ror_50 import ResnetBuilder

from data import load_train_data, load_test_data

#K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 224
img_cols = 224

smooth = 1.


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    # imgs_train, imgs_mask_train = load_train_data()
    imgs_train=np.load('train_accumulator.npy')
    # imgs_train = preprocess(imgs_train)
    #imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    #imgs_mask_train = imgs_mask_train.astype('float32')
    #imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    #model = get_unet()
    model= ResnetBuilder.build_ror_50((224,224,3),2,3)
    #model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    '''
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    '''
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test=np.load('train_accumulator.npy')
    #imgs_test, imgs_id_test = load_test_data()
#    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    #model.load_weights('best_annealing_model_50_butterfly25.hdf5')
    model.load_weights('best_annealing_adameminus4_model_50_butterfly4.hdf5')
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test)
    np.save('imgs_mask_test.npy', imgs_mask_test)
    print(imgs_mask_test)
    '''
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        image.save(os.path.join(pred_dir, str(image_id) + '_pred.png'))
    '''
if __name__ == '__main__':
    train_and_predict()
