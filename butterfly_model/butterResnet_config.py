import keras
from keras import backend as K
#from ror_50 import ResnetBuilder
from resnet_Butterfly import ResnetBuilder
from keras.preprocessing import image
#import cv2
import numpy as np 
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
modelCheckpoint=ModelCheckpoint('/home/ankit/installData/ME/butterfly_onsegmented/best_butterResnet_38_{epoch}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto',period=10)
def scheduler(epoch):
    if epoch<150:
        return 0.0001
    elif epoch<200:
        return 0.0001*(0.5*int((epoch-150)/10)+1)
    elif epoch<350:
        return 0.0001
    else:
        return 0.0001-0.000001*int((epoch-350+1)/5)
 
 

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


ordering ='tf'
K.set_image_dim_ordering(ordering)
#model = ResnetBuilder.build_butterRor_10((224, 224,3), 2,3)

'''model = keras.applications.resnet50.ResNet50(include_top=True, 
                                     weights=None, 
                                     input_tensor=None, 
                                     input_shape=(227,227,3))
'''
model = ResnetBuilder.build_butternet_38((224, 224,3), 2)


#shuffleList=np.load('finalMelanomaList.npy')

accumulator = np.load('train_accumulator.npy');
accumulator=accumulator.reshape(800,224,224,3)

# In[27]:

train_datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
	rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range = 0.25)
train_datagen.fit(accumulator)

val_accumulator = np.load('val_accumulator_final_100.npy')
val_accumulator=val_accumulator.reshape(100,224,224,3)


validation_datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
        )
validation_datagen.fit(val_accumulator) #need to do this for training and val

#path ='/home/iplab1/Desktop/resnet2/KeraResnetData/train_images'
path = '/home/ankit/installData/segmentedImages/train'
train_generator = train_datagen.flow_from_directory(
        path,
	target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')


path = '/home/ankit/installData/segmentedImages/val'
#path = '/home/iplab1/Desktop/resnet2/KeraResnetData/val_images'
validation_generator = validation_datagen.flow_from_directory(
        path,
	target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')



# In[33]:

#model.load_weights('best_annealing_adam_model_10_butterResnet413.hdf5')
#model.load_weights('mod_butterROR_10_600Epoch.h5')
optim = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)

change_lr = LearningRateScheduler(scheduler)

model.compile(loss='categorical_crossentropy',
              optimizer=optim,
              #loss=dice_coef_loss, 
	      metrics=[dice_coef,'accuracy'])




print model.summary()




model.fit_generator(
        train_generator,
        samples_per_epoch=800,
        #initial_epoch = 601,
        nb_epoch=600,
        validation_data=validation_generator,
        nb_val_samples=100,
	callbacks=[modelCheckpoint,change_lr])


# In[ ]:

model.save_weights('mod_butterResnet_38_600Epoch.h5')


# In[ ]:


