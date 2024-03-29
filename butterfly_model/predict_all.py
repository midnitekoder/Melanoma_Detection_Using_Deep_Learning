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
#from resnet_Butterfly import ResnetBuilder
from sklearn.metrics import jaccard_similarity_score
from data import load_train_data, load_test_data
from resnet_Butterfly import ResnetBuilder
#K.set_image_data_format('channels_last')  # TF dimension ordering in this code
from sklearn import metrics
img_rows = 224
img_cols = 224

smooth = 1.


print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)
# imgs_train, imgs_mask_train = load_train_data()
imgs_train=np.load('train_accumulator.npy')
#imgs_train = preprocess(imgs_train)
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
#model= ResnetBuilder.build_butternet_10((224, 224,3), 2)
#model= ResnetBuilder.build_ror_50((224,224,3),2,3)
#model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
model = ResnetBuilder.build_butternet_38((224, 224,3), 2)
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
imgs_test=np.load('val_accumulator_final_100.npy')
#imgs_test, imgs_id_test = load_test_data()
#imgs_test = preprocess(imgs_test)

imgs_test = imgs_test.astype('float32')

imgs_test -= mean
imgs_test /= std

print('-'*30)
print('Loading saved weights...')
print('-'*30)
#model.load_weights('best_model_50_butterfly463.hdf5')
#model.load_weights('best_annealing_adameminus4_model_50_butterfly4.hdf5')
#model.load_weights('best_annealing_rmsprop0point1_model_50_butterfly235.hdf5')
#model.load_weights('best_annealing_rmsprop0point1_model_6_butterResnet33.hdf5')
clist=[]
dict={}
dict['accuracy']='accuracy'
dict['sensitivity']='sensitivity'
dict['specificity']='specificity'
dict['jaccard']='jaccard'
dict['roc_auc']='roc_auc'
clist.append(dict)

for i in range(0,60):
	try:
		temp='best_butterResnet_38_'+str(i*10+9)+'.hdf5'
		print(temp)
		model.load_weights(temp)
		print('-'*30)
		print('Predicting masks on test data...')
		print('-'*30)
		imgs_mask_test = model.predict(imgs_test)

		np.save('imgs_mask_test.npy', imgs_mask_test)
		print(imgs_mask_test)
		predicted_labels=[]
		for each in imgs_mask_test:
			if each[0]>each[1]:
				predicted_labels.append(0)
			else:
				predicted_labels.append(1)
		print ('predicted: actual')
		val_accumulator_label=np.load('val_accumulator_final_100_label.npy')
		tp=0
		tn=0
		fp=0
		fn=0
		for i in range(0,100):
			print (predicted_labels[i],val_accumulator_label[i])
			if predicted_labels[i]==0:
				if predicted_labels[i]==val_accumulator_label[i]:
					tn=tn+1
				else:
					fn=fn+1
			else:
				if predicted_labels[i]==val_accumulator_label[i]:
					tp=tp+1
				else:
					fp=fp+1
		dict={}
		accuracy=float(tp+tn)/(tp+tn+fp+fn)
		sensitivity=float(tp)/(tp+fn)
		specificity=float(tn)/(tn+fp)
		jaccard=metrics.jaccard_similarity_score(val_accumulator_label, predicted_labels, normalize=True, sample_weight=None)
		fpr, tpr, thresholds = metrics.roc_curve(val_accumulator_label, predicted_labels, pos_label=1)
		roc_auc=metrics.auc(fpr, tpr)
		dict['accuracy']=accuracy
		dict['sensitivity']=sensitivity
		dict['specificity']=specificity
		dict['jaccard']=jaccard
		dict['roc_auc']=roc_auc
		clist.append(dict)
		print('accuracy',accuracy)
		print('sensitivity',sensitivity)
		print('specificty',specificity)
		print('roc auc',roc_auc)

		print('jaccard score',jaccard)
	except:
		print("exception",i)

import csv
with open('results_METRICS_butterResnet_38_decay_inc_lr.csv', 'wb') as csvfile:		#filename
	#cfile = csv.writer(csvfile, delimiter=' ')
	w = csv.DictWriter(csvfile, clist[0].keys())
	w.writerows(clist)
	csvfile.close()
