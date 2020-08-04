# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 21:32:27 2020

@author: amanm
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 01:26:09 2020

@author: amanm
"""
# importing libraries

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from keras import initializers
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import GridSearchCV




# importing files
test_data=pd.read_csv('test.csv')
train_data=pd.read_csv('train.csv')

#spliting the x and y values
train_label=train_data.iloc[:,0]
train_pixels=train_data.iloc[:,1:786]


#converting then to array
train=np.array(train_pixels).astype(float)
train=train.reshape(train_pixels.shape[0],28,28,1)

test=np.array(test_data).astype(float)
test=test.reshape(test_data.shape[0],28,28,1)

train_label=np.array(train_label)
train_label=train_label.reshape(42000,1)

#Normalize
train=train/255.0
test=test/255.0

#Label Encoding
Y_tr=to_categorical(train_label,num_classes=10)

model=[0]*15
#LeNet model
for i in range(15):
    model[i]=tf.keras.models.Sequential()
    model[i].add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',strides=(1,1),kernel_initializer='glorot_normal'))
    model[i].add(tf.keras.layers.BatchNormalization(axis=3))
    model[i].add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',strides=(1,1),kernel_initializer='glorot_normal'))
    model[i].add(tf.keras.layers.BatchNormalization(axis=3))
    model[i].add(tf.keras.layers.Conv2D(32,(5,5),activation='relu',strides=(2,2),padding='same',kernel_initializer='glorot_normal'))
    model[i].add(tf.keras.layers.BatchNormalization(axis=3))
    model[i].add(tf.keras.layers.Dropout(rate=0.4))
    
    model[i].add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',strides=(1,1),kernel_initializer='glorot_normal'))
    model[i].add(tf.keras.layers.BatchNormalization(axis=3))
    model[i].add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',strides=(1,1),kernel_initializer='glorot_normal'))
    model[i].add(tf.keras.layers.BatchNormalization(axis=3))
    model[i].add(tf.keras.layers.Conv2D(64,(5,5),activation='relu',strides=(2,2),padding='same',kernel_initializer='glorot_normal'))
    model[i].add(tf.keras.layers.BatchNormalization(axis=3))
    model[i].add(tf.keras.layers.Dropout(rate=0.4))
    
    model[i].add(tf.keras.layers.Conv2D(128,kernel_size=4,activation='relu',kernel_initializer='glorot_normal'))
    model[i].add(tf.keras.layers.BatchNormalization(axis=3))
    model[i].add(tf.keras.layers.Flatten())
    model[i].add(tf.keras.layers.Dropout(0.4))
    model[i].add(tf.keras.layers.Dense(10,activation='softmax',kernel_initializer='glorot_normal'))
    model[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  
#using adam optmizer and categorical crossentropy as loss


learning_decay = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.9935, 
                                            min_lr=0.00001)

#Data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images



#model fitting
history=[0]*15
epochs=50
for j in range(15):
    X_train,X_val,Y_train,Y_val=train_test_split(train,Y_tr,test_size=0.1)
    history[j]=model[j].fit_generator(datagen.flow(X_train,Y_train, batch_size=64),
    epochs = epochs, steps_per_epoch = X_train.shape[0]//64,  
    validation_data = (X_val,Y_val), callbacks=[learning_decay], verbose=0)
  #  recog=model.fit_generator(datagen.flow(X_train,Y_train,batch_size=65),epochs=50,validation_data=(X_val,Y_val),steps_per_epoch=X_train.shape[0]//65,callbacks=[learning_decay])
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))


#ensemble 15 model predictions
    
results=np.zeros((test.shape[0],10))
for j in range(15):
    results=results+model[j].predict(test)
#predictions=results
results=np.argmax(results,axis=1)
results=pd.Series(results)
results=pd.Series(results,name='Label')
submission=pd.concat([pd.Series(range(1,28001),name='ImageId'),results],axis=1)
submission.to_csv('MNSIT-ensemble.csv',index=False)
    


#model fit without augmentation
#model.fit(train,train_label,epochs=100,batch_size=25)
#print(model.summary())

#predictions=model.predict(test)
#pred=np.argmax(predictions,axis=1)
#df=pd.DataFrame(np.arange(1,28001),columns=['ImageId'])
#df['Label']=pred
#df.reset_index(drop=True,inplace=True)
#
#df.to_csv('Submissionrms.csv',index=False)