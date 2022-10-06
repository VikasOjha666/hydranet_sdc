# -*- coding: utf-8 -*-
"""
Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TwnqL34fbIgGvOpSOKda_AiSXYMvR_IK
"""


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
import random
import pandas as pd
import cv2
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from imgaug import augmenters as iaa

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def get_filenames(path):
    filename=path.split()[-1].split(sep='\\')[-1]
    return filename

columns=['center','left','right','steering','throttle','reverse','speed']
data_csv=pd.read_csv('data/driving_log.csv',names=columns)

data_csv['center']=data_csv['center'].apply(get_filenames)
#data_csv['left']=data_csv['left'].apply(get_filenames)
#data_csv['right']=data_csv['right'].apply(get_filenames)
hist,bins=np.histogram(data_csv['steering'],25)

#Drop the extra forward steering samples.
remove_list=[]
for i in range(25):
    temp=[]
    for j in range(len(data_csv['steering'])):
      if data_csv['steering'][j]>=bins[i] and data_csv['steering'][j]<=bins[i+1]:
        temp.append(j)
    temp=shuffle(temp)
    temp=temp[400:]
    remove_list.extend(temp)
data_csv.drop(data_csv.index[remove_list],inplace=True)

#We are using the center image and hence we are creating the pair array of paths and their corresponding steering angles.
img_path=[]
steering_angle=[]
for i in range(len(data_csv)):
    data=data_csv.iloc[i]
    img=data[0]
    str_angle=float(data[3])
    img_path.append(os.path.join('data/IMG/',img.strip()))
    steering_angle.append(str_angle)
img_path=np.asarray(img_path)
steering_angle=np.asarray(steering_angle)

X_train,X_valid,y_train,y_valid=train_test_split(img_path,steering_angle,test_size=0.2,random_state=6)
X_train.shape

def zoom(image):
  zoom=iaa.Affine(scale=(1,1.3))
  image=zoom.augment_image(image)
  return image

def pan(image):
  pan=iaa.Affine(translate_percent={"x":(-0.1,0.1),"y":(-0.1,0.1)})
  image=pan.augment_image(image)
  return image

def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

def img_random_flip(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return image, steering_angle

def random_augument(image,steering_angle):
  image=mpimg.imread(image)
  if np.random.rand()<0.5:
    image=pan(image)
  if np.random.rand()<0.5:
    image=zoom(image)
  if np.random.rand()<0.5:
    image=img_random_brightness(image)
  if np.random.rand()<0.5:
    image,steering_angle=img_random_flip(image,steering_angle)
  return image,steering_angle


def batch_generator(image_paths,steering_angle,batch_size,is_training):
  while True:
    batch_img=[]
    batch_steering=[]

    for i in range(batch_size):
      random_index=random.randint(0,(len(image_paths)-1))
      if is_training:
        im,steering=random_augument(image_paths[random_index],steering_angle[random_index])
      else:
        im=mpimg.imread(image_paths[random_index])
        steering=steering_angle[random_index]
      im=img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield np.asarray(batch_img),np.asarray(batch_steering)

def img_preprocess(img):
  img=img[60:135,:,:]
  img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img=cv2.GaussianBlur(img,(3,3),0)
  img=cv2.resize(img,(200,66))
  img=img/255
  return img

model=Sequential()
model.add(Conv2D(24,kernel_size=(5,5),strides=(2,2),input_shape=(66,200,3),activation='elu'))
model.add(Conv2D(36,kernel_size=(5,5),strides=(2,2),activation='elu'))
model.add(Conv2D(48,kernel_size=(5,5),strides=(2,2),activation='elu'))
model.add(Conv2D(64,kernel_size=(3,3),activation='elu'))
model.add(Dropout(0.2))
model.add(Conv2D(64,kernel_size=(3,3),activation='elu'))



model.add(Flatten())
model.add(Dense(100,activation='elu'))
model.add(Dropout(0.25))

model.add(Dense(50,activation='elu'))

model.add(Dense(10,activation='elu'))

model.add(Dense(1))

adam=Adam(lr=1e-3)
model.compile(loss='mse',optimizer=adam)

history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch=300,
                                  epochs=10,
                                  validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training','validation'])
# plt.title('Loss')
# plt.xlabel('Epoch')

model.save('SDCModifiedTrue.h5')

# from google.colab import files
# files.download('SDCModifiedTrue.h5')