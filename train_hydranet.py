import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,Input
from keras.optimizers import Adam
import random
import pandas as pd
import cv2
import os
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import pandas as pd
from keras.callbacks import ModelCheckpoint,EarlyStopping



data=open('throttle_str_info.txt','r').readlines()
data=list(map(lambda x:x.strip('\n'),data))
data=list(map(lambda x:'./imgs/'+x,data))
img_paths=list(map(lambda x:x.split('<sep>')[0],data))
throttles=list(map(lambda x:x.split('<sep>')[1],data))
throttles=list(map(lambda x:float(x)/2,throttles))
str_angles=list(map(lambda x:x.split('<sep>')[-1],data))
str_angles=list(map(lambda x:float(x),str_angles))


data_df=pd.DataFrame({"img_paths":img_paths,"throttles":throttles,"steering_angles":str_angles})
data_df_d=data_df.loc[data_df.throttles>0]
img_paths=list(data_df_d["img_paths"])
throttles=list(data_df_d["throttles"])
steering_angles=list(data_df_d["steering_angles"])



img_paths,str_angles,throttles=shuffle(img_paths,steering_angles,throttles)

img_paths_train=img_paths[:int(0.9*len(img_paths))]
img_paths_test=img_paths[int(0.9*len(img_paths)):]

str_angles_train=str_angles[:int(0.9*len(str_angles))]
str_angles_test=str_angles[int(0.9*len(str_angles)):]

throttles_train=throttles[:int(0.9*len(throttles))]
throttles_test=throttles[int(0.9*len(throttles)):]


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

def img_preprocess(img):
  img=img[60:135,:,:]
  img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img=cv2.GaussianBlur(img,(3,3),0)
  img=cv2.resize(img,(200,66))
  img=img/255
  return img

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

def batch_generator(image_paths,steering_angle,throttles,batch_size,is_training):
  while True:
    batch_img=[]
    batch_steering=[]
    batch_throttle=[]
    
  
    for i in range(batch_size):
      random_index=random.randint(0,(len(image_paths)-1))
      if is_training:
        im,steering=random_augument(image_paths[random_index],steering_angle[random_index])
        throttle=throttles[random_index]
      else:
        im=mpimg.imread(image_paths[random_index])
        steering=steering_angle[random_index]
        throttle=throttles[random_index]
      im=img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
      batch_throttle.append(throttle)
      
    yield np.asarray(batch_img),[np.asarray(batch_steering),np.asarray(batch_throttle)]

#Define the hydranet.

#The backbone of hydranet.
input_node=Input(shape=((66,200,3)))
conv1=Conv2D(filters=24,kernel_size=(5,5),strides=(2,2),activation='elu')(input_node)
conv2=Conv2D(filters=36,kernel_size=(5,5),strides=(2,2),activation='elu')(conv1)
conv3=Conv2D(filters=48,kernel_size=(5,5),strides=(2,2),activation='elu')(conv2)
conv4=Conv2D(filters=64,kernel_size=(3,3),activation='elu')(conv3)
drp1=Dropout(0.2)(conv4)
conv5=Conv2D(filters=64,kernel_size=(3,3),activation='elu')(drp1)
flat_feat=Flatten()(conv5)


#Branch for steering prediction.
st_layer1=Dense(100,activation='elu')(flat_feat)
st_layer2=Dense(50,activation='elu')(st_layer1)
st_layer3=Dense(10,activation='elu')(st_layer2)
out1=Dense(1,name='steer_out')(st_layer3)

#Branch for predicting throttle.

th_layer1=Dense(100,activation='elu')(flat_feat)
th_layer2=Dense(50,activation='elu')(th_layer1)
th_layer3=Dense(10,activation='elu')(th_layer2)
out2=Dense(1,name='throttle_out')(th_layer3)

hydranet=Model(inputs=input_node,outputs=[out1,out2])


hydranet.compile(optimizer='adam',
                  loss={'steer_out': 'mse', 
                        'throttle_out': 'mse'},
                  loss_weights={'steer_out': 1, 
                                'throttle_out': 0.2})





mc = ModelCheckpoint(
    filepath='hydranet.h5',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


es=EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=4,
    verbose=0,
    mode="min",
    baseline=None,
    restore_best_weights=False,
)

history = hydranet.fit_generator(batch_generator(img_paths_train, str_angles_train,throttles_train, 200, 1),
                                  steps_per_epoch=1200, 
                                  epochs=20,
                                  validation_data=batch_generator(img_paths_test, str_angles_test,throttles_test, 100, 0),
                                  validation_steps=200,
                                  verbose=True,
                                 callbacks=[mc,es],
                                  shuffle = 1)