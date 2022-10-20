import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
import random
import pandas as pd
import cv2
import os
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
from sklearn.utils import shuffle



# In[4]:


data=open('throttle_str_info.txt','r').readlines()
data=list(map(lambda x:x.strip('\n'),data))
data=list(map(lambda x:'./imgs/'+x,data))
img_paths=list(map(lambda x:x.split('<sep>')[0],data))
str_angles=list(map(lambda x:x.split('<sep>')[-1],data))
str_angles=list(map(lambda x:float(x),str_angles))

img_paths,str_angles=shuffle(img_paths,str_angles)

X_train,X_valid,y_train,y_valid=train_test_split(img_paths,str_angles,test_size=0.2,random_state=6)


# In[55]:


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

def img_random_flip(image):
    image = cv2.flip(image,1)
    return image

def random_augument(image,str_ang):
  image=mpimg.imread(image)
  if np.random.rand()<0.5:
    image=pan(image)
  if np.random.rand()<0.5:
    image=zoom(image)
  if np.random.rand()<0.5:
    image=img_random_brightness(image)
  if np.random.rand()<0.5:
    image=img_random_flip(image)
    str_ang=-str_ang
  return image,str_ang


def batch_generator(image_paths,str_angles,batch_size,is_training):
  while True:
    batch_img=[]
    batch_steering=[]
  
    for i in range(batch_size):
      random_index=random.randint(0,(len(image_paths)-1))
      if is_training:
        im,str_ang=random_augument(image_paths[random_index],str_angles[random_index])

      else:
        im=mpimg.imread(image_paths[random_index])
        str_ang=str_angles[random_index]
      im=img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(str_ang)
    yield np.asarray(batch_img),np.asarray(batch_steering)
    
def img_preprocess(img):
  img=img[60:135,:,:]
  img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img=cv2.GaussianBlur(img,(3,3),0)
  img=cv2.resize(img,(200,66))
  img=img/255
  return img


# In[53]:


model=Sequential()
model.add(Conv2D(24,kernel_size=(5,5),strides=(2,2),input_shape=(66,200,3),activation='elu'))
model.add(Conv2D(36,kernel_size=(5,5),strides=(2,2),activation='elu'))
model.add(Conv2D(48,kernel_size=(5,5),strides=(2,2),activation='elu'))
model.add(Conv2D(64,kernel_size=(3,3),activation='elu'))
model.add(Dropout(0.2))
model.add(Conv2D(64,kernel_size=(3,3),activation='elu'))



model.add(Flatten())
model.add(Dense(100,activation='elu'))
#model.add(Dropout(0.25))

model.add(Dense(50,activation='elu'))

model.add(Dense(10,activation='elu'))

model.add(Dense(1))

adam=Adam(lr=1e-3)
model.compile(loss='mse',optimizer=adam)


# In[56]:


history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch=600, 
                                  epochs=5,
                                  validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)




model.save('SDCsteer2.h5')

