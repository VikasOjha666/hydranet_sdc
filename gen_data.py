import socketio
from flask import Flask
import eventlet
import numpy as np
from keras.models import load_model
from multiprocessing import Pool
import base64
from io import BytesIO
from PIL import Image
import cv2
import os
import uuid
import asyncio

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sio=socketio.Server()
app=Flask(__name__)
speed_limit=5
model_steer=load_model('SDCModifiedTrue.h5')


if os.path.exists('imgs') is False:
    os.mkdir('imgs')
f=open('throttle_info.txt','w')


def save_image(img,throttle,str_ang):
    
    filename = str(uuid.uuid4())
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./imgs/'+filename+'.jpg',img)
    f.write(filename+'.jpg'+'<sep>'+str(throttle)+'<sep>'+str(str_ang)+'\n')

def img_preprocess(img):

  img=img[60:135,:,:]
  img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
  img=cv2.GaussianBlur(img,(3,3),0)
  img=cv2.resize(img,(200,66))
  img=img/255
  return img

@sio.on('telemetry')
def telemetry(sid,data):
    speed=float(data['speed'])
    image=Image.open(BytesIO(base64.b64decode(data['image'])))
    
    image=np.asarray(image)
    img_c=image.copy()
    image=img_preprocess(image)
    image=np.array([image])

    steering_angle=float(model_steer.predict(image))
    #throttle=float(model_thr.predict(image))
    throttle=1.0-speed/speed_limit
    
    save_image(img_c,throttle,steering_angle)
    print('{} {} {}'.format(steering_angle,throttle,speed))

    
    send_control(steering_angle,throttle)




@sio.on('connect')
def connect(sid,environ):
    print('Connected')
    send_control(0,0)

@sio.on('disconnect')
def disconnect(sid):
    print('Disconnected')
    f.close()
def send_control(steering_angle,throttle):
    sio.emit('steer',data={
    'steering_angle':steering_angle.__str__(),
    'throttle':throttle.__str__()
    })

if __name__=='__main__':
    app=socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(('',4567)),app)
