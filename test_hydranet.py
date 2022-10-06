import socketio
from flask import Flask
import eventlet
import numpy as np
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sio=socketio.Server()
app=Flask(__name__)
speed_limit=5
hydranet=load_model('hydranet.h5')
#model_thr=load_model('SDCThrottle.h5')



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
    image=img_preprocess(image)
    image=np.array([image])
    mdl_pred=hydranet.predict(image)
    steering_angle=float(mdl_pred[0])
    throttle=float(mdl_pred[1])
    #throttle=1.0-speed/speed_limit
    print('{} {} {}'.format(steering_angle,throttle,speed))

    
    send_control(steering_angle,throttle)




@sio.on('connect')
def connect(sid,environ):
    print('Connected')
    send_control(0,0.2)

def send_control(steering_angle,throttle):
    sio.emit('steer',data={
    'steering_angle':steering_angle.__str__(),
    'throttle':throttle.__str__()
    })

if __name__=='__main__':
    app=socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(('',4567)),app)
