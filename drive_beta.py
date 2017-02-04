import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import scipy.misc as sp

sio = socketio.Server()
app = Flask(__name__)
model = None
img_gen = None
shape = (100,200)
prev_angle=0

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car    
    throttle = data["throttle"]    
    # The current speed of the car    
    speed = float(data["speed"])    
    
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = sp.imresize(image_array,size=shape) 
    image_array = image_array[30:96,:]                    
    transformed_image_array = image_array[None, :, :, :]    
    
    #steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    steering_angle = float(model.predict_generator(img_gen.flow(transformed_image_array, batch_size=1), val_samples=1))
    
    # The driving model
    global prev_angle
    _diff = steering_angle-prev_angle
    diff = abs(_diff)    
    if(diff>=0 and diff<0.05):
        prev_angle = steering_angle
    elif(diff>=0.05 and diff<0.95):
        print("Angle stablizer ","from: ",format(steering_angle,'.3f'))
        steering_angle = prev_angle+_diff*(0.95-diff)
        prev_angle = steering_angle
        print("                ","  to: ",format(steering_angle,'.3f'))
    else:        
        print("Angle stablizer ","from: ",format(steering_angle,'.3f'))
        steering_angle = prev_angle+_diff*0.1
        prev_angle = steering_angle
        print("                ","  to: ",format(steering_angle,'.3f'))        
    if (diff<0.1 and steering_angle<0.5): 
        throttle = 0.4
    else:
        if(speed<5):
            throttle = 0.4 #Boost when the car start with large degree
        elif (diff<0.15 and steering_angle>0.6):
            throttle = 0.2
        else:
            throttle = 0.1 #3.75<
    print("Current Angle: ",format(steering_angle*25,'.2f'),"(",format(steering_angle,'.3f'),")"," Throttle: ", throttle,"\n")  
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    img_gen = ImageDataGenerator()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
'''
else:
        if(speed<5):
            throttle = 0.4 #Boost when the car start with large degree
        elif (abs(steering_angle)<0.15): # 2.5< <3.75degree
            throttle = 0.3
        else:
            throttle = 0.2 #3.75<
    print("Current Angle: ",format(steering_angle*25,'.2f'),"(",format(steering_angle,'.3f'),")"," Throttle: ", throttle,"\n")  
    send_control(steering_angle, throttle)
'''