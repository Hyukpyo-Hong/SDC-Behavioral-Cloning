import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import scipy.misc as sp
sio = socketio.Server()
app = Flask(__name__)
model = None
shape = (100, 200)
prev_angle = 0


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
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
        image_array = sp.imresize(image_array, size=shape)
        image_array = (image_array[30:96, :]) / 255. - .5
        steering_angle = float(model.predict(
            image_array[None, :, :, :], batch_size=1))
        
        global prev_angle
        _diff = steering_angle - prev_angle
        diff = abs(_diff)
        if(speed < 15):
            throttle = 0.6  # Boost when the car start with large degree                   
        elif(diff>0.05):        	
        	print("Angle stablizing. Original Angle: ",format(steering_angle, '.3f'))
        	steering_angle = prev_angle + _diff*0.7
        	throttle = -0.3        	
        else:
            if (abs(steering_angle) < 0.1):
            	throttle = 0.5
            elif (abs(steering_angle) < 0.15): 
            	throttle = 0.2
            elif (abs(steering_angle) < 0.2): 
                throttle = 0.1
            elif (abs(steering_angle) < 0.25): 
                throttle = 0.05
            else:
                throttle = 0.01  # 3.75<
        prev_angle = steering_angle


        print("Angle: ", format(steering_angle, '.3f'),
              " Throttle: ", throttle, " Speed: ", speed)

        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

'''
        global prev_angle
        _diff = steering_angle - prev_angle
        diff = abs(_diff)
        if(diff >= 0 and diff < 0.05):
            prev_angle = steering_angle
        elif(diff >= 0.05 and diff < 0.95):
            print("Angle stablizer ", "from: ", format(steering_angle, '.3f'))
            steering_angle = prev_angle + _diff * (1.2 - diff)
            prev_angle = steering_angle
            print("                ", "  to: ", format(steering_angle, '.3f'))
        else:
            print("Angle stablizer ", "from: ", format(steering_angle, '.3f'))
            steering_angle = prev_angle + _diff * 0.1
            prev_angle = steering_angle
            print("                ", "  to: ", format(steering_angle, '.3f'))
'''
