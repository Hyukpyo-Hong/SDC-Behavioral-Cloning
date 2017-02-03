import numpy as np
import csv
import tensorflow as tf

tf.python.control_flow_ops = tf

#Resizing
from scipy.misc import imread
import scipy.misc as sp


def flip_merge(data):

    
    length = len(data['imgc'])
    X_train=[]
    y_train=[]
     
    for i in range(3):        
        if data[0][i]=='center':
            for i, loc in zip(range(length),data['imgc']):
                if(i==0):
                    continue
                else:
                    image = sp.imresize(imread(loc),size=shape)
                    X_train.append(image[30:96,:])                      
                    y_train.append(data['angle'][i])
                    print("Center camera resizing",i,"/",length-1)
                    
        elif data[0][i]=='left':
            for i, loc in zip(range(length),data['imgl']):
                if(i==0):
                    continue
                else:
                    image = sp.imresize(imread(loc),size=shape)
                    X_train.append(image[30:96,:])
                    y_train.append(data['angle'][i]+shift)
                    print("Left camera resizing",i,"/",length-1)
                    
        elif data[0][i]=='right':
            for i, loc in zip(range(length),data['imgr']):
                if(i==0):
                    continue
                else:
                    image = sp.imresize(imread(loc),size=shape)
                    X_train.append(image[30:96,:])         
                    y_train.append(data['angle'][i]-shift)
                    print("Right camera resizing",i,"/",length-1)            
                    
    print("Fliping..")    
    y_train = np.array(y_train).astype(np.float32)    
    a=[]
    for i in range(len(X_train)):
        a.append(np.fliplr(X_train[i]))
    X_train = np.array(X_train)
    X_train = np.vstack((X_train,a))
    y_train = np.concatenate((y_train,-y_train))
    print("X_train shape: ",X_train.shape)
    print("y_train shape: ",y_train.shape)
    return X_train, y_train
    
#Model
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D   
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam

def model():
    model = Sequential()
    #input(80,160,3) output(24,88,24)
    model.add(Convolution2D(24, 5, 5,input_shape=(66, 200, 3),subsample=(2, 2)))    
    model.add(Dropout(0.5)) 
    model.add(Convolution2D(36, 5, 5,subsample=(2, 2))) 
    model.add(Dropout(0.5)) 
    model.add(Convolution2D(48, 5, 5,subsample=(2, 2))) 
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3))      
    model.add(Convolution2D(64, 3, 3))      
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

#Data Loading



epoch = 2
shift = 0.25
shape = (100,200)
X_train = []
y_train = []

def save():
    #dummy
    data = np.genfromtxt('./a.csv',dtype=[('imgc','U110'),('imgl','U110'),('imgr','U110'),('angle','f8')],delimiter=",",usecols=(0,1,2,3))
    #udacity
    #data = np.genfromtxt('./data/driving_log.csv',dtype=[('imgc','U110'),('imgl','U110'),('imgr','U110'),('angle','f8')],delimiter=",",usecols=(0,1,2,3))
    X_train, y_train = flip_merge(data)
    np.save("X_train",X_train)
    np.save("y_train",y_train)
    return X_train, y_train
    
def load():
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")    
    return X_train, y_train
    

#X_train, y_train = save()
print(X_train)
X_train, y_train = load()
print(X_train)


#Shuffle
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train) 

#Complie
model = model()
model.compile(Adam(lr=0.001), loss='mse', metrics=['accuracy'])

def generator(X_train, y_train, batch_size):
    batch_train = np.zeros((batch_size, 66, 200, 3))
    batch_angle = np.zeros(batch_size)    
    while True:
        for i in range(batch_size):
            rdx = np.random.choice(len(y_train))
            batch_train[i] = X_train[rdx]
            batch_angle[i] = y_train[rdx]            
        yield batch_train, batch_angle
        
   
#generator = generator(X_train, y_train, 128)
#history = model.fit_generator(generator, len(X_train), nb_epoch=4)
#Train
history = model.fit(X_train, y_train, nb_epoch=epoch, validation_split=0.1)



#Save Model
from keras.models import load_model
import json

json_str = model.to_json()
with open('model.json','w') as f:
    json.dump(json_str, f)
model.save_weights('model.h5')
print("Saved model to disk")