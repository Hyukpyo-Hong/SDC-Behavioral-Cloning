import numpy as np
import csv
import tensorflow as tf

tf.python.control_flow_ops = tf

# Train Parameter
epoch = 20
shift = 0.25  # Additional value to if there're images from left and right camera
shape = (100, 200)  # Shape of resize before crop
validate_portion = 0.01
learning_rate = 0.001
data_create_or_load = 1  # 0: Create new Dataset and save, 1: Load previous Dataset
batch_size = 256

# Model
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.regularizers import l2


def model():
    model = Sequential()
    elu = ELU(alpha=1.0)    
    model.add(Convolution2D(24, 5, 5,input_shape=(66, 200, 3),subsample=(2, 2),name='C1'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5,subsample=(2, 2), name='C2'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5,subsample=(2, 2),name='C3'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3,name='C4'))
    model.add(elu)
    model.add(Convolution2D(64, 3, 3,W_regularizer=l2(0.01),name='C5'))
    model.add(elu)
    model.add(Flatten())
    model.add(Dense(100,name='L1'))
    model.add(elu)
    model.add(Dense(50,name='L2'))    
    model.add(elu)
    model.add(Dense(10,name='L3'))
    model.add(Dense(1,name='L4'))
    return model

# Functions
from scipy.misc import imread
import scipy.misc as sp
import cv2

def flip_merge(data):
	length = len(data['imgc'])
	X_train = []
	y_train = []
	count = 0
	aug_left_count=0
	aug_right_count=0
	rows = 66
	cols = 200
	m_right = np.float32([[1,0,10],[0,1,0]])
	m_left = np.float32([[1,0,-10],[0,1,0]])
	m_right2 = np.float32([[1,0,30],[0,1,0]])
	m_left2 = np.float32([[1,0,-30],[0,1,0]])
	for i in range(3):
		if data[0][i] == 'center': 
			for i, loc in zip(range(length), data['imgc']):
				if(i == 0):
					continue
				else:
					try:
						image = sp.imresize(imread(loc), size=shape)
						image = image[30:96, :]
						X_train.append(image)
						y_train.append(data['angle'][i])
						if data['angle'][i] >0.3:							
							temp = cv2.warpAffine(image,m_right2,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]+0.35)
							aug_right_count+=1
						elif data['angle'][i] >0.1:							
							temp = cv2.warpAffine(image,m_right,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]+0.15)
							aug_right_count+=1
						elif data['angle'][i] <-0.3:							
							temp = cv2.warpAffine(image,m_left2,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]-0.35)
							aug_left_count+=1
						elif data['angle'][i] <-0.1:							
							temp = cv2.warpAffine(image,m_left,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]-0.15)
							aug_left_count+=1
						print("Center camera resizing", i, "/", length - 1," (+ left Augmented: ",aug_left_count,", right Augmented: ",aug_right_count)
					except OSError:
						count += 1
						pass
		elif data[0][i] == 'left':
			for i, loc in zip(range(length), data['imgl']):
				if(i == 0):
					continue
				elif(len(loc) == 0):
					break
				else:
					try:
						image = sp.imresize(imread(loc), size=shape)
						X_train.append(image[30:96, :])
						y_train.append(data['angle'][i] + shift)
						'''
						if data['angle'][i] >0.1:							
							temp = cv2.warpAffine(image,m_right,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]+0.1)
							temp = cv2.warpAffine(image,m_right2,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]+0.3)
							aug_right_count+=1
						elif data['angle'][i] <-0.1:							
							temp = cv2.warpAffine(image,m_left,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]-0.1)
							temp = cv2.warpAffine(image,m_left2,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]-0.3)
							aug_left_count+=1
						'''
						print("left camera resizing", i, "/", length - 1," (+ left Augmented: ",aug_left_count,", right Augmented: ",aug_right_count)
					except OSError:
						count += 1
						pass
		elif data[0][i] == 'right':
			for i, loc in zip(range(length), data['imgr']):
				if(i == 0):
					continue
				elif(len(loc) == 0):
					break
				else:
					try:
						image = sp.imresize(imread(loc), size=shape)
						X_train.append(image[30:96, :])
						y_train.append(data['angle'][i] - shift)
						'''
						if data['angle'][i] >0.1:							
							temp = cv2.warpAffine(image,m_right,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]+0.1)
							temp = cv2.warpAffine(image,m_right2,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]+0.3)
							aug_right_count+=1
						elif data['angle'][i] <-0.1:							
							temp = cv2.warpAffine(image,m_left,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]-0.1)
							temp = cv2.warpAffine(image,m_left2,(cols,rows))
							X_train.append(temp)
							y_train.append(data['angle'][i]-0.3)
							aug_left_count+=1
						'''
						print("Right camera resizing", i, "/", length - 1," (+ left Augmented: ",aug_left_count,", right Augmented: ",aug_right_count)
					except OSError:
						count += 1
						pass
	print(count, "files don't exist. Total number of imagaes is ", len(X_train) * 2)
	print("Fliping..")
	y_train = np.array(y_train).astype(np.float32)	
	a = []
	for i in range(len(X_train)):
		a.append(np.fliplr(X_train[i]))
	X_train = np.array(X_train)
	X_train = np.vstack((X_train, a))
	y_train = np.concatenate((y_train, -y_train))
	
	print("X_train shape: ", X_train.shape)
	print("y_train shape: ", y_train.shape)
	return X_train, y_train

import h5py
def save():
	#dummy
	#data = np.genfromtxt('./dummy.csv',dtype=[('imgc','U110'),('imgl','U110'),('imgr','U110'),('angle','f8')],delimiter=",",usecols=(0,1,2,3))
	#Real
	data = np.genfromtxt('./data/driving_log.csv',dtype=[('imgc','U110'),('imgl','U110'),('imgr','U110'),('angle','f8')],delimiter=",",usecols=(0,1,2,3))
	#data = np.genfromtxt('./DATAmine/driving_log.csv',dtype=[('imgc','U110'),('imgl','U110'),('imgr','U110'),('angle','f8')],delimiter=",",usecols=(0,1,2,3))
	X_train, y_train = flip_merge(data)

	try:
		with h5py.File('X_train.h5', 'w') as hf:
			hf.create_dataset("X_train",  data=X_train)
	except:
		print("X_Train Cannot be Save")
		pass	
	np.save("y_train", y_train)
	return X_train, y_train
	

def load():	
	with h5py.File('X_train.h5', 'r') as hf:
		X_train = hf['X_train'][:]


	y_train = np.load("y_train.npy")
	return X_train, y_train


def generator(X_train, y_train, batch_size):
	batch_train = np.zeros((batch_size, 66, 200, 3))
	batch_angle = np.zeros(batch_size)
	while True:
		for i in range(batch_size):
			batch_train[i] = X_train[i]
			batch_angle[i] = y_train[i]
			#batch_train[i] = batch_train[i]/255.-.5  # Nomalization
		yield batch_train, batch_angle

# Data initialize
X_train = []
y_train = []
if(data_create_or_load == 0):
	X_train, y_train = save()
else:
	X_train, y_train = load()


# Shuffle and Split into Train and Validate set
from sklearn.utils import shuffle
from math import ceil
print("Shuffling..")
X_train, y_train = shuffle(X_train, y_train)
training_idx = ceil(len(X_train) * (1 - validate_portion))
X_validate = X_train[training_idx:]
y_validate = y_train[training_idx:]
X_train = X_train[:training_idx]
y_train = y_train[:training_idx]

print("Training Set: ", X_train.shape)
print("Validation Set: ", X_validate.shape)


# Complie
model = model()
model.compile(Adam(lr=learning_rate), loss='mse', metrics=['accuracy'])


# Train
model.fit_generator(generator(X_train, y_train, batch_size),
	samples_per_epoch=len(X_train),
	nb_epoch=epoch,
	validation_data=generator(X_train, y_train, batch_size),
	nb_val_samples=len(X_validate))

# Save Model
from keras.models import load_model

model.save('model.h5')
print("Saved model to disk")
# print(model.summary())
