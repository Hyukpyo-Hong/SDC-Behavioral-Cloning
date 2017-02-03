def model_1():
	model = Sequential()
	model.add(Convolution2D(24, 5, 5,input_shape=(80, 160, 3),subsample=(2, 2)))
	model.add(Convolution2D(36, 5, 5,subsample=(2, 2)))
	model.add(Convolution2D(48, 5, 5,subsample=(2, 2)))
	model.add(Convolution2D(64, 3, 3))
	model.add(Convolution2D(64, 3, 3))
	model.add(Flatten())
	model.add(Dense(100,activation='relu'))
	model.add(Dense(50,activation='relu'))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

def model_2():
	model = Sequential()
	#input(80,160,3) output(24,88,24)
	model.add(Convolution2D(24, 5, 5,input_shape=(80, 160, 3),subsample=(2, 2)))	
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

def model_3():
    model = Sequential()
    elu = ELU(alpha=1.0)
    #input(38,160,3) output(36,88,24)
    model.add(Convolution2D(24, 3, 3,input_shape=(38, 160, 3),name='C1'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5,name='C2'))
    model.add(elu)              
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5,name='C3'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3,name='C4'))
    model.add(elu)
    model.add(Convolution2D(64, 3, 3,name='C5'))
    model.add(elu)
    model.add(Flatten())
    model.add(Dense(100,activation='tanh',name='L1'))
    model.add(Dense(50,activation='tanh',name='L2'))
    model.add(Dense(10,name='L3'))
    model.add(Dense(1,name='L4'))
    return model

def model_4():	
    model = Sequential()
    elu = ELU(alpha=1.0)    
    model.add(Convolution2D(24, 5, 5,input_shape=(66, 200, 3),subsample=(2, 2),name='C1'))    
    model.add(Convolution2D(36, 5, 5,subsample=(2, 2), name='C2'))    
    model.add(Convolution2D(48, 5, 5,subsample=(2, 2),name='C3'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3,name='C4'))    
    model.add(Convolution2D(64, 3, 3,name='C5'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100,name='L1'))        
    model.add(Dense(50,name='L2'))        
    model.add(Dense(10,name='L3'))
    model.add(Dense(1,name='L4'))
    return model

def model5():
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
    model.add(Convolution2D(64, 3, 3,name='C5'))
    model.add(elu)
    model.add(Flatten())
    model.add(Dense(100,name='L1'))
    model.add(elu)
    model.add(Dense(50,name='L2'))    
    model.add(elu)
    model.add(Dense(10,name='L3'))
    model.add(Dense(1,name='L4'))
    return model

 1. Augment Images (Merge left, right 0.25 and flip) 
    model 4
    lr 0.0001 
    epoch 5
    
    : only right 1~2


 2. Augment Images (Merge left, right 0.25 and flip) 
    !model 5 
    lr 0.0001 
    epoch 5
    
    :Little right-foward and off lane
    

3. Augment Images (Merge left, right 0.25 and flip) 
    model 5
    !lr 0.001       
    epoch 5
    
    : Until black bridge !!!


4. Augment Images (Merge left, right !0.15 and flip) 
    model 5
    lr 0.001       
    epoch 5
    
    : before bridge, off the lane at left turn

5. Augment Images (Merge left, right !0.2 and flip) 
    model 5
    lr 0.001       
    epoch 5
    
    : before bridge, off the lane at left turn