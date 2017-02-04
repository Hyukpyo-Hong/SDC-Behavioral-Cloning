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

def model_6():
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
    model.add(Convolution2D(64, 3, 3,name='C5'))
    model.add(elu)
    model.add(Flatten())
    model.add(Dense(100,activation='tanh',name='L1'))
    model.add(Dense(50,activation='tanh',name='L2'))
    model.add(Dense(10,name='L3'))
    model.add(Dense(1,name='L4'))
    return model

def model_7():
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
    model.add(Dense(50,,name='L2'))
    model.add(elu)
    model.add(Dense(10,activation='tanh'name='L3'))
    model.add(Dense(1,name='L4'))
    return model

def model8():
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
    model.add(elu)
    model.add(Dense(1,name='L4'))
    return model

def model_9():
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
    
    :left-foward and off lane
    

3. Augment Images (Merge left, right 0.25 and flip) 
    model 5
    !lr 0.001       
    epoch 5
    
    : Until black bridge !!!

3-1. Augment Images (Merge left, right 0.25 and flip) 
    model 5
    lr 0.001       
    !epoch 7
    loss: 0.0149 - acc: 0.1802
    : Fail first right turn !!!

3-2. Augment Images (Merge left, right 0.25 and flip) 
    model !2
    lr 0.001       
    epoch 2
    
    : only left

3-3.!!Augment Images (Merge left, right 0.25 and flip) 
    model 5
    lr 0.001       
    !epoch 9 Generator
    loss: 0.0112 - acc: 0.1797 
    : Little ziazag, slow start, sucess 2 circles, 15mile/h  !!!

3-4.!! Augment Images (Merge left, right 0.25 and flip) 
    model 5
    lr 0.001       
    !epoch 7 Generator
    loss: 0.0273 - acc: 0.1562 
    : Faster and stable than 3-3 failed at bridge

3-5. Augment Images (Merge left, right 0.25 and flip) 
    model 5
    lr 0.001       
    !epoch 9 Generator
    loss: 0.0212 - acc: 0.1953
    : Little ziazag, slow start, failed first right turn

3-6. Augment Images (Merge left, right 0.25 and flip) 
    model 5
    !lr 0.01       
    epoch 9 Generator
    loss: 0.1515 - acc: 0.1185 
    : just Left turn

3-7. Augment Images (Merge left, right 0.25 and flip) 
    model 5
    !lr 0.001       
    epoch 13 Generator
    
    : less zigzag slow start(-6angel)

3-8. Augment Images (Merge left, right !!0.2 and flip) 
    model 5
    !lr 0.001       
    epoch 13 Generator
    loss: 0.0085 - acc: 0.1875   

    right turn fail / slow start 

3-8. Augment Images (Merge left, right 0.25 and flip) 
    model 5
    !lr 0.001       
    epoch 8 Generator
    loss: 0.0264 - acc: 0.1484 

    slow start(5) / fail after bridge left corner
    
4. Augment Images (Merge left, right !0.15 and flip) 
    model 5
    lr 0.001       
    epoch 5
    
    : before bridge, off the lane at left turn

5. Augment Images (Merge left, right !0.2 and flip) 
    model 5
    lr 0.001       
    epoch 5
    
    : only right turn (why?)

5-1. Augment Images (Merge left, right 0.2 and flip) 
    model !2
    lr 0.001       
    epoch 2
    loss: 0.0096 - acc: 0.1953  
    : almost stop

6. Augment Images (Merge left, right 0.25 and flip) 
    model !6
    !lr 0.001       
    epoch 7 Generator
    
    loss: 0.0520 - acc: 0.1719     
	slow start(5) / fail after bridge left corner

7. Augment Images (Merge left, right 0.25 and flip) 
    !model 7
    lr 0.001       
    epoch 7 Generator
	loss: 0.0747 - acc: 0.1641   
    only right
    
7-1. Augment Images (Merge left, right 0.25 and flip) 
    model 7
    !lr 0.0001       
    epoch 7 Generator
    
    only left
    
8. Augment Images (Merge left, right 0.25 and flip) 
    model 8
    lr 0.001       
    epoch 7 Generator
    loss: 0.0462 - acc: 0.2031  
    
    so many ziazag

8. Augment Images (Merge left, right 0.25 and flip) 
    model 9
    lr 0.001       
    epoch 7 Generator
    


    


