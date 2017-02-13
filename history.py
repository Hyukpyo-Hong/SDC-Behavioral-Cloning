def model1():
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

def model2():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, input_shape=(
        66, 200, 3), subsample=(2, 2), name='C1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), name='C2'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), name='C3'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, name='C4'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, name='C5'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100, name='L1'))
    model.add(Activation('relu'))
    model.add(Dense(50, name='L2'))
    model.add(Activation('relu'))
    model.add(Dense(10, name='L3'))
    model.add(Dense(1, name='L4'))
    return model

def model3():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, input_shape=(
        66, 200, 3), subsample=(2, 2), name='C1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), name='C2'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), name='C3'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, name='C4'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, name='C5'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100, name='L1'))
    model.add(Activation('relu'))
    model.add(Dense(50, name='L2'))
    model.add(Activation('relu'))
    model.add(Dense(10, name='L3'))
    model.add(Dense(1, name='L4'))
    return model    

def model4():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, input_shape=(
        66, 200, 3), subsample=(2, 2), name='C1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), name='C2'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), name='C3'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, name='C4'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, name='C5'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, name='L1'))
    model.add(Activation('relu'))
    model.add(Dense(50, name='L2'))
    model.add(Activation('relu'))
    model.add(Dense(10, name='L3'))
    model.add(Dense(1, name='L4'))
    return model        

1-1.!!Augment Images (Merge left, right 0.25 and flip) 
    model 1
    lr 0.001       
    !epoch 9 Generator
    loss: 0.0112 - acc: 0.1797 
    : Little ziazag, slow start, sucess 2 circles, 15mile/h  !!!

1-2.!! Augment Images (Merge left, right 0.25 and flip) 
    model 1
    lr 0.001       
    !epoch 7 Generator
    loss: 0.0273 - acc: 0.1562 
    : Faster and stable than 3-3 failed at bridge


2-1.Augment Images (Merge left, right 0.25 and flip) 
    !model 2 (relu)
    lr 0.001       
    epoch 10 Generator
    sharp curv failed

3.Augment Images (Merge left, right 0.25 and flip) 
    !model 3 (reduce dropout)
    lr 0.001       
    epoch 10 Generator
    no nomalization

    strait fail curv fail
    

3-1.Augment Images (Merge left, right 0.25 and flip) 
    !model 3 (reduce dropout)
    lr 0.001       
    epoch 10 Generator
    nomalization

    curv fail / zigzag


4-1.Augment Images (Merge left, right 0.25 and flip) 
    !model 4 (add dropout)
    lr 0.001       
    epoch 10 Generator
    nomalization

    curv fail / redueced zigzag than 3-1

4-1.Augment Images (Merge left, right 0.25 and flip) 
    !model 4 (add dropout)
    lr 0.001       
    epoch 10 Generator
    nomalization / remove flipping

    fail to enter bridge / so so 

5-1.Augment(left / right +-30px, +-0.3) Images (Merge left, right 0.25 and flip) 
    model 4 (add dropout)
    lr 0.001       
    epoch 10 Generator
    nomalization

    fail to enter bridge / little stable

5-2.Augment(angle<> 0.1 left / right +-30px, +-0.3) Images (Merge left, right 0.25 and flip) 
    model 4 (add dropout)
    lr 0.001       
    !epoch 5 Generator
    nomalization

    pass bridge, then fail to left turn

5-2.Augment(angle<> 0.1 left / right +-20px, +-0.25) Images (Merge left, right 0.25 and flip) 
    model 4 (add dropout)
    lr 0.001       
    !epoch 10 Generator
    nomalization

    soso

5-3.Augment(ange<> 0.01, left / right +-20px, +-0.2 / +-30px, +-0.3) Images (Merge left, right 0.25 and flip) 
    model 4 (add dropout)
    lr 0.001       
    epoch 10 Generator
    nomalization

    zigzag fial    
    
5-4.Augment(angle<> 0.1 left / right +-20px, +-0.2 / +-30px, +-0.25) Images (Merge left, right 0.25 and flip) 
    model 4 (add dropout)
    lr 0.001       
    epoch 5 Generator
    nomalization

        
    


