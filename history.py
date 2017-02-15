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

def model5():
    model = Sequential()
    elu = ELU(alpha=1.0)    
    model.add(Convolution2D(24, 5, 5,input_shape=(66, 200, 3),subsample=(2, 2),b_regularizer=l1l2(l1=0.01, l2=0.01),W_regularizer=l1l2(l1=0.01, l2=0.01),name='C1'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5,subsample=(2, 2),b_regularizer=l1l2(l1=0.01, l2=0.01),W_regularizer=l1l2(l1=0.01, l2=0.01), name='C2'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5,subsample=(2, 2),b_regularizer=l1l2(l1=0.01, l2=0.01),W_regularizer=l1l2(l1=0.01, l2=0.01),name='C3'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3,b_regularizer=l1l2(l1=0.01, l2=0.01),W_regularizer=l1l2(l1=0.01, l2=0.01),name='C4'))
    model.add(elu)
    model.add(Convolution2D(64, 3, 3,b_regularizer=l1l2(l1=0.01, l2=0.01),W_regularizer=l1l2(l1=0.01, l2=0.01),name='C5'))
    model.add(elu)
    model.add(Flatten())
    model.add(Dense(100,name='L1'))
    model.add(elu)
    model.add(Dense(50,name='L2'))    
    model.add(elu)
    model.add(Dense(10,name='L3'))
    model.add(Dense(1,name='L4'))
    return model

def model6():
    model = Sequential()
    elu = ELU(alpha=1.0)    
    model.add(Convolution2D(24, 5, 5,input_shape=(66, 200, 3),subsample=(2, 2),W_regularizer=l1l2(l1=0.01, l2=0.01),name='C1'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5,subsample=(2, 2),W_regularizer=l1l2(l1=0.01, l2=0.01), name='C2'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5,subsample=(2, 2),W_regularizer=l1l2(l1=0.01, l2=0.01),name='C3'))
    model.add(elu)
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3,W_regularizer=l1l2(l1=0.01, l2=0.01),name='C4'))
    model.add(elu)
    model.add(Convolution2D(64, 3, 3,W_regularizer=l1l2(l1=0.01, l2=0.01),name='C5'))
    model.add(elu)
    model.add(Flatten())
    model.add(Dense(100,name='L1'))
    model.add(elu)
    model.add(Dense(50,name='L2'))    
    model.add(elu)
    model.add(Dense(10,name='L3'))
    model.add(Dense(1,name='L4'))
    return model

def model7():
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
    model.add(Dense(100,W_regularizer=l2(0.01),name='L1'))
    model.add(elu)
    model.add(Dense(50,name='L2'))    
    model.add(elu)
    model.add(Dense(10,name='L3'))
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
    model.add(Convolution2D(64, 3, 3,W_regularizer=l2(0.01),name='C5'))
    model.add(elu)
    model.add(Flatten())
    model.add(Dense(100,W_regularizer=l2(0.01),name='L1'))
    model.add(elu)
    model.add(Dense(50,name='L2'))    
    model.add(elu)
    model.add(Dense(10,name='L3'))
    model.add(Dense(1,name='L4'))
    return model

def model9():
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

def model10():
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
    model.add(Convolution2D(64, 3, 3,W_regularizer=l2(0.01),name='C4'))
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

    zigzag fail
    
5-4.Augment(angle<> 0.1 left / right +-20px, +-0.2 / +-30px, +-0.25) Images (Merge left, right 0.25 and flip) 
    model 4 (add dropout)
    lr 0.001       
    epoch 5 Generator
    nomalization

    fail

<Add images>
6-1.Augment(angle<> 0.1 left / right +-20px, +-0.2 / +-30px, +-0.3) Images (Merge left, right 0.25 and flip) 
    model 4 (add dropout)
    lr 0.001       
    epoch 5 Generator
    nomalization
        
    zigzag bridge crash


6-2.Augment(angle<> 0.1 left / right +-10px, +-0.2 / +-30px, +-0.3) Images (Merge left, right 0.25 and flip) 
    model 4 (add dropout)
    lr 0.001       
    epoch 10 Generator
    nomalization

    little zigzag fail first left turn 

   
6-2.Augment(angle<> 0.1 left / right +-10px, +-0.1 / +-30px, +-0.3) Images (Merge left, right 0.25 and flip) 
    model 4 (add dropout)    
    lr 0.001       
    !epoch 7 Generator
    !mean/std nomalization

    little zizgazg success first left / fail enter bridge

6-3 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:(angle<> 0.1 left,right +-10px,+-0.1 / +-30px, +-0.3) 
    model 4 (add dropout)    
    lr 0.001       
    batch_size : 128
    epoch: 7, Generator
    nomalization: mean/std 
    Creating image : True
    
    strait zigzag (0~3) : 2
    success strait : False
    first left turn :
    enter brideg :
    left turn :
    right turn :

6-4 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:(angle<> 0.1 left,right +-10px,+-0.1 / +-30px, +-0.3) 
    model 4 (add dropout)    
    lr 0.001       
    batch_size : 256
    epoch: 7, Generator
    nomalization: mean/std 
    Creating image : False
    
    strait zigzag (0~3) : 2
    success strait : True
    first left turn : True
    enter brideg : False
    left turn :
    right turn :


6-4 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:(angle<> 0.1 only center Camera add: +-10px,+-0.1 / +-30px, +-0.3) 
    model 4 (add dropout)    
    lr 0.001       
    epoch: 7, Generator
    nomalization: mean/std 
    Creating image : True
    batch_size : 256
    strait zigzag (0~3) : 2
    first left turn : True
    enter brideg :True
    left turn :False
    right turn :    

7-1 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:(angle<> 0.1 only center Camera add: +-10px,+-0.1 /  +-30px, +-0.3) 
    model !2!    
    lr 0.001       
    epoch: 9, Generator
    nomalization: mean/std 
    Creating image : False
    batch_size : 256
    strait zigzag (0~3) : 2
    first left turn : True
    enter brideg :True
    left turn :False
    right turn :    

7-2 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:(angle<> 0.1 only center Camera add: +-15px,+-0.15 
    model !2!     
    lr 0.001       
    epoch: 9, Generator
    nomalization: mean/std 

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 2
    first left turn : False
    enter brideg :
    left turn :
    right turn :

7-3 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:(angle<> 0.1 only center Camera add: +-15px,+-0.15 
    model !1!   
    lr 0.001       
    epoch: 9, Generator
    nomalization: mean/std 
    
    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 2
    first left turn : True
    enter brideg :True
    left turn :False
    right turn :        

7-4 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:
    model !1! 
    lr 0.001       
    epoch: 9, Generator
    nomalization: mean/std 

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 2
    first left turn : False
    enter brideg :
    left turn :
    right turn :    

7-4 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:
    model !5! W / b l1, l2 
    lr 0.001       
    epoch: 9, Generator
    nomalization: mean/std 

    batch_size : 256
    Creating image : False
    
    strait zigzag (0~3) : Only strait
    first left turn : 
    enter brideg :
    left turn :
    right turn :    

9-1 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:
    model !7! W l2 on desne
    lr 0.001       
    epoch: 9, Generator
    nomalization: mean/std 

    batch_size : 256
    Creating image : False
    
    strait zigzag (0~3) : 2
    first left turn : False
    enter brideg :
    left turn :
    right turn :    

9-1 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:
    model 1! 
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : False
    
    strait zigzag (0~3) : 1
    first left turn : False
    enter brideg :
    left turn :
    right turn :    

9-1 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:angle<> 0.1 only center Camera add: +-10px,+-0.1 /  +-30px, +-0.3) 
    model 1! 
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : False
    
    strait zigzag (0~3) : 1
    first left turn : True
    enter brideg :True
    left turn :True
    right turn :   True (crash side)
    final loss:0.0188


9-2 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:angle<> 0.1 only center Camera add: +-10px,+-0.1 /  +-30px, +-0.3) 
    model 1! 
    lr 0.001       
    epoch: 15, Generator
    nomalization: None

    batch_size : 256
    Creating image : False
    
    strait zigzag (0~3) : 1
    first left turn : True
    enter brideg :True
    left turn :True
    right turn :   True (little crash side)
    final loss:0.0243

9-3 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:angle<> 0.1 only center Camera add: +-10px,+-0.1 /  +-30px, +-0.3) 
    model 1! 
    lr 0.001       
    epoch: 11, Generator
    nomalization: Mean/Std

    batch_size : 256
    Creating image : False
    
    strait zigzag (0~3) : 3
    first left turn : False
    enter brideg :
    left turn :
    right turn : 
    final loss:0.002

9-3 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:angle<> 0.1 only center Camera add: +-10px,+-0.1 /  +-30px, +-0.3) 
    model 1! 
    lr 0.001       
    epoch: 11, Generator
    nomalization: Mean

    batch_size : 256
    Creating image : False
    
    strait zigzag (0~3) : 3
    first left turn : True
    enter brideg :True
    left turn :False
    right turn : 
    final loss:

9-4 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:angle<> 0.1 only center Camera add: +-10px,+-0.1 /  +-30px, +-0.3) 
    model 1! 
    lr 0.001       
    epoch: 11, Generator
    nomalization: /255-.5

    batch_size : 256
    Creating image : False
    
    strait zigzag (0~3) : 2
    first left turn : True
    enter brideg :True
    left turn :False
    right turn : 
    final loss:   

9-5 Image: Mydata/ Flip
    Augment:angle<> 0.1 only center Camera add: +-10px,+-0.1 /  +-30px, +-0.3) 
    model 1! 
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 1
    first left turn : False
    enter brideg :
    left turn :
    right turn :   
    final loss:0.0116

9-6 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:angle<> 0.1 only center Camera add: +-10px,+-0.1 /  +-30px, +-0.3) 
    model 2! 
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 1
    first left turn : True
    enter brideg :False
    left turn :
    right turn :
    final loss:0.03    

9-6 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment:angle<> 0.1 only center Camera add: +-10px,+-0.1 /  +-30px, +-0.3) 
    model 8!  l2
    lr 0.001       
    epoch: 15, Generator
    nomalization: None

    batch_size : 256
    Creating image : False
    
    strait zigzag (0~3) : 0
    first left turn : False
    enter brideg :
    left turn :
    right turn :
    final loss:1.718

9-7! Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment: only center Camera add: angle<>0.1 +-10px,+-0.15 / angle<>0.3  +-30px, +-0.35) 
    model 9!  l2
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 2
    first left turn : True
    enter brideg : True
    left turn :True
    right turn :True
    final loss:0.4252 

9-8 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment: only center Camera add: angle<>0.1 +-10px,+-0.15 / angle<>0.3  +-30px, +-0.35) 
    model 9!  l2
    lr 0.001       
    epoch: 15, Generator
    nomalization: None

    batch_size : 256
    Creating image : False
    
    strait zigzag (0~3) : 3
    first left turn : True
    enter brideg : True
    left turn :True
    right turn :True
    final loss:0.46

9-9 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment: only center Camera add: angle<>0.2 +-10px,+-0.2 / angle<>0.3  +-30px, +-0.35) 
    model 9!  l2
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 3
    first left turn : True
    enter brideg : True
    left turn :True
    right turn :True
    final loss:0.49

9-10 Image: Udacity (<>0.1 exclude) / +- 0.25 on left,right camera / Flip
    Augment: only center Camera add: angle<>0.1 +-10px,+-0.15 / angle<>0.3  +-30px, +-0.35) 
    model 9!  l2
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 3
    first left turn : False
    enter brideg : 
    left turn :
    right turn :
    final loss:

9-11 Image: Udacity (<>0.1 exclude) / +- 0.25 on left,right camera / Flip
    Augment: only center Camera add: angle<>0.3  +-30px, +-0.35) 
    model 9!  l2
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 3
    first left turn : False
    enter brideg : 
    left turn :
    right turn :
    final loss:

9-12 Image: Udacity  / +- 0.25 on left,right camera / Flip
    Augment: only center Camera add:  / angle<>0.3  +-30px, +-0.35) 
    model 10!  l2 x 2
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : False
    first left turn : 
    enter brideg : 
    left turn :
    right turn :
    final loss:

9-13 Image: Udacity  / +- 0.25 on left,right camera / Flip
    Augment: only center Camera add:  / angle<>0.3  +-30px, +-0.35) 
    model 9!  l2
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 3
    first left turn : True
    enter brideg : True
    left turn :True
    right turn :True
    final loss:   0.5

9-14 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment: only center Camera add: angle<>0.1 +-10px,+-0.05 / angle<>0.3  +-30px, +-0.35) 
    model 9!  l2
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 2
    first left turn : True
    enter brideg : True
    left turn : True(crash)
    right turn :False
    final loss:0.4985     

9-14 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment: only center Camera add: angle<>0.1 +-10px,+-0.15 / angle<>0.3  +-30px, +-0.35) 
    model 9!  l2(0.05)
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 2
    first left turn : True
    enter brideg : True
    left turn : True(crash)
    right turn :True
    final loss:1.881        

9-14 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment: only center Camera add: angle<>0.1 +-10px,+-0.15 / angle<>0.3  +-30px, +-0.35) 
    model 9!  l2(0.2)
    lr 0.001       
    epoch: 11, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 3
    first left turn : False
    enter brideg : 
    left turn : 
    right turn :
    final loss:2.995

9-15 Image: Udacity / +- 0.25 on left,right camera / Flip
    Augment: only center Camera add: angle<>0.1 +-10px,+-0.15 / angle<>0.3  +-30px, +-0.35) 
    model 9!  l2
    lr 0.001       
    epoch: 20, Generator
    nomalization: None

    batch_size : 256
    Creating image : True
    
    strait zigzag (0~3) : 2
    first left turn : True for new simulator
    enter brideg : True
    left turn :True
    right turn :True
    final loss:0.3